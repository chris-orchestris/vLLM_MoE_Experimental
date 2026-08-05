# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


def confidence_prefix_lengths(
    confidence: torch.Tensor,
    threshold: float | torch.Tensor,
    min_batch_size: int = 1,
) -> torch.Tensor:
    """Return the causal prefix ending before the first low-confidence token."""
    num_steps = confidence.shape[1]
    if confidence.shape[0] < min_batch_size:
        return torch.full(
            (confidence.shape[0],),
            num_steps,
            dtype=torch.int32,
            device=confidence.device,
        )
    positions = torch.arange(
        num_steps, dtype=torch.int32, device=confidence.device
    ).unsqueeze(0)
    return torch.where(confidence < threshold, positions, num_steps).amin(dim=1)


def confidence_budgeted_prefix_lengths(
    confidence: torch.Tensor, lengths: torch.Tensor, budget: int
) -> torch.Tensor:
    """Select the highest-value causal draft prefixes within a batch budget."""
    num_reqs, num_steps = confidence.shape
    num_candidates = num_reqs * num_steps
    if budget >= num_candidates:
        return lengths
    if budget == 0:
        return torch.zeros_like(lengths)

    positions = torch.arange(
        num_steps, dtype=torch.int32, device=confidence.device
    ).unsqueeze(0)
    eligible = positions < lengths.unsqueeze(1)
    marginal_survival = confidence.cumprod(dim=1).masked_fill(~eligible, -1)
    selected_indices = marginal_survival.flatten().topk(budget).indices
    selected = torch.zeros(num_candidates, dtype=torch.bool, device=confidence.device)
    selected.scatter_(0, selected_indices, True)
    return (selected.view_as(eligible) & eligible).sum(dim=1, dtype=torch.int32)


def parse_confidence_draft_budgets(
    value: str, max_num_reqs: int, num_steps: int
) -> dict[int, int]:
    """Parse comma-separated ``batch_size:total_drafts`` budget entries."""
    budgets: dict[int, int] = {}
    if not value:
        return budgets
    for entry in value.split(","):
        try:
            batch_text, budget_text = entry.split(":", maxsplit=1)
            batch_size = int(batch_text)
            budget = int(budget_text)
        except ValueError as error:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_DRAFT_BUDGETS must contain "
                "batch_size:total_drafts entries"
            ) from error
        if not 1 <= batch_size <= max_num_reqs:
            raise ValueError(
                "DSpark confidence draft-budget batch size must be in "
                f"[1, {max_num_reqs}], got {batch_size}"
            )
        if not 0 <= budget <= batch_size * num_steps:
            raise ValueError(
                f"DSpark confidence draft budget for batch {batch_size} must "
                f"be in [0, {batch_size * num_steps}], got {budget}"
            )
        if batch_size in budgets:
            raise ValueError(
                f"Duplicate DSpark confidence draft budget for batch {batch_size}"
            )
        budgets[batch_size] = budget
    return budgets


def parse_confidence_batch_thresholds(
    value: str, max_num_reqs: int
) -> dict[int, float]:
    """Parse comma-separated ``batch_size:threshold`` override entries."""
    thresholds: dict[int, float] = {}
    if not value:
        return thresholds
    for entry in value.split(","):
        try:
            batch_text, threshold_text = entry.split(":", maxsplit=1)
            batch_size = int(batch_text)
            threshold = float(threshold_text)
        except ValueError as error:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_BATCH_THRESHOLDS must contain "
                "batch_size:threshold entries"
            ) from error
        if not 1 <= batch_size <= max_num_reqs:
            raise ValueError(
                "DSpark confidence threshold batch size must be in "
                f"[1, {max_num_reqs}], got {batch_size}"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"DSpark confidence threshold for batch {batch_size} must be "
                f"in [0, 1], got {threshold}"
            )
        if batch_size in thresholds:
            raise ValueError(
                f"Duplicate DSpark confidence threshold for batch {batch_size}"
            )
        thresholds[batch_size] = threshold
    return thresholds


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        self.confidence_threshold = envs.VLLM_DSPARK_CONFIDENCE_THRESHOLD
        self.confidence_min_batch_size = envs.VLLM_DSPARK_CONFIDENCE_MIN_BATCH_SIZE
        self.confidence_draft_budgets = parse_confidence_draft_budgets(
            envs.VLLM_DSPARK_CONFIDENCE_DRAFT_BUDGETS,
            self.max_num_reqs,
            self.num_speculative_steps,
        )
        self.confidence_batch_thresholds = parse_confidence_batch_thresholds(
            envs.VLLM_DSPARK_CONFIDENCE_BATCH_THRESHOLDS,
            self.max_num_reqs,
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("VLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1]")
        if not 1 <= self.confidence_min_batch_size <= self.max_num_reqs:
            raise ValueError(
                "VLLM_DSPARK_CONFIDENCE_MIN_BATCH_SIZE must be in "
                f"[1, {self.max_num_reqs}]"
            )
        confidence_thresholds = [self.confidence_threshold] * (self.max_num_reqs + 1)
        for batch_size, threshold in self.confidence_batch_thresholds.items():
            confidence_thresholds[batch_size] = threshold
        self.confidence_thresholds = torch.tensor(
            confidence_thresholds,
            dtype=torch.float32,
            device=device,
        )
        confidence_enabled = bool(self.confidence_draft_budgets) or any(
            confidence_thresholds[batch_size] > 0
            for batch_size in range(
                self.confidence_min_batch_size, self.max_num_reqs + 1
            )
        )
        if confidence_enabled:
            block_size = (
                getattr(self.draft_model_config.hf_config, "dspark_block_size", 0) or 0
            )
            if self.num_speculative_steps != block_size:
                raise ValueError(
                    "DSpark confidence scheduling requires num_speculative_tokens "
                    f"to equal dspark_block_size ({block_size}); got "
                    f"{self.num_speculative_steps}"
                )
            self.draft_lengths: torch.Tensor | None = torch.full(
                (self.max_num_reqs,),
                self.num_speculative_steps,
                dtype=torch.int32,
                device=device,
            )
        else:
            self.draft_lengths = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if (
            self.draft_lengths is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        ):
            logger.warning(
                "DSpark confidence scheduling requires the real request count; "
                "running the draft eagerly instead of padding a full CUDA graph."
            )
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if self.draft_lengths is not None and not getattr(
            model, "confidence_head_loaded", False
        ):
            raise ValueError(
                "DSpark confidence scheduling requires a loaded confidence head"
            )
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        use_confidence = (
            self.draft_lengths is not None
            and num_reqs >= self.confidence_min_batch_size
        )
        markov_embeds: list[torch.Tensor] | None = [] if use_confidence else None

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if markov_embeds is not None:
                markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.draft_lengths is not None and markov_embeds is None:
            self.draft_lengths[:num_reqs].fill_(n_spec)
        elif markov_embeds is not None:
            confidence = self.model.compute_confidence(
                sample_hidden.view(num_reqs, n_spec, -1),
                torch.stack(markov_embeds, dim=1),
            )
            assert self.draft_lengths is not None
            lengths = confidence_prefix_lengths(
                confidence,
                self.confidence_thresholds[num_reqs],
                self.confidence_min_batch_size,
            )
            budget = self.confidence_draft_budgets.get(num_reqs)
            if budget is not None:
                lengths = confidence_budgeted_prefix_lengths(
                    confidence, lengths, budget
                )
            lengths = get_tp_group().broadcast(lengths, src=0)
            self.draft_lengths[:num_reqs].copy_(lengths)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        with record_function_or_nullcontext("dspark: backbone"):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with record_function_or_nullcontext("dspark: markov_sample"):
            self._sample_sequential(num_reqs, head_hidden)
