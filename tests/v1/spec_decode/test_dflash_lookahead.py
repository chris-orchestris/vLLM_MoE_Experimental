# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.models.deepseek_v4.ampere.dspark import DSparkDeepseekV4ForCausalLM
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.input_batch import get_speculative_draft_valid_mask
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    confidence_budgeted_prefix_lengths,
    confidence_prefix_lengths,
    parse_confidence_batch_thresholds,
    parse_confidence_draft_budgets,
)
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3


def _dflash_speculative_config(num_speculative_tokens: int) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )


def _create_dflash_scheduler(num_speculative_tokens: int) -> Scheduler:
    speculative_config = _dflash_speculative_config(num_speculative_tokens)
    model_config = speculative_config.target_model_config
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dflash_prefill_reserves_lookahead_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    assert scheduler.num_lookahead_tokens == NUM_SPECULATIVE_TOKENS + 1

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    # prefill block + one lookahead block
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


def test_dflash_first_prefill_query_window_fits_allocated_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    query_positions = range(BLOCK_SIZE, BLOCK_SIZE + scheduler.num_lookahead_tokens)

    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in query_positions)


def test_dflash_drafter_window_reserves_bonus_token():
    # DFlash's drafter window is num_spec + 1 (the extra slot is the bonus token),
    # so max_seq_len + num_spec + 1 must stay within the draft model's max len.
    input_fits_in_drafter = GPUModelRunner._input_fits_in_drafter
    dflash_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=_dflash_speculative_config(NUM_SPECULATIVE_TOKENS),
    )
    # window = 4, so 96 fits (96 + 4 == 100) but 97 does not (97 + 4 == 101)
    assert input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=96))
    assert not input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=97))
    assert not input_fits_in_drafter(dflash_runner, None)  # no metadata

    # Other drafters don't reserve the bonus token, so 97 fits (97 + 3 == 100).
    plain_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    assert input_fits_in_drafter(plain_runner, SimpleNamespace(max_seq_len=97))


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_uses_first_failure_as_causal_prefix():
    confidence = torch.tensor(
        [
            [0.9, 0.8, 0.4, 0.95, 0.99],
            [0.7, 0.6, 0.5, 0.4, 0.3],
            [0.1, 0.9, 0.9, 0.9, 0.9],
        ]
    )

    lengths = confidence_prefix_lengths(confidence, 0.5)

    assert lengths.tolist() == [2, 3, 0]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_preserves_k_below_minimum_batch_size():
    confidence = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]])

    lengths = confidence_prefix_lengths(confidence, threshold=0.5, min_batch_size=2)

    assert lengths.tolist() == [5]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_budget_selects_best_causal_prefixes():
    confidence = torch.tensor(
        [
            [0.9, 0.9, 0.9, 0.9, 0.9],
            [0.8, 0.8, 0.8, 0.8, 0.8],
            [0.7, 0.7, 0.7, 0.7, 0.7],
        ]
    )
    lengths = torch.full((3,), 5, dtype=torch.int32)

    budgeted = confidence_budgeted_prefix_lengths(confidence, lengths, 13)

    assert budgeted.tolist() == [5, 5, 3]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_draft_budget_parser():
    assert parse_confidence_draft_budgets("2:9,3:13", 4, 5) == {2: 9, 3: 13}
    with pytest.raises(ValueError, match="batch 3 must be in"):
        parse_confidence_draft_budgets("3:16", 4, 5)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_batch_threshold_parser():
    assert parse_confidence_batch_thresholds("2:0.5,3:0", 4) == {
        2: 0.5,
        3: 0.0,
    }
    with pytest.raises(ValueError, match="batch 3 must be in"):
        parse_confidence_batch_thresholds("3:1.1", 4)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_dspark_confidence_checkpoint_name_maps_to_loaded_head():
    model = object.__new__(DSparkDeepseekV4ForCausalLM)
    model.num_dspark_layers = 3

    mapped = model._remap_dspark_name("mtp.2.confidence_head.proj.weight")

    assert mapped == "model.confidence_head.proj.weight"
    assert model._remap_dspark_name("mtp.1.confidence_head.proj.weight") is None


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_draft_token_handler_returns_variable_placeholder_lengths():
    handler = object.__new__(DraftTokensHandler)
    handler.req_ids = ["first", "second", "third"]
    handler.num_draft_tokens = 5
    handler.draft_tokens_np = None
    handler.draft_lengths_np = np.array([5, 2, 0], dtype=np.int32)
    handler.copy_event = SimpleNamespace(synchronize=lambda: None)

    output = handler.get_draft_tokens()

    assert output.req_ids == handler.req_ids
    assert output.draft_token_ids == [[-1] * 5, [-1] * 2, []]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_async_draft_lengths_follow_reordered_request_state_indices():
    draft_lengths = torch.tensor([2, 5, 0], dtype=torch.int32)
    expanded_idx_mapping = torch.tensor(
        [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    expanded_local_pos = torch.arange(6).repeat(3)

    valid = get_speculative_draft_valid_mask(
        draft_lengths, expanded_idx_mapping, expanded_local_pos
    )

    assert valid.view(3, 6).sum(dim=1).tolist() == [1, 3, 6]
