# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_ampere_dspark_does_not_request_native_mtp_hidden_buffer():
    from vllm.models.deepseek_v4.ampere.model import _needs_mtp_hidden_states

    dspark = SimpleNamespace(
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
    )
    eagle = SimpleNamespace(
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
    )
    draft_model = SimpleNamespace(
        use_eagle=lambda: False,
        uses_draft_model=lambda: True,
    )

    assert not _needs_mtp_hidden_states(None)
    assert not _needs_mtp_hidden_states(dspark)
    assert _needs_mtp_hidden_states(eagle)
    assert _needs_mtp_hidden_states(draft_model)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_ampere_releases_nonfinal_aux_reconstruction(monkeypatch: pytest.MonkeyPatch):
    from vllm.models.deepseek_v4.ampere import model as ampere_model

    recon_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    num_tokens = 2
    hidden_size = 8
    hc_mult = 4

    class FakeLayer:
        def __init__(self, index: int):
            self.index = index

        def __call__(self, hidden_states, positions, input_ids, *mhc_state):
            if self.index > 0:
                assert recon_refs[self.index - 1]() is None
            return (
                torch.full((num_tokens, hidden_size), float(self.index)),
                torch.zeros(num_tokens, hc_mult, hidden_size),
                torch.zeros(num_tokens, hc_mult, 1),
                torch.zeros(num_tokens, hc_mult, hc_mult),
            )

        def hc_post(self, hidden_states, residual, post_mix, res_mix):
            reconstruction = torch.full(
                (num_tokens, hc_mult, hidden_size),
                float(self.index + 1),
            )
            recon_refs.append(weakref.ref(reconstruction))
            return reconstruction

    monkeypatch.setattr(
        ampere_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(ampere_model, "_nan_probe", lambda *args: None)
    monkeypatch.setattr(ampere_model, "_NAN_PASS", [0])

    model = SimpleNamespace(
        use_mega_moe=False,
        hc_mult=hc_mult,
        start_layer=0,
        end_layer=3,
        layers=[FakeLayer(index) for index in range(3)],
        aux_hidden_state_layers=(1, 2, 3),
        _mtp_hidden_buffer=None,
        hc_head_op=lambda hidden_states, *args: hidden_states.mean(dim=1),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        norm=lambda hidden_states: hidden_states,
    )

    with torch.inference_mode():
        output, aux_hidden_states = ampere_model.DeepseekV4Model.forward(
            model,
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros(num_tokens, hidden_size),
        )

    assert len(recon_refs) == 3
    assert all(reference() is None for reference in recon_refs)
    assert [state[0, 0].item() for state in aux_hidden_states] == [1.0, 2.0, 3.0]
    assert output[0, 0].item() == 3.0


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
