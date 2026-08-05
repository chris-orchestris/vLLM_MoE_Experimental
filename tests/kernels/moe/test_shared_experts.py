# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def test_discard_orphaned_output_clears_only_the_current_slot():
    shared_experts = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts.enable_dbo = False
    orphan = torch.ones(1)
    other_slot = torch.ones(1) * 2
    shared_experts._output = [orphan, other_slot]

    assert shared_experts.discard_orphaned_output()
    assert shared_experts._output[0] is None
    assert shared_experts._output[1] is other_slot
    assert not shared_experts.discard_orphaned_output()


def test_prefer_aux_stream_over_modular_kernel_overlap(monkeypatch):
    shared_experts = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts._moe_config = type(
        "MoEConfig",
        (),
        {
            "moe_parallel_config": type(
                "ParallelConfig",
                (),
                {
                    "enable_eplb": False,
                    "all2all_backend": "allgather_reducescatter",
                    "use_fi_nvl_two_sided_kernels": False,
                },
            )()
        },
    )()
    shared_experts._stream = object()
    shared_experts._mk_can_overlap_shared_experts = lambda: True
    hidden_states = torch.empty(8, 1)
    monkeypatch.setattr(shared_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")

    monkeypatch.delenv("VLLM_MOE_PREFER_SHARED_EXPERTS_STREAM", raising=False)
    assert (
        shared_experts._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )

    monkeypatch.setenv("VLLM_MOE_PREFER_SHARED_EXPERTS_STREAM", "1")
    assert (
        shared_experts._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    assert (
        shared_experts._determine_shared_experts_order(torch.empty(9, 1))
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )
