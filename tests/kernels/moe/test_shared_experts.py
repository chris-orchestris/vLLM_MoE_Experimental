# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
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
