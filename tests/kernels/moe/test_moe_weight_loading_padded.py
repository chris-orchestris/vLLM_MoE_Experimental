# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FusedMoEFactory weight loading with padded hidden dimensions.

When using DeepEP backends or NIXL EP with models like nemotron_h,
hidden_size may be rounded up (e.g., 2688 -> 3072) for backend requirements.
Weight parameters are created with the padded size, but checkpoint weights
have the original unpadded size. These tests verify that weight loading
correctly handles this mismatch.
"""

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def _make_uneven_tp_config(tp_rank: int) -> FusedMoEConfig:
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(), tp_size=4, tp_rank=tp_rank
    )
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size=2048,
        num_local_experts=256,
        num_logical_experts=256,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )


class TestUnevenTpIntermediatePartitions:
    def test_config_selects_rank_specific_width_and_start(self, monkeypatch):
        monkeypatch.setenv("VLLM_MOE_TP_INTERMEDIATE_PARTITIONS", "640,640,384,384")
        config = _make_uneven_tp_config(tp_rank=2)

        assert config.tp_intermediate_partition_sizes == (640, 640, 384, 384)
        assert config.tp_intermediate_partition_start == 1280
        assert config.intermediate_size_per_partition == 384
        assert config.intermediate_size_per_partition_unpadded == 384

    @pytest.mark.parametrize(
        ("tp_rank", "expected_start", "expected_width"),
        [
            (0, 0, 576),
            (1, 576, 576),
            (2, 1152, 448),
            (3, 1600, 448),
        ],
    )
    def test_config_accepts_64_aligned_topology_partitions(
        self, monkeypatch, tp_rank, expected_start, expected_width
    ):
        monkeypatch.setenv("VLLM_MOE_TP_INTERMEDIATE_PARTITIONS", "576,576,448,448")
        config = _make_uneven_tp_config(tp_rank=tp_rank)

        assert config.tp_intermediate_partition_sizes == (576, 576, 448, 448)
        assert config.tp_intermediate_partition_start == expected_start
        assert config.intermediate_size_per_partition == expected_width
        assert config.intermediate_size_per_partition_unpadded == expected_width

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("640,640,384", "one width per TP rank"),
            ("640,640,512,384", "must sum"),
            ("600,600,424,424", "multiples of 64"),
        ],
    )
    def test_config_rejects_unsafe_partition_specs(self, monkeypatch, spec, message):
        monkeypatch.setenv("VLLM_MOE_TP_INTERMEDIATE_PARTITIONS", spec)
        with pytest.raises(ValueError, match=message):
            _make_uneven_tp_config(tp_rank=0)

    @pytest.mark.parametrize(
        ("checkpoint_extent", "expected"),
        [
            (2048, (1280, 384)),
            (1024, (640, 192)),
            (64, (40, 12)),
        ],
    )
    def test_packed_checkpoint_bounds_are_proportional(
        self, monkeypatch, checkpoint_extent, expected
    ):
        monkeypatch.setenv("VLLM_MOE_TP_INTERMEDIATE_PARTITIONS", "640,640,384,384")
        layer = object.__new__(RoutedExperts)
        layer.moe_config = _make_uneven_tp_config(tp_rank=2)

        assert layer._tp_intermediate_shard_bounds(checkpoint_extent, 2) == expected

    def test_w13_and_packed_w2_load_the_same_logical_slice(self, monkeypatch):
        monkeypatch.setenv("VLLM_MOE_TP_INTERMEDIATE_PARTITIONS", "640,640,384,384")
        layer = object.__new__(RoutedExperts)
        layer.moe_config = _make_uneven_tp_config(tp_rank=2)

        full_w13 = torch.arange(2048 * 2, dtype=torch.float32).reshape(2048, 2)
        local_w13 = torch.zeros(2 * 384, 2)
        layer._load_w13(local_w13, 0, "w1", full_w13, tp_rank=2)
        assert torch.equal(local_w13[:384], full_w13[1280:1664])
        assert torch.count_nonzero(local_w13[384:]) == 0

        full_w2_packed = torch.arange(2 * 1024, dtype=torch.float32).reshape(2, 1024)
        local_w2_packed = torch.zeros(2, 192)
        layer._load_w2(local_w2_packed, 1, full_w2_packed, tp_rank=2)
        assert torch.equal(local_w2_packed, full_w2_packed[:, 640:832])


class TestGetHiddenDim:
    """Unit tests for _get_hidden_dim."""

    def test_2d_non_transposed_w2(self):
        # w2: shard_dim=1 (intermediate), hidden=0
        assert RoutedExperts._get_hidden_dim(shard_dim=1, ndim=2) == 0

    def test_2d_non_transposed_w13(self):
        # w1/w3: shard_dim=0 (intermediate), hidden=1
        assert RoutedExperts._get_hidden_dim(shard_dim=0, ndim=2) == 1

    def test_2d_transposed_w2(self):
        # transposed w2: shard_dim=0, hidden=1
        assert RoutedExperts._get_hidden_dim(shard_dim=0, ndim=2) == 1

    def test_2d_transposed_w13(self):
        # transposed w1/w3: shard_dim=1, hidden=0
        assert RoutedExperts._get_hidden_dim(shard_dim=1, ndim=2) == 0

    def test_3d_non_transposed_w2(self):
        # 3D w2: shard_dim=2, hidden=1
        assert RoutedExperts._get_hidden_dim(shard_dim=2, ndim=3) == 1

    def test_3d_non_transposed_w13(self):
        # 3D w1/w3: shard_dim=1, hidden=2
        assert RoutedExperts._get_hidden_dim(shard_dim=1, ndim=3) == 2

    def test_3d_transposed_w2(self):
        # transposed 3D w2: shard_dim=1, hidden=2
        assert RoutedExperts._get_hidden_dim(shard_dim=1, ndim=3) == 2

    def test_3d_transposed_w13(self):
        # transposed 3D w1/w3: shard_dim=2, hidden=1
        assert RoutedExperts._get_hidden_dim(shard_dim=2, ndim=3) == 1

    def test_1d_returns_zero(self):
        # 1D per-channel scales: always returns 0
        assert RoutedExperts._get_hidden_dim(shard_dim=0, ndim=1) == 0
        assert RoutedExperts._get_hidden_dim(shard_dim=1, ndim=1) == 0

    def test_invalid_shard_dim_raises(self):
        # shard_dim outside the data dimensions should raise
        with pytest.raises(ValueError, match="not a valid data dimension"):
            RoutedExperts._get_hidden_dim(shard_dim=0, ndim=3)


class TestOrientFusedWeight:
    """Unit tests for _orient_fused_weight.

    E=8 experts, hidden=3072, intermediate=1024.
    """

    HIDDEN = 3072

    def test_w13_standard_orientation_is_untouched(self):
        weight = torch.randn(8, 2048, self.HIDDEN)
        result = RoutedExperts._orient_fused_weight(weight, "w1", self.HIDDEN)
        assert result.shape == (8, 2048, self.HIDDEN)

    def test_w13_transposed_checkpoint_is_normalised(self):
        # e.g. Qwen3 VL MoE stores [experts, hidden, 2 * intermediate]
        weight = torch.randn(8, self.HIDDEN, 2048)
        result = RoutedExperts._orient_fused_weight(weight, "w3", self.HIDDEN)
        assert result.shape == (8, 2048, self.HIDDEN)

    def test_w2_standard_orientation_is_untouched(self):
        weight = torch.randn(8, self.HIDDEN, 1024)
        result = RoutedExperts._orient_fused_weight(weight, "w2", self.HIDDEN)
        assert result.shape == (8, self.HIDDEN, 1024)

    def test_w2_transposed_checkpoint_is_normalised(self):
        weight = torch.randn(8, 1024, self.HIDDEN)
        result = RoutedExperts._orient_fused_weight(weight, "w2", self.HIDDEN)
        assert result.shape == (8, self.HIDDEN, 1024)

    def test_w13_per_channel_scale_is_untouched(self):
        # A fused per-channel scale has no hidden dim, so transposing it would
        # leave chunk()/TP sharding operating on the wrong axis.
        scale = torch.randn(8, 2048, 1)
        result = RoutedExperts._orient_fused_weight(scale, "w1", self.HIDDEN)
        assert result.shape == (8, 2048, 1)
        assert result.chunk(2, dim=1)[0].shape == (8, 1024, 1)

    def test_w2_per_channel_scale_is_untouched(self):
        scale = torch.randn(8, self.HIDDEN, 1)
        result = RoutedExperts._orient_fused_weight(scale, "w2", self.HIDDEN)
        assert result.shape == (8, self.HIDDEN, 1)

    def test_block_scale_is_untouched(self):
        # Block scales are [experts, 2 * intermediate / block, hidden / block]
        scale = torch.randn(8, 16, 24)
        result = RoutedExperts._orient_fused_weight(scale, "w1", self.HIDDEN)
        assert result.shape == (8, 16, 24)


class TestNarrowExpertDataForPadding:
    """Unit tests for _narrow_expert_data_for_padding."""

    def test_no_narrowing_when_shapes_match(self):
        expert_data = torch.zeros(1024, 1024)
        loaded_weight = torch.randn(1024, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == loaded_weight.shape
        assert result.data_ptr() == expert_data.data_ptr()

    def test_narrow_w2_hidden_dim(self):
        # w2: (hidden_size, intermediate_size) - hidden_size padded at dim 0
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688, 1024)

    def test_narrow_w13_hidden_dim(self):
        # w1/w3: (intermediate_size, hidden_size) - hidden_size padded at dim 1
        expert_data = torch.zeros(2048, 3072)
        loaded_weight = torch.randn(2048, 2688)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=1
        )
        assert result.shape == (2048, 2688)

    def test_narrow_transposed_w2(self):
        # transposed w2: (intermediate_size, hidden_size) - hidden at dim 1
        expert_data = torch.zeros(1024, 3072)
        loaded_weight = torch.randn(1024, 2688)
        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=0, ndim=2)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=hidden_dim
        )
        assert result.shape == (1024, 2688)

    def test_narrow_3d_full_load(self):
        # 3D tensor for full_load path: w2 (num_experts, hidden_size, intermediate)
        expert_data = torch.zeros(8, 3072, 1024)
        loaded_weight = torch.randn(8, 2688, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=1
        )
        assert result.shape == (8, 2688, 1024)

    def test_narrow_1d_scale(self):
        # 1D scale tensor: per-channel w2 scale (hidden_size,)
        expert_data = torch.zeros(3072)
        loaded_weight = torch.randn(2688)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688,)

    def test_scalar_weight_no_op(self):
        # 0-dim tensor should be a no-op
        expert_data = torch.zeros(3072)
        loaded_weight = torch.tensor(1.0)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        # ndim == 0, so no narrowing
        assert result.shape == (3072,)

    def test_no_narrowing_when_loaded_weight_larger(self):
        # Guard: don't narrow if loaded_weight is larger than expert_data
        expert_data = torch.zeros(2688, 1024)
        loaded_weight = torch.randn(3072, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        assert result.shape == (2688, 1024)
        assert result.data_ptr() == expert_data.data_ptr()

    def test_negative_hidden_dim_is_noop(self):
        # Negative hidden_dim should be a safe no-op (0 <= check)
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=-1
        )
        # -1 fails the 0 <= check, so no narrowing
        assert result.shape == (3072, 1024)
        assert result.data_ptr() == expert_data.data_ptr()

    def test_only_narrows_hidden_dim(self):
        # Verify that only the specified hidden_dim is narrowed,
        # even when other dimensions also differ
        expert_data = torch.zeros(3072, 2048)
        loaded_weight = torch.randn(2688, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        # Only dim 0 (hidden) should be narrowed; dim 1 stays at 2048
        assert result.shape == (2688, 2048)

    def test_narrowed_data_shares_storage(self):
        # Verify narrowing returns a view (writes go to original tensor)
        expert_data = torch.zeros(3072, 1024)
        loaded_weight = torch.randn(2688, 1024)
        result = RoutedExperts._narrow_expert_data_for_padding(
            expert_data, loaded_weight, hidden_dim=0
        )
        result.copy_(loaded_weight)
        # The first 2688 rows of expert_data should now have loaded_weight
        assert torch.equal(expert_data[:2688, :], loaded_weight)
        # Padded region should remain zero
        assert torch.equal(expert_data[2688:, :], torch.zeros(3072 - 2688, 1024))


class TestWeightLoadingWithPaddedHiddenSize:
    """Integration-style tests that simulate padded weight loading."""

    def test_load_w2_with_padding(self):
        """Simulate loading w2 weights when hidden_size is padded."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        expert_data_full = torch.zeros(padded_hidden, intermediate)
        loaded_weight = torch.randn(original_hidden, intermediate)

        # w2 non-transposed: shard_dim=1, hidden_dim=0
        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=1, ndim=2)
        expert_data = RoutedExperts._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:original_hidden, :], loaded_weight)
        assert torch.equal(
            expert_data_full[original_hidden:, :],
            torch.zeros(padded_hidden - original_hidden, intermediate),
        )

    def test_load_w13_with_padding(self):
        """Simulate loading w1/w3 weights when hidden_size is padded."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        # w1/w3: (intermediate_size, hidden_size)
        expert_data_full = torch.zeros(intermediate, padded_hidden)
        loaded_weight = torch.randn(intermediate, original_hidden)

        # w1 non-transposed: shard_dim=0, hidden_dim=1
        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=0, ndim=2)
        expert_data = RoutedExperts._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:, :original_hidden], loaded_weight)
        assert torch.equal(
            expert_data_full[:, original_hidden:],
            torch.zeros(intermediate, padded_hidden - original_hidden),
        )

    def test_load_transposed_w2_with_padding(self):
        """Simulate loading transposed w2 (GPTQ) with padded hidden_size."""
        padded_hidden = 3072
        original_hidden = 2688
        intermediate = 1024

        # transposed w2: (intermediate_size, hidden_size), shard_dim=0
        expert_data_full = torch.zeros(intermediate, padded_hidden)
        loaded_weight = torch.randn(intermediate, original_hidden)

        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=0, ndim=2)
        expert_data = RoutedExperts._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full[:, :original_hidden], loaded_weight)

    def test_no_padding_is_noop(self):
        """Verify that when sizes match, behavior is unchanged."""
        hidden = 2048
        intermediate = 1024

        expert_data_full = torch.zeros(hidden, intermediate)
        loaded_weight = torch.randn(hidden, intermediate)

        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=1, ndim=2)
        expert_data = RoutedExperts._narrow_expert_data_for_padding(
            expert_data_full, loaded_weight, hidden_dim=hidden_dim
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(expert_data_full, loaded_weight)

    def test_narrow_shard_dim(self):
        """Simulate loading w2 when both hidden_size and intermediate_size
        are padded.
        """
        padded_hidden = 3072
        original_hidden = 2688
        padded_intermediate = 1024
        original_intermediate = 896

        expert_data_full = torch.zeros(padded_hidden, padded_intermediate)
        loaded_weight = torch.randn(original_hidden, original_intermediate)

        shard_dim = 1
        hidden_dim = RoutedExperts._get_hidden_dim(shard_dim=shard_dim, ndim=2)
        expert_data = RoutedExperts._narrow_expert_data_for_padding(
            expert_data_full,
            loaded_weight,
            hidden_dim=hidden_dim,
            shard_dim=shard_dim,
        )
        expert_data.copy_(loaded_weight)

        assert torch.equal(
            expert_data_full[:original_hidden, :original_intermediate],
            loaded_weight,
        )
        assert torch.equal(
            expert_data_full[original_hidden:, :],
            torch.zeros(padded_hidden - original_hidden, padded_intermediate),
        )
        assert torch.equal(
            expert_data_full[:original_hidden, original_intermediate:],
            torch.zeros(original_hidden, padded_intermediate - original_intermediate),
        )

    def test_bnb_shape_mismatch_raises(self):
        """BnB + padded hidden_size should raise via weight_loader."""
        from unittest.mock import MagicMock

        num_experts = 1
        padded_packed = 3072  # padded packed size
        original_packed = 2688  # original packed size

        # Build a param that looks like a BnB 4-bit MoE weight.
        param_data = torch.zeros(num_experts, padded_packed, 1, dtype=torch.uint8)
        param = torch.nn.Parameter(param_data, requires_grad=False)
        param.use_bitsandbytes_4bit = True

        loaded_weight = torch.randint(0, 255, (original_packed, 1), dtype=torch.uint8)

        # Minimal RoutedExperts mock so weight_loader reaches the BnB path.
        moe = MagicMock(spec=RoutedExperts)
        moe.quant_config = None
        moe.quant_method = MagicMock()
        moe.quant_method.__class__.__name__ = "BitsAndBytesMethod"
        moe._expert_map = None
        moe.tp_rank = 0

        # Call the real weight_loader (unbound) with our mock as self.
        with pytest.raises(ValueError, match="BitsAndBytes"):
            RoutedExperts.weight_loader(
                moe,
                param,
                loaded_weight,
                weight_name="w2",
                shard_id="w2",
                expert_id=0,
            )


class TestPerTensorScaleCoercion:
    """Regression test for shape-(1,) per-tensor scales (issue #43297).

    llm-compressor NVFP4 emits per-tensor weight and input scales as
    shape-(1,) tensors. `_to_scalar` collapses them to a 0-D scalar so the
    scalar-slot assignments in the weight loader neither broadcast nor raise.
    """

    def test_collapses_to_scalar(self):
        # shape-(1,) and 0-D both reduce to a 0-D scalar.
        for loaded_weight in (torch.tensor([0.5]), torch.tensor(0.5)):
            scalar = RoutedExperts._to_scalar(loaded_weight)
            assert scalar.shape == ()
            assert scalar.item() == pytest.approx(0.5)

    def test_rejects_non_scalar(self):
        # numel > 1 must fail loudly instead of silently picking an element.
        with pytest.raises(RuntimeError):
            RoutedExperts._to_scalar(torch.tensor([0.1, 0.2]))
