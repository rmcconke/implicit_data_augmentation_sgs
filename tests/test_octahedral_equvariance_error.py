import pytest
import torch
import torch.nn as nn

import sys
sys.path.append("./src/")
from eval import octahedral_equivariance_error

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=[1, 3, 6, 9])
def num_channels(request):
    return request.param


@pytest.fixture
def sample_input(device, num_channels):
    """Standard 3D input: [B, C, D, H, W] with cubic spatial dims."""
    torch.manual_seed(42)
    return torch.randn(4, num_channels, 8, 8, 8, device=device)


# ============================================================
# Mock Models
# ============================================================

class EquivariantModel(nn.Module):
    """Perfectly equivariant: scalar multiplication."""
    def forward(self, x):
        return x * 2.0


class NonEquivariantModel(nn.Module):
    """Deliberately breaks equivariance by scaling first spatial slice."""
    def forward(self, x):
        x = x.clone()
        x[:, :, 0, :, :] *= 2.0
        return x


class LearnableEquivariantModel(nn.Module):
    """Channel-wise scaling (equivariant)."""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x):
        return x * self.scale


class AddConstantModel(nn.Module):
    """Adds a constant (equivariant for scalars, not for vectors)."""
    def forward(self, x):
        return x + 1.0


class IdentityModel(nn.Module):
    """Identity function - perfectly equivariant."""
    def forward(self, x):
        return x


# ============================================================
# Basic Functionality Tests
# ============================================================

class TestBasicFunctionality:

    def test_output_shape(self, sample_input, num_channels):
        """Error tensor should be [N, 24, C, D, H, W]."""
        model = EquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        N, C, D, H, W = sample_input.shape
        assert error.shape == (N, 24, C, D, H, W)

    def test_output_dtype_matches_input(self, device, num_channels):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(2, num_channels, 4, 4, 4, device=device, dtype=dtype)
            model = EquivariantModel()

            error = octahedral_equivariance_error(x, model)

            assert error.dtype == dtype

    def test_output_device_matches_input(self, sample_input):
        """Output should be on same device as input."""
        model = EquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        assert error.device == sample_input.device

    def test_non_negative_error(self, sample_input):
        """Absolute error should be non-negative everywhere."""
        model = NonEquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        assert (error >= 0).all()


# ============================================================
# Equivariance Property Tests
# ============================================================

class TestEquivarianceProperties:

    def test_equivariant_model_zero_error(self, sample_input):
        """Perfectly equivariant model (scalar multiply) should have zero error."""
        model = EquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)

    def test_identity_model_zero_error(self, sample_input):
        """Identity model should have zero error."""
        model = IdentityModel()

        error = octahedral_equivariance_error(sample_input, model)

        assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)

    def test_non_equivariant_model_nonzero_error(self, sample_input):
        """Non-equivariant model should have non-zero error."""
        model = NonEquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        assert error.sum() > 0

    def test_identity_rotation_zero_error(self, sample_input):
        """Identity rotation (index 0) should always have zero error for any model."""
        model = NonEquivariantModel()

        error = octahedral_equivariance_error(sample_input, model)

        # error is [N, 24, C, D, H, W], identity is at index 0
        identity_error = error[:, 0]

        assert torch.allclose(identity_error, torch.zeros_like(identity_error), atol=1e-5)

    def test_scalar_field_add_constant_equivariant(self, device):
        """Adding constant to scalar field (C=1) is equivariant."""
        x = torch.randn(2, 1, 4, 4, 4, device=device)
        model = AddConstantModel()

        error = octahedral_equivariance_error(x, model)

        assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)

    def test_learnable_equivariant_zero_error(self, device, num_channels):
        """Channel-wise scaling is equivariant."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device)
        model = LearnableEquivariantModel(num_channels).to(device)

        error = octahedral_equivariance_error(x, model)

        assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)


# ============================================================
# Batching Tests
# ============================================================

class TestBatching:

    def test_batch_size_none_uses_full_batch(self, sample_input):
        """batch_size=None should process all samples at once."""
        model = EquivariantModel()

        error = octahedral_equivariance_error(sample_input, model, batch_size=None)

        N = sample_input.shape[0]
        assert error.shape[0] == N

    def test_batch_size_one(self, sample_input):
        """batch_size=1 should give same result as full batch."""
        model = NonEquivariantModel()

        error_full = octahedral_equivariance_error(sample_input, model, batch_size=None)
        error_one = octahedral_equivariance_error(sample_input, model, batch_size=1)

        assert torch.allclose(error_full, error_one, atol=1e-5)

    def test_varying_batch_sizes_consistent(self, sample_input):
        """Different batch sizes should produce identical results."""
        model = NonEquivariantModel()

        error_bs1 = octahedral_equivariance_error(sample_input, model, batch_size=1)
        error_bs2 = octahedral_equivariance_error(sample_input, model, batch_size=2)
        error_bs3 = octahedral_equivariance_error(sample_input, model, batch_size=3)
        error_full = octahedral_equivariance_error(sample_input, model, batch_size=100)

        assert torch.allclose(error_bs1, error_bs2, atol=1e-5)
        assert torch.allclose(error_bs2, error_bs3, atol=1e-5)
        assert torch.allclose(error_bs3, error_full, atol=1e-5)

    def test_batch_size_larger_than_input(self, device, num_channels):
        """batch_size > num_samples should work correctly."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device)
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model, batch_size=100)

        assert error.shape[0] == 2
        assert error.shape[1] == 24


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:

    def test_single_sample(self, device, num_channels):
        """Should work with a single sample."""
        x = torch.randn(1, num_channels, 4, 4, 4, device=device)
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model)

        assert error.shape == (1, 24, num_channels, 4, 4, 4)

    def test_minimal_spatial_dims(self, device, num_channels):
        """Should work with 1x1x1 spatial dimensions."""
        x = torch.randn(2, num_channels, 1, 1, 1, device=device)
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model)

        assert error.shape == (2, 24, num_channels, 1, 1, 1)


# ============================================================
# Numerical Stability Tests
# ============================================================

class TestNumericalStability:

    def test_large_values(self, device, num_channels):
        """Should handle large input values."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device) * 1e6
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model)

        assert torch.isfinite(error).all()

    def test_small_values(self, device, num_channels):
        """Should handle small input values."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device) * 1e-6
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model)

        assert torch.isfinite(error).all()

    def test_mixed_sign_values(self, device, num_channels):
        """Should handle mixed positive/negative values correctly."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device)
        model = EquivariantModel()

        error = octahedral_equivariance_error(x, model)

        assert (error >= 0).all()


# ============================================================
# Consistency Tests
# ============================================================

class TestConsistency:

    def test_deterministic_output(self, sample_input):
        """Same inputs should produce same outputs."""
        model = NonEquivariantModel()

        error1 = octahedral_equivariance_error(sample_input, model)
        error2 = octahedral_equivariance_error(sample_input, model)

        assert torch.allclose(error1, error2)

    def test_cloned_input_same_result(self, sample_input):
        """Using cloned input should match original."""
        model = EquivariantModel()

        error_orig = octahedral_equivariance_error(sample_input, model)
        error_clone = octahedral_equivariance_error(sample_input.clone(), model)

        assert torch.allclose(error_orig, error_clone)


# ============================================================
# Gradient Tests
# ============================================================

class TestGradients:

    def test_gradients_flow_through(self, device, num_channels):
        """Gradients should flow through the error computation."""
        x = torch.randn(2, num_channels, 4, 4, 4, device=device, requires_grad=True)
        model = LearnableEquivariantModel(channels=num_channels).to(device)

        error = octahedral_equivariance_error(x, model)
        loss = error.mean()
        loss.backward()

        assert x.grad is not None
        assert model.scale.grad is not None

    def test_no_grad_context(self, sample_input):
        """Should work in no_grad context."""
        model = EquivariantModel()

        with torch.no_grad():
            error = octahedral_equivariance_error(sample_input, model)

        assert error.shape[0] == sample_input.shape[0]
        assert error.shape[1] == 24


# ============================================================
# Error Magnitude Tests
# ============================================================

class TestErrorMagnitude:

    def test_error_scales_with_non_equivariance(self, device):
        """Larger non-equivariant perturbations should give larger errors."""
        x = torch.randn(2, 3, 4, 4, 4, device=device)

        class SmallPerturbModel(nn.Module):
            def forward(self, x):
                x = x.clone()
                x[:, :, 0, :, :] *= 1.1
                return x

        class LargePerturbModel(nn.Module):
            def forward(self, x):
                x = x.clone()
                x[:, :, 0, :, :] *= 2.0
                return x

        small_error = octahedral_equivariance_error(x, SmallPerturbModel())
        large_error = octahedral_equivariance_error(x, LargePerturbModel())

        assert large_error.mean() > small_error.mean()

    def test_all_rotations_same_error_for_symmetric_breaking(self, device):
        """For spatially symmetric non-equivariance, most non-identity rotations 
        should have non-zero error (though some may be zero depending on symmetry)."""
        torch.manual_seed(123)
        x = torch.randn(2, 3, 4, 4, 4, device=device)
        model = NonEquivariantModel()

        error = octahedral_equivariance_error(x, model)

        # Skip identity (index 0), check that at least some rotations have non-zero error
        non_identity_errors = error[:, 1:].mean(dim=(0, 2, 3, 4, 5))
        assert (non_identity_errors > 0).any(), "At least some rotations should have non-zero error"
        # Most rotations should have error (not all will due to symmetry)
        assert (non_identity_errors > 0).sum() > 10, "Most rotations should have non-zero error"


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])