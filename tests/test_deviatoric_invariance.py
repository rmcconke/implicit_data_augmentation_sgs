import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.append("./src/")
from deviatoric import to_orthonormal, frobenius_norm, invariant_criterion
from transformations import rotate_octahedral_exact

# ============================================================
# Helpers
# ============================================================

def full_tensor(components):
    """Rebuild the 3x3 deviatoric tensor from the five stored components."""
    d_xx, d_yy, xy, xz, yz = components.unbind(dim=1)
    d_zz = -d_xx - d_yy
    rows = [
        torch.stack([d_xx, xy, xz], dim=1),
        torch.stack([xy, d_yy, yz], dim=1),
        torch.stack([xz, yz, d_zz], dim=1),
    ]
    return torch.stack(rows, dim=1)


@pytest.fixture
def components():
    torch.manual_seed(42)
    return torch.randn(16, 5)


@pytest.fixture
def field():
    torch.manual_seed(7)
    return torch.randn(3, 5, 6, 6, 6)


# ============================================================
# Basis Tests
# ============================================================

class TestOrthonormalBasis:

    def test_matches_frobenius_norm(self, components):
        """Sum of squared coefficients equals the Frobenius norm of the full tensor."""
        expected = full_tensor(components).pow(2).sum(dim=(1, 2))

        assert torch.allclose(to_orthonormal(components).pow(2).sum(dim=1), expected, atol=1e-5)

    def test_raw_components_do_not_match_frobenius(self, components):
        """The stored five components are not an orthonormal representation."""
        expected = full_tensor(components).pow(2).sum(dim=(1, 2))

        assert not torch.allclose(components.pow(2).sum(dim=1), expected, atol=1e-3)

    def test_frobenius_norm_helper(self, components):
        expected = full_tensor(components).pow(2).sum(dim=(1, 2)).sqrt()

        assert torch.allclose(frobenius_norm(components), expected, atol=1e-5)

    def test_preserves_shape(self, field):
        assert to_orthonormal(field).shape == field.shape

    def test_component_dim_argument(self, field):
        """dim selects which axis holds the five components."""
        moved = field.movedim(1, 2)

        assert torch.allclose(to_orthonormal(moved, dim=2).movedim(2, 1), to_orthonormal(field), atol=1e-6)


# ============================================================
# Rotational Invariance Tests
# ============================================================

class TestRotationalInvariance:

    def test_loss_invariant_under_all_rotations(self, field):
        """MSE on the orthonormal basis is unchanged by every element of the group."""
        target = torch.randn_like(field)
        criterion = invariant_criterion(F.mse_loss)
        reference = criterion(field, target)

        rotated_pred = rotate_octahedral_exact(field, mode="whole_group")
        rotated_target = rotate_octahedral_exact(target, mode="whole_group")
        for g in range(24):
            assert torch.allclose(criterion(rotated_pred[:, g], rotated_target[:, g]), reference, atol=1e-5)

    def test_raw_mse_is_not_invariant(self, field):
        """The unweighted five-component MSE changes under rotation."""
        target = torch.randn_like(field)
        reference = F.mse_loss(field, target)

        rotated_pred = rotate_octahedral_exact(field, mode="whole_group")
        rotated_target = rotate_octahedral_exact(target, mode="whole_group")
        losses = [F.mse_loss(rotated_pred[:, g], rotated_target[:, g]) for g in range(24)]

        assert not all(torch.allclose(l, reference, atol=1e-5) for l in losses)

    def test_correlation_invariant_under_rotation(self, field):
        """The uncentered correlation is unchanged on the orthonormal basis."""
        target = torch.randn_like(field)

        def correlation(pred, true):
            pred, true = to_orthonormal(pred), to_orthonormal(true)
            return (pred * true).sum() / (pred.pow(2).sum().sqrt() * true.pow(2).sum().sqrt())

        reference = correlation(field, target)
        rotated_pred = rotate_octahedral_exact(field, mode="whole_group")
        rotated_target = rotate_octahedral_exact(target, mode="whole_group")
        for g in range(24):
            assert torch.allclose(correlation(rotated_pred[:, g], rotated_target[:, g]), reference, atol=1e-5)

    def test_frobenius_norm_invariant_under_rotation(self, field):
        """Every voxel keeps its norm; the group only moves voxels and mixes components."""
        reference = frobenius_norm(field).flatten().sort().values

        rotated = rotate_octahedral_exact(field, mode="whole_group")
        for g in range(24):
            assert torch.allclose(frobenius_norm(rotated[:, g]).flatten().sort().values, reference, atol=1e-5)


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
