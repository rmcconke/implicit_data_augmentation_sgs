"""
Comprehensive tests for rotate_octahedral_exact function.

Tests cover:
- Octahedral group properties (24 unique rotations, closure, inverses)
- Vector field rotations (C=3)
- Symmetric tensor field rotations (C=6)
- Full tensor field rotations (C=9)
- Pointwise rotations
- Stress tests with random fields
- whole_group mode for applying all 24 rotations
"""

import torch
import math
import sys
import pytest
from collections import Counter

sys.path.append("./src")
from transformations import euler_angles_to_matrix, get_all_octahedral_angles, rotate_octahedral_exact


# ============ Helper functions ============

def tensor6_to_matrix(t):
    """Convert 6-channel tensor [xx, yy, zz, xy, xz, yz] to 3x3 symmetric matrix."""
    T = torch.zeros(3, 3)
    T[0, 0], T[1, 1], T[2, 2] = t[0], t[1], t[2]
    T[0, 1] = T[1, 0] = t[3]
    T[0, 2] = T[2, 0] = t[4]
    T[1, 2] = T[2, 1] = t[5]
    return T


def matrix_to_tensor6(T):
    """Convert 3x3 symmetric matrix to 6-channel tensor."""
    return torch.tensor([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def tensor9_to_matrix(t):
    """Convert 9-channel tensor to 3x3 matrix (row-major order)."""
    return t.reshape(3, 3)


def matrix_to_tensor9(T):
    """Convert 3x3 matrix to 9-channel tensor."""
    return T.flatten()


def rotate_tensor(T, R):
    """Rotate a 3x3 tensor: T' = R @ T @ R.T"""
    return R @ T @ R.T


def find_inverse_rotation(matrices, i):
    """Find index of inverse rotation for rotation i."""
    R_inv = matrices[i].T
    for k in range(len(matrices)):
        if torch.allclose(matrices[k], R_inv, atol=1e-5):
            return k
    return None


def find_composed_rotation(matrices, i, j):
    """Find index of composed rotation R_j @ R_i."""
    R_composed = matrices[j] @ matrices[i]
    R_composed_int = torch.round(R_composed).int()
    for k in range(len(matrices)):
        if torch.allclose(torch.round(matrices[k]).int().float(), R_composed_int.float()):
            return k
    return None


# ============ Octahedral group tests ============

class TestOctahedralGroup:
    """Tests for the octahedral rotation group properties."""

    def test_correct_number_of_rotations(self):
        """Test that we have exactly 24 rotations."""
        angles = get_all_octahedral_angles()
        assert len(angles) == 24, f"Expected 24 rotations, got {len(angles)}"

    def test_all_rotations_unique(self):
        """Test that all 24 rotation matrices are unique."""
        angles = get_all_octahedral_angles()
        R_mats = euler_angles_to_matrix(angles)

        unique_mats = set()
        for R in R_mats:
            R_rounded = torch.round(R * 1000) / 1000
            key = tuple(R_rounded.flatten().tolist())
            unique_mats.add(key)

        assert len(unique_mats) == 24, f"Expected 24 unique matrices, got {len(unique_mats)}"

    def test_all_orthogonal(self):
        """Test that all rotation matrices are orthogonal (R^T R = I)."""
        angles = get_all_octahedral_angles()
        R_mats = euler_angles_to_matrix(angles)

        for i, R in enumerate(R_mats):
            should_be_I = R.T @ R
            assert torch.allclose(should_be_I, torch.eye(3), atol=1e-5), \
                f"Matrix {i} is not orthogonal"

    def test_all_proper_rotations(self):
        """Test that all rotation matrices have determinant = 1."""
        angles = get_all_octahedral_angles()
        R_mats = euler_angles_to_matrix(angles)

        for i, R in enumerate(R_mats):
            det = torch.det(R)
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), \
                f"Matrix {i} has det={det:.4f}, not a proper rotation"

    def test_no_duplicates(self):
        """Test that there are no duplicate rotation matrices."""
        angles = get_all_octahedral_angles()
        R_mats = euler_angles_to_matrix(angles)

        seen = {}
        duplicates = []

        for i, R in enumerate(R_mats):
            R_rounded = torch.round(R * 1000) / 1000
            key = tuple(R_rounded.flatten().tolist())

            if key in seen:
                duplicates.append((i, seen[key]))
            else:
                seen[key] = i

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate pairs: {duplicates}"

    def test_rotation_angle_distribution(self):
        """Test that the rotation angles follow expected distribution for octahedral group."""
        angles = get_all_octahedral_angles()
        R_mats = euler_angles_to_matrix(angles)

        rotation_angles = []
        for R in R_mats:
            trace = torch.trace(R).item()
            trace_clamped = max(-1, min(3, trace))
            angle = math.acos((trace_clamped - 1) / 2)
            rotation_angles.append(round(math.degrees(angle)))

        angle_counts = Counter(rotation_angles)
        assert angle_counts[0] == 1, f"Expected 1 identity rotation, got {angle_counts[0]}"

    def test_group_closure(self):
        """Test that the product of any two rotations is in the group."""
        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(matrices)):
            for j in range(len(matrices)):
                composed_idx = find_composed_rotation(matrices, i, j)
                assert composed_idx is not None, \
                    f"Product of rotations {i} and {j} not found in group"

    def test_inverses_exist(self):
        """Test that every rotation has an inverse in the group."""
        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(matrices)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found in group"


# ============ Vector field tests (C=3) ============

class TestVectorField:
    """Tests for vector field rotations."""

    def test_uniform_x_vector_rotation(self):
        """Test that uniform +X vectors are rotated correctly through all 24 octahedral rotations."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]
            expected = R[:, 0]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), \
                f"Rotation {i}: expected={expected.tolist()}, actual={actual.tolist()}"

    def test_uniform_y_vector_rotation(self):
        """Test that uniform +Y vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i][:, 1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_uniform_z_vector_rotation(self):
        """Test that uniform +Z vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i][:, 2]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_negative_x_vector_rotation(self):
        """Test that uniform -X vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = -1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = -matrices[i][:, 0]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_diagonal_vector_rotation(self):
        """Test that uniform [1,1,1] vectors are rotated correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 1.0
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            expected = matrices[i] @ torch.tensor([1.0, 1.0, 1.0])

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_spatial_permutation(self):
        """Test that spatial positions are permuted correctly through all 24 octahedral rotations."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = torch.round(matrices[i]).int()

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")

            actual_pos = None
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        if rotated[0, :, d, h, w].abs().max() > 0.5:
                            actual_pos = (d, h, w)

            coord = torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32)
            new_coord = (R.float() @ coord + 0.5).clamp(0, 1)
            expected_pos = tuple(new_coord.int().tolist())

            assert actual_pos == expected_pos, \
                f"Rotation {i}: actual={actual_pos}, expected={expected_pos}"

    def test_multiple_positions(self):
        """Test that vectors at two corners are both transformed correctly."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 1, 1, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")

            expected_vec1 = matrices[i][:, 0]
            expected_vec2 = matrices[i][:, 1]

            found_vec1, found_vec2 = False, False
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        vec = rotated[0, :, d, h, w]
                        if torch.allclose(vec, expected_vec1, atol=1e-5):
                            found_vec1 = True
                        if torch.allclose(vec, expected_vec2, atol=1e-5):
                            found_vec2 = True

            assert found_vec1, f"Rotation {i}: didn't find rotated +X vector"
            assert found_vec2, f"Rotation {i}: didn't find rotated +Y vector"

    def test_identity_rotation(self):
        """Test that identity rotation leaves field unchanged."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 0, 1, 0] = 2.0
        field[0, 2, 1, 0, 1] = 3.0

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated, field, atol=1e-5)

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11
        angle_i, angle_j = angles[i:i+1], angles[j:j+1]

        rotated_once = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angle_j, rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None, "Composed rotation not found in group"

        rotated_composed = rotate_octahedral_exact(
            field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise"
        )

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original field."""
        field = torch.zeros(1, 3, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 1, 0, 1] = 2.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle_i = angles[i:i+1]

            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found"

            rotated = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"


# ============ Vector field stress tests (C=3) ============

class TestVectorFieldStress:
    """Stress tests for vector field rotations with random data."""

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random fields."""
        torch.manual_seed(42)
        field = torch.randn(1, 3, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_composition(self):
        """Test that R_j(R_i(field)) = R_{ji}(field) for random fields."""
        torch.manual_seed(43)
        field = torch.randn(1, 3, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(0, len(angles), 4):
            for j in range(0, len(angles), 4):
                composed_idx = find_composed_rotation(matrices, i, j)
                assert composed_idx is not None

                rotated_twice = rotate_octahedral_exact(
                    rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise"),
                    angles[j:j+1], rotate_channels=True, mode="pairwise"
                )
                rotated_composed = rotate_octahedral_exact(
                    field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise"
                )

                assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5), \
                    f"Rotations {i}, {j}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(44)
        field = torch.randn(1, 3, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_vector_magnitude_preservation(self):
        """Test that rotation preserves the sum of vector magnitudes."""
        torch.manual_seed(45)
        field = torch.randn(1, 3, 4, 4, 4)
        original_magnitude_sum = torch.norm(field, dim=1).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_magnitude_sum = torch.norm(rotated, dim=1).sum()

            assert torch.allclose(original_magnitude_sum, rotated_magnitude_sum, atol=1e-5), \
                f"Rotation {i}"

    def test_random_field_dot_product_preservation(self):
        """Test that rotation preserves sum of dot products between two fields."""
        torch.manual_seed(46)
        field1 = torch.randn(1, 3, 4, 4, 4)
        field2 = torch.randn(1, 3, 4, 4, 4)
        original_dot_sum = (field1 * field2).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated1 = rotate_octahedral_exact(field1, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated2 = rotate_octahedral_exact(field2, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_dot_sum = (rotated1 * rotated2).sum()

            assert torch.allclose(original_dot_sum, rotated_dot_sum, atol=1e-5), f"Rotation {i}"


# ============ Symmetric tensor field tests (C=6) ============

class TestSymmetricTensorField:
    """Tests for symmetric tensor field rotations."""

    def test_xx_tensor_rotation(self):
        """Test that T = diag(1,0,0) (xx only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([1., 0., 0., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_yy_tensor_rotation(self):
        """Test that T = diag(0,1,0) (yy only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 1., 0., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_zz_tensor_rotation(self):
        """Test that T = diag(0,0,1) (zz only) rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 0., 1., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_xy_tensor_rotation(self):
        """Test that off-diagonal xy component rotates correctly."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 3] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([0., 0., 0., 1., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_tensor_invariance(self):
        """Test that identity tensor T = diag(1,1,1) is invariant under all rotations."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 1.0
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        expected = torch.tensor([1., 1., 1., 0., 0., 0.])

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_general_tensor_rotation(self):
        """Test a general symmetric tensor with all components nonzero."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = 2.0
        field[:, 2] = 3.0
        field[:, 3] = 0.5
        field[:, 4] = 0.3
        field[:, 5] = 0.2

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor6_to_matrix(torch.tensor([1., 2., 3., 0.5, 0.3, 0.2]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor6(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_rotation(self):
        """Test that identity rotation leaves tensor field unchanged."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 3, 1, 0, 1] = 2.0

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")

        assert torch.allclose(rotated, field, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original tensor field."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 0, 0, 0] = 2.0
        field[0, 3, 0, 0, 0] = 0.5

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle_i = angles[i:i+1]

            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None, f"Inverse of rotation {i} not found"

            rotated = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        field = torch.zeros(1, 6, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 3, 0, 0, 0] = 0.5

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11
        angle_i, angle_j = angles[i:i+1], angles[j:j+1]

        rotated_once = rotate_octahedral_exact(field, angle_i, rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angle_j, rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None, "Composed rotation not found in group"

        rotated_composed = rotate_octahedral_exact(
            field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise"
        )

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)


# ============ Symmetric tensor field stress tests (C=6) ============

class TestSymmetricTensorFieldStress:
    """Stress tests for symmetric tensor field rotations."""

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random symmetric tensor fields."""
        torch.manual_seed(47)
        field = torch.randn(1, 6, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(49)
        field = torch.randn(1, 6, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_trace_preservation(self):
        """Test that rotation preserves the sum of traces over the field."""
        torch.manual_seed(50)
        field = torch.randn(1, 6, 4, 4, 4)
        original_trace_sum = (field[:, 0] + field[:, 1] + field[:, 2]).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_trace_sum = (rotated[:, 0] + rotated[:, 1] + rotated[:, 2]).sum()

            assert torch.allclose(original_trace_sum, rotated_trace_sum, atol=1e-5), f"Rotation {i}"


# ============ Full tensor field tests (C=9) ============

class TestFullTensorField:
    """Tests for full (non-symmetric) tensor field rotations."""

    def test_xx_tensor_rotation(self):
        """Test that T with only (0,0) component rotates correctly."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[0, 0] = 1.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_asymmetric_tensor_rotation(self):
        """Test an asymmetric tensor where T[0,1] != T[1,0]."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 1] = 1.0
        field[:, 3] = 2.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = torch.zeros(3, 3)
            T_in[0, 1] = 1.0
            T_in[1, 0] = 2.0
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor9(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_identity_tensor_invariance(self):
        """Test that identity tensor T = I is invariant under all rotations."""
        field = torch.zeros(1, 9, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 4] = 1.0
        field[:, 8] = 1.0

        angles = get_all_octahedral_angles()
        expected = matrix_to_tensor9(torch.eye(3))

        for i in range(len(angles)):
            angle = angles[i:i+1]

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"


# ============ Full tensor field stress tests (C=9) ============

class TestFullTensorFieldStress:
    """Stress tests for full tensor field rotations."""

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random full tensor fields."""
        torch.manual_seed(54)
        field = torch.randn(1, 9, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_norm_preservation(self):
        """Test that rotation preserves the L2 norm of the field."""
        torch.manual_seed(56)
        field = torch.randn(1, 9, 4, 4, 4)
        original_norm = torch.norm(field)

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_norm = torch.norm(rotated)

            assert torch.allclose(original_norm, rotated_norm, atol=1e-5), f"Rotation {i}"

    def test_random_field_trace_preservation(self):
        """Test that rotation preserves the sum of traces over the field."""
        torch.manual_seed(57)
        field = torch.randn(1, 9, 4, 4, 4)
        original_trace_sum = (field[:, 0] + field[:, 4] + field[:, 8]).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rotated_trace_sum = (rotated[:, 0] + rotated[:, 4] + rotated[:, 8]).sum()

            assert torch.allclose(original_trace_sum, rotated_trace_sum, atol=1e-5), f"Rotation {i}"


# ============ Pointwise tensor field tests ============

class TestTensorFieldPointwise:
    """Tests for pointwise (batched) tensor rotations."""

    def test_vector_pointwise_rotation(self):
        """Pointwise vectors [B, 3] rotate correctly."""
        B, C = 3, 3
        field = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 3.0]
        ])

        angles = get_all_octahedral_angles()[:B]
        matrices = euler_angles_to_matrix(angles)

        field5d = field[:, :, None, None, None]
        rotated5d = rotate_octahedral_exact(field5d, angles, rotate_channels=True, mode="pairwise")
        rotated = rotated5d[:, :, 0, 0, 0]

        for i in range(B):
            expected = matrices[i] @ field[i]
            assert torch.allclose(rotated[i], expected, atol=1e-5), f"Vector {i} failed"

    def test_symmetric_tensor_pointwise_rotation(self):
        """Pointwise symmetric tensors [B, 6] rotate correctly."""
        B, C = 3, 6
        field = torch.tensor([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [0.5, 1.5, 2.5, -0.1, -0.2, -0.3],
            [2.0, 1.0, 0.0, 0.5, 0.4, 0.3]
        ])

        angles = get_all_octahedral_angles()[:B]
        matrices = euler_angles_to_matrix(angles)

        field5d = field[:, :, None, None, None]
        rotated5d = rotate_octahedral_exact(field5d, angles, rotate_channels=True, mode="pairwise")
        rotated = rotated5d[:, :, 0, 0, 0]

        assert rotated.shape == field.shape, "Output shape mismatch"

        for i in range(B):
            T_in = tensor6_to_matrix(field[i])
            T_out = rotate_tensor(T_in, matrices[i])
            expected = matrix_to_tensor6(T_out)
            assert torch.allclose(rotated[i], expected, atol=1e-5), f"Tensor {i} failed"


# ============ whole_group mode tests ============

class TestWholeGroupMode:
    """Tests for mode='whole_group' which applies all 24 rotations to each sample."""

    def test_output_shape_single_sample(self):
        """Single sample [1, C, D, H, W] -> [1, 24, C, D, H, W]."""
        x = torch.randn(1, 3, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (1, 24, 3, 4, 4, 4)

    def test_output_shape_batch(self):
        """Batch [N, C, D, H, W] -> [N, 24, C, D, H, W]."""
        x = torch.randn(5, 6, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (5, 24, 6, 4, 4, 4)

    def test_output_shape_unbatched(self):
        """Unbatched [C, D, H, W] -> [24, C, D, H, W]."""
        x = torch.randn(3, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (24, 3, 4, 4, 4)

    def test_angles_parameter_ignored(self):
        """Angles parameter should be ignored in whole_group mode."""
        x = torch.randn(2, 3, 4, 4, 4)
        dummy_angles = torch.zeros(2, 3)
        result = rotate_octahedral_exact(x, angles=dummy_angles, mode="whole_group")
        assert result.shape == (2, 24, 3, 4, 4, 4)

    def test_identity_rotation_at_index_zero(self):
        """First rotation (index 0) should be identity."""
        x = torch.randn(2, 3, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert torch.allclose(result[:, 0], x, atol=1e-5)

    def test_consistency_with_pairwise(self):
        """whole_group should match pairwise applied to each rotation."""
        x = torch.randn(2, 3, 4, 4, 4)
        angles = get_all_octahedral_angles()

        whole_group_result = rotate_octahedral_exact(x, mode="whole_group")

        for j in range(24):
            angle_j = angles[j:j+1]
            for i in range(x.shape[0]):
                pairwise_result = rotate_octahedral_exact(
                    x[i:i+1], angle_j, mode="pairwise"
                )
                assert torch.allclose(
                    whole_group_result[i, j], pairwise_result[0], atol=1e-5
                ), f"Mismatch at sample {i}, rotation {j}"

    def test_all_24_rotations_unique(self):
        """All 24 rotations should produce different results (for generic input)."""
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")

        for i in range(24):
            for j in range(i + 1, 24):
                assert not torch.allclose(result[0, i], result[0, j], atol=1e-5), \
                    f"Rotations {i} and {j} produced identical results"

    def test_norm_preservation(self):
        """Each rotation should preserve the L2 norm."""
        x = torch.randn(2, 3, 4, 4, 4)
        original_norms = torch.linalg.vector_norm(x.flatten(1), dim=1)

        result = rotate_octahedral_exact(x, mode="whole_group")

        for i in range(x.shape[0]):
            for j in range(24):
                rotated_norm = torch.linalg.vector_norm(result[i, j].flatten())
                assert torch.allclose(original_norms[i], rotated_norm, atol=1e-5), \
                    f"Norm not preserved for sample {i}, rotation {j}"

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(2, 3, 4, 4, 4, dtype=dtype)
            result = rotate_octahedral_exact(x, mode="whole_group")
            assert result.dtype == dtype

    def test_device_preservation(self):
        """Output should be on same device as input."""
        x = torch.randn(2, 3, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.device == x.device

    def test_vector_field_rotation(self):
        """Vector field (C=3) channels should rotate correctly."""
        x = torch.zeros(1, 3, 2, 2, 2)
        x[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        result = rotate_octahedral_exact(x, mode="whole_group", rotate_channels=True)

        for j in range(24):
            expected_vec = matrices[j][:, 0]
            actual_vec = result[0, j, :, 0, 0, 0]
            assert torch.allclose(actual_vec, expected_vec, atol=1e-5), \
                f"Vector rotation {j} incorrect"

    def test_symmetric_tensor_identity_invariance(self):
        """Identity tensor (C=6) should be invariant under all rotations."""
        x = torch.zeros(1, 6, 2, 2, 2)
        x[:, 0] = 1.0
        x[:, 1] = 1.0
        x[:, 2] = 1.0

        result = rotate_octahedral_exact(x, mode="whole_group", rotate_channels=True)

        expected = torch.tensor([1., 1., 1., 0., 0., 0.])
        for j in range(24):
            actual = result[0, j, :, 0, 0, 0]
            assert torch.allclose(actual, expected, atol=1e-5), \
                f"Identity tensor not invariant under rotation {j}"

    def test_no_channel_rotation(self):
        """With rotate_channels=False, only spatial rotation should occur."""
        x = torch.zeros(1, 3, 2, 2, 2)
        x[0, 0, 0, 0, 0] = 1.0

        result = rotate_octahedral_exact(x, mode="whole_group", rotate_channels=False)

        for j in range(24):
            assert torch.isclose(result[0, j, 0].max(), torch.tensor(1.0), atol=1e-5)
            assert torch.allclose(result[0, j, 1], torch.zeros(2, 2, 2), atol=1e-5)
            assert torch.allclose(result[0, j, 2], torch.zeros(2, 2, 2), atol=1e-5)

    def test_single_voxel(self):
        """Should work with 1x1x1 spatial dimensions."""
        x = torch.randn(2, 3, 1, 1, 1)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (2, 24, 3, 1, 1, 1)

    def test_scalar_field(self):
        """Scalar field (C=1) should only have spatial rotation."""
        x = torch.randn(2, 1, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (2, 24, 1, 4, 4, 4)

        original_sum = x.sum()
        for j in range(24):
            assert torch.allclose(result[:, j].sum(), original_sum, atol=1e-4)

# ============ Traceless symmetric tensor field tests (C=5) ============

def tensor5_to_matrix(t):
    """Convert 5-channel traceless symmetric tensor [dev_xx, dev_yy, xy, xz, yz] to 3x3 matrix."""
    T = torch.zeros(3, 3)
    T[0, 0] = t[0]
    T[1, 1] = t[1]
    T[2, 2] = -t[0] - t[1]
    T[0, 1] = T[1, 0] = t[2]
    T[0, 2] = T[2, 0] = t[3]
    T[1, 2] = T[2, 1] = t[4]
    return T


def matrix_to_tensor5(T):
    """Convert 3x3 traceless symmetric matrix to 5-channel tensor."""
    return torch.tensor([T[0, 0], T[1, 1], T[0, 1], T[0, 2], T[1, 2]])


class TestTracelessSymmetricTensorField:
    """Tests for traceless symmetric tensor field rotations (C=5)."""

    def test_deviatoric_xx_rotation(self):
        """Test that dev_xx = 1 (i.e. diag(1, 0, -1)) rotates correctly."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 0] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([1., 0., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), \
                f"Rotation {i}: expected={expected.tolist()}, actual={actual.tolist()}"

    def test_deviatoric_yy_rotation(self):
        """Test that dev_yy = 1 (i.e. diag(0, 1, -1)) rotates correctly."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 1] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([0., 1., 0., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_xy_rotation(self):
        """Test that xy component rotates correctly."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 2] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([0., 0., 1., 0., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_xz_rotation(self):
        """Test that xz component rotates correctly."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 3] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([0., 0., 0., 1., 0.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_yz_rotation(self):
        """Test that yz component rotates correctly."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 4] = 1.0

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([0., 0., 0., 0., 1.]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_general_traceless_tensor_rotation(self):
        """Test a general traceless symmetric tensor with all components nonzero."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = -0.5
        field[:, 2] = 0.3
        field[:, 3] = 0.2
        field[:, 4] = -0.1

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            angle = angles[i:i+1]
            R = matrices[i]

            T_in = tensor5_to_matrix(torch.tensor([1.0, -0.5, 0.3, 0.2, -0.1]))
            T_out = rotate_tensor(T_in, R)
            expected = matrix_to_tensor5(T_out)

            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            actual = rotated[0, :, 0, 0, 0]

            assert torch.allclose(actual, expected, atol=1e-5), f"Rotation {i}"

    def test_trace_remains_zero(self):
        """Test that the trace remains zero after rotation."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[:, 0] = 1.0
        field[:, 1] = -0.5

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            angle = angles[i:i+1]
            rotated = rotate_octahedral_exact(field, angle, rotate_channels=True, mode="pairwise")
            # trace = dev_xx + dev_yy + (-dev_xx - dev_yy) = 0 by construction
            # but check the expanded version
            dev_xx = rotated[0, 0, 0, 0, 0]
            dev_yy = rotated[0, 1, 0, 0, 0]
            zz = -dev_xx - dev_yy
            trace = dev_xx + dev_yy + zz
            assert torch.allclose(trace, torch.tensor(0.0), atol=1e-5), \
                f"Rotation {i}: trace = {trace.item()}"

    def test_identity_rotation(self):
        """Test that identity rotation leaves field unchanged."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 2, 1, 0, 1] = 0.5

        angles = get_all_octahedral_angles()
        identity_angle = angles[0:1]

        rotated = rotate_octahedral_exact(field, identity_angle, rotate_channels=True, mode="pairwise")
        assert torch.allclose(rotated, field, atol=1e-5)

    def test_inverse_rotations(self):
        """Test that applying a rotation and its inverse returns the original field."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 1, 0, 0, 0] = -0.5
        field[0, 2, 0, 0, 0] = 0.3

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_rotation_composition(self):
        """Test that applying two rotations equals applying their composition."""
        field = torch.zeros(1, 5, 2, 2, 2)
        field[0, 0, 0, 0, 0] = 1.0
        field[0, 2, 0, 0, 0] = 0.5

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        i, j = 5, 11

        rotated_once = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
        rotated_twice = rotate_octahedral_exact(rotated_once, angles[j:j+1], rotate_channels=True, mode="pairwise")

        composed_idx = find_composed_rotation(matrices, i, j)
        assert composed_idx is not None

        rotated_composed = rotate_octahedral_exact(
            field, angles[composed_idx:composed_idx+1], rotate_channels=True, mode="pairwise"
        )

        assert torch.allclose(rotated_twice, rotated_composed, atol=1e-5)


class TestTracelessSymmetricTensorFieldStress:
    """Stress tests for traceless symmetric tensor field rotations."""

    def test_random_field_inverse(self):
        """Test that R^{-1}(R(field)) = field for random traceless symmetric tensor fields."""
        torch.manual_seed(60)
        field = torch.randn(1, 5, 4, 4, 4)

        angles = get_all_octahedral_angles()
        matrices = euler_angles_to_matrix(angles)

        for i in range(len(angles)):
            inv_idx = find_inverse_rotation(matrices, i)
            assert inv_idx is not None

            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            restored = rotate_octahedral_exact(
                rotated, angles[inv_idx:inv_idx+1], rotate_channels=True, mode="pairwise"
            )

            assert torch.allclose(restored, field, atol=1e-5), f"Rotation {i}"

    def test_random_field_frobenius_norm_preservation(self):
        """Test that the Frobenius norm of each tensor is preserved under rotation."""
        torch.manual_seed(62)
        field = torch.randn(1, 5, 4, 4, 4)

        # Frobenius norm from 5 components: xx^2 + yy^2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2
        # where zz = -xx - yy
        xx, yy, xy, xz, yz = field[:, 0], field[:, 1], field[:, 2], field[:, 3], field[:, 4]
        zz = -xx - yy
        original_frob = (xx**2 + yy**2 + zz**2 + 2*xy**2 + 2*xz**2 + 2*yz**2).sum()

        angles = get_all_octahedral_angles()

        for i in range(len(angles)):
            rotated = rotate_octahedral_exact(field, angles[i:i+1], rotate_channels=True, mode="pairwise")
            rxx, ryy, rxy, rxz, ryz = rotated[:, 0], rotated[:, 1], rotated[:, 2], rotated[:, 3], rotated[:, 4]
            rzz = -rxx - ryy
            rotated_frob = (rxx**2 + ryy**2 + rzz**2 + 2*rxy**2 + 2*rxz**2 + 2*ryz**2).sum()

            assert torch.allclose(original_frob, rotated_frob, atol=1e-4), f"Rotation {i}"


class TestTracelessWholeGroup:
    """Tests for whole_group mode with traceless symmetric tensors."""

    def test_output_shape(self):
        """Test output shape for C=5 whole_group."""
        x = torch.randn(2, 5, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert result.shape == (2, 24, 5, 4, 4, 4)

    def test_identity_at_index_zero(self):
        """First rotation should be identity."""
        x = torch.randn(2, 5, 4, 4, 4)
        result = rotate_octahedral_exact(x, mode="whole_group")
        assert torch.allclose(result[:, 0], x, atol=1e-5)

    def test_consistency_with_pairwise(self):
        """whole_group should match pairwise for C=5."""
        x = torch.randn(2, 5, 4, 4, 4)
        angles = get_all_octahedral_angles()

        whole_group_result = rotate_octahedral_exact(x, mode="whole_group")

        for j in range(24):
            angle_j = angles[j:j+1]
            for i in range(x.shape[0]):
                pairwise_result = rotate_octahedral_exact(
                    x[i:i+1], angle_j, mode="pairwise"
                )
                assert torch.allclose(
                    whole_group_result[i, j], pairwise_result[0], atol=1e-5
                ), f"Mismatch at sample {i}, rotation {j}"

# ============ Main entry point ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])