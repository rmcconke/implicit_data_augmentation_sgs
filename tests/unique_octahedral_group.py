import torch
import sys
sys.path.append("./src/")
from transformations import euler_angles_to_matrix
import math

def get_all_octahedral_angles() -> torch.Tensor:
    """Return the 24 discrete rotations of the cube in Y-X-Y convention."""
    values = []
    
    # Group 1: Rotations with middle angle = 0 (no tilt)
    # Just 4 rotations around Z axis (avoid gimbal lock by using only first Y)
    for k in range(4):
        values.append((0, 0, k * math.pi / 2))
    
    # Group 2: Tilt to +X face (middle angle = π/2)
    for k in range(4):
        values.append((0, math.pi / 2, k * math.pi / 2))
    
    # Group 3: Tilt to -X face (middle angle = 3π/2)
    for k in range(4):
        values.append((0, 3 * math.pi / 2, k * math.pi / 2))
    
    # Group 4: Flip upside down (middle angle = π)
    for k in range(4):
        values.append((0, math.pi, k * math.pi / 2))
    
    # Group 5: Tilt to +Y face (first angle = π/2, middle angle = π/2)
    for k in range(4):
        values.append((math.pi / 2, math.pi / 2, k * math.pi / 2))
    
    # Group 6: Tilt to -Y face (first angle = π/2, middle angle = 3π/2)
    for k in range(4):
        values.append((math.pi / 2, 3 * math.pi / 2, k * math.pi / 2))
    
    return torch.tensor(values)

def test_rotation_matrices():
    """Test if the rotation matrices form a valid group for cube symmetry."""
    
    # Get your rotation matrices
    angles = get_all_octahedral_angles()
    R_mats = euler_angles_to_matrix(angles)  # Assuming this is from pytorch3d or similar
    
    print(f"Number of rotations: {len(R_mats)}")
    
    # Test 1: Check orthogonality (R^T R = I)
    for i, R in enumerate(R_mats):
        should_be_I = R.T @ R
        if not torch.allclose(should_be_I, torch.eye(3), atol=1e-5):
            print(f"Matrix {i} is not orthogonal!")
            print(R)
    
    # Test 2: Check determinant = 1 (proper rotations)
    for i, R in enumerate(R_mats):
        det = torch.det(R)
        if not torch.allclose(det, torch.tensor(1.0), atol=1e-5):
            print(f"Matrix {i} has det={det:.4f}, not a proper rotation!")
    
    # Test 3: Check group closure (all products should be in the group)
    unique_mats = set()
    for R in R_mats:
        unique_mats.add(tuple(R.flatten().tolist()))
    
    print(f"Unique matrices: {len(unique_mats)}")
    
    # Test 4: Verify it's actually 24 (not 48 or other)
    if len(unique_mats) != 24:
        print(f"ERROR: Expected 24 unique rotations, got {len(unique_mats)}")
    
    return R_mats

def debug_duplicate_rotations():
    angles = get_all_octahedral_angles()
    R_mats = euler_angles_to_matrix(angles)
    
    # Find duplicates
    seen = {}
    duplicates = []
    
    for i, R in enumerate(R_mats):
        # Round to avoid floating point issues
        R_rounded = torch.round(R * 1000) / 1000
        key = tuple(R_rounded.flatten().tolist())
        
        if key in seen:
            duplicates.append((i, seen[key]))
            print(f"\n{'='*60}")
            print(f"Rotation {i} is duplicate of {seen[key]}")
            print(f"\nAngles {i}: {angles[i]}")
            print(f"Angles {seen[key]}: {angles[seen[key]]}")
            
            print(f"\nMatrix {i}:")
            print(R)
            print(f"\nMatrix {seen[key]}:")
            print(R_mats[seen[key]])
            
            print(f"\nDifference (should be ~0):")
            print(R - R_mats[seen[key]])
            print(f"Max absolute difference: {torch.max(torch.abs(R - R_mats[seen[key]])).item():.2e}")
            print('='*60)
        else:
            seen[key] = i
    
    print(f"\nFound {len(duplicates)} duplicate pairs")
    return duplicates

    
test_rotation_matrices()
print("\n" + "="*60)
print("CHECKING FOR DUPLICATES")
print("="*60)
debug_duplicate_rotations()

import torch
import math
import sys
sys.path.append("./src/")
from transformations import euler_angles_to_matrix

def analyze_octahedral_rotations():
    """Analyze the octahedral rotation group in detail."""
    
    angles = get_all_octahedral_angles()
    R_mats = euler_angles_to_matrix(angles)
    
    print("="*70)
    print("OCTAHEDRAL GROUP ANALYSIS")
    print("="*70)
    
    # Find duplicates with detailed info
    seen = {}
    duplicates = []
    
    for i, R in enumerate(R_mats):
        R_rounded = torch.round(R * 1000) / 1000
        key = tuple(R_rounded.flatten().tolist())
        
        if key in seen:
            duplicates.append((i, seen[key]))
        else:
            seen[key] = i
    
    print(f"\nTotal angles provided: {len(angles)}")
    print(f"Unique rotation matrices: {len(seen)}")
    print(f"Duplicate pairs found: {len(duplicates)}")
    
    # Print duplicates
    if duplicates:
        print("\n" + "="*70)
        print("DUPLICATE ROTATIONS:")
        print("="*70)
        for dup_idx, orig_idx in duplicates:
            print(f"\nIndex {dup_idx} duplicates Index {orig_idx}:")
            print(f"  Angles[{dup_idx}]: ({angles[dup_idx][0]/math.pi:.2f}π, {angles[dup_idx][1]/math.pi:.2f}π, {angles[dup_idx][2]/math.pi:.2f}π)")
            print(f"  Angles[{orig_idx}]: ({angles[orig_idx][0]/math.pi:.2f}π, {angles[orig_idx][1]/math.pi:.2f}π, {angles[orig_idx][2]/math.pi:.2f}π)")
            print(f"  Matrix:\n{R_mats[dup_idx]}")
    
    # Check what's missing
    print("\n" + "="*70)
    print("CHECKING OCTAHEDRAL GROUP PROPERTIES:")
    print("="*70)
    
    # The octahedral group should have:
    # - 1 identity
    # - 6 face rotations (90° and 270° around each of 3 axes)
    # - 3 face rotations (180° around each of 3 axes)
    # - 8 vertex rotations (120° and 240° around each of 4 body diagonals)
    # - 6 edge rotations (180° around each of 6 edge-to-edge axes)
    # Total: 1 + 6 + 3 + 8 + 6 = 24
    
    identity_count = 0
    rotation_types = []
    
    for i, R in enumerate(R_mats):
        # Check if identity
        if torch.allclose(R, torch.eye(3), atol=1e-4):
            identity_count += 1
            rotation_types.append((i, "Identity"))
        else:
            # Check rotation angle via trace
            trace = torch.trace(R).item()
            angle = math.acos((trace - 1) / 2)
            rotation_types.append((i, f"{math.degrees(angle):.1f}°"))
    
    print(f"\nIdentity matrices: {identity_count}")
    
    # Count rotation angles
    from collections import Counter
    angle_counts = Counter([rt[1] for rt in rotation_types])
    print("\nRotation angle distribution:")
    for angle, count in sorted(angle_counts.items()):
        print(f"  {angle}: {count} rotations")
    
    return angles, R_mats, duplicates

# Run analysis
angles, R_mats, duplicates = analyze_octahedral_rotations()
print(f"Duplicates: {duplicates}")
assert duplicates == [], "Duplicates found in octahedral group"
