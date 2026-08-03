import torch
import time
import sys
sys.path.append("./src/")
from transformations import get_all_octahedral_angles, euler_angles_to_matrix, rotate_octahedral_exact, _apply_channel_rotation

B = 20000
C = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.randn(B, C, 1, 1, 1, device=device)
R = torch.randn(B, 3, 3, device=device)

# Warmup
for _ in range(3):
    _ = _apply_channel_rotation(X, R)
    if device == 'cuda':
        torch.cuda.synchronize()

# Time it
start = time.perf_counter()
for _ in range(100):
    result = _apply_channel_rotation(X, R)
if device == 'cuda':
    torch.cuda.synchronize()
print(f"_apply_channel_rotation: {(time.perf_counter() - start)*10:.2f} ms per call")

# Compare to raw bmm
flat = X.view(B, C, -1)
start = time.perf_counter()
for _ in range(100):
    result = torch.bmm(R, flat)
if device == 'cuda':
    torch.cuda.synchronize()
print(f"raw bmm: {(time.perf_counter() - start)*10:.2f} ms per call")

angles = get_all_octahedral_angles()
start = time.perf_counter()
for _ in range(100):
    R = euler_angles_to_matrix(angles)
print(f"euler_angles_to_matrix: {(time.perf_counter() - start)*10:.2f} ms")

B = 10000
C = 3
X = torch.randn(B, C, 1, 1, 1)
R = torch.randn(B, 3, 3)
flat = X[:, :, 0, 0, 0]  # [B, 3]

# Warmup
for _ in range(3):
    _ = torch.einsum('bij,bj->bi', R, flat)

start = time.perf_counter()
for _ in range(100):
    result = torch.einsum('bij,bj->bi', R, flat)
print(f"einsum: {(time.perf_counter() - start)*10:.2f} ms per call")

input_tensor = torch.randn(10000, 9)


octahedral_angles = get_all_octahedral_angles().to(input_tensor.device)
random_angles = octahedral_angles[torch.randint(0, len(octahedral_angles), (input_tensor.shape[0],))]

time_start = time.time()
rotated_inputs = rotate_octahedral_exact(
    input_tensor,
    random_angles,
    rotate_channels=True,
    mode="pairwise",
)

print(f"Time taken: {time.time() - time_start} seconds")






