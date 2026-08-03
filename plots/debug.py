import numpy as np
import sys
sys.path.append('./src/')
from transformations import get_all_octahedral_angles, euler_angles_to_matrix

angles_all = get_all_octahedral_angles()
mats = euler_angles_to_matrix(angles_all).numpy()
target = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

for i, M in enumerate(mats):
    if np.allclose(M, target, atol=1e-6):
        print(f"Index {i}: angles = {angles_all[i].tolist()}")
        print(M)