"""Region-averaged barycentric coordinates of the resolved filtered
Reynolds stress anisotropy tensor, per the four training regions."""
import glob, os
import numpy as np
import tqdm

NUMPY_DIR = "/orcd/data/abodner/002/ryley/numpy_4000_sgs"
FS, NY = 4, 16

REGIONS = [
    (1000, "Near-wall",      "channel_nearwall"),
    (1000, "Channel-center", "channel_middle"),
    (5200, "Near-wall",      "channel5200_nearwall"),
    (5200, "Channel-center", "channel5200_middle"),
]

def box_filter_downsample(arr, factor):
    """Simple box filter downsampling by factor"""
    c, n, _, _ = arr.shape
    new_n = n // factor
    
    # Reshape and average
    reshaped = arr[:, :new_n*factor, :new_n*factor, :new_n*factor]
    reshaped = reshaped.reshape(c, new_n, factor, new_n, factor, new_n, factor)
    return reshaped.mean(axis=(2, 4, 6))

def region_barycentric(prefix):
    files = sorted(glob.glob(os.path.join(NUMPY_DIR, f"{prefix}_box_x*_z*_time_*.npy")))
    assert files, f"No files matching {prefix}"

    sum_u  = np.zeros((NY, 3))
    sum_uu = np.zeros((NY, 3, 3))
    n_per_y = 0

    for f in tqdm.tqdm(files):
        # raw (3, 64, 64, 64) with axes (c, x, y, z) -> filtered (3, 16, 16, 16)
        u = box_filter_downsample(np.load(f), FS)
        u = u.transpose(2, 0, 1, 3).reshape(NY, 3, -1)   # (y, c, V=x*z)
        sum_u  += u.sum(axis=2)
        sum_uu += np.einsum("yiv,yjv->yij", u, u)
        n_per_y += u.shape[2]

    mean_u  = sum_u  / n_per_y
    mean_uu = sum_uu / n_per_y
    R = mean_uu - np.einsum("yi,yj->yij", mean_u, mean_u)             # resolved Reynolds stress
    b = R / np.einsum("yii->y", R)[:, None, None] - np.eye(3) / 3.0   # anisotropy tensor
    evals = np.linalg.eigvalsh(b)[:, ::-1]                            # descending λ1, λ2, λ3

    C1c = evals[:, 0] - evals[:, 1]
    C2c = 2 * (evals[:, 1] - evals[:, 2])
    C3c = 3 * evals[:, 2] + 1
    return C1c.mean(), C2c.mean(), C3c.mean()


if __name__ == "__main__":
    print(f"{'Re_tau':>7} {'Region':>16} {'C1c':>7} {'C2c':>7} {'C3c':>7}")
    for re, name, prefix in REGIONS:
        c1, c2, c3 = region_barycentric(prefix)
        print(f"{re:>7} {name:>16} {c1:>7.3f} {c2:>7.3f} {c3:>7.3f}")