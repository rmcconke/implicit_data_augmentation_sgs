"""Sanity checks for the per-y Reynolds-stress accumulator in
region_barycentric. Synthetic data only; no disk I/O."""
import numpy as np

NY, NX, NZ = 16, 16, 16
N_BOXES = 5


def streaming_stats(boxes):
    """Per-box accumulate, matching the region_barycentric implementation."""
    sum_u  = np.zeros((NY, 3))
    sum_uu = np.zeros((NY, 3, 3))
    n_per_y = 0
    for box in boxes:  # each box has axes (c, x, y, z)
        u = box.transpose(2, 0, 1, 3).reshape(NY, 3, -1)  # (y, c, V=x*z)
        sum_u  += u.sum(axis=2)
        sum_uu += np.einsum("yiv,yjv->yij", u, u)
        n_per_y += u.shape[2]
    mean_u = sum_u / n_per_y
    R = sum_uu / n_per_y - np.einsum("yi,yj->yij", mean_u, mean_u)
    return mean_u, R


def naive_stats(boxes):
    """Independent implementation: stack all samples, mean-subtract, cov."""
    stack = np.stack(boxes, axis=0)  # (n_boxes, c, x, y, z)
    per_y = stack.transpose(3, 1, 0, 2, 4).reshape(NY, 3, -1)  # (y, c, V)
    mean_u = per_y.mean(axis=2)
    fluct = per_y - mean_u[:, :, None]
    R = np.einsum("yiv,yjv->yij", fluct, fluct) / fluct.shape[2]
    return mean_u, R


def test_matches_naive():
    """Streaming and mean-subtracted implementations agree on random data."""
    rng = np.random.default_rng(0)
    boxes = [rng.standard_normal((3, NX, NY, NZ)) for _ in range(N_BOXES)]
    mean_s, R_s = streaming_stats(boxes)
    mean_n, R_n = naive_stats(boxes)
    assert np.allclose(mean_s, mean_n), "means differ from naive"
    assert np.allclose(R_s, R_n), "Reynolds stress differs from naive"


def test_y_varying_mean():
    """u_x = y_index everywhere, no fluctuations. Mean should recover the
    profile; Reynolds stress should be zero. Catches spatial-axis mixups."""
    boxes = []
    for _ in range(3):
        u = np.zeros((3, NX, NY, NZ))
        u[0] = np.arange(NY, dtype=float)[None, :, None]  # u_x = y-index
        boxes.append(u)
    mean, R = streaming_stats(boxes)
    expected = np.zeros((NY, 3))
    expected[:, 0] = np.arange(NY)
    assert np.allclose(mean, expected), f"mean profile wrong: {mean[:, 0]}"
    assert np.allclose(R, 0, atol=1e-8), f"expected zero R, got max |R|={np.abs(R).max():.2e}"


def test_prescribed_anisotropy():
    """Diagonal Reynolds stress (4, 1, 1); check we recover each component."""
    rng = np.random.default_rng(2)
    target = np.array([4.0, 1.0, 1.0])
    scale = np.sqrt(target)
    boxes = [rng.standard_normal((3, NX, NY, NZ)) * scale[:, None, None, None]
             for _ in range(50)]
    _, R = streaming_stats(boxes)
    diag = np.diagonal(R, axis1=1, axis2=2).mean(axis=0)
    assert np.allclose(diag, target, rtol=0.05), f"{diag} != {target}"


def test_isotropic_barycentric():
    """Isotropic Gaussian velocity should give barycentric C3c near 1
    and C1c + C2c + C3c = 1 exactly."""
    rng = np.random.default_rng(1)
    boxes = [rng.standard_normal((3, NX, NY, NZ)) for _ in range(200)]
    _, R = streaming_stats(boxes)
    b = R / np.einsum("yii->y", R)[:, None, None] - np.eye(3) / 3.0
    evals = np.linalg.eigvalsh(b)[:, ::-1]
    C1c = evals[:, 0] - evals[:, 1]
    C2c = 2 * (evals[:, 1] - evals[:, 2])
    C3c = 3 * evals[:, 2] + 1
    assert np.allclose(C1c + C2c + C3c, 1.0), "barycentric coords don't sum to 1"
    assert C3c.mean() > 0.95, f"expected C3c near 1, got {C3c.mean():.3f}"


if __name__ == "__main__":
    test_matches_naive()
    test_y_varying_mean()
    test_prescribed_anisotropy()
    test_isotropic_barycentric()
    print("all tests passed")