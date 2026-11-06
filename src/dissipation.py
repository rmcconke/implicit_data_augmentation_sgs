import torch

from deviatoric import to_orthonormal

# Published JHTDB channel parameters. Re_tau = u_tau / nu with half-height 1, so
# the nominal "5200" dataset is really 5186; the box geometry in
# dataset_creation/ reproduces both values to within a wall unit.
WALL_UNITS = {
    'channel':      {'u_tau': 0.0499,      'nu': 5.0e-5},
    'channel5200':  {'u_tau': 4.14872e-2,  'nu': 8.0e-6},
}


def strain_deviator(grad_hat):
    """Deviatoric strain rate from the 9 normalized velocity gradient components."""
    g = grad_hat
    s11, s22, s33 = g[:, 0], g[:, 4], g[:, 8]
    trace = (s11 + s22 + s33) / 3
    return torch.stack([
        s11 - trace,
        s22 - trace,
        0.5 * (g[:, 1] + g[:, 3]),
        0.5 * (g[:, 2] + g[:, 6]),
        0.5 * (g[:, 5] + g[:, 7]),
    ], dim=1)


def sgs_dissipation(tau_hat, grad_hat, G, delta):
    """eps_SGS = -tau^d_ij S_ij in dimensional units.

    tau_hat is normalized by delta^2 G^2 and grad_hat by G, so the contraction
    carries a factor delta^2 G^3. The orthonormal basis makes the five stored
    components a true Frobenius inner product.
    """
    contraction = (to_orthonormal(tau_hat) * to_orthonormal(strain_deviator(grad_hat))).sum(dim=1)
    return -contraction * delta**2 * G**3


def wall_scaled(eps, dataset_prefix):
    """Nondimensionalize a dissipation by nu / u_tau^4."""
    p = WALL_UNITS['channel5200' if '5200' in dataset_prefix else 'channel']
    return eps * p['nu'] / p['u_tau']**4
