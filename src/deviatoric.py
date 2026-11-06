import torch

# Orthonormal basis for symmetric traceless tensors, expressed in the five stored
# components (d_xx, d_yy, xy, xz, yz) with d_zz = -d_xx - d_yy. The plain sum of
# squares of those five is not the Frobenius norm and is not rotationally
# invariant; the sum of squares of the coefficients below is both.
_SQRT2 = 2.0 ** 0.5
_SQRT32 = 1.5 ** 0.5

ORTHONORMAL_BASIS = torch.tensor([
    [1 / _SQRT2, -1 / _SQRT2, 0.0, 0.0, 0.0],
    [_SQRT32, _SQRT32, 0.0, 0.0, 0.0],
    [0.0, 0.0, _SQRT2, 0.0, 0.0],
    [0.0, 0.0, 0.0, _SQRT2, 0.0],
    [0.0, 0.0, 0.0, 0.0, _SQRT2],
])


def to_orthonormal(tensor, dim=1):
    """Map the five stored deviatoric components onto the orthonormal basis.

    Euclidean operations on the result (MSE, dot products, norms) become
    rotationally invariant. The component dimension is 1 for both pointwise
    [B, 5] and field [B, 5, D, H, W] inputs.
    """
    basis = ORTHONORMAL_BASIS.to(device=tensor.device, dtype=tensor.dtype)
    return (tensor.movedim(dim, -1) @ basis.T).movedim(-1, dim)


def frobenius_norm(tensor, dim=1):
    """Frobenius norm of the deviatoric stress held in the five stored components."""
    return to_orthonormal(tensor, dim=dim).pow(2).sum(dim=dim).sqrt()


def invariant_criterion(criterion):
    """Wrap a loss so it acts on the orthonormal basis rather than the raw components."""
    def wrapped(pred, target):
        return criterion(to_orthonormal(pred), to_orthonormal(target))
    return wrapped
