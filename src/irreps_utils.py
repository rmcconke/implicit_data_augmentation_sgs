import torch

from e3nn.io import CartesianTensor
#torch.set_default_dtype(torch.float64)


_CARTESIAN = CartesianTensor("ij")
_CARTESIAN_VECTOR = CartesianTensor("i")
_SYMMETRIC_CARTESIAN = CartesianTensor("ij=ji")


def _cartesian_matrix_to_irreps(matrix: torch.Tensor, cartesian_tensor: CartesianTensor) -> torch.Tensor:
    spatial_axes = list(range(3, matrix.ndim))
    permute_order = [0] + spatial_axes + [1, 2]
    cartesian = matrix.permute(*permute_order)
    irreps = cartesian_tensor.from_cartesian(cartesian)
    return irreps.moveaxis(-1, 1)

def _cartesian_vector_to_irreps(vector: torch.Tensor, cartesian_tensor: CartesianTensor) -> torch.Tensor:
    spatial_axes = list(range(1, vector.ndim))
    permute_order = [0] + spatial_axes
    cartesian = vector.permute(*permute_order)
    irreps = cartesian_tensor.from_cartesian(cartesian)
    return irreps.moveaxis(-1, 1)


def _irreps_to_cartesian_vector(irreps: torch.Tensor, cartesian_tensor: CartesianTensor) -> torch.Tensor:
    irreps = irreps.moveaxis(1, -1)
    cartesian = cartesian_tensor.to_cartesian(irreps)
    return cartesian.moveaxis(1, -1)


def _irreps_to_cartesian_matrix(irreps: torch.Tensor, cartesian_tensor: CartesianTensor) -> torch.Tensor:
    irreps = irreps.moveaxis(1, -1)
    cartesian = cartesian_tensor.to_cartesian(irreps)
    n_dims = cartesian.ndim
    permute_order = [0, n_dims - 2, n_dims - 1] + list(range(1, n_dims - 2))
    return cartesian.permute(*permute_order)


def gradient_channels_to_matrix(gradients: torch.Tensor) -> torch.Tensor:
    """
    Interpret 9-channel tensors as velocity gradients arranged row-major:
    [
      du/dx, du/dy, du/dz,
      dv/dx, dv/dy, dv/dz,
      dw/dx, dw/dy, dw/dz
    ]
    -> matrix with shape [..., 3, 3].
    """
    if gradients.shape[1] != 9:
        raise ValueError(f"Expected 9 channels for gradient tensors, got {gradients.shape[1]}")
    spatial_shape = gradients.shape[2:]
    return gradients.reshape(gradients.shape[0], 3, 3, *spatial_shape)

def matrix_to_gradient_channels(matrix: torch.Tensor) -> torch.Tensor:
    spatial_shape = matrix.shape[3:]
    return matrix.reshape(matrix.shape[0], 9, *spatial_shape)

def gradients_to_irreps(gradients: torch.Tensor) -> torch.Tensor:
    matrix = gradient_channels_to_matrix(gradients)
    return _cartesian_matrix_to_irreps(matrix, _CARTESIAN)

def velocity_to_irreps(velocity: torch.Tensor) -> torch.Tensor:
    return _cartesian_vector_to_irreps(velocity, _CARTESIAN_VECTOR)

def irreps_to_velocity(irreps: torch.Tensor) -> torch.Tensor:
    return _irreps_to_cartesian_vector(irreps, _CARTESIAN_VECTOR)

def irreps_to_symmetric_tensor(irreps: torch.Tensor) -> torch.Tensor:
    matrix = _irreps_to_cartesian_matrix(irreps, _SYMMETRIC_CARTESIAN)
    xx = matrix[:, 0, 0]
    yy = matrix[:, 1, 1]
    zz = matrix[:, 2, 2]
    xy = matrix[:, 0, 1]
    xz = matrix[:, 0, 2]
    yz = matrix[:, 1, 2]
    return torch.stack([xx, yy, zz, xy, xz, yz], dim=1)

def irreps_to_2ndorder_tensor(irreps: torch.Tensor) -> torch.Tensor:
    matrix = _irreps_to_cartesian_matrix(irreps, _CARTESIAN)
    xx = matrix[:, 0, 0]
    yy = matrix[:, 1, 1]
    zz = matrix[:, 2, 2]
    xy = matrix[:, 0, 1]
    xz = matrix[:, 0, 2]
    yz = matrix[:, 1, 2]

    yx = matrix[:, 1, 0]
    zx = matrix[:, 2, 0]
    zy = matrix[:, 2, 1]
    
    return torch.stack([xx, yy, zz, xy, xz, yz, yx, zx, zy], dim=1)

