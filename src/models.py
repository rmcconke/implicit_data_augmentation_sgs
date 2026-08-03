import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from typing import Any, Callable, Dict, List, Sequence, Optional

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import BatchNorm, Gate

from irreps_utils import (
    gradients_to_irreps,
    irreps_to_symmetric_tensor,
    gradient_channels_to_matrix,
    matrix_to_gradient_channels,
    irreps_to_2ndorder_tensor,
    velocity_to_irreps,
    irreps_to_velocity,
)

from _models._escnn import sgsESCNN, srESCNN
from _models._eigenframe import sgsPointwiseEigenframeMLP
from _models._ecnn import sgsECNN

def get_model(model_config):
    if model_config.model_type == 'LinearInterpolator':
        return LinearInterpolator(
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srCNN':
        return srCNN(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srUpsampleCNN':
        return srUpsampleCNN(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels,
            upscale_factor=model_config.upscale_factor,
            is_3d=model_config.is_3d
        )
    elif model_config.model_type == 'srECNN':
        return srECNN(
            input_irreps=model_config.input_irreps,
            output_irreps=model_config.output_irreps,
            hidden_irreps=model_config.hidden_irreps,
            kernel_size=model_config.kernel_size,
            is_3d=model_config.is_3d,
            group=model_config.group,
            num_radial_basis=model_config.num_radial_basis,
            lmax=model_config.lmax,
            steps=model_config.steps,
            activation=model_config.activation,
            conv_block_kwargs=getattr(model_config, 'conv_block_kwargs', None),
        )
    elif model_config.model_type == 'srMLP':
        return srMLP(
            in_channels=model_config.input_channels,
            out_channels=model_config.output_channels,
            hidden_channels=model_config.hidden_channels
        )
    elif model_config.model_type == 'sgsCNN':
        return sgsCNN(
            hidden_channels=model_config.hidden_channels,
            kernel_size=model_config.kernel_size
        )
    elif model_config.model_type == 'sgsECNN':
        return sgsECNN(
            input_irreps=model_config.input_irreps,
            output_irreps=model_config.output_irreps,
            hidden_irreps=model_config.hidden_irreps,
            n_layers=model_config.n_layers,
            kernel_size=model_config.kernel_size,
            num_radial_basis=model_config.num_radial_basis,
            lmax=model_config.lmax,
            diameter=model_config.diameter,
            steps=model_config.steps,
            conv_block_kwargs=model_config.conv_block_kwargs,
        )   
    elif model_config.model_type == 'sgsESCNN':
        return sgsESCNN(
            n_reg_repr=model_config.n_reg_repr,
            n_layers=model_config.n_layers,
        )
    elif model_config.model_type == 'sgsPointwiseMLP':
        return sgsPointwiseMLP(
            layers=model_config.layers
        )
    elif model_config.model_type == 'sgsPointwiseEigenframeMLP':
        return sgsPointwiseEigenframeMLP(
            layers=model_config.layers,
        )

    else:
        raise ValueError(f"Model type {model_config.model_type} not supported")




class EquivariantUpsampleBlock(nn.Module):
    """
    Closely follows srCNN stage: Upsample -> Conv -> Activation -> Conv -> Activation.
    Uses E3NNConvBlock for each conv/activation pair.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        scale_factor: float,
        irreps_sh: o3.Irreps,
        diameter: float,
        num_radial_basis: int,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        apply_batchnorm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=False)
        self.block1 = E3NNConvBlock(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
            apply_batchnorm=apply_batchnorm,
            activation=activation,
        )
        self.block2 = E3NNConvBlock(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh=irreps_sh,
            diameter=diameter,
            num_radial_basis=num_radial_basis,
            steps=steps,
            apply_batchnorm=apply_batchnorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.block1(x)
        x = self.block2(x)
        return x




class LinearInterpolator(nn.Module):
    def __init__(self, upscale_factor, is_3d: bool = False):
        super(LinearInterpolator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, 
                                    mode='trilinear' if is_3d else 'bilinear', 
                                    align_corners=True)
        self.dummy_param = nn.Parameter(torch.tensor(0.0)) 
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        return x + 0.0 * self.dummy_param


class srCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int = 4,
                 is_3d: bool = False) -> None:
        super().__init__()
        self.is_3d = is_3d
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        self.upsample_mode = 'trilinear' if is_3d else 'bilinear'

        # Input feature extraction
        self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

        # Build progressive upsample factors (e.g., 8 -> [2,2,2], 6 -> [2,3], 3 -> [3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        # Per-stage upsample + refinement blocks
        upsample_blocks = []
        for factor in self.upsample_factors:
            upsample_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=factor, mode=self.upsample_mode, align_corners=False),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
            ))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        # Output projection
        self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        
        # Project probes to same output channels 
        self.intermediate_projection = conv_layer(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode='reflect')

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward:
            x:  (N,C,D,H,W) if 3D, or (N,C,H,W) if 2D
            return_intermediate: if True, also return feature taps and 1x1 projections per scale.

        Returns:
            y  (and optionally feats, proj) where feats/proj have keys: 'lr', f'x{cum}'
        """
        x = self.relu(self.conv1(x))

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor

        x = self.conv_out(x)
        return x
    
class srUpsampleCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 upscale_factor: int = 4,
                 is_3d: bool = False) -> None:
        super().__init__()
        self.is_3d = is_3d
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        self.upsample_mode = 'trilinear' if is_3d else 'bilinear'

        # Input feature extraction
        self.conv1 = conv_layer(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

        # Build progressive upsample factors (e.g., 8 -> [2,2,2], 6 -> [2,3], 3 -> [3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        # Per-stage upsample + refinement blocks
        upsample_blocks = []
        for factor in self.upsample_factors:
            upsample_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=factor, mode=self.upsample_mode, align_corners=False),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                conv_layer(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
                nn.ReLU(inplace=True),
            ))
        self.upsample_blocks = nn.ModuleList(upsample_blocks)

        # Output projection
        self.conv_out = conv_layer(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        
        # Project probes to same output channels 
        self.intermediate_projection = conv_layer(hidden_channels, out_channels, kernel_size=1, bias=True, padding_mode='reflect')

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward:
            x:  (N,C,D,H,W) if 3D, or (N,C,H,W) if 2D
            return_intermediate: if True, also return feature taps and 1x1 projections per scale.

        Returns:
            y  (and optionally feats, proj) where feats/proj have keys: 'lr', f'x{cum}'
        """
        # Upsample input for global residual
        x_up = F.interpolate(x, scale_factor=self.upscale_factor, mode=self.upsample_mode, align_corners=False)
        
        x = self.relu(self.conv1(x))

        cumulative = 1
        for factor, upsample_block in zip(self.upsample_factors, self.upsample_blocks):
            x = upsample_block(x)
            cumulative *= factor

        x = self.conv_out(x)
        return x + x_up

class srECNN(nn.Module):
    def __init__(
        self,
        input_irreps,
        output_irreps,
        hidden_irreps,
        upscale_factor: int,
        is_3d: bool = True,
        group: str = "so3",
        *,
        lmax: int = 3,
        num_radial_basis: int = 6,
        kernel_diameter: float = 3.0,
        steps: Sequence[float] = (1.0, 1.0, 1.0),
        activation: str = "relu",
        conv_block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not is_3d:
            raise NotImplementedError(
                "srECNN currently supports only 3D inputs when using e3nn."
            )

        self.is_3d = True
        self.group = group
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"

        self.steps = tuple(float(s) for s in steps)
        self.irreps_in = o3.Irreps(input_irreps)
        if isinstance(hidden_irreps, (str, o3.Irreps)):
            self.irreps_hidden = o3.Irreps(hidden_irreps)
        else:
            raise TypeError("srECNN expects a single hidden_irreps specification (str or Irreps).")
        self.irreps_out = o3.Irreps(output_irreps)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.kernel_diameter = float(kernel_diameter)
        self.num_radial_basis = int(num_radial_basis)
        if conv_block_kwargs is None:
            conv_block_kwargs = {}
        self.apply_batchnorm = conv_block_kwargs.get("apply_batchnorm", True)
        self.activation_name = conv_block_kwargs.get("activation", "silu")
        self.input_block = E3NNConvBlock(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            diameter=self.kernel_diameter,
            num_radial_basis=self.num_radial_basis,
            steps=self.steps,
            apply_batchnorm=self.apply_batchnorm,
            activation=self.activation_name,
        )

        # Progressive factors (e.g., 8 -> [2, 2, 2], 6 -> [2, 3])
        upsample_factors: list[int] = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors

        self.upsample_blocks = nn.ModuleList(
            [
                EquivariantUpsampleBlock(
                    irreps=self.irreps_hidden,
                    scale_factor=factor,
                    irreps_sh=self.irreps_sh,
                    diameter=self.kernel_diameter,
                    num_radial_basis=self.num_radial_basis,
                    steps=self.steps,
                    apply_batchnorm=self.apply_batchnorm,
                    activation=self.activation_name,
                )
                for factor in self.upsample_factors
            ]
        )

        self.output_linear = Linear(self.irreps_hidden, self.irreps_out)
        self.intermediate_linear = Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x:  tensor of shape (N, C, D, H, W)
        If return_intermediate=True: also return hidden feature taps and 1x1 projections per scale.
        """
        hidden = self.input_block(x)

        cumulative = 1
        for factor, stage in zip(self.upsample_factors, self.upsample_blocks):
            hidden = stage(hidden)
            cumulative *= factor

        y = _apply_linear(self.output_linear, hidden)
        return y


# -----------------------------
# SGS models
# -----------------------------

class sgsPointwiseMLP(nn.Module):
    def __init__(self, in_channels: int=9, out_channels: int=5, layers: Sequence[int]=[8,8]):
        super().__init__()
        if isinstance(layers, int):
            hidden_layers = [layers]
        else:
            hidden_layers = list(layers)

        modules: List[nn.Module] = []
        prev_dim = in_channels
        for hidden_dim in hidden_layers:
            modules.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.output_layer = nn.Linear(prev_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class sgsCNN(nn.Module):
    def __init__(self, in_channels=9, out_channels=5, hidden_channels: Sequence[int] = (32, 32, 32), kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        layers: List[nn.Module] = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers += [
                nn.Conv3d(prev_ch, ch, kernel_size, padding=padding),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True)
            ]
            prev_ch = ch

        layers.append(nn.Conv3d(prev_ch, out_channels, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 5, D, H, W): [dev_xx, dev_yy, xy, xz, yz]

