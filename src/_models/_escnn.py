import torch
import torch.nn as torch_nn
from escnn import gspaces
from escnn import nn
torch.set_default_dtype(torch.float32)

# Setup
r3_act = gspaces.octaOnR3()
O = r3_act.fibergroup

# Build representations
vec = O.irrep(1)
rank2_full = vec.tensor(vec)

# COB for full 9-component tensor (input)
COB_full = torch.tensor(rank2_full.change_of_basis, dtype=torch.float32)
COB_inv_full = torch.tensor(rank2_full.change_of_basis_inv, dtype=torch.float32)

# Convention: [xx, yy, zz, xy, xz, yz] -> [xx, xy, xz, yx, yy, yz, zx, zy, zz]
#              0   1   2   3   4   5       0   1   2   3   4   5   6   7   8

expand_sym_to_full = torch.zeros(9, 6)
expand_sym_to_full[0, 0] = 1.0  # xx -> xx
expand_sym_to_full[4, 1] = 1.0  # yy -> yy
expand_sym_to_full[8, 2] = 1.0  # zz -> zz
expand_sym_to_full[1, 3] = 1.0  # xy -> xy
expand_sym_to_full[3, 3] = 1.0  # xy -> yx
expand_sym_to_full[2, 4] = 1.0  # xz -> xz
expand_sym_to_full[6, 4] = 1.0  # xz -> zx
expand_sym_to_full[5, 5] = 1.0  # yz -> yz
expand_sym_to_full[7, 5] = 1.0  # yz -> zy

contract_full_to_sym = torch.zeros(6, 9)
contract_full_to_sym[0, 0] = 1.0  # xx
contract_full_to_sym[1, 4] = 1.0  # yy
contract_full_to_sym[2, 8] = 1.0  # zz
contract_full_to_sym[3, 1] = 0.5  # (xy + yx)/2
contract_full_to_sym[3, 3] = 0.5
contract_full_to_sym[4, 2] = 0.5  # (xz + zx)/2
contract_full_to_sym[4, 6] = 0.5
contract_full_to_sym[5, 5] = 0.5  # (yz + zy)/2
contract_full_to_sym[5, 7] = 0.5

# Build COB for symmetric tensors: 6 symmetric Cartesian <-> 6 symmetric irreps
# Symmetric irreps are [A1, T2(3), E(2)] = indices [0, 4,5,6, 7,8] from full irreps
sym_irrep_indices = [0, 4, 5, 6, 7, 8]
print(f"torch default dtype: {torch.get_default_dtype()}")
COB_inv_sym = (COB_inv_full @ expand_sym_to_full)[sym_irrep_indices, :]  # 6x6
COB_sym = torch.linalg.inv(COB_inv_sym)  # 6x6

P = torch.tensor([[0., 1., 0.],
                  [0., 0., 1.],
                  [1., 0., 0.]], dtype=torch.float32)
P9 = torch.kron(P, P)
P9_inv = torch.kron(P.T, P.T)

# Full chain: your Cartesian -> ESCNN Cartesian -> irrep basis
COB_inv_full_corrected = COB_inv_full @ P9       # your xyz -> irreps
COB_full_corrected = P9_inv @ COB_full            # irreps -> your xyz
COB_inv_sym_corrected = (COB_inv_full_corrected @ expand_sym_to_full)[sym_irrep_indices, :]
COB_sym_corrected = torch.linalg.inv(COB_inv_sym_corrected)

# Convention: [dev_xx, dev_yy, xy, xz, yz] -> full 9-component tensor
#              0       1       2   3   4
# Full order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
#              0   1   2   3   4   5   6   7   8

expand_traceless_to_full = torch.zeros(9, 5)
expand_traceless_to_full[0, 0] = 1.0   # xx <- dev_xx
expand_traceless_to_full[4, 1] = 1.0   # yy <- dev_yy
expand_traceless_to_full[8, 0] = -1.0  # zz <- -dev_xx - dev_yy
expand_traceless_to_full[8, 1] = -1.0  
expand_traceless_to_full[1, 2] = 1.0   # xy
expand_traceless_to_full[3, 2] = 1.0   # yx = xy
expand_traceless_to_full[2, 3] = 1.0   # xz
expand_traceless_to_full[6, 3] = 1.0   # zx = xz
expand_traceless_to_full[5, 4] = 1.0   # yz
expand_traceless_to_full[7, 4] = 1.0   # zy = yz

# Traceless symmetric irreps: indices 1-5 (exclude 0 which is A1/trace)
traceless_sym_irrep_indices = [4, 5, 6, 7, 8]

COB_inv_traceless_corrected = (COB_inv_full_corrected @ expand_traceless_to_full)[traceless_sym_irrep_indices, :]
COB_traceless_corrected = torch.linalg.inv(COB_inv_traceless_corrected)  # 5x5
COB_traceless_corrected = COB_sym_corrected[[0, 1, 3, 4, 5], :][:, 1:]

class sgsESCNN(torch_nn.Module):
    def __init__(self, kernel_size=3, batchnorm=False, n_reg_repr = 2, n_layers = 3):
        super().__init__()

        self.feat_type_in = nn.FieldType(r3_act, [
            r3_act.irrep(0),
            r3_act.irrep(1),
            r3_act.irrep(-1),
            r3_act.irrep(2),
        ])

        self.feat_type_hid = nn.FieldType(
            r3_act, n_reg_repr * [r3_act.regular_repr]
        )

        self.feat_type_out = nn.FieldType(r3_act, [
            r3_act.irrep(-1),
            r3_act.irrep(2),
        ])

        self.layers = torch_nn.ModuleList()
        in_type = self.feat_type_in
        for _ in range(n_layers):
            modules = [
                nn.R3Conv(in_type, self.feat_type_hid, kernel_size=kernel_size,padding=1)
            ]
            if batchnorm:
                modules.append(nn.InnerBatchNorm(self.feat_type_hid))
            modules.append(nn.ReLU(self.feat_type_hid))

            self.layers.append(nn.SequentialModule(*modules))
            in_type = self.feat_type_hid

        self.to_out = nn.R3Conv(self.feat_type_hid, self.feat_type_out, kernel_size=1)

    def forward(self, inputs):
        inputs = torch.einsum('ij,bjdhw->bidhw', COB_inv_full_corrected.to(inputs.device), inputs)
        inputs = inputs.flip(-1).flip(-2)
        inputs = inputs.transpose(-1, -3)
        x_geom = nn.GeometricTensor(inputs, self.feat_type_in)
        for layer in self.layers:
            x_geom = layer(x_geom)
        x_geom = self.to_out(x_geom)
        y = x_geom.tensor.transpose(-1, -3)
        y = y.flip(-2).flip(-1)
        y = torch.cat([torch.zeros_like(y[:, 0:1]), y], dim=1)
        y = torch.einsum('ij,bjdhw->bidhw', COB_sym_corrected.to(y.device), y)
        y_traceless = torch.cat([
            y[:, 0:1],
            y[:, 1:2],
            y[:, 3:4],
            y[:, 4:5],
            y[:, 5:6],
        ], dim=1)
        return y_traceless


class srESCNN(torch_nn.Module):
    """
    Equivariant Super-Resolution CNN for vector fields using ESCNN.
    Uses octahedral symmetry group. Mirrors srCNN architecture.
    
    Input: (B, 3, D, H, W) - vector field in Cartesian coordinates
    Output: (B, 3, D*scale, H*scale, W*scale) - upscaled vector field
    """
    
    def __init__(
        self,
        upscale_factor: int = 4,
        n_reg_repr: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        
        self.upscale_factor = int(upscale_factor)
        assert self.upscale_factor >= 1, "upscale_factor must be >= 1"
        
        # Vector field type: irrep(1) is the 3D vector representation
        self.feat_type_in = nn.FieldType(r3_act, [r3_act.irrep(1)])
        
        # Hidden: use regular representation (group convolution)
        self.feat_type_hid = nn.FieldType(r3_act, n_reg_repr * [r3_act.regular_repr])
        
        # Output: back to vector
        self.feat_type_out = nn.FieldType(r3_act, [r3_act.irrep(1)])
        
        # Build progressive upsample factors (e.g., 8 -> [2,2,2], 6 -> [2,3])
        upsample_factors = []
        remaining = self.upscale_factor
        while remaining > 1 and remaining % 2 == 0:
            upsample_factors.append(2)
            remaining //= 2
        if remaining > 1:
            upsample_factors.append(remaining)
        self.upsample_factors = upsample_factors
        
        # Input feature extraction (like conv1 + relu in srCNN)
        self.conv_in = nn.R3Conv(
            self.feat_type_in,
            self.feat_type_hid,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.relu_in = nn.ReLU(self.feat_type_hid, inplace=True)
        
        # Upsample blocks: each does Upsample -> Conv -> ReLU -> Conv -> ReLU
        self.upsample_blocks = torch_nn.ModuleList()
        for factor in self.upsample_factors:
            block = nn.SequentialModule(
                nn.R3Upsampling(self.feat_type_hid, scale_factor=factor, mode='trilinear', align_corners=False),
                nn.R3Conv(
                    self.feat_type_hid,
                    self.feat_type_hid,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                ),
                nn.ReLU(self.feat_type_hid, inplace=True),
                nn.R3Conv(
                    self.feat_type_hid,
                    self.feat_type_hid,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                ),
                nn.ReLU(self.feat_type_hid, inplace=True),
            )
            self.upsample_blocks.append(block)
        
        # Output projection
        self.conv_out = nn.R3Conv(
            self.feat_type_hid,
            self.feat_type_out,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, D, H, W) vector field
            
        Returns:
            (B, 3, D*scale, H*scale, W*scale) upscaled vector field
        """
        # Transform from your xyz convention to ESCNN convention
        # Vector components: permute (x,y,z) -> (y,z,x)
        x = torch.einsum('ij,bjdhw->bidhw', P.to(x.device), x)
        # Spatial axes: flip and transpose to match ESCNN
        x = x.flip(-1).flip(-2)              # (..., X, Y, Z) -> (..., X, -Y, -Z)
        x = x.transpose(-1, -3)              # (..., X, -Y, -Z) -> (..., -Z, -Y, X)
        
        x_geom = nn.GeometricTensor(x, self.feat_type_in)
        
        # Input feature extraction
        x_geom = self.relu_in(self.conv_in(x_geom))
        
        # Progressive upsampling
        for upsample_block in self.upsample_blocks:
            x_geom = upsample_block(x_geom)
        
        # Output projection
        x_geom = self.conv_out(x_geom)
        
        # Transform back from ESCNN convention to your xyz convention
        y = x_geom.tensor.transpose(-1, -3)  # (..., X, -Y, -Z)
        y = y.flip(-2).flip(-1)              # (..., X, Y, Z)
        # Vector components: permute back (y,z,x) -> (x,y,z)
        y = torch.einsum('ij,bjdhw->bidhw', P.T.to(y.device), y)

        return y