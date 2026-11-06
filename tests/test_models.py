from dataclasses import dataclass
from typing import Callable
import torch
from tqdm import tqdm
import sys
sys.path.append("./src/")
from escnn import nn as enn
from models import sgsESCNN, sgsECNN, sgsPointwiseEigenframeMLP, sgsPointwiseMLP, srESCNN
from transformations import rotate_octahedral_exact
from transformations import get_all_octahedral_angles
import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False,specialized_code=False,optimize_einsums=False,)
torch.set_default_dtype(torch.float32)
import pytest
import matplotlib.pyplot as plt

from eval import octahedral_equivariance_error

DEVICE = torch.device("cuda:0")

def build_sgsESCNN(device="cpu"):
    return sgsESCNN(kernel_size=3, batchnorm=False, n_reg_repr=3, n_layers=1).to(device).eval()

def build_sgsECNN(device="cpu"):
    return sgsECNN(input_irreps="0e+1o+2e",
                   output_irreps="2e",
                   hidden_irreps="8x0e + 8x1o + 8x2e",
                   n_layers=1,
                   kernel_size=3,
                   num_radial_basis=3,
                   lmax=2,
                   steps=(1.0, 1.0, 1.0),
                   diameter=3.0,
                   conv_block_kwargs={"apply_batchnorm": False, "activation_name": "relu"}).to(device).eval()

def build_sgsPointwiseEigenframeMLP(device="cpu"):
    return sgsPointwiseEigenframeMLP(layers=[20]).to(device).eval()

def build_sgsPointwiseMLP(device="cpu"):
    return sgsPointwiseMLP(in_channels=9, out_channels=5, layers=[20]).to(device).eval()

def build_srESCNN(device="cpu"):
    return srESCNN(upscale_factor=4, n_reg_repr=2, kernel_size=3).to(device).eval()

def sample_input_output(pointwise, vector, device="cpu"):
    if vector:
        coords = torch.linspace(0, 2*torch.pi, 64)
        z, yc, xc = torch.meshgrid(coords, coords, coords, indexing='ij')
        vx = torch.sin(xc)
        vy = torch.sin(yc)
        vz = torch.sin(z)
        y = torch.stack([vx, vy, vz], dim=0).unsqueeze(0)
        x = torch.nn.functional.avg_pool3d(y, kernel_size=4, stride=4)
        return x.to(device), y.to(device)
    if pointwise:
        x = torch.randn(1, 9)
        y = compute_traceless_symmetric_part(x, pointwise)
        return x.to(device), y.to(device)
    else:
        x = torch.randn(1, 9, 14, 14, 14)
        y = compute_traceless_symmetric_part(x, pointwise)
        return x.to(device), y.to(device)

def compute_traceless_symmetric_part(x, pointwise):
    if not pointwise:
        B = x.shape[0]
        spatial = x.shape[2:]
        A = x.view(B, 3, 3, *spatial)
        A = A.permute(0, 3, 4, 5, 1, 2)
        A_sym = 0.5 * (A + A.transpose(-1, -2))
        xx = A_sym[..., 0, 0]
        yy = A_sym[..., 1, 1]
        zz = A_sym[..., 2, 2]
        xy = A_sym[..., 0, 1]
        xz = A_sym[..., 0, 2]
        yz = A_sym[..., 1, 2]
        trace = (xx + yy + zz) / 3
        dev_xx = xx - trace
        dev_yy = yy - trace
        return torch.stack([dev_xx, dev_yy, xy, xz, yz], dim=1)
    else:
        A = x.view(-1, 3, 3)
        A_sym = 0.5 * (A + A.transpose(-1, -2))
        xx = A_sym[:, 0, 0]
        yy = A_sym[:, 1, 1]
        zz = A_sym[:, 2, 2]
        xy = A_sym[:, 0, 1]
        xz = A_sym[:, 0, 2]
        yz = A_sym[:, 1, 2]
        trace = (xx + yy + zz) / 3
        dev_xx = xx - trace
        dev_yy = yy - trace
        return torch.stack([dev_xx, dev_yy, xy, xz, yz], dim=1)

@dataclass
class ModelSpec:
    name: str
    build: Callable
    pointwise: bool
    is_equivariant: bool
    vector: bool

MODEL_SPECS = [
    ModelSpec("sgsESCNN", build_sgsESCNN, pointwise=False, is_equivariant=True, vector=False),
    ModelSpec("srESCNN", build_srESCNN, pointwise=False, is_equivariant=True, vector=True),
    #ModelSpec("sgsECNN", build_sgsECNN, pointwise=False, is_equivariant=True, vector=False),
    ModelSpec("sgsPointwiseEigenframeMLP", build_sgsPointwiseEigenframeMLP, pointwise=True, is_equivariant=True, vector=False),
    ModelSpec("sgsPointwiseMLP", build_sgsPointwiseMLP, pointwise=True, is_equivariant=False, vector=False),
]

ANGLES = get_all_octahedral_angles()

@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.name)
def test_equivariance(spec):
    model = spec.build(device="cpu")
    x, _ = sample_input_output(spec.pointwise, spec.vector, device="cpu")

    g_list = [ANGLES[10], ANGLES[11], ANGLES[12]]
    for g in g_list:
        y = model(x)
        g_of_y = rotate_octahedral_exact(y, g)

        diff = (y - g_of_y).abs().max().item()
        assert diff > 1e-5, f"{spec.name}: transform has no effect"

        err = octahedral_equivariance_error(x, model).abs().max().item()
        print(f"{spec.name} equivariance error {err:.2e}")

        if spec.is_equivariant:
            assert err < 1e-5, f"{spec.name} equivariance error {err:.2e}"
        else:
            assert err > 1e-5, f"{spec.name} unexpectedly equivariant"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.name)
def test_memorization(spec):
    model = spec.build(device=DEVICE)
    model.train()
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    x, y = sample_input_output(spec.pointwise, spec.vector, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in tqdm(range(200), desc=f"Training {spec.name}"):
        y_pred = model(x)
        loss = (y - y_pred).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print("Losses:", loss.item())
    plt.plot(losses)
    plt.savefig(f"tests/outputs/losses_{spec.name}.png")
    plt.close()

    assert loss.item() < 1e-2, "Memorization failed"