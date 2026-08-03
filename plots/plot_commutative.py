import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import yaml
from box import Box

sys.path.append('./src/')
from data import get_test_file_lists, NumpyTimeSeriesDataset
from models import get_model
from transformations import rotate_octahedral_exact

torch.set_default_dtype(torch.float32)

# -------- config --------
DEVICE = 'cuda:0'
OUT_DIR = 'outputs/plots_final/commutative'
SLICE_Z = 7

MODELS = {
    'nearwall': {
        'aug':   'outputs/CNNaug_data/26_06_04_17:18:58_strong_minnow_nearwall_boxfilter_4x_sgs',
        'noaug': 'outputs/CNNnoaug_data/26_06_04_17:55:22_caring_guinea_nearwall_boxfilter_4x_sgs',
    },
    'middle': {
        'aug':   'outputs/CNNaug_data_middle/26_06_26_11:33:34_cosmic_weasel_middle_boxfilter_4x_sgs',
        'noaug': 'outputs/CNNnoaug_data_middle/26_06_26_11:23:23_just_ghost_middle_boxfilter_4x_sgs',
    },
}

EVAL_PARAMS = {
    'nearwall': {
        'input_prefix': 'channel_nearwall_grad_fs4',
        'target_prefix': 'channel_nearwall_sgs_fs4',
        'pattern': ['z4'],
        'times': [25, 26, 27, 28, 29],
    },
    'middle': {
        'input_prefix': 'channel_middle_grad_fs4',
        'target_prefix': 'channel_middle_sgs_fs4',
        'pattern': ['z4'],
        'times': [25, 26, 27, 28, 29],
    },
}

ANGLE = torch.tensor([[np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2]])

CMAP_FIELD = 'PuOr'
CMAP_GRAD  = 'RdBu'
CMAP_DIFF  = 'viridis'

VMIN_FIELD = -1.0
VMAX_FIELD =  1.0
VMIN_GRAD  = -1.0
VMAX_GRAD  =  1.0
VMIN_DIFF  =  0.0
VMAX_DIFF  =  1.0

GRAD_PANELS = {'x', 'gx'}

PANEL_SIZE_INCHES = 1.2


# -------- helpers --------
def load_config(run_dir):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        return Box(yaml.safe_load(f))


def load_model(run_dir, device):
    cfg = load_config(run_dir)
    ckpt = torch.load(os.path.join(run_dir, f'{cfg.barcode}_checkpoint.pt'),
                      map_location=device)
    model = get_model(cfg.model).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt['model'].items()})
    model.eval()
    return model, cfg, ckpt['scaling_factors']


def get_one_box(eval_params, scaling_factors, pointwise, device):
    input_files, target_files = get_test_file_lists(Box(eval_params))
    dataset = NumpyTimeSeriesDataset(
        input_file_list=[input_files[0]],
        target_file_list=[target_files[0]],
        scaling_factors=scaling_factors,
        augmentation_group=None,
        pointwise=pointwise,
        precompute_eigenframe_inputs=False,
        device=device,
    )
    return dataset.get_all_data()


def predict_field(model, cfg, inputs):
    G = 14
    with torch.inference_mode():
        if cfg.pointwise:
            n_voxels = G ** 3
            pred = model(inputs[:n_voxels]).reshape(G, G, G, 5).permute(3, 0, 1, 2)
        else:
            pred = model(inputs[0:1])[0]
    return pred


def tau33(field):
    return -field[0] - field[1]


def reshape_input_for_field(inputs, cfg):
    G = 14
    if cfg.pointwise:
        return inputs[:G ** 3].reshape(G, G, G, 9).permute(3, 0, 1, 2).unsqueeze(0)
    return inputs[0:1]


def save_panel(arr, path, vmin, vmax, cmap):
    fig, ax = plt.subplots(figsize=(PANEL_SIZE_INCHES, PANEL_SIZE_INCHES))
    ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_colorbar(vmin, vmax, cmap, label, path, ticks=None):
    fig, ax = plt.subplots(figsize=(0.5, PANEL_SIZE_INCHES * 2.2))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=ax)
    if ticks is not None:
        cbar.set_ticks(ticks)
    cbar.ax.set_xlabel(label, fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# -------- main --------
def make_commutative_panels_for_region(region):
    aug_dir   = MODELS[region]['aug']
    noaug_dir = MODELS[region]['noaug']

    if not aug_dir or not noaug_dir:
        print(f'Skipping region "{region}" (model paths not set)')
        return

    eval_params = EVAL_PARAMS[region]
    out_subdir = os.path.join(OUT_DIR, region)
    os.makedirs(out_subdir, exist_ok=True)

    print(f'\n=== Region: {region} ===')

    _, ref_cfg, ref_scaling = load_model(aug_dir, DEVICE)
    inputs, _ = get_one_box(eval_params, ref_scaling, ref_cfg.pointwise, DEVICE)

    x_field  = reshape_input_for_field(inputs, ref_cfg)
    gx_field = rotate_octahedral_exact(x_field, ANGLE, rotate_channels=True, mode="pairwise")

    panels = {}

    for tag, run_dir in [('noaug', noaug_dir), ('aug', aug_dir)]:
        model, cfg, scaling = load_model(run_dir, DEVICE)
        local_inputs, _ = get_one_box(eval_params, scaling, cfg.pointwise, DEVICE)

        fx = predict_field(model, cfg, local_inputs)

        local_x_field  = reshape_input_for_field(local_inputs, cfg)
        local_gx_field = rotate_octahedral_exact(local_x_field, ANGLE, rotate_channels=True, mode="pairwise")

        with torch.inference_mode():
            if cfg.pointwise:
                local_gx_pw = local_gx_field[0].permute(1, 2, 3, 0).reshape(-1, 9)
                fgx_pw = model(local_gx_pw)
                G = 14
                fgx = fgx_pw.reshape(G, G, G, 5).permute(3, 0, 1, 2)
            else:
                fgx = model(local_gx_field)[0]

        gfx = rotate_octahedral_exact(fx.unsqueeze(0), ANGLE, rotate_channels=True, mode="pairwise")[0]

        panels[f'fx_{tag}']  = tau33(fx).cpu().numpy()[:, :, SLICE_Z]
        panels[f'fgx_{tag}'] = tau33(fgx).cpu().numpy()[:, :, SLICE_Z]
        panels[f'gfx_{tag}'] = tau33(gfx).cpu().numpy()[:, :, SLICE_Z]

        del model
        torch.cuda.empty_cache()

    panels['x']  = x_field[0, 8].cpu().numpy()[:, :, SLICE_Z]
    panels['gx'] = gx_field[0, 8].cpu().numpy()[:, :, SLICE_Z]

    panels['diff_noaug'] = np.abs(panels['fgx_noaug'] - panels['gfx_noaug'])
    panels['diff_aug']   = np.abs(panels['fgx_aug']   - panels['gfx_aug'])

    for name, arr in panels.items():
        is_diff = name.startswith('diff')
        is_grad = name in GRAD_PANELS
        if is_grad:
            cmap, vmin, vmax = CMAP_GRAD, VMIN_GRAD, VMAX_GRAD
        elif is_diff:
            cmap, vmin, vmax = CMAP_DIFF, VMIN_DIFF, VMAX_DIFF
        else:
            cmap, vmin, vmax = CMAP_FIELD, VMIN_FIELD, VMAX_FIELD
        path = os.path.join(out_subdir, f'{name}.pdf')
        save_panel(arr.T, path, vmin, vmax, cmap)
        print(f'  Saved {path}  min={arr.min():.3g}  max={arr.max():.3g}')


def make_commutative_panels():
    os.makedirs(OUT_DIR, exist_ok=True)
    for region in MODELS:
        make_commutative_panels_for_region(region)

    save_colorbar(VMIN_FIELD, VMAX_FIELD, CMAP_FIELD,
                  r'$\hat\tau^{\mathrm{d}}_{33}$',
                  os.path.join(OUT_DIR, 'colorbar_field.pdf'),
                  ticks=[VMIN_FIELD, 0, VMAX_FIELD])
    save_colorbar(VMIN_GRAD, VMAX_GRAD, CMAP_GRAD,
                  r'$\partial u_i / \partial x_j$',
                  os.path.join(OUT_DIR, 'colorbar_grad.pdf'),
                  ticks=[VMIN_GRAD, 0, VMAX_GRAD])
    save_colorbar(VMIN_DIFF, VMAX_DIFF, CMAP_DIFF,
                  r'diff.',
                  os.path.join(OUT_DIR, 'colorbar_diff.pdf'),
                  ticks=[VMIN_DIFF, VMAX_DIFF / 2, VMAX_DIFF])


if __name__ == '__main__':
    make_commutative_panels()