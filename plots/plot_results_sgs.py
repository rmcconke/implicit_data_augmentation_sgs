"""Generate a 3x6 magnitude figure (3 generalization tests x 6 columns: target + Clark + 4 models)
using the nearwall-trained models. Outputs are in the non-dimensional space the models operate in.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from box import Box

sys.path.append('./src/')
from data import get_test_file_lists, NumpyTimeSeriesDataset
from models import get_model

torch.set_default_dtype(torch.float32)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
linewidth = 1
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['xtick.major.width'] = linewidth
plt.rcParams['ytick.major.width'] = linewidth
# -------- config --------
DEVICE = 'cuda:0'
OUT_DIR = 'outputs/plots_final'
SLICE_Z = 7   # middle of the 14^3 interior

MODEL_DIRS = {
    'Eigenframe': 'outputs/pointwiseEigenFrame_params/26_05_23_16:39:46_live_kitten_nearwall_boxfilter_4x_sgs',
    'MLP (aug)':  'outputs/pointwiseAug_params/26_05_21_11:13:54_ace_burro_nearwall_boxfilter_4x_sgs',
    'ESCNN':      'outputs/ESCNN_params/26_05_21_11:07:09_nice_flea_nearwall_boxfilter_4x_sgs',
    'CNN (aug)':  'outputs/CNNaug_params/26_05_21_12:26:23_well_pony_nearwall_boxfilter_4x_sgs',
}

EVAL_SETS = {
    'spatiotemporal': {'input_prefix': 'channel_nearwall_grad_fs4',
                       'target_prefix': 'channel_nearwall_sgs_fs4',
                       'pattern': ['z4'], 'times': [25, 26, 27, 28, 29]},
    'anisotropy':     {'input_prefix': 'channel_middle_grad_fs4',
                       'target_prefix': 'channel_middle_sgs_fs4',
                       'pattern': ['z4'], 'times': [25, 26, 27, 28, 29]},
    'reynolds':       {'input_prefix': 'channel5200_nearwall_grad_fs4',
                       'target_prefix': 'channel5200_nearwall_sgs_fs4',
                       'pattern': ['z0', 'z1', 'z2', 'z3', 'z4'],
                       'times': list(range(11))},
}


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


def clark_prediction(grad_input):
    """
    Clark model on normalized inputs.
    Accepts pointwise (B, 9) or field (1, 9, D, H, W).
    Returns (B, 5) for pointwise or (5, D, H, W) for field.
    """
    if grad_input.shape[-1] == 9 and grad_input.ndim == 2:
        G = grad_input.view(-1, 3, 3)
    else:
        G = grad_input[0].permute(1, 2, 3, 0).reshape(*grad_input.shape[2:], 3, 3)

    tau = torch.einsum('...ik,...jk->...ij', G, G) / 12.0

    trace = (tau[..., 0, 0] + tau[..., 1, 1] + tau[..., 2, 2]) / 3.0
    tau[..., 0, 0] -= trace
    tau[..., 1, 1] -= trace
    tau[..., 2, 2] -= trace

    out = torch.stack([tau[..., 0, 0], tau[..., 1, 1],
                       tau[..., 0, 1], tau[..., 0, 2], tau[..., 1, 2]], dim=-1)

    if out.ndim == 2:
        return out
    return out.permute(3, 0, 1, 2)


def get_one_box(eval_params, scaling_factors, pointwise, device, idx = 11):
    """Return inputs, targets, and the path of the first box of an eval dataset."""
    input_files, target_files = get_test_file_lists(Box(eval_params))
    dataset = NumpyTimeSeriesDataset(
        input_file_list=[input_files[idx]],
        target_file_list=[target_files[idx]],
        scaling_factors=scaling_factors,
        augmentation_group=None,
        pointwise=pointwise,
        precompute_eigenframe_inputs=False,
        device=device,
    )
    inputs, targets = dataset.get_all_data()
    return inputs, targets, input_files[0]


def predict_field(model, cfg, inputs):
    """Predict and reshape to (5, 14, 14, 14)."""
    G = 14
    with torch.inference_mode():
        if cfg.pointwise:
            n_voxels = G ** 3
            input0 = inputs[:n_voxels]
            pred = model(input0).reshape(G, G, G, 5).permute(3, 0, 1, 2)
        else:
            pred = model(inputs[0:1])[0]
    return pred.cpu().numpy()


def reshape_target(targets, cfg):
    G = 14
    if cfg.pointwise:
        return targets[:G ** 3].reshape(G, G, G, 5).permute(3, 0, 1, 2).cpu().numpy()
    return targets[0].cpu().numpy()


def reshape_input_for_clark(inputs, cfg):
    G = 14
    if cfg.pointwise:
        return inputs[:G ** 3].reshape(G, G, G, 9).permute(3, 0, 1, 2).unsqueeze(0)
    return inputs[0:1]


def magnitude(field):
    """Magnitude of the symmetric deviatoric tensor from its 5 stored components.
    field: (5, D, H, W). Returns (D, H, W)."""
    t11, t22, t12, t13, t23 = field[0], field[1], field[2], field[3], field[4]
    t33 = -t11 - t22
    return np.sqrt(t11**2 + t22**2 + t33**2 + 2.0*(t12**2 + t13**2 + t23**2))


# -------- main --------
def make_combined_figure():
    ref_dir = list(MODEL_DIRS.values())[0]
    _, ref_cfg, ref_scaling = load_model(ref_dir, DEVICE)

    target_scale = float(ref_scaling['target_mean_magnitude'])
    input_scale  = float(ref_scaling['input_mean_magnitude'])

    test_rows = {}
    for test_name, eval_params in EVAL_SETS.items():
        print(f'--- {test_name} ---')
        inputs, targets, _ = get_one_box(
            eval_params, ref_scaling, ref_cfg.pointwise, DEVICE)
        target_field = reshape_target(targets, ref_cfg)

        clark_input = reshape_input_for_clark(inputs, ref_cfg)
        clark_out   = clark_prediction(clark_input)
        if clark_out.ndim == 2:
            G = 14
            clark_field = clark_out[:G**3].reshape(G, G, G, 5).permute(3, 0, 1, 2).cpu().numpy()
        else:
            clark_field = clark_out.cpu().numpy()

        pred_fields = {}
        for name, run_dir in MODEL_DIRS.items():
            model, cfg, scaling = load_model(run_dir, DEVICE)
            local_inputs, _, _ = get_one_box(eval_params, scaling, cfg.pointwise, DEVICE)
            pred_fields[name] = predict_field(model, cfg, local_inputs)
            del model
            torch.cuda.empty_cache()

        target_field = target_field * target_scale
        clark_field  = clark_field  * (input_scale ** 2)
        for name in pred_fields:
            pred_fields[name] = pred_fields[name] * target_scale

        test_rows[test_name] = [
            ('Target',     target_field),
            ('Clark',      clark_field),
            ('Eigenframe', pred_fields['Eigenframe']),
            ('MLP (aug)',  pred_fields['MLP (aug)']),
            ('ESCNN',      pred_fields['ESCNN']),
            ('CNN (aug)',  pred_fields['CNN (aug)']),
        ]

    test_names = list(EVAL_SETS.keys())
    col_names  = [r[0] for r in test_rows[test_names[0]]]

    VMIN_VMAX_PER_TEST = {
        'spatiotemporal': (0, .2),
        'anisotropy':     (0, .2),
        'reynolds':       (0, .2),
    }

    # ---------- layout ----------
    # 6 image columns + 1 narrow colorbar column
    fig = plt.figure(figsize=(10, 5))
    gs  = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.08],  # 6 equal image cols + thin cbar col
        wspace=0.05,
        hspace=0.15,
    )

    im_ref = None  # will hold the last imshow for the colorbar
    for i, test_name in enumerate(test_names):
        row   = test_rows[test_name]
        vmin, vmax = VMIN_VMAX_PER_TEST[test_name]
        mags  = [magnitude(field)[:, :, SLICE_Z] for _, field in row]
        for j, mag in enumerate(mags):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(mag, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_names[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(test_name.capitalize(), fontsize=10, rotation=0,
                              ha='right', va='center', labelpad=20)
            if i == len(test_names) - 1 and j == len(col_names) - 1:
                im_ref = im  # grab any image for the colorbar (all share vmin/vmax)

    # Colorbar spanning all 3 rows in the last (7th) column
    cbar_ax = fig.add_subplot(gs[:, 6])
    cbar    = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'$|\hat{\tau}^\mathrm{d}|$', fontsize=9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'predictions_magnitude.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    make_combined_figure()