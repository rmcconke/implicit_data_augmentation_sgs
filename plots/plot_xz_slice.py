import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================== CONFIGURATION ==================
# Paths/prefixes must match download_channel_slices.py
datasets = [
    {
        'npy_path': '/orcd/data/abodner/002/ryley/numpy_channel_xzslice/channel_xz_slice_0.005.npy',
        'plot_name': 'channel_re1000_xz_velocity_magnitude.png',
    },
    {
        'npy_path': '/orcd/data/abodner/002/ryley/numpy_channel5200_xzslice/channel5200_xz_slice_0.005.npy',
        'plot_name': 'channel5200_re5200_xz_velocity_magnitude.png',
    },
]

plot_output_dir = 'outputs/plots_final'
cmap = 'plasma'
dpi = 200
fig_size = 8
vmax = 1.2  # fixed colorbar max for both plots
vmin = 1 # fixed colorbar min for both plots
# =====================================================


def plot_velocity_magnitude(npy_path, plot_name):
    """Load a (3, Nx, Nz) velocity slice and save a plasma velocity-magnitude image."""
    vel_data = np.load(npy_path)
    u, v, w = vel_data[0], vel_data[1], vel_data[2]
    vel_mag = np.sqrt(u**2 + v**2 + w**2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        vel_mag.T,
        origin='lower',
        cmap=cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )

    ax.axis('off')  # no axis, no title

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|u|')

    os.makedirs(plot_output_dir, exist_ok=True)
    out_path = f'{plot_output_dir}/{plot_name}'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    for cfg in datasets:
        plot_velocity_magnitude(cfg['npy_path'], cfg['plot_name'])

    print("Done.")