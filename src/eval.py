import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformations import unstandardize, rand_so3_angles, get_all_octahedral_angles, rotate_3d, rotate_octahedral_exact, euler_angles_to_matrix

def eval_test_set(config, loader, model, criterion, scaling_factors):
    device = next(model.parameters()).device
    test_loss = 0.0
    total_samples = 0

    for inputs, targets in tqdm.tqdm(loader, desc="Plain eval test set"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = model(inputs)

        batch_size = inputs.size(0)
        loss = criterion(
            unstandardize(config.task_type, 'output', pred, scaling_factors),
            unstandardize(config.task_type, 'output', targets, scaling_factors)
        ).item()
        
        # Weight by batch size since criterion returns mean over batch
        test_loss += loss * batch_size
        total_samples += batch_size

    return test_loss / total_samples


def eval_equivariance(config, loader, model, compute_spectra, scaling_factors):
    device = next(model.parameters()).device
    abs_equiv_error = 0.0
    total_samples = 0

    spec_error = []
    for inputs, targets in tqdm.tqdm(loader, desc="Equivariance eval test set"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred = model(inputs)
        
        equiv_error = octahedral_equivariance_error(
            inputs, pred, targets, model, batch_size=config.rotation_batch_size
        )
        equiv_error = unstandardize(
            config.task_type, 'output', equiv_error, scaling_factors, equiv_error=True
        )
        
        if compute_spectra:
            k, spec_i = compute_isotropic_spectrum(equiv_error.flatten(0, 1), Lx=1)
            spec_error.append((spec_i, inputs.size(0)))  # Store with batch size

        equiv_error_pointwise = torch.linalg.norm(equiv_error, dim=2)
        
        batch_size = inputs.size(0)
        abs_equiv_error += torch.mean(equiv_error_pointwise).item() * batch_size
        total_samples += batch_size

    abs_equiv_error = abs_equiv_error / total_samples
    
    if compute_spectra:
        # Weighted average of spectra
        total_weight = sum(bs for _, bs in spec_error)
        spec_error = sum(spec * bs for spec, bs in spec_error) / total_weight
        return abs_equiv_error, (k, spec_error)
    else:
        return abs_equiv_error


def octahedral_equivariance_error(x, model, batch_size=None):
    """
    Compute |f(g·x) - g·f(x)| for the 24 rotations in the octahedral group.

    Returns tensor shaped [N, 24, C, D, H, W]
    """
    num_samples = x.shape[0]
    batch_size = num_samples if batch_size is None else min(num_samples, batch_size)

    fx = model(x)
    abs_errors = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        x_batch = x[start:end]
        fx_batch = fx[start:end]
        B = x_batch.shape[0]

        gx = rotate_octahedral_exact(x_batch, mode="whole_group_flat")   # [B*24, C, D, H, W]
        gfx = rotate_octahedral_exact(fx_batch, mode="whole_group_flat") # [B*24, C, D, H, W]

        fgx = model(gx)  # [B*24, C, D, H, W]
        
        diff = torch.abs(fgx - gfx)
        # Reshape [B*24, ...] -> [B, 24, ...]
        diff = diff.view(B, 24, *diff.shape[1:])
        abs_errors.append(diff)

    error = torch.cat(abs_errors, dim=0)
    return error


def mean_octahedral_equivariance_error(model, inputs, batch_size=None):
    """Scalar equivariance error: |f(g.x) - g.f(x)| averaged over group, batch, components, space."""
    error = octahedral_equivariance_error(inputs, model, batch_size=batch_size)
    return torch.mean(torch.abs(error))
    

def visualize_predictions(model, val_loader, output_path):
    model.eval()
    torch.manual_seed(42)  # Same 3 examples every time
    
    with torch.no_grad():
        # Get first batch
        inputs, targets = next(iter(val_loader))
        
        # Take first 3 samples and move to device
        inputs = inputs[:3]
        targets = targets[:3]
        predictions = model(inputs)
        
        # Plot
        fig, axes = plt.subplots(3, 6, figsize=(15, 8))
        
        for i in range(3):
            # Convert to numpy and handle channels
            inp = inputs[i].cpu().numpy()
            tgt = targets[i].cpu().numpy()
            pred = predictions[i].cpu().numpy()
            #pred_then_rotated = torch.rot90(predictions[i],k=1,dims=(1,2))[0].cpu().numpy()
            pred_then_rotated_oct = rotate_octahedral_exact(predictions[i], 1, rotate_channels=True, mode="cartesian")
            rotated_inputs = rotate_octahedral_exact(inputs[i], 1, rotate_channels=True, mode="cartesian")
            #rotated_then_predicted = model(torch.rot90(inputs[i].unsqueeze(0),k=1,dims=(2,3)))[0][0].cpu().numpy()
            rotated_then_predicted = model(rotated_inputs)[0].cpu().numpy()
            
            # Compute difference for diverging colormap
            diff = rotated_then_predicted - pred_then_rotated_oct
            
            if inp.ndim == 3:  # Remove channel dim if needed
                inp = inp.squeeze() if inp.shape[0] == 1 else inp.transpose(1,2,0)
                tgt = tgt.squeeze() if tgt.shape[0] == 1 else tgt.transpose(1,2,0)
                pred = pred.squeeze() if pred.shape[0] == 1 else pred.transpose(1,2,0)
            
            # Compute shared bounds for all images except difference
            #shared_vmax = max(np.abs(inp).max(), np.abs(tgt).max(), np.abs(pred).max(), 
            #                 np.abs(pred_then_rotated).max(), np.abs(rotated_then_predicted).max())
            #diff_vmax = np.abs(diff).max()
            shared_vmax = 5
            diff_vmax = 1
            # All images except difference use same colorbar bounds
            axes[i,0].imshow(inp, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,1].imshow(tgt, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,2].imshow(pred, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,3].imshow(pred_then_rotated, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            axes[i,4].imshow(rotated_then_predicted, cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
            im = axes[i,5].imshow(diff, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
            
            # Add colorbar for the difference plot
            if i == 0:  # Only add colorbar to first row
                cbar = plt.colorbar(im, ax=axes[i,5], shrink=0.8)
                cbar.set_label('Difference')

            axes[i,0].set_title('Input')
            axes[i,1].set_title('Target')
            axes[i,2].set_title('Prediction')
            axes[i,3].set_title('Pred → Rot 90°')
            axes[i,4].set_title('Rot 90° → Pred')
            axes[i,5].set_title('Difference')
            
            for ax in axes[i]:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def visualize_equivariance(inputs, pred, targets, model, angle=torch.tensor([0,0,np.pi/2])):
    #angle = get_all_octahedral_angles()[0]
    _, Cin, Din, Hin, Win = inputs.shape
    rotated_input = rotate_octahedral_exact(inputs, angle, rotate_channels=True, mode="cartesian")
    rotated_pred = rotate_octahedral_exact(model(inputs), angle, rotate_channels=True, mode="cartesian")
    rotated_then_pred = model(rotated_input)
    indices = [(0,0,0), (5,6,7), (11,5,12)]

    print(f'Example pairs')
    for spatial_index in indices:
        print(f'Predicted then rotated: {rotated_then_pred[...,*spatial_index].detach().cpu().numpy()}')
        print(f'Rotated then predicted: {rotated_pred[...,*spatial_index].detach().cpu().numpy()}')
        print(f'Difference: {rotated_then_pred[...,*spatial_index].detach().cpu().numpy() - rotated_pred[...,*spatial_index].detach().cpu().numpy()}')   



def plot_equivariance(model, loader, output_path, title, angle=torch.tensor([0,0,np.pi/2])):
    model.eval()
    with torch.no_grad():
        # ----- fetch -----
        inputs, targets = next(iter(loader))
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)
        pred = model(inputs)
        #angle = get_all_octahedral_angles()[0]
        _, Cin, Din, Hin, Win = inputs.shape
        _, Cout, _, _, _ = pred.shape

        rotated_input = rotate_octahedral_exact(inputs, angle, rotate_channels=True, mode="cartesian")
        rotated_pred = rotate_octahedral_exact(pred, angle, rotate_channels=True, mode="cartesian")
        rotated_then_pred = model(rotated_input)

        idx=0
        slice_idx= 7
        fig, axes = plt.subplots(6,Cin,figsize=(30,30))

        for i in range(Cin):
            im=axes[0,i].imshow(inputs[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[0,i].set_title(f'Input {i}')
            axes[0,i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        for i in range(Cin):
            axes[1,i].imshow(rotated_input[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[1,i].set_title(f'Rotated Input {i}')
            axes[1,i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        for i in range(Cout):
            axes[2,i].imshow(pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[2,i].set_title(f'Prediction {i}')
            axes[2,i].axis('off')
        for i in range(Cout):
            im=axes[3,i].imshow(rotated_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[3,i].set_title(f'Rotated Pred {i}')
            axes[3,i].axis('off')
            plt.colorbar(im, ax=axes[3, i], fraction=0.046)
        for i in range(Cout):
            im=axes[4,i].imshow(rotated_then_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[4,i].set_title(f'Rotated Then Pred {i}')
            axes[4,i].axis('off')
            plt.colorbar(im, ax=axes[4, i], fraction=0.046)
        for i in range(Cout):
            im=axes[5,i].imshow(rotated_pred[idx][i][:,:,slice_idx].detach().cpu().numpy() - rotated_then_pred[idx][i][:,:,slice_idx].detach().cpu().numpy(), cmap='RdBu_r',origin='lower')
            axes[5,i].set_title(f'Difference {i}')
            axes[5,i].axis('off')
            plt.colorbar(im, ax=axes[5, i], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_predictions_3D(inputs, preds, targets, output_path, slice_idx=None):
    """
    inputs: (9, D, H, W)
    preds: (5, D, H, W) - components: xx, yy, xy, xz, yz (zz reconstructed from trace)
    targets: (5, D, H, W) - components: xx, yy, xy, xz, yz (zz reconstructed from trace)
    slice_idx: int, index along D axis. If None, uses center slice.
    """
    inp = inputs.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()
    
    D = inp.shape[1]
    if slice_idx is None:
        slice_idx = D // 2
    
    # Extract slices
    inp_slice = inp[:, slice_idx, :, :]    # (9, H, W)
    pred_slice = pred[:, slice_idx, :, :]  # (5, H, W)
    tgt_slice = tgt[:, slice_idx, :, :]    # (5, H, W)
    
    # Reconstruct full 6-component tensors from 5 components
    # Input order: xx, yy, xy, xz, yz (indices 0, 1, 2, 3, 4)
    # Output order: xx, yy, zz, xy, xz, yz
    def reconstruct_full_tensor(t5):
        xx, yy = t5[0], t5[1]
        zz = -xx - yy  # traceless: xx + yy + zz = 0
        return np.stack([xx, yy, zz, t5[2], t5[3], t5[4]], axis=0)
    
    pred_full = reconstruct_full_tensor(pred_slice)  # (6, H, W)
    tgt_full = reconstruct_full_tensor(tgt_slice)    # (6, H, W)
    
    # Compute magnitudes for symmetric tensors: sqrt(xx² + yy² + zz² + 2xy² + 2xz² + 2yz²)
    def tensor_magnitude(t):
        return np.sqrt(t[0]**2 + t[1]**2 + t[2]**2 + 2*t[3]**2 + 2*t[4]**2 + 2*t[5]**2)
    
    pred_mag = tensor_magnitude(pred_full)
    tgt_mag = tensor_magnitude(tgt_full)
    
    labels_inp = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    labels_out = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    
    fig, axes = plt.subplots(3, 10, figsize=(40, 12))
    
    # Row 0: Input (9 components + 1 empty) - each column has its own symmetric colorbar
    for c in range(9):
        vmax = np.abs(inp_slice[c]).max()
        if vmax == 0:
            vmax = 1.0
        im = axes[0, c].imshow(inp_slice[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, c].set_title(f'Input {labels_inp[c]}')
        axes[0, c].axis('off')
        plt.colorbar(im, ax=axes[0, c], fraction=0.046)
    axes[0, 9].axis('off')
    
    # Compute per-column symmetric vmax for output (shared between target and pred rows)
    vmax_per_col = []
    for c in range(6):
        col_vmax = max(np.abs(tgt_full[c]).max(), np.abs(pred_full[c]).max())
        if col_vmax == 0:
            col_vmax = 1.0
        vmax_per_col.append(col_vmax)
    
    # Magnitude colorbar range (shared between target and pred)
    mag_max = max(tgt_mag.max(), pred_mag.max())
    if mag_max == 0:
        mag_max = 1.0
    
    # Row 1: Target (6 components + magnitude)
    for c in range(6):
        vmax = vmax_per_col[c]
        im = axes[1, c].imshow(tgt_full[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, c].set_title(f'Target {labels_out[c]}')
        axes[1, c].axis('off')
        plt.colorbar(im, ax=axes[1, c], fraction=0.046)
    im = axes[1, 6].imshow(tgt_mag, cmap='viridis', vmin=0, vmax=mag_max)
    axes[1, 6].set_title('Target |·|')
    axes[1, 6].axis('off')
    plt.colorbar(im, ax=axes[1, 6], fraction=0.046)
    for c in range(7, 10):
        axes[1, c].axis('off')
    
    # Row 2: Prediction (6 components + magnitude)
    for c in range(6):
        vmax = vmax_per_col[c]
        im = axes[2, c].imshow(pred_full[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, c].set_title(f'Pred {labels_out[c]}')
        axes[2, c].axis('off')
        plt.colorbar(im, ax=axes[2, c], fraction=0.046)
    im = axes[2, 6].imshow(pred_mag, cmap='viridis', vmin=0, vmax=mag_max)
    axes[2, 6].set_title('Pred |·|')
    axes[2, 6].axis('off')
    plt.colorbar(im, ax=axes[2, 6], fraction=0.046)
    for c in range(7, 10):
        axes[2, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
        
def plot_spectra(model, loader, output_path, title):
    # TODO: only uses first batch right now
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)
        pred = model(inputs)
        error = abs(pred- targets)
        oct_equiv_err_abs, oct_equiv_err_rel = octahedral_equivariance_error(inputs, pred, targets, model, batch_size=1)

        equiv_err_spectra_abs = []
        equiv_err_spectra_rel = []
        vel_spectra_in = []
        vel_spectra_pred = []
        vel_spectra_tgt = []
        error_spectra = []
        for i in tqdm.tqdm(range(oct_equiv_err_abs.shape[0]), desc="Computing spectra"):
            k, spec = compute_isotropic_spectrum(oct_equiv_err_abs[i].cpu().numpy(), Lx=0.0256)
            equiv_err_spectra_abs.append(spec)
            k_rel, spec_rel = compute_isotropic_spectrum(oct_equiv_err_rel[i].cpu().numpy(), Lx=0.0256, tke_normalize=True)
            equiv_err_spectra_rel.append(spec_rel)

        for i in tqdm.tqdm(range(targets.shape[0]), desc="Computing spectra"):
            k_in, spec_in = compute_isotropic_spectrum(inputs[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_in.append(spec_in)
            k_tgt, spec_tgt = compute_isotropic_spectrum(targets[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_tgt.append(spec_tgt)
            k_pred, spec_pred = compute_isotropic_spectrum(pred[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            vel_spectra_pred.append(spec_pred)
            k_mse, spec_mse = compute_isotropic_spectrum(error[i].cpu().numpy(), Lx=0.0256, tke_normalize=False)
            error_spectra.append(spec_mse)

        equiv_err_spectra_abs = np.stack(equiv_err_spectra_abs, axis=0)
        equiv_err_spectra_rel = np.stack(equiv_err_spectra_rel, axis=0)
        vel_spectra_in = np.stack(vel_spectra_in, axis=0)
        vel_spectra_tgt = np.stack(vel_spectra_tgt, axis=0)
        vel_spectra_pred = np.stack(vel_spectra_pred, axis=0)
        error_spectra = np.stack(error_spectra, axis=0)
        mean_equiv_err_spectra_abs = np.mean(equiv_err_spectra_abs, axis=0)
        mean_equiv_err_spectra_rel = np.mean(equiv_err_spectra_rel, axis=0)
        mean_vel_spectra_in = np.mean(vel_spectra_in, axis=0)
        mean_vel_spectra_pred = np.mean(vel_spectra_pred, axis=0)
        mean_vel_spectra_tgt = np.mean(vel_spectra_tgt, axis=0)
        mean_error_spectra = np.mean(error_spectra, axis=0)


        fig, axes = plt.subplots(2, 2, figsize=(12, 12),sharex=True)
        axes[0,0].loglog(k, mean_equiv_err_spectra_abs)
        axes[0,0].loglog(k_mse, mean_error_spectra)

        axes[0,0].set_title('Equivariance Error (Absolute)')
        axes[0,0].set_xlabel('Wavenumber')
        axes[0,0].set_ylabel('Normalized Spectrum')
        axes[0,1].loglog(k, mean_equiv_err_spectra_rel)

        axes[0,1].set_title('Equivariance Error (Relative)')
        axes[0,1].set_xlabel('Wavenumber')
        axes[0,1].set_ylabel('Normalized Spectrum')
        axes[1,0].loglog(k_in, mean_vel_spectra_in)
        axes[1,0].set_title('Velocity Spectrum (In, Tgt, Pred)')
        axes[1,0].loglog(k_tgt, mean_vel_spectra_tgt)
        axes[1,0].loglog(k_pred, mean_vel_spectra_pred)
        axes[1,1].loglog(k_mse, mean_error_spectra)
        axes[1,1].set_title('Error Spectrum')
        axes[1,1].set_xlabel('Wavenumber')
        axes[1,1].set_ylabel('Normalized Spectrum')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_spectrum(k, spec, output_path, title):
    # TODO: only uses first batch right now


    fig, axes = plt.subplots(1, 1, figsize=(12, 12),sharex=True)
    axes.loglog(k, spec)
    axes.set_title('Equivariance Error (Absolute)')
    axes.set_xlabel('Wavenumber')
    axes.set_ylabel('Normalized Spectrum')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def visualize_intermediate_features(model, loader, output_path, title, center_mode: str = "cell"):
    """
    Visualize intermediate features from MultiScaleResidualSR model.
    Shows XY, XZ, and YZ slices for each intermediate output scale.
    
    Args:
        model: MultiScaleResidualSR model
        loader: data loader 
        output_path: where to save the visualization
        title: plot title
        center_mode: "cell" or "node" for coordinate interpretation
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))

    model.eval()
    with torch.no_grad():
        # Get first sample
        inputs, targets = next(iter(loader))
        input_sample = inputs[0:1].to(next(model.parameters()).device)   # (1, Cin, Din, Hin, Win)
        
        # Get intermediate features and final prediction
        try:
            final_pred, intermediate_outputs, raw_features = model(input_sample, return_intermediate=True)
        except ValueError:
            # Fallback for models that don't return raw features
            final_pred, intermediate_outputs = model(input_sample, return_intermediate=True)
            raw_features = None
        
        # Convert to numpy
        inp = input_sample[0].detach().cpu().numpy()  # (Cin, Din, Hin, Win)
        
        # Get input dimensions for reference slicing
        Cin, Din, Hin, Win = inp.shape
        cin_d, cin_h, cin_w = center_index(Din, center_mode), center_index(Hin, center_mode), center_index(Win, center_mode)
        
        # Prepare intermediate outputs for visualization
        intermediate_data = {}
        for scale_name, features in intermediate_outputs.items():
            feat_np = features[0].detach().cpu().numpy()  # (C, D, H, W)
            C, D, H, W = feat_np.shape
            
            # Map center indices to this scale
            c_d = map_index_from_src_to_dst(cin_d, Din, D, center_mode)
            c_h = map_index_from_src_to_dst(cin_h, Hin, H, center_mode)
            c_w = map_index_from_src_to_dst(cin_w, Win, W, center_mode)
            
            # Compute magnitude across channels for each slice
            xy_mag = np.sqrt(np.sum(feat_np[:, c_d, :, :]**2, axis=0))  # (H, W)
            xz_mag = np.sqrt(np.sum(feat_np[:, :, c_h, :]**2, axis=0))  # (D, W)  
            yz_mag = np.sqrt(np.sum(feat_np[:, :, :, c_w]**2, axis=0))  # (D, H)
            
            intermediate_data[scale_name] = {
                'xy': xy_mag,
                'xz': xz_mag, 
                'yz': yz_mag,
                'indices': (c_d, c_h, c_w),
                'shape': (D, H, W)
            }
        
        # Also add input for comparison
        xy_mag_inp = np.sqrt(np.sum(inp[:, cin_d, :, :]**2, axis=0))
        xz_mag_inp = np.sqrt(np.sum(inp[:, :, cin_h, :]**2, axis=0))
        yz_mag_inp = np.sqrt(np.sum(inp[:, :, :, cin_w]**2, axis=0))
        
        intermediate_data['input'] = {
            'xy': xy_mag_inp,
            'xz': xz_mag_inp,
            'yz': yz_mag_inp,
            'indices': (cin_d, cin_h, cin_w),
            'shape': (Din, Hin, Win)
        }
        
        # Create figure - one row per scale, 3 columns (XY, XZ, YZ)
        num_scales = len(intermediate_data)
        fig, axes = plt.subplots(num_scales, 3, figsize=(12, 4*num_scales))
        
        # Handle single row case
        if num_scales == 1:
            axes = axes.reshape(1, -1)
        
        # Determine global colormap range
        all_mags = []
        for data in intermediate_data.values():
            all_mags.extend([data['xy'].max(), data['xz'].max(), data['yz'].max()])
        vmin, vmax = 0.0, max(all_mags)
        
        # Plot each scale
        scale_names = ['input'] + [k for k in intermediate_data.keys() if k != 'input']
        for row, scale_name in enumerate(scale_names):
            data = intermediate_data[scale_name]
            d_idx, h_idx, w_idx = data['indices']
            D, H, W = data['shape']
            
            # XY slice
            im_xy = axes[row, 0].imshow(data['xy'], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{scale_name.upper()}: XY (d={d_idx}, {H}×{W})')
            axes[row, 0].axis('off')
            
            # XZ slice  
            im_xz = axes[row, 1].imshow(data['xz'], cmap='plasma', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{scale_name.upper()}: XZ (h={h_idx}, {D}×{W})')
            axes[row, 1].axis('off')
            
            # YZ slice
            im_yz = axes[row, 2].imshow(data['yz'], cmap='cividis', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'{scale_name.upper()}: YZ (w={w_idx}, {D}×{H})')
            axes[row, 2].axis('off')
        
        # Add colorbar to the last row
        cbar = plt.colorbar(im_yz, ax=axes[-1, :], shrink=0.8, aspect=30)
        cbar.set_label('Feature Magnitude')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_feature_refinement(model, loader, output_path, title, center_mode: str = "cell"):
    """
    Visualize raw features at each upsampling scale and the final output.
    Shows how features evolve through the upsampling process.
    
    Args:
        model: MultiScaleResidualSR model
        loader: data loader 
        output_path: where to save the visualization
        title: plot title
        center_mode: "cell" or "node" for coordinate interpretation
    """
    def grid_coords(n: int, mode: str):
        if mode == "cell":
            return (np.arange(n) + 0.5) / n
        elif mode == "node":
            if n == 1:
                return np.array([0.0])
            return np.linspace(0.0, 1.0, n)
        else:
            raise ValueError("center_mode must be 'cell' or 'node'")

    def center_index(n: int, mode: str):
        coords = grid_coords(n, mode)
        return int(np.argmin(np.abs(coords - 0.5)))

    def map_index_from_src_to_dst(i_src: int, n_src: int, n_dst: int, mode: str):
        x_src = grid_coords(n_src, mode)[i_src]
        coords_dst = grid_coords(n_dst, mode)
        return int(np.argmin(np.abs(coords_dst - x_src)))

    model.eval()
    with torch.no_grad():
        # Get first sample
        inputs, targets = next(iter(loader))
        input_sample = inputs[0:1].to(next(model.parameters()).device)   # (1, Cin, Din, Hin, Win)
        
        # Get intermediate features and final prediction
        try:
            final_pred, intermediate_outputs, raw_features = model(input_sample, return_intermediate=True)
        except (ValueError, AttributeError):
            print("Model doesn't support raw feature extraction, skipping refinement visualization")
            return
        
        # Convert to numpy
        inp = input_sample[0].detach().cpu().numpy()  # (Cin, Din, Hin, Win)
        target = targets[0].cpu().numpy() if len(targets) > 0 else None
        final_pred_np = final_pred[0].detach().cpu().numpy()
        
        # Get input dimensions for reference slicing
        Cin, Din, Hin, Win = inp.shape
        cin_d = center_index(Din, center_mode)
        
        # Prepare data for each scale
        scale_data = {}
        
        # Add input
        inp_xy = np.sqrt(np.sum(inp[:, cin_d, :, :]**2, axis=0))
        scale_data['input'] = {
            'data': inp_xy,
            'title': f'Input\n{inp_xy.shape}',
            'shape': (Din, Hin, Win)
        }
        
        # Add raw features for each scale
        scale_names = sorted(raw_features.keys(), key=lambda x: int(x.replace('x', '0')) if x != 'lr' else 0)
        for scale_name in scale_names:
            raw_feat = raw_features[scale_name][0].detach().cpu().numpy()  # (hidden_channels, D, H, W)
            C, D, H, W = raw_feat.shape
            
            # Map center index to this scale
            c_d = map_index_from_src_to_dst(cin_d, Din, D, center_mode)
            
            # Compute magnitude across channels for XY slice
            raw_feat_xy = np.sqrt(np.sum(raw_feat[:, c_d, :, :]**2, axis=0))
            
            scale_data[scale_name] = {
                'data': raw_feat_xy,
                'title': f'{scale_name.upper()}\n{H}×{W}, {C}ch',
                'shape': (D, H, W)
            }
        
        # Add final output
        _, D_final, H_final, W_final = final_pred_np.shape
        c_d_final = map_index_from_src_to_dst(cin_d, Din, D_final, center_mode)
        final_xy = np.sqrt(np.sum(final_pred_np[:, c_d_final, :, :]**2, axis=0))
        scale_data['final'] = {
            'data': final_xy,
            'title': f'Final Output\n{final_xy.shape}',
            'shape': (D_final, H_final, W_final)
        }
        
        # Add target if available
        if target is not None:
            _, D_tgt, H_tgt, W_tgt = target.shape
            c_d_tgt = map_index_from_src_to_dst(cin_d, Din, D_tgt, center_mode)
            target_xy = np.sqrt(np.sum(target[:, c_d_tgt, :, :]**2, axis=0))
            scale_data['target'] = {
                'data': target_xy,
                'title': f'Target\n{target_xy.shape}',
                'shape': (D_tgt, H_tgt, W_tgt)
            }
        
        # Create figure
        num_cols = len(scale_data)
        fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4))
        
        # Handle single column case
        if num_cols == 1:
            axes = [axes]
        
        # Plot each scale
        plot_order = ['input'] + scale_names + ['final'] + (['target'] if target is not None else [])
        
        for col, scale_name in enumerate(plot_order):
            data_info = scale_data[scale_name]
            
            # Use different colormaps for raw features vs outputs
            if scale_name in scale_names:
                # Raw features - use plasma colormap with their own range
                cmap = 'plasma'
                vmin, vmax = 0, data_info['data'].max()
            else:
                # Input, output, target - use viridis with shared range
                cmap = 'viridis' 
                # Calculate shared range for input/output/target
                shared_data = [scale_data[k]['data'] for k in ['input', 'final'] + (['target'] if target is not None else [])]
                vmin, vmax = 0.0, max(d.max() for d in shared_data)
            
            axes[col].imshow(data_info['data'], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[col].set_title(data_info['title'])
            axes[col].axis('off')
        
        plt.suptitle(f'{title}\nShowing XY slices at center depth')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def compute_isotropic_spectrum(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=True, crop_first_bin = True):
    # computes mean spectrum for a bunch of fields
    spectra = []
    for i in tqdm.tqdm(range(field.shape[0]), desc="Computing spectra"):
        k, spec = compute_single_isotropic_spectrum(field[i].cpu().numpy(), Lx=0.0256)
        spectra.append(spec)
    spectra = np.stack(spectra, axis=0)
    mean_spectra = np.mean(spectra, axis=0)
    return k, mean_spectra

def compute_single_isotropic_spectrum(field, Lx, Ly=None, Lz=None, tke_normalize=False, spectral_dealias=True, crop_first_bin = True):
    """
    field: (3, Nx, Ny, Nz) velocity field [m/s]
    Returns:
      k_plot : 1D array of kappa (rad/length)
      E_plot : normalized spectrum (unitless), integrates to 1 over kept bins
    """
    # TODO: fix normalization stuff in this for comparing velocities.
    if Ly is None: 
        Ly = Lx
    if Lz is None:
        Lz = Lx

    C, Nx, Ny, Nz = field.shape

    field_hat = np.fft.fftn(field, axes=(1,2,3), norm='forward') # Forward divides by N
    E3   = 0.5 * np.sum(field_hat * np.conj(field_hat), axis=0).real 

    # Wavenumbers (radians/length)
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()

    # Bin setup
    dk = float(min(abs(kx[1]-kx[0]), abs(ky[1]-ky[0]), abs(kz[1]-kz[0])))
    edges   = np.arange(0.0, k_mag.max() + dk, dk)
    centers = 0.5*(edges[1:] + edges[:-1])

    if spectral_dealias:
        # Mimic spectral solver: 2/3 rule per direction
        kx_cut = (2/3) * np.max(np.abs(kx))
        ky_cut = (2/3) * np.max(np.abs(ky))
        kz_cut = (2/3) * np.max(np.abs(kz))
        k_cut_mag = min(kx_cut, ky_cut, kz_cut)  # Conservative: all directions satisfied
        mask = (centers > 0) & (centers <= k_cut_mag)
    else:
        mask = centers > 0

    # Shell sum only for masked bins
    E_k_masked = []
    widths = np.diff(edges)[mask]
    E3_flat = E3.reshape(-1)

    shell, _ = np.histogram(k_mag, bins=edges, weights=E3_flat)
    shell_masked = shell[mask] / widths
    E_k_masked = shell_masked

    # Normalize by masked-bin TKE so ∫ E_norm dk = 1
    if tke_normalize:
        TKE_masked = np.sum(E_k_masked * widths, axis=0)  # (T,)
        E_k_norm = (E_k_masked.T / TKE_masked).T
        E_1D = E_k_norm.mean(axis=0)
    else:
        E_1D = E_k_masked.mean(axis=0)

    if crop_first_bin:
        return centers[mask][1:], E_k_masked[1:]    
    else:
        return centers[mask], E_k_masked
