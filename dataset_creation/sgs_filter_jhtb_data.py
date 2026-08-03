import numpy as np
import os
import tqdm
import glob


def box_filter_downsample(arr, factor):
    """Simple box filter downsampling by factor"""
    c, n, _, _ = arr.shape
    new_n = n // factor
    
    # Reshape and average
    reshaped = arr[:, :new_n*factor, :new_n*factor, :new_n*factor]
    reshaped = reshaped.reshape(c, new_n, factor, new_n, factor, new_n, factor)
    return reshaped.mean(axis=(2, 4, 6))


def process_sgs_data(input_prefix, numpy_dir, delta, fs=4, eps=1e-8):
    """
    Process velocity data to compute normalized SGS tensor and velocity gradients.
    
    Parameters
    ----------
    input_prefix : str
        Prefix for input files (e.g., 'channel_nearwall')
    numpy_dir : str
        Directory containing numpy files
    delta : float
        Filter width in physical units (= coarse grid spacing)
    fs : int
        Filter/downsampling factor
    eps : float
        Regularization for |S| -> 0
    S_mag_warn : float
        Warn if strain rate magnitude falls below this value
    """
    # Find only raw velocity files
    files = glob.glob(os.path.join(numpy_dir, f"{input_prefix}_box_x*_z*_time_*.npy"))
    
    print(f"{input_prefix}: found {len(files)} files, reprocessing")
    
    if len(files) == 0:
        return
    
    dx = delta  # coarse grid spacing = filter width
    
    for file in tqdm.tqdm(files):
        data = np.load(file)
        filtered = box_filter_downsample(data, fs)
        
        # Compute SGS tensor (6 components: 11,22,33,12,13,23)       
        sgs_full = np.array([
            box_filter_downsample(data[i:i+1]*data[j:j+1], fs)[0] - filtered[i]*filtered[j]
            for i, j in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
        ])

        # Make deviatoric (subtract trace from diagonal components)
        trace = (sgs_full[0] + sgs_full[1] + sgs_full[2]) / 3
        sgs_full[0] -= trace
        sgs_full[1] -= trace
        # Drop sgs_full[2] (zz) since it can be reconstructed as -sgs_full[0] - sgs_full[1]

        # Extract interior points only for SGS to match gradient shape
        # Keep only 5 components: 11, 22, 12, 13, 23 (indices 0, 1, 3, 4, 5)
        sgs = sgs_full[[0, 1, 3, 4, 5], 1:-1, 1:-1, 1:-1]

        
        # Compute velocity gradients (9 components, interior points only)
        u, v, w = filtered[0], filtered[1], filtered[2]
        grad = np.array([
            np.gradient(u, dx, axis=0)[1:-1,1:-1,1:-1],  # du/dx
            np.gradient(u, dx, axis=1)[1:-1,1:-1,1:-1],  # du/dy
            np.gradient(u, dx, axis=2)[1:-1,1:-1,1:-1],  # du/dz
            np.gradient(v, dx, axis=0)[1:-1,1:-1,1:-1],  # dv/dx
            np.gradient(v, dx, axis=1)[1:-1,1:-1,1:-1],  # dv/dy
            np.gradient(v, dx, axis=2)[1:-1,1:-1,1:-1],  # dv/dz
            np.gradient(w, dx, axis=0)[1:-1,1:-1,1:-1],  # dw/dx
            np.gradient(w, dx, axis=1)[1:-1,1:-1,1:-1],  # dw/dy
            np.gradient(w, dx, axis=2)[1:-1,1:-1,1:-1],  # dw/dz
        ])
        
        # Compute strain rate magnitude: |S| = sqrt(2 * S_ij * S_ij)
        S11 = grad[0]
        S22 = grad[4]
        S33 = grad[8]
        S12 = 0.5 * (grad[1] + grad[3])
        S13 = 0.5 * (grad[2] + grad[6])
        S23 = 0.5 * (grad[5] + grad[7])

        S_sq = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)
        W12 = 0.5 * (grad[1] - grad[3])
        W13 = 0.5 * (grad[2] - grad[6])
        W23 = 0.5 * (grad[5] - grad[7])
        W_sq = 2*(W12**2 + W13**2 + W23**2)
        G = np.sqrt(S_sq + W_sq)
        
        # Warn if S_mag is too small
        n_small = np.sum(G < 1e-6)
        if n_small > 0:
            print(f"  Warning: {file} has {n_small} points with G < 1e-6 (min: {G.min():.2e})")
        
        G = G + eps
        
        # Normalize
        grad_normalized = grad / G
        sgs_normalized = sgs / (delta**2 * G**2)
        
        # Save
        np.save(file.replace(f'{input_prefix}_box', f'{input_prefix}_sgs_fs{fs}_box'), sgs_normalized)
        np.save(file.replace(f'{input_prefix}_box', f'{input_prefix}_grad_fs{fs}_box'), grad_normalized)
        np.save(file.replace(f'{input_prefix}_box', f'{input_prefix}_G_fs{fs}_box'), G)


if __name__ == '__main__':
    numpy_dir = "/orcd/data/abodner/002/ryley/numpy_4000_sgs"
    
    filter_widths = {
        'channel_nearwall': 4.0026e-02,
        'channel_middle': 4.0026e-02,
        'channel5200_nearwall': 7.7132e-03,
        'channel5200_middle': 7.7132e-03,
    }
    
    fs = 4
    
    for prefix, delta in filter_widths.items():
        process_sgs_data(prefix, numpy_dir, delta, fs=fs)