import numpy as np
import os
import tqdm
import multiprocessing as mp
from functools import partial
import glob 


def box_filter_downsample(arr, factor):
    """Simple box filter downsampling by factor"""
    c, n, _, _ = arr.shape
    new_n = n // factor
    
    # Reshape and average
    reshaped = arr[:, :new_n*factor, :new_n*factor, :new_n*factor]
    reshaped = reshaped.reshape(c, new_n, factor, new_n, factor, new_n, factor)
    return reshaped.mean(axis=(2, 4, 6))


if __name__ == '__main__':  
    numpy_dir = '/home/rmcconke/orcd/scratch/numpy_4000'
    input_prefix = 'channel_nearwall'
    fs = 4

    files = [f for f in glob.glob(os.path.join(numpy_dir, f"{input_prefix}*.npy")) 
            if "filtered" not in f and "grad" not in f and "sgs" not in f]
    for file in tqdm.tqdm(files):
        data = np.load(file)
        filtered = box_filter_downsample(data, fs)
        np.save(file.replace(input_prefix, f'{input_prefix}_filtered_fs{fs}'), filtered)


