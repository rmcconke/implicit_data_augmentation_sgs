from sgs_filter_jhtb_data import process_sgs_data
from sgs_filter_ratios import DELTA_FS4

# Control: normalize by one box-averaged G rather than the local G at
# each voxel, so the gradient field keeps its relative amplitude across the box.
if __name__ == '__main__':
    numpy_dir = "/orcd/data/abodner/002/ryley/numpy_4000_sgs"

    for prefix, delta in DELTA_FS4.items():
        print(f"{prefix}: box-mean normalization at fs=4")
        process_sgs_data(prefix, numpy_dir, delta, fs=4, box_mean_norm=True, suffix='bm4')
