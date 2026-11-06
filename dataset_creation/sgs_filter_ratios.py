from sgs_filter_jhtb_data import process_sgs_data

# Filter widths at ratio 4, from sgs_filter_jhtb_data.py. Delta is the coarse grid
# spacing, so it scales linearly with the downsampling factor.
DELTA_FS4 = {
    'channel_nearwall': 4.0026e-02,
    'channel_middle': 4.0026e-02,
    'channel5200_nearwall': 7.7132e-03,
    'channel5200_middle': 7.7132e-03,
}

if __name__ == '__main__':
    numpy_dir = "/orcd/data/abodner/002/ryley/numpy_4000_sgs"

    for fs in [2, 8]:
        for prefix, delta_fs4 in DELTA_FS4.items():
            delta = delta_fs4 * fs / 4
            print(f"{prefix} at fs={fs}: delta={delta:.6e}, interior={64 // fs - 2}^3")
            process_sgs_data(prefix, numpy_dir, delta, fs=fs)
