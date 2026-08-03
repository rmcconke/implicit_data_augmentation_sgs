import numpy as np
import os
from multiprocessing import Pool
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
from dotenv import load_dotenv

load_dotenv()

# ================== SHARED CONFIGURATION ==================
auth_token = os.getenv('JHTDB_AUTH_TOKEN')
spatial_interpolation = "lag8"
num_processes = 10

# Domain is the same size (8*pi x 2 x 3*pi) for both 'channel' (Re1000)
# and 'channel5200' (Re5200)
domain_xmin, domain_xmax = 0.0, 8 * np.pi
domain_ymin, domain_ymax = -1.0, 1.0
domain_zmin, domain_zmax = 0.0, 3 * np.pi
z_fixed = (domain_zmin + domain_zmax) / 2  # 1.5*pi, same fixed plane for both

# ================== PER-DATASET CONFIGURATION ==================
configs = [
    {
        'dataset_title': 'channel',
        'output_path': '/orcd/data/abodner/002/ryley/numpy_channel_xyslice',
        'save_prefix': 'channel_xy_slice_0.005',
        'dx': 0.005,
        'slice_length_x': 2.0,
        'T_fixed': 0.0,                    # continuous time value for 'channel'
    },
    {
        'dataset_title': 'channel5200',
        'output_path': '/orcd/data/abodner/002/ryley/numpy_channel5200_xyslice',
        'save_prefix': 'channel5200_xy_slice_0.005',
        'dx': 0.005,
        'slice_length_x': 2.0,
        'T_fixed': 1,                      # channel5200 uses integer snapshot index (1-11)
    },
]
# =================================================================


def create_chunks(N_total, max_chunk_size=64):
    """Break a large 1D grid into API-friendly chunks."""
    if N_total <= max_chunk_size:
        return [(0, N_total)]

    num_chunks = int(np.ceil(N_total / max_chunk_size))
    chunk_size = N_total // num_chunks
    remainder = N_total % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        current_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_size
        chunks.append((start, end))
        start = end

    return chunks


def get_all_chunks_for_slice(Nx, Ny, xmin, xmax, ymin, ymax, z_value):
    """Get all chunk definitions for a 2D x-y slice at a fixed z."""
    x_points = np.linspace(xmin, xmax, Nx)
    y_points = np.linspace(ymin, ymax, Ny)

    x_chunks = create_chunks(Nx, 64)
    y_chunks = create_chunks(Ny, 64)

    all_chunks = []
    chunk_id = 0

    for x_start, x_end in x_chunks:
        for y_start, y_end in y_chunks:
            x_chunk = x_points[x_start:x_end]
            y_chunk = y_points[y_start:y_end]

            X_chunk, Y_chunk = np.meshgrid(x_chunk, y_chunk, indexing='ij')
            Z_chunk = np.full_like(X_chunk, z_value)
            points_chunk = np.stack([X_chunk.flatten(), Y_chunk.flatten(), Z_chunk.flatten()], axis=1)

            all_chunks.append({
                'chunk_id': chunk_id,
                'points': points_chunk,
                'shape': (len(x_chunk), len(y_chunk)),
                'indices': {
                    'x': (x_start, x_end),
                    'y': (y_start, y_end),
                }
            })
            chunk_id += 1

    return all_chunks


def download_chunk(args):
    """Download a single chunk"""
    current_time, dataset, spatial_interpolation, output_path, save_prefix, chunk_info = args

    chunk_id = chunk_info['chunk_id']
    filename = f'{output_path}/{save_prefix}_chunk{chunk_id}.npy'

    if os.path.exists(filename):
        print(f'Chunk {chunk_id} already exists, skipping')
        return

    try:
        result_vel = getData(dataset,
                        'velocity',
                        current_time,
                        'none',
                        spatial_interpolation,
                        'field',
                        chunk_info['points'])

        vel_chunk = np.array(result_vel).reshape(*chunk_info['shape'], 3).transpose(2, 0, 1)
        np.save(filename, vel_chunk)
        print(f'Downloaded chunk {chunk_id}')

    except Exception as e:
        print(f'Error downloading chunk {chunk_id}: {e}')


def stitch_chunks(Nx, Ny, xmin, xmax, ymin, ymax, z_fixed, output_path, save_prefix):
    """Stitch all chunks together into the final 2D slice."""

    final_filename = f'{output_path}/{save_prefix}.npy'
    if os.path.exists(final_filename):
        print(f'Final file {final_filename} already exists, skipping stitching')
        return True

    all_chunks = get_all_chunks_for_slice(Nx, Ny, xmin, xmax, ymin, ymax, z_fixed)

    vel_data_full = np.zeros((3, Nx, Ny))

    all_chunks_exist = True
    for chunk_info in all_chunks:
        chunk_id = chunk_info['chunk_id']
        chunk_filename = f'{output_path}/{save_prefix}_chunk{chunk_id}.npy'

        if not os.path.exists(chunk_filename):
            print(f'Missing chunk file: {chunk_filename}')
            all_chunks_exist = False
            continue

        vel_chunk = np.load(chunk_filename)

        x_start, x_end = chunk_info['indices']['x']
        y_start, y_end = chunk_info['indices']['y']

        vel_data_full[:, x_start:x_end, y_start:y_end] = vel_chunk

        os.remove(chunk_filename)

    if all_chunks_exist:
        np.save(final_filename, vel_data_full)
        print(f'Successfully stitched and saved {final_filename}')
        return True
    else:
        print(f'Not all chunks available')
        return False


def download_slice(cfg):
    """Download (if needed) and return the path to the stitched x-y slice for one dataset config."""
    dataset_title = cfg['dataset_title']
    output_path = cfg['output_path']
    save_prefix = cfg['save_prefix']
    dx = cfg['dx']
    dy = dx
    slice_length_x = cfg['slice_length_x']
    T_fixed = cfg['T_fixed']

    domain_center_x = (domain_xmin + domain_xmax) / 2
    xmin = domain_center_x - slice_length_x / 2
    xmax = domain_center_x + slice_length_x / 2
    ymin, ymax = domain_ymin, domain_ymax

    Nx = int(np.round((xmax - xmin) / dx)) + 1
    Ny = int(np.round((ymax - ymin) / dy)) + 1
    xmax = xmin + (Nx - 1) * dx
    ymax = ymin + (Ny - 1) * dy

    print(f"\n=== {dataset_title} ===")
    print(f"  Slice bounds: x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}], z={z_fixed:.4f} (fixed)")
    print(f"  Grid dimensions: {Nx} x {Ny} = {Nx*Ny:,} points")
    print(f"  Time: t={T_fixed}")

    final_filename = f'{output_path}/{save_prefix}.npy'

    if not os.path.exists(final_filename):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dataset = turb_dataset(dataset_title=dataset_title,
                                 output_path=output_path,
                                 auth_token=auth_token)

        chunks = get_all_chunks_for_slice(Nx, Ny, xmin, xmax, ymin, ymax, z_fixed)
        print(f"  Total download tasks: {len(chunks)}")

        all_download_tasks = [
            (T_fixed, dataset, spatial_interpolation, output_path, save_prefix, chunk_info)
            for chunk_info in chunks
        ]

        with Pool(num_processes) as pool:
            pool.map(download_chunk, all_download_tasks)

        stitch_chunks(Nx, Ny, xmin, xmax, ymin, ymax, z_fixed, output_path, save_prefix)
    else:
        print(f"  Already exists: {final_filename}")

    expected_shape = (3, Nx, Ny)
    data = np.load(final_filename)
    if data.shape != expected_shape:
        print(f'  WARNING: shape {data.shape} != expected {expected_shape}')
    else:
        print(f'  Verified: {final_filename}, shape {data.shape}')

    return final_filename


# ================== MAIN EXECUTION ==================

if __name__ == '__main__':
    for cfg in configs:
        download_slice(cfg)

    print("\nDone.")