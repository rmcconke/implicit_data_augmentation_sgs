import numpy as np
import os
from multiprocessing import Pool
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
# ================== USER CONFIGURATION ==================
# Domain parameters
dx = 0.005  # Grid spacing in x
dy = dx    # Grid spacing in y  
dz = dx    # Grid spacing in z

# Full domain bounds
domain_xmin, domain_xmax = 0.0, 8 * np.pi
domain_ymin, domain_ymax = -1.0, 1.0
domain_zmin, domain_zmax = 0.0, 3 * np.pi

# Box configuration - square box spanning full channel height
box_side_length = 2.0  # Side length in x and z directions
# y will span full channel: -1 to 1

# Time parameters
T_start = 0.0
T_end = 1.0
num_timesteps = 1

# API parameters
jhtdb_dataset_title = "channel"  # Your dataset name
auth_token = os.getenv('JHTDB_AUTH_TOKEN')   # Your auth token
spatial_interpolation = "lag8"   # Interpolation method

# Output parameters
output_path = f'/data/NFS/potato/turbulence/numpy'
save_prefix = "channel_full_rich_0.005"

# Parallel processing
num_processes = 10
# ========================================================

def create_chunks(N_total, max_chunk_size=64):
    """
    Create chunk indices to break down a large grid into API-friendly pieces
    Returns list of (start_idx, end_idx) tuples
    """
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

def get_all_chunks_for_box(Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Get all chunk definitions for a box
    Returns list of chunk info dictionaries
    """
    # Create full coordinate arrays
    x_points = np.linspace(xmin, xmax, Nx)
    y_points = np.linspace(ymin, ymax, Ny)
    z_points = np.linspace(zmin, zmax, Nz)
    
    # Get chunking for each dimension
    x_chunks = create_chunks(Nx, 64)
    y_chunks = create_chunks(Ny, 64)
    z_chunks = create_chunks(Nz, 64)
    
    all_chunks = []
    chunk_id = 0
    
    for x_start, x_end in x_chunks:
        for y_start, y_end in y_chunks:
            for z_start, z_end in z_chunks:
                # Extract chunk coordinates
                x_chunk = x_points[x_start:x_end]
                y_chunk = y_points[y_start:y_end]
                z_chunk = z_points[z_start:z_end]
                
                # Create meshgrid for this chunk
                X_chunk, Y_chunk, Z_chunk = np.meshgrid(x_chunk, y_chunk, z_chunk, indexing='ij')
                points_chunk = np.stack([X_chunk.flatten(), Y_chunk.flatten(), Z_chunk.flatten()], axis=1)
                
                all_chunks.append({
                    'chunk_id': chunk_id,
                    'points': points_chunk,
                    'shape': (len(x_chunk), len(y_chunk), len(z_chunk)),
                    'indices': {
                        'x': (x_start, x_end),
                        'y': (y_start, y_end), 
                        'z': (z_start, z_end)
                    }
                })
                chunk_id += 1
    
    return all_chunks

def download_chunk(args):
    """Download a single chunk"""
    t_idx, current_time, dataset, spatial_interpolation, output_path, save_prefix, chunk_info = args
    
    chunk_id = chunk_info['chunk_id']
    filename = f'{output_path}/{save_prefix}_time{t_idx}_chunk{chunk_id}.npy'
    
    if os.path.exists(filename):
        print(f'Chunk {chunk_id} for time {t_idx} already exists, skipping')
        return
    
    try:
        result_vel = getData(dataset, 
                        'velocity',
                        current_time,
                        'none',
                        spatial_interpolation,
                        'field',
                        chunk_info['points'])
        
        # Reshape and transpose to match expected format: (3, nx, ny, nz)
        vel_chunk = np.array(result_vel).reshape(*chunk_info['shape'], 3).transpose(3, 0, 1, 2)
        np.save(filename, vel_chunk)
        print(f'Downloaded chunk {chunk_id} for time {t_idx}')
        
    except Exception as e:
        print(f'Error downloading chunk {chunk_id} for time {t_idx}: {e}')

def stitch_chunks_for_timestep(t_idx, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, output_path, save_prefix):
    """Stitch all chunks together for one timestep"""
    
    final_filename = f'{output_path}/{save_prefix}_time{t_idx}.npy'
    if os.path.exists(final_filename):
        print(f'Final file {final_filename} already exists, skipping stitching')
        return True
    
    # Get chunk info
    all_chunks = get_all_chunks_for_box(Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax)
    
    # Initialize full array
    vel_data_full = np.zeros((3, Nx, Ny, Nz))
    
    # Load and stitch each chunk
    all_chunks_exist = True
    for chunk_info in all_chunks:
        chunk_id = chunk_info['chunk_id']
        chunk_filename = f'{output_path}/{save_prefix}_time{t_idx}_chunk{chunk_id}.npy'
        
        if not os.path.exists(chunk_filename):
            print(f'Missing chunk file: {chunk_filename}')
            all_chunks_exist = False
            continue
            
        # Load chunk
        vel_chunk = np.load(chunk_filename)
        
        # Place in full array
        x_start, x_end = chunk_info['indices']['x']
        y_start, y_end = chunk_info['indices']['y']
        z_start, z_end = chunk_info['indices']['z']
        
        vel_data_full[:, x_start:x_end, y_start:y_end, z_start:z_end] = vel_chunk
        
        # Clean up chunk file
        os.remove(chunk_filename)
    
    if all_chunks_exist:
        # Save final stitched file
        np.save(final_filename, vel_data_full)
        print(f'Successfully stitched and saved {final_filename}')
        return True
    else:
        print(f'Not all chunks available for time {t_idx}')
        return False

# ================== MAIN EXECUTION ==================

# Calculate box bounds
# x and z: centered in domain with specified side length
domain_center_x = (domain_xmin + domain_xmax) / 2  # 4π
domain_center_z = (domain_zmin + domain_zmax) / 2  # 1.5π

xmin = domain_center_x - box_side_length / 2  # 4π - 1
xmax = domain_center_x + box_side_length / 2  # 4π + 1
zmin = domain_center_z - box_side_length / 2  # 1.5π - 1  
zmax = domain_center_z + box_side_length / 2  # 1.5π + 1

# y: full channel height
ymin = domain_ymin  # -1
ymax = domain_ymax  # +1

# Calculate grid dimensions
Nx = int(np.round((xmax - xmin) / dx)) + 1
Ny = int(np.round((ymax - ymin) / dy)) + 1
Nz = int(np.round((zmax - zmin) / dz)) + 1

# Adjust bounds to match exact grid spacing
xmax = xmin + (Nx - 1) * dx
ymax = ymin + (Ny - 1) * dy
zmax = zmin + (Nz - 1) * dz

print(f"Domain configuration:")
print(f"  Full domain: x=[0, {8*np.pi:.2f}], y=[-1, 1], z=[0, {3*np.pi:.2f}]")
print(f"  Box bounds: x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}], z=[{zmin:.3f}, {zmax:.3f}]")
print(f"  Box dimensions: {xmax-xmin:.3f} x {ymax-ymin:.3f} x {zmax-zmin:.3f}")
print(f"  Grid spacing: dx={dx}, dy={dy}, dz={dz}")
print(f"  Grid dimensions: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} points")
print(f"  Time range: {T_start} to {T_end} with {num_timesteps} timesteps")

# Check if chunking is needed
if Nx > 64 or Ny > 64 or Nz > 64:
    chunks = get_all_chunks_for_box(Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax)
    print(f"  Chunking: Grid will be split into {len(chunks)} chunks")
    max_chunk_points = max(len(chunk['points']) for chunk in chunks)
    print(f"  Maximum points per chunk: {max_chunk_points:,} (limit: {64**3:,})")
else:
    print(f"  No chunking needed (grid fits in single API call)")

# Create output directory
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Initialize dataset (you'll need to import your API functions)
# from your_turbulence_api import turb_dataset, getData
dataset = turb_dataset(dataset_title=jhtdb_dataset_title, 
                         output_path=output_path, 
                         auth_token=auth_token)

# Create time array
times = np.linspace(T_start, T_end, num_timesteps)

print("\nStep 1: Downloading all chunks...")

# Get all chunks for the box
chunks = get_all_chunks_for_box(Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax)

# Create all download tasks
all_download_tasks = []
for t_idx, current_time in enumerate(times):
    for chunk_info in chunks:
        task = (t_idx, current_time, dataset, spatial_interpolation, 
               output_path, save_prefix, chunk_info)
        all_download_tasks.append(task)

print(f"Total download tasks: {len(all_download_tasks)}")

# Download all chunks in parallel
with Pool(num_processes) as pool:
    pool.map(download_chunk, all_download_tasks)

print("\nStep 2: Stitching chunks together...")

# Stitch chunks for each timestep
for t_idx, current_time in enumerate(times):
    stitch_chunks_for_timestep(t_idx, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, output_path, save_prefix)

print("\nStep 3: Verifying that all files exist...")
looks_good = True
for t_idx, current_time in enumerate(times):
    filename = f'{output_path}/{save_prefix}_time{t_idx}.npy'
    if not os.path.exists(filename):
        print(f'Missing file: {filename}')
        looks_good = False
    else:
        # Check file shape
        data = np.load(filename)
        expected_shape = (3, Nx, Ny, Nz)
        if data.shape != expected_shape:
            print(f'Wrong shape for {filename}: got {data.shape}, expected {expected_shape}')
            looks_good = False

if looks_good: 
    print(f'All files downloaded and verified successfully!')
    print(f'Files saved as: {output_path}/{save_prefix}_time{{t_idx}}.npy')
    print(f'Data shape: (3, {Nx}, {Ny}, {Nz}) - [velocity_components, x, y, z]')
else:
    print('Some files are missing or have incorrect shapes.')