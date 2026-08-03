from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
import os
import numpy as np
from glob import glob
import time
from functools import wraps
import random
import shutil
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

random.seed(42)

# Load environment variables from .env file
load_dotenv()

# Domain dimensions (channel)
auth_token = os.getenv('JHTDB_AUTH_TOKEN')
jhtdb_dataset_title = 'channel5200'
output_path = f'/orcd/data/abodner/002/ryley/numpy_4000_sgs'
n_procs = 32  # Number of parallel workers
Nxy = 64
Nz = 64

box_side_length = 0.123
domain_min_x = 0
domain_max_x = 8*np.pi
domain_min_z = 0
domain_max_z = 3*np.pi

n_times = 11
number_boxes_x = 15
number_boxes_z = 5

times = range(1,n_times+1)
box_dict = {'channel5200_nearwall': {'box_min_y': 0.8763934417, 'box_max_y': 0.999805},
            'channel5200_middle': {'box_min_y': -0.0617, 'box_max_y': 0.0617}}

box_spacing_x = (domain_max_x - domain_min_x) / number_boxes_x
box_spacing_z = (domain_max_z - domain_min_z) / number_boxes_z

box_centers_x = np.linspace(domain_min_x+box_spacing_x/2, domain_max_x-box_spacing_x/2, number_boxes_x)
box_centers_z = np.linspace(domain_min_z+box_spacing_z/2, domain_max_z-box_spacing_z/2, number_boxes_z)

box_mins_x = box_centers_x - box_side_length/2
box_maxs_x = box_centers_x + box_side_length/2
box_mins_z = box_centers_z - box_side_length/2
box_maxs_z = box_centers_z + box_side_length/2

print(f'X range: [{min(box_mins_x):.4f}, {max(box_maxs_x):.4f}]')
print(f'Z range: [{min(box_mins_z):.4f}, {max(box_maxs_z):.4f}]')


# Write provenance file
provenance = {
    'created': datetime.now().isoformat(),
    'dataset': {
        'source': 'Johns Hopkins Turbulence Database (JHTDB)',
        'name': jhtdb_dataset_title,
        'url': 'https://turbulence.pha.jhu.edu/',
        'flow': 'Turbulent channel flow',
        'Re_tau': 5200,
    },
    'query_params': {
        'interpolation': 'lag8',
        'temporal_interpolation': 'none',
    },
    'grid': {
        'Nxy': Nxy,
        'Nz': Nz,
        'box_side_length': box_side_length,
        'box_side_length_description': 'Non-dimensional units (channel half-height = 1)',
        'dx_plus_approx': 10,
        'box_side_wall_units_approx': 640,
    },
    'domain': {
        'x_range': [domain_min_x, domain_max_x],
        'z_range': [domain_min_z, domain_max_z],
        'x_range_description': '[0, 8*pi]',
        'z_range_description': '[0, 3*pi]',
    },
    'boxes': {
        'number_boxes_x': number_boxes_x,
        'number_boxes_z': number_boxes_z,
        'box_spacing_x': box_spacing_x,
        'box_spacing_z': box_spacing_z,
        'regions': box_dict,
    },

    'notes': {
        'filter_ratio': 4,
        'LES_grid_size': 16,
        'delta_plus_LES': 40,
        'description': 'DNS data for SGS model training. Filter ratio of 4 gives 16^3 LES grid with delta+ ~ 40.',
    }
}

provenance_file = os.path.join(output_path, 'provenance.json')
with open(provenance_file, 'w') as f:
    json.dump(provenance, f, indent=2)
print(f'Wrote provenance file: {provenance_file}')

dataset = turb_dataset(dataset_title=jhtdb_dataset_title, 
                        output_path=output_path, 
                        auth_token=auth_token)


def create_query_points_from_box(Nxy, Nz, xmin, xmax, ymin, ymax, zmin, zmax):
    """Create all query points for the domain"""
    x_points = np.linspace(xmin, xmax, Nxy)
    y_points = np.linspace(ymin, ymax, Nxy)
    z_points = np.linspace(zmin, zmax, Nz)
    X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points, (Nxy, Nxy, Nz), (x_points, y_points, z_points)


def fetch_single_box(args):
    """Fetch a single box - returns True on success, False on failure"""
    save_prefix, t_idx, t, box_num_x, box_num_z, box_min_y, box_max_y = args
    
    output_file = f'{output_path}/{save_prefix}_box_x{box_num_x}_z{box_num_z}_time_{t_idx}.npy'
    
    # Skip if already exists
    if os.path.exists(output_file):
        return True
    
    box_min_x = box_mins_x[box_num_x]
    box_max_x = box_maxs_x[box_num_x]
    box_min_z = box_mins_z[box_num_z]
    box_max_z = box_maxs_z[box_num_z]
    
    try:
        query_points, (nx, ny, nz), _ = create_query_points_from_box(
            Nxy, Nz, box_min_x, box_max_x, box_min_y, box_max_y, box_min_z, box_max_z
        )
        
        result_vel = getData(dataset, 
                             'velocity',
                             t,
                             'none',
                             'lag8',
                             'field',
                             query_points)
        
        vel_data = np.array(result_vel).reshape(nx, ny, nz, 3).transpose(3, 0, 1, 2)
        np.save(output_file, vel_data)
        print(f'Saved: {save_prefix}_box_x{box_num_x}_z{box_num_z}_time_{t_idx}')
        return True
    except Exception as e:
        print(f'Failed: {save_prefix}_box_x{box_num_x}_z{box_num_z}_time_{t_idx} - {e}')
        return False


def get_missing_boxes(save_prefix, box_min_y, box_max_y):
    """Generate list of all missing boxes"""
    missing = []
    for t_idx, t in enumerate(times):
        for box_num_x in range(number_boxes_x):
            for box_num_z in range(number_boxes_z):
                output_file = f'{output_path}/{save_prefix}_box_x{box_num_x}_z{box_num_z}_time_{t_idx}.npy'
                if not os.path.exists(output_file):
                    missing.append((save_prefix, t_idx, t, box_num_x, box_num_z, box_min_y, box_max_y))
    return missing


for box in box_dict:
    box_min_y = box_dict[box]['box_min_y']
    box_max_y = box_dict[box]['box_max_y']
    save_prefix = box
    
    all_boxes_are_there = False
    while not all_boxes_are_there:
        # Get list of missing boxes
        missing_boxes = get_missing_boxes(save_prefix, box_min_y, box_max_y)
        
        if not missing_boxes:
            all_boxes_are_there = True
            print(f'All boxes are there for {save_prefix}!')
            break
        
        print(f'Processing {len(missing_boxes)} missing boxes for {save_prefix}...')
        
        # Process missing boxes in parallel
        with ThreadPoolExecutor(max_workers=n_procs) as executor:
            futures = [executor.submit(fetch_single_box, args) for args in missing_boxes]
            
            # Wait for all to complete
            for future in as_completed(futures):
                pass  # Results handled in fetch_single_box
        
        # Check if all boxes are there now (loop will retry if not)
        missing_boxes = get_missing_boxes(save_prefix, box_min_y, box_max_y)
        if not missing_boxes:
            all_boxes_are_there = True
            print(f'All boxes are there for {save_prefix}!')

print(f'\nDone! Total boxes per region: {number_boxes_x} x {number_boxes_z} x {n_times} = {number_boxes_x * number_boxes_z * n_times}')
print(f'Total boxes: {2 * number_boxes_x * number_boxes_z * n_times}')