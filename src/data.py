import torch
import numpy as np
import os
from typing import Optional
from torch.utils.data import Dataset
from global_config import numpy_dir
from utils import get_timesteps_available, full_filename
import random
from transformations import apply_so3_augmentation, apply_octahedral_augmentation
import re
import glob
from pathlib import Path


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class NumpyTimeSeriesDataset(Dataset):
    def __init__(
        self,
        input_file_list,
        target_file_list,
        scaling_factors = None,
        scale: bool = True,
        augmentation_group: Optional[str] = None,
        pointwise: bool = False,
        precompute_eigenframe_inputs: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda:0'),
    ):
        self.numpy_dir = numpy_dir
        self.inputs = []
        self.targets = []
        self.pointwise = pointwise
        self.precompute_eigenframe_inputs = precompute_eigenframe_inputs
        print(f"Precompute eigenframe inputs: {precompute_eigenframe_inputs}")
        self.scale_flag = scale

        for input_file, target_file in zip(input_file_list, target_file_list):
            input_numpy = np.load(os.path.join(numpy_dir, input_file)).astype(np.float32)
            target_numpy = np.load(os.path.join(numpy_dir, target_file)).astype(np.float32)
            self.inputs.append(input_numpy)
            self.targets.append(target_numpy)
            
        self.inputs = torch.from_numpy(np.array(self.inputs)).to(dtype).to(device)
        self.targets = torch.from_numpy(np.array(self.targets)).to(dtype).to(device)
        print(f"After loading - inputs: {self.inputs.shape}, targets: {self.targets.shape}") 

        if scale:
            if 'sgs' in target_file_list[0]:
                self.scaling_factors = self.scale_tensor(scaling_factors)
            elif 'filtered' in input_file_list[0]:
                self.scaling_factors = self.scale_velocity(scaling_factors)
            else:
                raise ValueError(f"Unknown scaling type for: {input_file_list[0]}, couldn't detect task type")

        if augmentation_group is not None:
            if augmentation_group == 'so3':
                self.inputs, self.targets = apply_so3_augmentation(self.inputs, self.targets)
            elif augmentation_group == 'oct':
                self.inputs, self.targets = apply_octahedral_augmentation(self.inputs, self.targets)
        
        if self.pointwise:
            B, C_in, X, Y, Z = self.inputs.shape
            B, C_out, X, Y, Z = self.targets.shape

            self.inputs = (
                self.inputs
                    .flatten(start_dim=2)      # (B, C, V)
                    .permute(0, 2, 1)          # (B, V, C)
                    .reshape(-1, C_in)            # (B*V, C)
            )
            self.targets = (
                self.targets
                    .flatten(start_dim=2)      # (B, C, V)
                    .permute(0, 2, 1)          # (B, V, C)
                    .reshape(-1, C_out)            # (B*V, C)
            )
            print(f"After pointwise reshape - inputs: {self.inputs.shape}, targets: {self.targets.shape}")  # <-- HERE

            if self.precompute_eigenframe_inputs:
                print("Precomputing eigenframe inputs...")
                filter_width = 0.0375  # TODO: pass this as parameter
                
                vel_grad_mat = self.inputs.reshape(-1, 3, 3)
                
                S = 0.5 * (vel_grad_mat + vel_grad_mat.transpose(-2, -1))
                W = 0.5 * (vel_grad_mat - vel_grad_mat.transpose(-2, -1))
                
                G = torch.sqrt((S * S).sum(dim=(-2, -1)) + (W * W).sum(dim=(-2, -1)))
                G_safe = G.clamp(min=1e-10)
                
                # Chunked eigh to avoid limits
                chunk_size = 10000
                eigenvalues_list = []
                eigenvectors_list = []
                for i in range(0, S.shape[0], chunk_size):
                    evals, evecs = torch.linalg.eigh(S[i:i + chunk_size])
                    eigenvalues_list.append(evals)
                    eigenvectors_list.append(evecs)
                eigenvalues = torch.cat(eigenvalues_list, dim=0).flip(-1)
                eigenvectors = torch.cat(eigenvectors_list, dim=0).flip(-1)
                
                # Vorticity
                omega = torch.stack([
                    vel_grad_mat[..., 2, 1] - vel_grad_mat[..., 1, 2],
                    vel_grad_mat[..., 0, 2] - vel_grad_mat[..., 2, 0],
                    vel_grad_mat[..., 1, 0] - vel_grad_mat[..., 0, 1],
                ], dim=-1)
                
                # Orient eigenvectors
                v1 = eigenvectors[..., :, 0]
                v3 = eigenvectors[..., :, 2]
                v1 = torch.where((v1 * omega).sum(dim=-1, keepdim=True) < 0, -v1, v1)
                v3 = torch.where((v3 * omega).sum(dim=-1, keepdim=True) < 0, -v3, v3)
                v2 = torch.linalg.cross(v3, v1)
                eigenvectors = torch.stack([v1, v2, v3], dim=-1)
                
                # Compute invariants (inputs only)
                omega_S1 = (omega * v1).sum(dim=-1)
                omega_S2 = (omega * v2).sum(dim=-1)
                omega_S3 = (omega * v3).sum(dim=-1)
                lambda3 = eigenvalues[..., 2]
                
                self.inputs = torch.stack([
                    lambda3 / G_safe,
                    omega_S1 / G_safe,
                    omega_S2 / G_safe,
                    omega_S3 / G_safe,
                ], dim=-1)
                
                # Store eigenvectors and G for inverse transform in forward pass
                self.eigenvectors = eigenvectors
                self.G = G_safe
                
                # Targets stay in physical space (no transformation)
                
                print(f"Eigenframe precomputation done. Inputs: {self.inputs.shape}, Targets: {self.targets.shape}")

    def scale_velocity(self, scaling_factors):
        if scaling_factors is not None:
            self.input_mean = scaling_factors['input_mean']
            self.input_std = scaling_factors['input_std']
            self.target_mean = scaling_factors['target_mean']
            self.target_std = scaling_factors['target_std']

        else:
            self.input_mean = self.inputs.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.input_std = self.inputs.std(dim=(0, 2, 3, 4), keepdim=True)
            self.target_mean = self.targets.mean(dim=(0, 2, 3, 4), keepdim=True)
            self.target_std = self.targets.std(dim=(0, 2, 3, 4), keepdim=True)
            
        self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-8)
        self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-8)
            
        return {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
        }

    def scale_tensor(self, scaling_factors):
        if scaling_factors is not None:
            self.input_mean_magnitude = scaling_factors['input_mean_magnitude']
            self.target_mean_magnitude = scaling_factors['target_mean_magnitude']
            
        else:
            self.input_mean_magnitude = torch.linalg.norm(self.inputs, dim=1, keepdim=True).mean(dim=(0, 2, 3, 4), keepdim=False)
            self.target_mean_magnitude = torch.linalg.norm(self.targets, dim=1, keepdim=True).mean(dim=(0, 2, 3, 4), keepdim=False)
        
        self.inputs = self.inputs / (self.input_mean_magnitude + 1e-8)
        self.targets = self.targets / (self.target_mean_magnitude + 1e-8)
        return {
            'input_mean_magnitude': self.input_mean_magnitude,
            'target_mean_magnitude': self.target_mean_magnitude,
        }

    def get_scaling_factors(self):
        return self.scaling_factors

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.precompute_eigenframe_inputs:
            # inputs: (4,), eigenvectors: (3,3), G: (1,)
            # Flatten and concat: 4 + 9 + 1 = 14
            combined = torch.cat([
                self.inputs[idx],
                self.eigenvectors[idx].flatten(),
                self.G[idx].unsqueeze(-1),
            ], dim=-1)
            return combined, self.targets[idx]
        else:
            return self.inputs[idx], self.targets[idx]
        

    def get_all_data(self):
        if self.precompute_eigenframe_inputs:
            return torch.cat([
                self.inputs,
                self.eigenvectors.flatten(start_dim=1),
                self.G.unsqueeze(-1),
            ], dim=-1), self.targets
        else:
            return self.inputs, self.targets


def get_files_from_pattern(prefix, pattern, times):
    files = []
    for z in pattern:
        pattern_glob = f'{numpy_dir}/{prefix}_box_x*_{z}_time_*.npy'
        for f in glob.glob(pattern_glob):
            time_val = int(Path(f).stem.split('_time_')[1])
            if time_val in times:
                files.append(f)
    return sorted(files)

def get_target_from_input(input_file, input_prefix, target_prefix):
    """Derive target filename by replacing the prefix."""
    return input_file.replace(f'/{input_prefix}_box_', f'/{target_prefix}_box_')

def extract_file_id(filepath):
    """Extract the unique identifier (box_x*_z*_time_*) from a filepath."""
    stem = Path(filepath).stem
    match = re.search(r'(box_x\d+_z\d+_time_\d+)', stem)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract file ID from {filepath}")

def qc_file_lists(train_input_files, train_target_files, val_input_files, val_target_files, input_prefix, target_prefix):
    """Run QC checks on train/val file lists."""
    
    # 1. Length checks
    assert len(train_input_files) == len(train_target_files), \
        f"Train input/target length mismatch: {len(train_input_files)} vs {len(train_target_files)}"
    assert len(val_input_files) == len(val_target_files), \
        f"Val input/target length mismatch: {len(val_input_files)} vs {len(val_target_files)}"
    assert len(train_input_files) > 0, "No training samples found"
    assert len(val_input_files) > 0, "No validation samples found"

    # 2. Input-target correspondence check
    for inp, tgt in zip(train_input_files, train_target_files):
        inp_id, tgt_id = extract_file_id(inp), extract_file_id(tgt)
        assert inp_id == tgt_id, f"Train input/target ID mismatch: {inp_id} vs {tgt_id}"
    for inp, tgt in zip(val_input_files, val_target_files):
        inp_id, tgt_id = extract_file_id(inp), extract_file_id(tgt)
        assert inp_id == tgt_id, f"Val input/target ID mismatch: {inp_id} vs {tgt_id}"

    # 3. Data leakage check
    train_ids = set(extract_file_id(f) for f in train_input_files)
    val_ids = set(extract_file_id(f) for f in val_input_files)
    overlap = train_ids & val_ids
    assert not overlap, f"Data leakage detected! Overlapping IDs: {overlap}"

    # 4. Prefix check
    for f in train_input_files + val_input_files:
        assert input_prefix in Path(f).stem, f"Input file missing input prefix: {f}"
    for f in train_target_files + val_target_files:
        assert target_prefix in Path(f).stem, f"Target file missing target prefix: {f}"

    print("QC passed: all checks OK")

def get_tv_file_lists(dataset_config):
    # Get input files
    train_input_files = get_files_from_pattern(dataset_config.input_prefix, dataset_config.train_pattern, dataset_config.train_times)
    val_input_files = get_files_from_pattern(dataset_config.input_prefix, dataset_config.val_pattern, dataset_config.val_times)

    # Derive corresponding target files
    train_target_files = [get_target_from_input(f, dataset_config.input_prefix, dataset_config.target_prefix) for f in train_input_files]
    val_target_files = [get_target_from_input(f, dataset_config.input_prefix, dataset_config.target_prefix) for f in val_input_files]

    # Verify target files exist
    for f in train_target_files + val_target_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Target file not found: {f}")

    print(f"Train samples: {len(train_input_files)}")
    print(f"Val samples: {len(val_input_files)}")

    # Subsample training data
    if dataset_config.train_fraction < 1.0:
        n_samples = int(len(train_input_files) * dataset_config.train_fraction)
        n_samples = max(1, n_samples)
        indices = random.sample(range(len(train_input_files)), n_samples)
        train_input_files = [train_input_files[i] for i in indices]
        train_target_files = [train_target_files[i] for i in indices]
        print(f"Train samples (after subsampling): {len(train_input_files)}")

    # Run QC
    qc_file_lists(
        train_input_files, train_target_files,
        val_input_files, val_target_files,
        dataset_config.input_prefix, dataset_config.target_prefix
    )

    return train_input_files, train_target_files, val_input_files, val_target_files



def tvt_split_list(list_to_split, train_split, val_split, test_split):
    n = len(list_to_split)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    train_list = list_to_split[:train_end]
    val_list = list_to_split[train_end:val_end]
    test_list = list_to_split[val_end:]
    
    return train_list, val_list, test_list

def get_test_file_lists(test_dataset_config):
    test_input_files = get_files_from_pattern(test_dataset_config.input_prefix, test_dataset_config.pattern, test_dataset_config.times)
    test_target_files = get_files_from_pattern(test_dataset_config.target_prefix, test_dataset_config.pattern, test_dataset_config.times)
    if len(test_input_files) != len(test_target_files):
        raise ValueError(f"Number of input and target files do not match for test sets")

    return test_input_files, test_target_files