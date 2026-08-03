import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses
from data import get_tv_file_lists, NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model
import argparse
from transformations import precompute_octahedral_augmentations
from eval_model import eval_model_after_training
import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False,specialized_code=False,optimize_einsums=False,)
import time

def train_model(config,device):
    print(f"WELCOME TO THE EXPERIMENT: {config.barcode}")

    print(f"Using seed: {config.seed}")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Model 
    model = get_model(config.model).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model param count: {param_count}')

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss

    # Train and val data loaders
    train_input_files, train_target_files, val_input_files, val_target_files, = get_tv_file_lists(config.dataset)
    print(f"Number of train samples: {len(train_input_files)}")
    print(f"Number of val samples: {len(val_input_files)}")
    
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        pointwise=config.pointwise,
        precompute_eigenframe_inputs = config.eigenframe_model,
        device=device,
    )

    if config.memorization_test:
        val_dataset = train_dataset
    else:
        val_dataset = NumpyTimeSeriesDataset(
            input_file_list=val_input_files,
            target_file_list=val_target_files,
            scaling_factors=train_dataset.scaling_factors,
            augmentation_group=config.val_aug_group,
            pointwise=config.pointwise,
            precompute_eigenframe_inputs = config.eigenframe_model,
            device=device,
        )

    # Training loop
    train_losses = []
    val_losses = []
    epoch_times = []  # ADD before the loop
    best_val_loss = float('inf') 
    best_epoch = 0

    train_inputs, train_targets = train_dataset.get_all_data()
    val_inputs, val_targets = val_dataset.get_all_data()

    n_train = train_inputs.shape[0]
    n_val = val_inputs.shape[0]

    if config.train_aug_group == "oct":
        print("Precomputing octahedral augmentations...")
        train_inputs, train_targets = precompute_octahedral_augmentations(train_inputs, train_targets)
        n_train_augmented = train_inputs.shape[0]
    else:
        n_train_augmented = n_train


    batch_size = config.batch_size
    for epoch in range(config.epochs):    
        t_epoch = time.perf_counter()  # ADD THIS  
        model.train()
        shuffle_indices = torch.randperm(n_train_augmented, device=train_inputs.device)[:n_train]

        total_loss = 0.0
        
        for i in tqdm(range(0, n_train, batch_size), desc="Training"):
            batch_idx = shuffle_indices[i : i + batch_size]
            
            batch_inputs = train_inputs[batch_idx]
            batch_targets = train_targets[batch_idx]

            pred = model(batch_inputs)
            loss = criterion(pred, batch_targets)
            
            total_loss += loss.detach() * batch_idx.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = (total_loss / n_train).item()

        # Validation loop
        if epoch % config['validation_interval'] == 0:
            plot_losses(train_losses, val_losses, best_epoch, os.path.join(config['directory'], f'{config.barcode}_loss_plot.png'))
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for i in tqdm(range(0, n_val, batch_size), desc="Validation"):
                    batch_idx = slice(i, i + batch_size)
                    pred = model(val_inputs[batch_idx])
                    total_val_loss += criterion(pred, val_targets[batch_idx]).detach() * (min(i + batch_size, n_val) - i)
                
                val_loss = (total_val_loss / n_val).item()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # Save model at either save interval or stopping criteria (which are early stopping + end of training)
        if  (epoch % config['save_interval'] == 0) or \
            (epoch > best_epoch + config.early_stopping_patience) or \
            (epoch == config.epochs - 1):

            print(f"----> Saving model from epoch {best_epoch} (val loss: {best_val_loss}). {juicy_word()}!")
            torch.save(
                {
                    "model": best_state_cpu,
                    "param_count": param_count,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "seed": config.seed,
                    "scaling_factors": train_dataset.get_scaling_factors()
                },
                os.path.join(config['directory'], f'{config.barcode}_checkpoint.pt'),
            )

        if epoch > best_epoch + config.early_stopping_patience:
            print(f"\n\n=== {juicy_word()}! {juicy_word()}! {juicy_word()}! EARLY STOPPING at epoch {epoch} (no improvement since {best_epoch}) ===\n")
            break

        epoch_times.append(time.perf_counter() - t_epoch)  # ADD THIS
        window = min(50, len(epoch_times))  # ADD THIS
        print(f"Epoch [{epoch:4d}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_times[-1]:.2f}s, Avg({window}): {sum(epoch_times[-window:])/window:.2f}s")


    del model
    torch.cuda.empty_cache()
    plot_losses(train_losses, val_losses, best_epoch, os.path.join(config['directory'], f'{config.barcode}_loss_plot.png'))

    # Test after training
    if config.test_after_training:
        print(f"----> Running test eval on model from Epoch {best_epoch}")
        eval_model_after_training(config['directory'], device)

    
if __name__ == "__main__":
    # Get arguments, create experiment config
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to experiment config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--subdirectory', type=str, default=None, help='Subdirectory to save the experiment to, convenient for tracking sweeps')
    args = parser.parse_args()
    device = args.device    
    config = new_experiment(args.config_file, args.subdirectory)
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS/Python 3.13

    # Launch main training loop
    train_model(config, device)
