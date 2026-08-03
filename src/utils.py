import yaml
from box import Box
import matplotlib.pyplot as plt
import random
import petname
import os
import global_config as global_config
import datetime
import logging
import sys
import builtins
import numpy as np
import shutil

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return Box(config)


def plot_losses(train_losses, val_losses, best_val_loss_last_save, filename="loss_plot.png"):
    """Plot training and validation loss curves and save the plot as an image file"""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    
    # Plot main losses on top subplot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_ylabel('Normalized Loss (log scale)')
    ax1.set_yscale('log') 
    ax1.set_title('Training and Validation Loss')
    
    ax1.axvline(x=best_val_loss_last_save, color='green', linestyle='--', label='Best Saved Validation Loss')
    
    ax1.legend(loc='upper right')
    plt.tight_layout() 
    
    plt.savefig(filename)
    plt.close()  

### DEPRECATED
def plot_equivariance_error(val_equiv_errors, filename="equiv_error.png"):
    """Plot equivariance error curve and save the plot as an image file"""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    
    # Plot main losses on top subplot
    ax1.plot(val_equiv_errors, label='Equivariance Error', color='blue')
    ax1.set_ylabel('Equivariance Error')
    ax1.set_title('Equivariance Error')
    
    plt.tight_layout() 
    
    plt.savefig(filename)
    plt.close()  


def plot_equivariance_errors(val_equiv_errors=None, val_rel_equiv_errors=None, 
                           val_so3_equiv_errors=None, val_so3_rel_equiv_errors=None,
                           filename="equiv_errors.png"):
    """
    Plot absolute and relative equivariance errors for both octahedral and SO(3) symmetries.
    
    Args:
        val_equiv_errors: List of octahedral absolute equivariance errors over epochs
        val_rel_equiv_errors: List of octahedral relative equivariance errors over epochs  
        val_so3_equiv_errors: List of SO(3) absolute equivariance errors over epochs
        val_so3_rel_equiv_errors: List of SO(3) relative equivariance errors over epochs
        filename: Output filename for the plot
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    # Plot absolute errors on the left
    if val_equiv_errors is not None:
        axs[0].plot(val_equiv_errors, color='blue', linewidth=2, label='Octahedral')
    if val_so3_equiv_errors is not None:
        axs[0].plot(val_so3_equiv_errors, color='red', linewidth=2, label='SO(3)')
    
    axs[0].set_ylabel('Mean Absolute Equivariance Error')
    axs[0].set_xlabel('Epoch')
    axs[0].set_title('Mean Absolute Equivariance Error')
    axs[0].set_yscale('log')
    axs[0].grid(True, alpha=0.3)
    if val_equiv_errors is not None or val_so3_equiv_errors is not None:
        axs[0].legend()

    # Plot relative errors on the right
    if val_rel_equiv_errors is not None:
        axs[1].plot(val_rel_equiv_errors, color='blue', linewidth=2, label='Octahedral')
    if val_so3_rel_equiv_errors is not None:
        axs[1].plot(val_so3_rel_equiv_errors, color='red', linewidth=2, label='SO(3)')
        
    axs[1].set_ylabel('Mean Relative Equivariance Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('Mean Relative Equivariance Error')
    axs[1].set_yscale('log')
    axs[1].grid(True, alpha=0.3)
    if val_rel_equiv_errors is not None or val_so3_rel_equiv_errors is not None:
        axs[1].legend()
    
    plt.tight_layout() 
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()  

def juicy_word(capitalize=True):
    # Comprehensive list of words describing delicious food
    FOOD_ADJECTIVES = [
    # Basic delicious words
    "tasty", "juicy", "scrumptious", "delicious", "yummy", "flavorful",

    # Rich and indulgent
    "luscious", "decadent", "sumptuous", "rich", "creamy", "buttery", 
    "velvety", "luxurious", "opulent", "indulgent",

    # Fresh and vibrant
    "zesty", "tangy", "crisp", "fresh", "bright", "zingy", "refreshing",
    "vibrant", "lively", "invigorating", "citrusy",

    # Satisfying and flavorful
    "savory", "aromatic", "fragrant", "hearty", "robust", "bold",
    "full-bodied", "well-seasoned", "piquant", "pungent", "spicy",

    # Sweet and appealing
    "delectable", "divine", "heavenly", "delightful", "sweet", "honeyed",
    "sugary", "candy-like", "syrupy", "nectarous", "saccharine",

    # Texture-focused
    "tender", "succulent", "moist", "fluffy", "crispy", "crunchy",
    "smooth", "silky", "chewy", "firm", "al-dente", "flaky",
    "crumbly", "gooey", "sticky", "creamy", "chunky",

    # Generally appetizing
    "appetizing", "mouthwatering", "tempting", "irresistible", "satisfying",
    "enticing", "alluring", "inviting", "appealing", "tantalizing",

    # Sophisticated descriptors
    "exquisite", "refined", "gourmet", "sublime", "ambrosial", "palatable",
    "toothsome", "epicurean", "artisanal", "sophisticated", "elegant",

    # Casual and fun
    "nom-worthy", "finger-licking", "lip-smacking", "addictive", "crowd-pleasing",
    "comfort-food-y", "guilty-pleasure", "binge-worthy", "crave-worthy",

    # International flair
    "umami-rich", "perfectly-spiced", "authentic", "traditional", "exotic",

    # Intensity descriptors
    "intense", "explosive", "powerful", "subtle", "nuanced", "complex",
    "layered", "multidimensional", "well-balanced", "harmonious",

    # Additional appealing words
    "memorable", "outstanding", "exceptional", "phenomenal", "spectacular",
    "magnificent", "wonderful", "amazing", "incredible", "fantastic",
    "marvelous", "splendid", "superb", "excellent", "perfect",
    "wholesome", "nourishing", "comforting", "soul-warming", "blissful"
    ]

    word = random.choice(FOOD_ADJECTIVES)
    return word.capitalize() if capitalize else word

def get_timesteps_available(numpy_dir,prefix):
    print(f"Getting timesteps available for {prefix} in {numpy_dir}")
    files = os.listdir(numpy_dir)
    if 'channel' in prefix:
        box_files = [f for f in files if f.startswith(prefix+'_box0') and f.endswith('.npy')]
    else:
        box_files = [f for f in files if f.startswith(prefix+'_boxnum0') and f.endswith('.npy')]
    timesteps = sorted([(f.split('time')[-1].split('.npy')[0]) for f in box_files])
    return timesteps

def full_filename(prefix, box_number, timestep):
    if 'channel' in prefix:
        return os.path.join(prefix + f'_box{box_number}_time{timestep}.npy')
    else:
        return os.path.join(prefix + f'_boxnum{box_number}_time{timestep}.npy')

def load_box_timeseries(prefix, box_number=1,numpy_dir='numpy_individual_arrays'):
    timesteps = get_timesteps_available(numpy_dir,prefix)
    image_shape = np.load(full_filename(numpy_dir, prefix, box_number, timesteps[0])).shape

    array = np.zeros((len(timesteps),*image_shape))
    for i, timestep in enumerate(timesteps):
        array[i] = np.load(full_filename(numpy_dir, prefix, box_number, timestep))
    return array

def new_experiment(config, subdirectory):
    """
    New experiments can be created from a config file or a config object directly.
    """
    # If config is a string, load the config from the file
    if isinstance(config, str):
        config = load_config(config)

    validate_config(config)


    # Auto populate some useful config entries
    if config.seed == 'random':
        config.seed = np.random.default_rng().integers(0, 2**32 - 1)
    
    if config.model.model_type == 'sgsPointwiseEigenframeMLP':
        config.eigenframe_model = True
        config.pointwise = True
    elif config.model.model_type == 'sgsPointwiseMLP':
        config.eigenframe_model = False
        config.pointwise = True
    else:
        config.eigenframe_model = False
        config.pointwise = False

    unique_name = petname.Generate(separator="_")
    config['barcode'] = f"{datetime.datetime.now().strftime('%y_%m_%d_%H:%M:%S')}_{unique_name}_{config['task_name']}"

    if subdirectory is not None:
        config['directory'] = os.path.join(global_config.experiment_outputs, subdirectory, config['barcode'])
    else:
        config['directory'] = os.path.join(global_config.experiment_outputs, config['barcode'])

    os.makedirs(config['directory'], exist_ok=False)

    setup_logging(os.path.join(config['directory'], f'{config.barcode}_training_log.txt'))

    # Dump config to yaml file
    with open(os.path.join(config['directory'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        
    shutil.copy('src/global_config.py', os.path.join(config['directory'], 'global_config.py'))

    return config

def validate_config(config):
    if config.train_aug_group is not None:
        if config.train_aug_group not in ["oct"]:
            raise ValueError(f"Invalid augmentation group: {config.train_aug_group}")
    if config.val_aug_group is not None:
        if config.val_aug_group not in ["oct"]:
            raise ValueError(f"Invalid augmentation group: {config.val_aug_group}")
    if config.test_aug_group is not None:
        if config.test_aug_group not in ["oct"]:
            raise ValueError(f"Invalid augmentation group: {config.test_aug_group}")
    if config.loss_type not in ['L1', 'L2']:
        raise ValueError(f"Invalid loss type: {config.loss_type}")

    if config.seed == 'random':
        pass
    elif isinstance(config.seed, int):
        pass
    else:
        raise ValueError(f"Invalid seed: {config.seed}")
    
    if config.early_stopping_patience < config.save_interval:
        raise ValueError(f"Hmmmmm, early stopping patience should be greater than save interval: {config.early_stopping_patience} < {config.save_interval}")

def setup_logging(log_file, level=logging.INFO, redirect_print=True):
    """
    Set up logging to both file and console.
    Optionally redirects print statements to logging (enabled by default).
    Returns a configured logger.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Close and remove any existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Redirect print to logging if requested
    if redirect_print:
        original_print = builtins.print
        
        def logging_print(*args, sep=' ', end='\n', file=None, flush=False):
            if file is not None:
                original_print(*args, sep=sep, end=end, file=file, flush=flush)
                return
            message = sep.join(map(str, args))
            logger.info(f"PRINT: {message}")
        
        builtins.print = logging_print
    
    return logger