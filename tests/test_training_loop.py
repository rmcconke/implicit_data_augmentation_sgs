import pytest
import os
import sys
sys.path.append("./src/")
from utils import load_config, new_experiment
from train import train_model

TEST_CONFIGS_DIR = 'tests/test_configs'

TEST_CONFIG_FILES = [
    "SGS_nearwall_cnn_aug.yaml",
    "SGS_nearwall_escnn.yaml",
    "SGS_nearwall_pointwiseEigenframe.yaml",
    "SGS_nearwall_pointwiseNN_aug.yaml",
]

@pytest.fixture(params=TEST_CONFIG_FILES, ids=lambda p: p)
def config_path(request):
    return os.path.join(TEST_CONFIGS_DIR, request.param)

def test_training_loop(config_path, device="cuda:0"):
    """Test that training completes without error for each config."""
    cfg = load_config(config_path)
    config = new_experiment(cfg, subdirectory="test_runs")
    train_model(config, device)