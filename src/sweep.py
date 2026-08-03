import argparse, itertools, copy, tempfile, yaml
from utils import load_config, new_experiment
from train import train_model

def set_dotted(cfg, dotted, value):
    """Set cfg.a.b.c given 'a.b.c'."""
    parts = dotted.split(".")
    obj = cfg
    for name in parts[:-1]:
        obj = getattr(obj, name)
    setattr(obj, parts[-1], value)

def sweep_model(sweep_config_path, device, subdirectory=None):
    sweep_cfg = load_config(sweep_config_path)
    base_cfg = load_config(sweep_cfg.base_config)

    grid = sweep_cfg.sweep          # mapping: "model.n_layers" -> [2,4,6,8]
    names = list(grid.keys())
    value_lists = [grid[n] for n in names]

    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_cfg)
        for name, val in zip(names, combo):
            set_dotted(cfg, name, val)

        config = new_experiment(cfg, subdirectory)

        train_model(config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subdirectory", type=str, default=None)
    args = parser.parse_args()
    sweep_model(args.sweep_config, args.device, args.subdirectory)
