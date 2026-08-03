import os
import re
from box import Box
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
# This script amalgamates the runs and saves the results to a csv file
import argparse
import matplotlib.pyplot as plt
import json
import datetime

def find_run_dirs(exp_dir: str):
    subdirs = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    run_dirs = []
    for d in subdirs:
        files = os.listdir(d)
        if any(f.endswith('_eval_results.npy') for f in files):
            run_dirs.append(d)
    return sorted(run_dirs)

def load_config(run_dir):
    setup_path = os.path.join(run_dir, 'config.yaml')
    with open(setup_path, 'r') as f:
        cfg = Box(yaml.safe_load(f))
    return cfg


def get_wall_clock_time(run_dir):
    barcode = os.path.basename(run_dir)
    log_path = os.path.join(run_dir, f"{barcode}_training_log.txt")
    with open(log_path, 'r') as f:
        log = f.read()

    # Find the last "Saving model" line and extract the best epoch number
    save_pattern = r"Saving model from epoch (\d+)"
    save_matches = re.findall(save_pattern, log)
    if not save_matches:
        raise ValueError(f"No 'Saving model' lines found in {log_path}")
    best_epoch = int(save_matches[-1])

    # Find the timestamp for that epoch
    epoch_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch \[\s*" + str(best_epoch) + r"\]"
    epoch_matches = re.findall(epoch_pattern, log, re.MULTILINE)
    if not epoch_matches:
        raise ValueError(f"Could not find epoch {best_epoch} in {log_path}")
    best_epoch_time = datetime.datetime.strptime(epoch_matches[0], "%Y-%m-%d %H:%M:%S")

    # Get the first epoch timestamp
    first_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch \["
    first_match = re.search(first_pattern, log, re.MULTILINE)
    if not first_match:
        raise ValueError(f"No epoch lines found in {log_path}")
    first_epoch_time = datetime.datetime.strptime(first_match.group(1), "%Y-%m-%d %H:%M:%S")

    return (best_epoch_time - first_epoch_time).total_seconds()


def add_wall_clock_times(results: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    def _get_time(barcode):
        run_dir = os.path.join(results_dir, barcode)
        try:
            return get_wall_clock_time(run_dir)
        except Exception as e:
            print(f"Warning: could not get wall clock time for {barcode}: {e}")
            return None

    results['wall_clock_time_s'] = results['barcode'].apply(_get_time)
    return results

def analyze_sweep(results_dir):
    run_dirs = find_run_dirs(results_dir)
    print(f"Found {len(run_dirs)} runs in {results_dir}")

    results_list = []

    for run_dir in tqdm(run_dirs, desc="Processing runs...."):
        run_config = load_config(run_dir)
        run_results = np.load(os.path.join(run_dir, f"{run_config.barcode}_eval_results.npy"), allow_pickle=True).item()
        flattened = pd.json_normalize(run_config, sep='_').to_dict(orient='records')[0]
        for key, value in run_results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            else:
                flattened[key] = value    
        results_list.append(flattened)

    results = pd.DataFrame(results_list)
    csv_path = os.path.join(results_dir, 'sweep_results.csv')
    results = add_wall_clock_times(results, results_dir)

    results.to_csv(csv_path, index=False)

    results = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results['param_count'], results['Re1000_nearwall_val_loss'], 'o-')

    ax.legend()
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss vs Number of Parameters')
    ax.set_xscale('log')
    fig.savefig(os.path.join(results_dir, 'sweep_results.png'))

    # Get model-related columns
    model_cols = [col for col in results.columns if 'model' in col.lower()]

    best_results = results.sort_values('Re1000_nearwall_val_loss').head(1).iloc[0]

    print("\n=== Best Run ===")
    print(f"{best_results['barcode']}")
    print("\nModel columns:")
    for col in model_cols:
        print(f"  {col}: {best_results[col]}")

    print(f"\nVal Loss: {best_results['Re1000_nearwall_val_loss']}")
    print(f"Param Count: {best_results['param_count']}")

    # Get model-related columns
    model_cols = [col for col in results.columns if 'model' in col.lower()]

    best_results = results.sort_values('Re1000_nearwall_val_loss').head(1).iloc[0]

    output = {
        "model_columns": {col: best_results[col] for col in model_cols},
        "val_loss": best_results['Re1000_nearwall_val_loss'],
        "param_count": best_results['param_count'],
        "model_barcode": best_results['barcode'],
    }

    print(output)

    with open(os.path.join(results_dir, 'best_results.json'), 'w') as f:
        json.dump(output, f, indent=4,default=str)
    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    args = parser.parse_args()

    FINAL_RESULTS_DIR = args.results_dir
    analyze_sweep(FINAL_RESULTS_DIR)