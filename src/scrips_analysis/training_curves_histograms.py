import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import os
import random
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import cm
from PIL import Image, ImageChops

########## Data Loading and Processing ##########
def exp_summary(exp_path):
    rows = []
    for root, dirs, files in os.walk(exp_path):
        if 'config.yml' in files:
            config_path = os.path.join(root, 'config.yml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                data = {**config, 'run_folder': root}  
                rows.append(data)

    df = pd.DataFrame(rows)
    columns_to_remove = ['_wandb', 'disable_cma_bounds', 'entity', 'experiment_folder_path',
    'group', 'initial_state_seed_radius',
       'initial_state_seed_type', 'log_freq', 'lower_bounds',
       'notes experiment', 'output_grid_size', 'pad_mode',
       'popsize', 'project_name', 'run_name', 'saving_path', 'search_space',
       'seed', 'sigma_init', 'target_image_path',
        'update_steps', 'upper_bounds', 'wandb_mode',]
    
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return df

def gather_data_npy(df, ath=True):
    system_name = df['system_name'].unique()
    if len(system_name) != 1:
        raise ValueError("More than one system name found in the dataframe.")
    system_name = system_name[0]

    use_key_name = 'key_name' in df.columns and df['key_name'].notna().any()

    data = defaultdict(dict)

    for target_space in df['target_space'].unique():
        space_df = df[df['target_space'] == target_space]

        if use_key_name:
            group_values = space_df['key_name'].unique()
        else:
            if system_name == 'schelling':
                param_series = space_df['want_similar'].dropna().apply(np.array)
            elif system_name == 'gray_scott':
                param_series = space_df['params_gray_scott'].dropna().apply(np.array)
            else:
                raise ValueError(f"Unknown system name: {system_name}")

            group_values = []
            for arr in param_series:
                if not any(np.allclose(arr, existing) for existing in group_values):
                    group_values.append(arr)
            group_values = np.array(group_values, dtype=object)  # preserve tuples

        for group_value in group_values:
            if use_key_name:
                group_rows = space_df[space_df['key_name'] == group_value]
                group_key = group_value
            else:
                sel = space_df.apply(
                    lambda r: np.allclose(r['want_similar'] if system_name == 'schelling' else r['params_gray_scott'],
                                          group_value),
                    axis=1
                )
                group_rows = space_df[sel]
                if np.ndim(group_value) == 0:
                    group_key = (float(group_value),)
                else:
                    group_key = tuple(group_value)

            if group_rows.empty:
                continue

            data[target_space][group_key] = {}

            for _, row in group_rows.iterrows():
                run_path = row['run_folder']
                try:
                    loss_path = f'{run_path}/gen_losses.npy'
                    sol_path  = f'{run_path}/gen_solutions.npy'
                    
                    losses = np.load(loss_path).clip(0, 1)
                    results = np.load(sol_path)
                    if system_name == 'schelling':
                        results = results.clip(0, 1)

                    if ath:
                        min_curve = np.minimum.accumulate(losses)
                        idx = np.argmax(losses[:, None] == min_curve[None, :], axis=0)
                        losses = min_curve
                        results = results[idx]

                    data[target_space][group_key][run_path] = {
                        'losses': losses,
                        'results': results
                    }

                except Exception as e:
                    print(f'Error loading data from {run_path}: {e}')

    # export a lightweight JSON copy
    json_ready = {}
    for tspace, groups in data.items():
        json_ready[tspace] = {}
        for gkey, runs in groups.items():
            gkey_str = str(gkey)
            json_ready[tspace][gkey_str] = {
                rp: {'losses': v['losses'].tolist(),
                     'results': v['results'].tolist()}
                for rp, v in runs.items()
            }

    #with open(f'{system_name}_data.json', 'w') as f:
    #    json.dump(json_ready, f, indent=4)

    return data

########## Plotters ##########
def plot_histogram(data, system_name):
    CMAP = plt.colormaps['Dark2']
    all_mse = []

    for param_group in data['embedding'].values():
        for run_path, run in param_group.items():
            if 'config.yml' not in os.listdir(run_path):
                continue

            with open(os.path.join(run_path, 'config.yml'), 'r') as f:
                run_config = yaml.safe_load(f)
                if system_name == 'schelling':
                    target_params = np.array([run_config.get('want_similar', 0)])
                elif system_name == 'gray_scott':
                    target_params = np.array(run_config.get('params_gray_scott', []))
                else:
                    continue

            results = np.array(run['results'])
            last_gen = results[-1]
            mse = np.mean((last_gen - target_params) ** 2)
            all_mse.append(mse)

    all_mse = np.array(all_mse)
    bins = np.histogram_bin_edges(all_mse, bins=10)
    #bins = np.histogram_bin_edges(all_mse, bins=50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_counts = np.zeros(len(bin_centers))
    bin_stds = np.zeros(len(bin_centers))

    for i in range(len(bins) - 1):
        in_bin = (all_mse >= bins[i]) & (all_mse < bins[i + 1])
        bin_vals = all_mse[in_bin]
        bin_counts[i] = len(bin_vals) / len(all_mse)  # relative frequency
        bin_stds[i] = np.std(bin_vals) if len(bin_vals) > 1 else 0.0

    plt.figure(figsize=(5, 3))
    # cornflowerblue royalblue
    plt.bar(bin_centers, bin_counts, width=np.diff(bins), color='darkgrey', alpha=1,
            edgecolor='white', linewidth=0.6, align='center', label='Histogram')
    
    # HERE ERROR BARS
    plt.errorbar(bin_centers, bin_counts, yerr=bin_stds, fmt='none', ecolor='black', capsize=3, lw=1, alpha=0.8,)

    # remove frame up and right
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['right'].set_visible(False)

    #ower limit to 0
    plt.ylim(0, 1.1 * max(bin_counts))

    #plt.xlabel('MSE causal-predicted parameters $\\theta$', fontsize=16)
    plt.xlabel('MSE causal parameters $(\\theta, \\theta\'$)', fontsize=16)
    #plt.ylabel('Relative Frequency', fontsize=16)
    plt.ylabel('Proportion of Runs', fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = 'visuals_paper/optimization_results'
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f'{output_path}/{system_name}_histogram.png', bbox_inches='tight', dpi=100)
    plt.close()

def plot_loss_gen(data, system_name, independent_groupping=True):
    all_losses = []

    for group_name, param_group in data['embedding'].items():
        for run_path, run in param_group.items():
            if 'config.yml' in os.listdir(run_path):
                with open(os.path.join(run_path, 'config.yml'), 'r') as f:
                    run_config = yaml.safe_load(f)
                    generations = run_config['generations']

                if independent_groupping:
                    if 'key_name' in run_config:          # <-- add this line
                        group_label = run_config['key_name']
                    elif system_name == 'schelling':
                        group_label = run_config.get('want_similar')
                    elif system_name == 'gray_scott':
                        group_label = tuple(run_config.get('params_gray_scott', []))
                    else:
                        group_label = 'unknown'

            losses = np.array(run['losses'])[-generations:]
            all_losses.append((group_label, losses))

    # Group losses by label
    grouped = {}
    for i in range(len(all_losses)):
        group_label, losses = all_losses[i]
        grouped.setdefault(group_label, []).append(losses)

    cmap = plt.colormaps['magma']
    plt.figure(figsize=(5, 3))

    #sort by group_label
    sorted_labels = sorted(grouped.keys(), key=lambda x: (isinstance(x, tuple), x))
    grouped = {label: grouped[label] for label in sorted_labels}
    
    all_losses = []

    for i, (group_label, losses_list) in enumerate(grouped.items()):
        lengths = [len(l) for l in losses_list]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent loss lengths in group '{group_label}': {set(lengths)}")

        losses_array = np.stack(losses_list)
        all_losses.append(losses_array)
        mean = losses_array.mean(axis=0)
        std = losses_array.std(axis=0)
        gens = np.arange(losses_array.shape[1])

        color = cmap(i / max(len(grouped) - 1, 1))

        if system_name == 'schelling':
            label = f"$y = {float(group_label):.1f}$"
        elif system_name == 'gray_scott':
            label = f'$\\mathcal{{y}}_{{{i}}}$'
        else:
            label = f'WRONG SYSTEM {system_name}'

        plt.plot(gens, mean, color=color, label= label)
        plt.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)

    if False: #plota mean of all losses in black and dashed
        all_losses = np.concatenate(all_losses, axis=0)
        mean = all_losses.mean(axis=0)
        std = all_losses.std(axis=0)
        gens = np.arange(all_losses.shape[1])
        plt.plot(gens, mean, color='black', linestyle='--', label='Mean of all runs')
        plt.fill_between(gens, mean - std, mean + std, color='black', alpha=0.2)

    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda x, _: f'{x:.2f}')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    #plt.gca().xaxis.set_major_formatter(fmt)
    plt.gca().yaxis.set_major_formatter(fmt)
    plt.xlabel('Generation', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Cosine Loss', fontsize=16)
    #plt.title('Mean Loss over Generations')
    
    #plt.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()

    output_path = f'visuals_paper/optimization_results'
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f'{system_name}_losses.png'), bbox_inches='tight', dpi=100)
    plt.close()

########## Wrapper ##########
def plot_loss_generation_histograms(exp_paths):
    dfs = []
    
    for exp_path in (exp_paths):
        df = exp_summary(exp_path)
        unique_system_name = df['system_name'].unique()
        
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    
    unique_system_name = df['system_name'].unique()
    
    if len(unique_system_name) > 1:
        raise ValueError("More than one system name found in the dataframe.")

    system_name = unique_system_name[0]
    data = gather_data_npy(df)    
    
    plot_loss_gen(data, system_name, independent_groupping=True)
    plot_histogram(data, system_name)
    pass


if __name__ == "__main__":
    
    plot_loss_generation_histograms(exp_paths = ['experiments_results/RD/RD-2025-05-16_16-57-14'])
    plot_loss_generation_histograms(exp_paths = ['experiments_results/SH/SH-2025-05-16_17-43-32'])
    
    pass