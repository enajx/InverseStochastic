from torchvision.io import read_image
from torchvision.transforms.functional import resize
import numpy as np
import subprocess
from pathlib import Path
import time
import glob
import re

def run_morpheus_blastocyst(params, xml_path, outdir, param_keys=None, supress_output=True):
    cmd = ["morpheus", "-f", xml_path, "--outdir", outdir]
    if params is not None:
        if isinstance(params, np.ndarray) and param_keys:
            for k, v in zip(param_keys, params):
                cmd += ["--set", f"{k}={v}"]
        elif isinstance(params, dict):
            for k, v in params.items():
                cmd += ["--set", f"{k}={v}"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL if supress_output else None)


def read_png(path, size=[224, 224], last=False):
    if last:
        path = Path(path)
        files = list(path.glob("plot_*.png"))
        if not files:
            raise FileNotFoundError("No matching PNG files found.")
        path = max(files, key=lambda p: int(p.stem.split("_")[1]))
        time.sleep(0.1)
        if path.stat().st_size == 0:
            raise ValueError(f"File {path} is empty.")
        # sleep for 1 second
        time.sleep(1)
        # print(path)
    img = read_image(path)
    img = resize(img, size)
    img = img.permute(1, 2, 0)  # 224, 224, 3 - uint8
    return img


def sample_params_normally(params, std_ratio, keys_to_sample=None, seed=None):
    """
    Generate a new dictionary with parameter values sampled from normal distributions
    around the original values.
    
    Args:
        params (dict): Original parameter dictionary
        std_ratio (float): Standard deviation as a ratio of the original value
        keys_to_sample (list, optional): List of parameter keys to sample. If None, all keys are sampled.
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        dict: New dictionary with sampled parameter values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # If no keys specified, sample all parameters
    if keys_to_sample is None:
        keys_to_sample = list(params.keys())
    
    sampled_params = {}
    for key, value in params.items():
        if key in keys_to_sample:
            # Calculate standard deviation as a ratio of the original value
            std = abs(value) * std_ratio
            
            # Sample from normal distribution
            new_value = np.random.normal(loc=value, scale=std)
            
            # Ensure non-negative values for biological parameters
            if new_value < 0:
                new_value = 0.0
            
            sampled_params[key] = new_value
        else:
            # Keep original value unchanged
            sampled_params[key] = value
    
    return sampled_params


def run_multiple_simulations_and_save_images(params_list, xml_path, base_outdir="temp_blastocyst", image_size=[224, 224], seed=None):
    """
    Run multiple Morpheus blastocyst simulations and save the last images in a folder.
    
    Args:
        params_list (list): List of parameter dictionaries to simulate
        xml_path (str): Path to the XML configuration file
        base_outdir (str): Base directory for simulation outputs (default: "temp_blastocyst")
        image_size (list): Size to resize images to (default: [224, 224])
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        list: List of paths to saved images
    """
    import os
    import yaml
    from pathlib import Path
    from datetime import datetime
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create unique run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_outdir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory within the run directory
    images_dir = run_dir / "last_images"
    images_dir.mkdir(exist_ok=True)
    
    saved_image_paths = []
    
    for i, params in enumerate(params_list):
        print(f"Running simulation {i+1}/{len(params_list)}...")
        
        # Create unique output directory for this simulation
        sim_outdir = run_dir / f"sim_{i+1:03d}"
        
        # Save parameters as YAML file
        params_file = sim_outdir / "parameters.yaml"
        sim_outdir.mkdir(exist_ok=True)
        with open(params_file, 'w') as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        
        # Run simulation
        tic = time.time()
        run_morpheus_blastocyst(
            params=params,
            xml_path=xml_path,
            outdir=str(sim_outdir),
        )
        toc = time.time()
        print(f"  Simulation took: {toc - tic:.2f} seconds")
        
        # Find and save the last image
        try:
            files = glob.glob(str(sim_outdir / "plot_*.png"))
            if not files:
                print(f"  Warning: No PNG files found in {sim_outdir}")
                continue
                
            last_file = max(files, key=lambda f: int(re.search(r"plot_(\d+)\.png", f).group(1)))
            
            # Read and resize the image
            img = read_image(last_file)
            img = resize(img, image_size)
            img = img.permute(1, 2, 0)  # Convert to HWC format
            
            # Extract original filename and save with simulation index prefix
            original_filename = Path(last_file).name  # e.g., "plot_123.png"
            image_filename = f"sim_{i+1:03d}_{original_filename}"
            image_path = images_dir / image_filename
            
            # Convert to PIL Image and save
            import torch
            from PIL import Image
            
            # Convert torch tensor to PIL Image
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().astype(np.uint8)
            else:
                img_np = img.astype(np.uint8)
            
            pil_img = Image.fromarray(img_np)
            pil_img.save(image_path)
            
            saved_image_paths.append(str(image_path))
            print(f"  Saved image: {image_path}")
            
        except Exception as e:
            print(f"  Error processing simulation {i+1}: {e}")
            continue
    
    print(f"\nCompleted {len(saved_image_paths)} simulations.")
    print(f"Images saved in: {images_dir}")
    
    # Create grid plot of all last images
    if saved_image_paths:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            # Calculate grid dimensions
            n_images = len(saved_image_paths)
            cols = min(5, n_images)  # Max 5 columns
            rows = (n_images + cols - 1) // cols  # Ceiling division
            
            # Create figure and grid
            fig = plt.figure(figsize=(3*cols, 3*rows))
            gs = gridspec.GridSpec(rows, cols, figure=fig)
            
            # Load and plot each image
            for i, image_path in enumerate(saved_image_paths):
                row = i // cols
                col = i % cols
                
                # Load image
                img = read_image(image_path)
                img = resize(img, image_size)
                img = img.permute(1, 2, 0)  # Convert to HWC format
                
                # Convert to numpy for matplotlib
                if hasattr(img, 'numpy'):
                    img_np = img.numpy().astype(np.uint8)
                else:
                    img_np = img.astype(np.uint8)
                
                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img_np)
                ax.set_title(f'Sim {i+1:03d}', fontsize=10)
                ax.axis('off')
            
            # Adjust layout and save
            plt.tight_layout()
            grid_plot_path = run_dir / "last_images_grid.png"
            plt.savefig(grid_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Grid plot saved: {grid_plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not create grid plot: {e}")
    
    return saved_image_paths


def print_parameters_comparison(original_params, params_list, precision=3):
    """
    Print a comparison table of original parameters vs multiple parameter sets.
    
    Args:
        original_params (dict): Original parameter dictionary
        params_list (list): List of parameter dictionaries to compare
        precision (int): Number of decimal places to display (default: 3)
    """
    if not params_list:
        print("No parameters to compare.")
        return
    
    # Get all parameter keys
    all_keys = list(original_params.keys())
    
    # Calculate column widths
    param_width = max(len(key) for key in all_keys) + 2
    value_width = precision + 8  # Account for decimal places and sign
    
    # Print header
    header = f"{'PARAMETER':<{param_width}} {'ORIGINAL':<{value_width}}"
    for i in range(len(params_list)):
        header += f" {'SET_' + str(i+1):<{value_width}}"
    
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Print each parameter
    for key in all_keys:
        original_val = original_params[key]
        row = f"{key:<{param_width}} {original_val:<{value_width}.{precision}f}"
        
        for params in params_list:
            val = params[key]
            row += f" {val:<{value_width}.{precision}f}"
        
        print(row)
    
    print("=" * len(header))
    print()


if __name__ == "__main__":

    params = {
        "vsg1": 1.202,  # Maximum rate of Gata6 synthesis caused by ERK activation
        "vsg2": 1,  # Maximum rate of GATA6 synthesis caused by its auto-activation
        "vsn1": 0.856,  # Basal rate of NANOG synthesis
        "vsn2": 1,  # Maximum rate of NANOG synthesis caused by its auto-activation
        "vsfr1": 2.8,  # Basal rate of FGFR2
        "vsfr2": 2.8,  # Maximum rate of FGFR2 synthesis caused by GATA6 activation
        "vex": 0.0,  # Basal rate of FGF4 synthesis
        "vsf": 0.6,  # Maximum rate of FGF4 synthesis caused by NANOG activation
        "va": 20,  # ERK activation rate
        "vin": 3.3,  # ERK inactivation rate
        "kdg": 1,  # GATA6 degradation rate
        "kdn": 1,  # NANOG degradation rate
        "kdfr": 1,  # FGFR2 degradation rate
        "kdf": 0.09,  # FGF4 degradation rate
        "Kag1": 0.28,  # Threshold constant for the activation of GATA6 synthesis by ERK
        "Kag2": 0.55,  # Threshold constant for GATA6 auto-activation
        "Kan": 0.55,  # Threshold constant for NANOG auto-activation
        "Kafr": 0.5,  # Threshold constant for the activation of FGFR2 synthesis by GATA6
        "Kaf": 5,  # Threshold constant for the activation of FGF4 synthesis by NANOG
        "Kig": 2,  # Threshold constant for the inhibition of GATA6 synthesis by NANOG
        "Kin1": 0.28,  # Threshold constant for the inhibition of NANOG synthesis by ERK
        "Kin2": 2,  # Threshold constant for the inhibition of NANOG synthesis by GATA6
        "Kifr": 0.5,  # Threshold constant for the inhibition of FGFR2 synthesis by NANOG
        "Ka": 0.7,  # Michaelis constant for activation of the ERK pathway
        "Ki": 0.7,  # Michaelis constant for inactivation of the ERK pathway
        "Kd": 2,  # Michaelis constant for activation of the ERK pathway by FGF4
        "r": 3,  # Hill coefficient for the activation of GATA6 synthesis by ERK
        "s": 4,  # Hill coefficient for GATA6 auto-activation
        "q": 4,  # Hill coefficient for the inhibition of GATA6 synthesis by NANOG
        "u": 3,  # Hill coefficient for the inhibition of NANOG synthesis by ERK
        "v": 4,  # Hill coefficient for NANOG auto-activation
        "w": 4,  # Hill coefficient for the inhibition of NANOG synthesis by GATA6
        "z": 4,  # Hill coefficient for the activation of FGF4 synthesis by NANOG
        "k": 0.32,
        "b": 2,
        "a": 1,
        "th": 0.5,
        "i": 1.5,
        "d": 0.4,
        "n": 4,
        "basal": 0.0,
        "same": 0.0,
        "dcm": 0.0,
    }

    ###################### 
    # Single simulation  #
    ######################

    # # sampled_params = sample_params_normally(params, std_ratio=0.1)
    # sampled_params = params
    
    # print("=" * 60)
    # print(f"{'PARAMETER':<15} {'ORIGINAL':<12} {'SAMPLED':<12} {'CHANGE %':<10}")
    # print("=" * 60)
    # for key, value in params.items():
    #     sampled_value = sampled_params[key]
    #     change_pct = ((sampled_value - value) / value * 100) if value != 0 else 0
    #     change_symbol = "↗" if change_pct > 0 else "↘" if change_pct < 0 else "="
    #     print(f"{key:<15} {value:<12.3f} {sampled_value:<12.3f} {change_symbol} {change_pct:>6.1f}%")
    # print("=" * 60)
    # print()


    # tic = time.time()
    # run_morpheus_blastocyst(
    #     params=params,
    #     xml_path="src/models/Blastocyst/Mammalian_Embryo_Development.xml",
    #     outdir="temp_blastocyst",
    # )
    # toc = time.time()
    # print(f"Simulation took: {toc - tic} seconds")


    # files = glob.glob("temp_blastocyst/plot_*.png")
    # last_file = max(files, key=lambda f: int(re.search(r"plot_(\d+)\.png", f).group(1)))
    # img = read_image(last_file)
    # img = resize(img, [224, 224])
    # img = img.permute(1, 2, 0)  # 224, 224, 3 - uint8
    # print(img.shape)
    # print(img.dtype)

    # import matplotlib.pyplot as plt

    # plt.imshow(img)
    # plt.show()



    # ###################### 
    # # Multiple simulations #
    # ######################
    n_simulations = 8
    keys_to_sample = ["vsg1", "vsg2", "vsn1", "vsn2", "vsfr1", "vsfr2"]
    # params_list = [params, params, params, params]
    params_list = [sample_params_normally(params, std_ratio=0.5, keys_to_sample=keys_to_sample) for _ in range(n_simulations)]
    print_parameters_comparison(params, params_list)

    run_multiple_simulations_and_save_images(params_list, "src/models/Blastocyst/Mammalian_Embryo_Development.xml", base_outdir="temp_blastocyst_n", image_size=[224, 224])