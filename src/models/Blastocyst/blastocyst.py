from torchvision.io import read_image
from torchvision.transforms.functional import resize
import numpy as np
import subprocess
from pathlib import Path
import time
import glob
import re
from datetime import datetime


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


def run_multiple_simulations_and_save_images(
    params_list, xml_path, run_dir="temp_blastocyst", image_size=[224, 224], seed=None
):
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

    run_dir = Path(run_dir)
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
        with open(params_file, "w") as f:
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
            fig = plt.figure(figsize=(3 * cols, 3 * rows))
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
                if hasattr(img, "numpy"):
                    img_np = img.numpy().astype(np.uint8)
                else:
                    img_np = img.astype(np.uint8)

                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img_np)
                ax.set_title(f"Sim {i+1:03d}", fontsize=10)
                ax.axis("off")

            # Adjust layout and save
            plt.tight_layout()
            grid_plot_path = run_dir / "last_images_grid.png"
            plt.savefig(grid_plot_path, dpi=150, bbox_inches="tight")
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


def compute_metrics_wrt_reference(reference_image_path, run_dir):
    """
    Compute distances between a reference image and simulation images using different metrics.

    Args:
        reference_image_path (str): Path to the reference image
        last_images_dir (str): Directory containing the last images from simulations

    Returns:
        dict: Dictionary with metric names as keys and lists of distances as values
    """
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from transformers import CLIPImageProcessor, CLIPModel
    from src.vision_models.clip_run import make_embedding_clip
    from src.trainers.fwd import frechet_wavelet_distance

    # Load reference image
    ref_img = read_png(reference_image_path)

    # Get all image files
    images_dir = Path(run_dir) / "last_images"
    image_files = [f for f in images_dir.glob("*.png") if f.name != "last_images_grid.png"]

    if not image_files:
        print("No images found in the last_images directory.")
        return {}

    print(f"Computing distances for {len(image_files)} images...")

    # Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Get reference embeddings
    ref_tensor = ref_img.unsqueeze(0).to(device)
    ref_embedding = make_embedding_clip(
        images=ref_tensor,
        model=model,
        processor=processor,
        do_rescale=True,
        device=device,
    )
    ref_embedding = torch.from_numpy(ref_embedding).to(device)

    results = {"clip_cosine": [], "pixel_mae": [], "frechet_wavelet": []}

    for img_path in image_files:
        # Load image
        img = read_png(str(img_path))
        img_tensor = img.unsqueeze(0).to(device)

        # CLIP cosine distance
        img_embedding = make_embedding_clip(
            images=img_tensor,
            model=model,
            processor=processor,
            do_rescale=True,
            device=device,
        )
        img_embedding = torch.from_numpy(img_embedding).to(device)
        cosine_sim = F.cosine_similarity(ref_embedding, img_embedding, dim=1)
        results["clip_cosine"].append(1 - cosine_sim.item())

        # Pixel MAE
        mae = torch.mean(torch.abs(img_tensor.float() - ref_tensor.float())).item()
        results["pixel_mae"].append(mae)

        # Fréchet wavelet distance
        # Remove batch dimension for single image comparison
        img_single = img_tensor.squeeze(0)  # [H, W, C]
        ref_single = ref_tensor.squeeze(0)  # [H, W, C]
        fwd = frechet_wavelet_distance(img_single.unsqueeze(0), ref_single, "haar", 2, True)
        results["frechet_wavelet"].append(fwd.item())

    return results


def print_distance_comparison(results, output_dir=None):
    """Print distance comparison table."""
    if not results:
        return

    # Get metric names from results
    metric_names = list(results.keys())

    # Print header
    header = f"{'METRIC':<15}"
    for i in range(len(results[metric_names[0]])):
        header += f" {'SIM_' + str(i+1):<12}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Print each metric's distances
    for metric in metric_names:
        values = results[metric]
        row = f"{metric:<15}"
        for val in values:
            row += f" {val:<12.4f}"
        print(row)

    # Print summary
    print("=" * len(header))
    for metric, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: mean={mean_val:.4f}, std={std_val:.4f}")
    print("=" * len(header))

    # Save results as text file if output directory provided
    if output_dir is not None:
        output_path = f"{output_dir}/distance_comparison.txt"
        with open(output_path, "w") as f:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for metric in metric_names:
                f.write(f"{metric:<15}")
                for val in results[metric]:
                    f.write(f" {val:<12.4f}")
                f.write("\n")


if __name__ == "__main__":

    params = {
        "vsg1": 1.202,  # Maximum rate of Gata6 synthesis caused by ERK activation
        "vsg2": 1,  # Maximum rate of GATA6 synthesis caused by its auto-activation
        "vsn1": 0.856,  # Basal rate of NANOG synthesis
        "vsn2": 1,  # Maximum rate of NANOG synthesis caused by its auto-activation
        # "vsfr1": 2.8,  # Basal rate of FGFR2
        # "vsfr2": 2.8,  # Maximum rate of FGFR2 synthesis caused by GATA6 activation
        # "vex": 0.0,  # Basal rate of FGF4 synthesis
        # "vsf": 0.6,  # Maximum rate of FGF4 synthesis caused by NANOG activation
        # "va": 20,  # ERK activation rate
        # "vin": 3.3,  # ERK inactivation rate
        # "kdg": 1,  # GATA6 degradation rate
        # "kdn": 1,  # NANOG degradation rate
        # "kdfr": 1,  # FGFR2 degradation rate
        # "kdf": 0.09,  # FGF4 degradation rate
        # "Kag1": 0.28,  # Threshold constant for the activation of GATA6 synthesis by ERK
        # "Kag2": 0.55,  # Threshold constant for GATA6 auto-activation
        # "Kan": 0.55,  # Threshold constant for NANOG auto-activation
        # "Kafr": 0.5,  # Threshold constant for the activation of FGFR2 synthesis by GATA6
        # "Kaf": 5,  # Threshold constant for the activation of FGF4 synthesis by NANOG
        # "Kig": 2,  # Threshold constant for the inhibition of GATA6 synthesis by NANOG
        # "Kin1": 0.28,  # Threshold constant for the inhibition of NANOG synthesis by ERK
        # "Kin2": 2,  # Threshold constant for the inhibition of NANOG synthesis by GATA6
        # "Kifr": 0.5,  # Threshold constant for the inhibition of FGFR2 synthesis by NANOG
        # "Ka": 0.7,  # Michaelis constant for activation of the ERK pathway
        # "Ki": 0.7,  # Michaelis constant for inactivation of the ERK pathway
        # "Kd": 2,  # Michaelis constant for activation of the ERK pathway by FGF4
        # "r": 3,  # Hill coefficient for the activation of GATA6 synthesis by ERK
        # "s": 4,  # Hill coefficient for GATA6 auto-activation
        # "q": 4,  # Hill coefficient for the inhibition of GATA6 synthesis by NANOG
        # "u": 3,  # Hill coefficient for the inhibition of NANOG synthesis by ERK
        # "v": 4,  # Hill coefficient for NANOG auto-activation
        # "w": 4,  # Hill coefficient for the inhibition of NANOG synthesis by GATA6
        # "z": 4,  # Hill coefficient for the activation of FGF4 synthesis by NANOG
        # "k": 0.32,
        # "b": 2,
        # "a": 1,
        # "th": 0.5,
        # "i": 1.5,
        # "d": 0.4,
        # "n": 4,
        # "basal": 0.0,
        # "same": 0.0,
        # "dcm": 0.0,
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
    #     print(
    #         f"{key:<15} {value:<12.3f} {sampled_value:<12.3f} {change_symbol} {change_pct:>6.1f}%"
    #     )
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
    # n_simulations = 2
    # keys_to_sample = ["vsg1", "vsg2", "vsn1", "vsn2", "vsfr1", "vsfr2"]
    # # params_list = [params, params, params, params]
    # params_list = [
    #     sample_params_normally(params, std_ratio=0.0, keys_to_sample=keys_to_sample)
    #     for _ in range(n_simulations)
    # ]
    # print_parameters_comparison(params, params_list)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_dir = Path("temp_blastocyst_n") / f"run_{timestamp}"
    # run_multiple_simulations_and_save_images(
    #     params_list,
    #     "src/models/Blastocyst/Mammalian_Embryo_Development.xml",
    #     run_dir=run_dir,
    #     image_size=[224, 224],
    # )

    # #################################
    # # Sensitivity analysis manually #
    # #################################

    # reference_image = "data/blastocyst_instances/3000/3000_4.png"

    # images_distances = compute_metrics_wrt_reference(reference_image, run_dir)
    # print_distance_comparison(images_distances, run_dir)

    # ######################################
    # # Sensitivity analysis per parameter #
    # ######################################
    import pandas as pd

    reference_image = "data/blastocyst_instances/3000/3000_4.png"
    # reference_images = [
    #     "data/blastocyst_instances/3000/3000_1.png",
    #     "data/blastocyst_instances/3000/3000_2.png",
    #     "data/blastocyst_instances/3000/3000_3.png",
    #     "data/blastocyst_instances/3000/3000_4.png",
    #     "data/blastocyst_instances/3000/3000_5.png",
    # ]
    experiment_base_path = "sensitivity_analysis"
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_base_path = experiment_base_path + "/" + experiment_id

    n_samples = 3
    std_sample = 0.0
    original_params = params.copy()
    metrics_data = []

    for key in original_params.keys():
        print(f"\nAnalyzing sensitivity for parameter: {key}\n")

        # Create all variations for one parameter (keeping others at original values)
        params_list_for_key = []
        sampled_values = np.random.normal(
            loc=original_params[key], scale=std_sample, size=n_samples
        )

        for value in sampled_values:
            current_params = original_params.copy()
            current_params[key] = value
            params_list_for_key.append(current_params)

        # Run all simulations for this parameter in one call
        run_dir = Path(experiment_base_path) / f"param_{key}_{value}"

        run_multiple_simulations_and_save_images(
            params_list_for_key,  # All variations of just this one parameter
            "src/models/Blastocyst/Mammalian_Embryo_Development.xml",
            run_dir=run_dir,
            image_size=[224, 224],
        )

        # Compute metrics for all simulations of this parameter
        images_distances = compute_metrics_wrt_reference(reference_image, run_dir)
        # clip_cosine_distances = []
        # pixel_mae_distances = []
        # frechet_wavelet_distances = []
        # for reference_image in reference_images:
        #     images_distances_ = compute_metrics_wrt_reference(reference_image, run_dir)
        #     clip_cosine_distances.append(np.mean(images_distances_["clip_cosine"]))
        #     pixel_mae_distances.append(np.mean(images_distances_["pixel_mae"]))
        #     frechet_wavelet_distances.append(np.mean(images_distances_["frechet_wavelet"]))

        # images_distances = {
        #     "clip_cosine": clip_cosine_distances,
        #     "pixel_mae": pixel_mae_distances,
        #     "frechet_wavelet": frechet_wavelet_distances,
        # }

        # Map results back to parameter values
        for i, value in enumerate(sampled_values):
            # Extract metrics for simulation i from each metric type
            for metric_name, metric_values in images_distances.items():
                metrics_data.append(
                    {
                        "parameter": key,
                        "value": value,
                        "simulation_id": i,
                        "metric_name": metric_name,
                        "metric_value": metric_values[
                            i
                        ],  # Get the specific value for this simulation
                    }
                )

    metrics_df = pd.DataFrame(metrics_data)
    # Save the dataframe
    metrics_df.to_csv(Path(experiment_base_path) / "metrics_df.csv", index=False)

    # Print sample of data structure for verification
    print("\nSample of metrics data structure:")
    print(metrics_df.head(10))
    print(f"\nTotal rows: {len(metrics_df)}")
    print(f"Parameters analyzed: {metrics_df['parameter'].unique()}")
    print(f"Metric types: {metrics_df['metric_name'].unique()}")
    print(f"Runs per parameter: {metrics_df['simulation_id'].nunique()}")

    # Compute mean and std distance values for each parameter across all metric types
    # First, compute mean and std for each parameter-metric combination
    mean_metrics_per_param = (
        metrics_df.groupby(["parameter", "metric_name"])["metric_value"].mean().reset_index()
    )
    std_metrics_per_param = (
        metrics_df.groupby(["parameter", "metric_name"])["metric_value"].std().reset_index()
    )

    # Create pivot tables for mean and std
    mean_pivot_table = mean_metrics_per_param.pivot(
        index="parameter", columns="metric_name", values="metric_value"
    )
    std_pivot_table = std_metrics_per_param.pivot(
        index="parameter", columns="metric_name", values="metric_value"
    )

    # Show detailed breakdown by metric type - MEAN
    print("\nBreakdown by metric (MEAN):")
    print(mean_pivot_table)

    # Show detailed breakdown by metric type - STD
    print("\nBreakdown by metric (STD):")
    print(std_pivot_table)

    # Save pivot tables to csv
    mean_pivot_table.to_csv(Path(experiment_base_path) / "mean_pivot_table.csv", index=True)
    std_pivot_table.to_csv(Path(experiment_base_path) / "std_pivot_table.csv", index=True)
