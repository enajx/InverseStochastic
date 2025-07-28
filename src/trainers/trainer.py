import os
import sys

import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import shutil
from pathlib import Path
import multiprocessing
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib import pyplot as plt
from transformers import AutoModel, CLIPImageProcessor, CLIPModel
import pathlib
import wandb
import time

from models.RD.RDBatch4Params import RD_GPU
from models.Schelling.schelling import run_schelling
from models.Blastocyst.blastocyst import run_morpheus_blastocyst, read_png

from trainers.cmaes import train

from utils.utils import load_image_as_tensor
from vision_models.clip_run import make_embedding_clip
from visualisations.visual_tools import make_image

from trainers.domain_losses import *


def _schelling_worker(want_similar, config):
    params = {
        "want_similar": want_similar,
        "n_groups": 2,
        "density": 0.9,
        "size": config["output_grid_size"][0],
        "max_run_duration": 100,
    }
    return run_schelling(params)


def run_model_with_population_parameters(population_parameters, config):
    population_parameters = np.array(population_parameters)
    if config["system_name"] == "gray_scott":
        rd = RD_GPU(
            param_batch=population_parameters,
            grid_size=config["output_grid_size"],
            seed=config["initial_state_seed_type"],
            seed_radius=config["initial_state_seed_radius"],
            pad_mode=config["pad_mode"],
            anisotropic=config["anisotropic"],
        )
        last_frame, _ = rd.run(
            config["update_steps"],
            save_all=False,
            show_progress=False,
            minimax_RD_output=config["minmax_RD_output"],
        )
        return last_frame

    elif config["system_name"] == "schelling":
        population_parameters = population_parameters.flatten()
        worker = partial(_schelling_worker, config=config)
        num_cores = psutil.cpu_count(logical=False)
        with multiprocessing.Pool(num_cores) as pool:
            results = pool.map(worker, population_parameters)

        assert isinstance(results[0], torch.Tensor)
        results = torch.stack(results)  # Stack tensors along a new batch dimension
        return results

    elif config["system_name"] == "blastocyst":
        target_images = []
        for i, param in enumerate(population_parameters):
            run_morpheus_blastocyst(
                params=param,
                xml_path="src/models/Blastocyst/Mammalian_Embryo_Development.xml",
                outdir=f"{config["run_folder_path"]}/temp/current_population/{i}",
                param_keys=list(config["params_blastocyst"].keys()),
            )
            target_image = read_png(
                f"{config['run_folder_path']}/temp/current_population/{i}", last=True
            )
            target_images.append(target_image)

        shutil.rmtree(Path(f"{config['run_folder_path']}/temp/current_population"))

        return torch.stack(target_images)


def compute_fitness(
    output_batch: np.array,
    target: np.array,
    config: dict,
    processor: Callable,
    model: Callable,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["target_space"] == "embedding":  # embedding space
        if config["visual_embedding"] == "clip":
            embeddings = make_embedding_clip(
                output_batch.float(),
                do_rescale=config["do_rescale"],
                processor=processor,
                model=model,
                device=device,
            )
        else:
            raise ValueError("visual_embedding must be clip")

        cosine_similarity = np.dot(embeddings, target)
        loss = 1 - cosine_similarity

    elif config["target_space"] == "pixel":
        assert target.shape[-1] == 3
        # loss = torch.mean((output_batch - target) ** 2, axis=(1, 2, 3))
        loss = torch.mean((output_batch.float() - target.float()) ** 2, axis=(1, 2, 3))
        loss = torch.sqrt(loss)

    # RD-specific losses
    elif config["target_space"] == "spectral_entropy":  # Spectral entropy loss
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)
        out_spec = compute_radial_power_spectrum(out_gray)
        tgt_spec = compute_radial_power_spectrum(tgt_gray)
        loss = torch.abs(spectral_entropy(out_spec) - spectral_entropy(tgt_spec))

    elif config["target_space"] == "dominant_wavelength":  # Dominant wavelength loss
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)

        out_spec = compute_radial_power_spectrum(out_gray)
        tgt_spec = compute_radial_power_spectrum(tgt_gray)
        loss = torch.abs(dominant_wavelength(out_spec) - dominant_wavelength(tgt_spec))

    elif config["target_space"] == "spectral_radial":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)

        out_spec = compute_radial_power_spectrum(out_gray)
        tgt_spec = compute_radial_power_spectrum(tgt_gray)
        loss = torch.sqrt(torch.mean((out_spec - tgt_spec) ** 2, dim=1))

    elif config["target_space"] == "edge_density":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)
        loss = torch.abs(edge_density(out_gray) - edge_density(tgt_gray))

    elif config["target_space"] == "local_contrast":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)
        loss = torch.abs(local_contrast(out_gray) - local_contrast(tgt_gray))

    elif config["target_space"] == "frequency_band_energy":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(output_batch.shape[0], -1, -1)
        out_spec = compute_radial_power_spectrum(out_gray)
        tgt_spec = compute_radial_power_spectrum(tgt_gray)
        loss = torch.abs(band_energy(out_spec) - band_energy(tgt_spec))

    elif config["target_space"] == "skeleton_difference":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(out_gray.shape[0], -1, -1)
        loss = skeleton_difference(out_gray, tgt_gray)

    elif config["target_space"] == "distance_transform":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(out_gray.shape[0], -1, -1)
        loss = distance_transform_loss(out_gray, tgt_gray)

    elif config["target_space"] == "orientation_variance":
        out_gray = to_grayscale(output_batch)
        tgt_gray = to_grayscale(target).unsqueeze(0).expand(out_gray.shape[0], -1, -1)
        loss = torch.abs(
            local_orientation_variance(out_gray) - local_orientation_variance(tgt_gray)
        )

    elif config["target_space"] == "fwd":

        from fwd import frechet_wavelet_distance

        # output_batch: [B, H, W, C], target: [H, W, C]
        # broadcast target across the batch dimension
        # loss shape: [B]

        wave = config["fwd_wave"]
        level = config["fwd_level"]
        log = config["fwd_log"]
        loss = frechet_wavelet_distance(output_batch, target, wave, level, log)

    # Schelling-specific losses
    elif config["target_space"] == "dissimilarity_index":
        out_lbl = rgb_to_label(output_batch)
        tgt_lbl = rgb_to_label(target.unsqueeze(0).expand(output_batch.shape[0], -1, -1, -1))
        loss = torch.abs(
            compute_dissimilarity_index(out_lbl) - compute_dissimilarity_index(tgt_lbl)
        )

    elif config["target_space"] == "boundary_length":
        out_lbl = rgb_to_label(output_batch)
        tgt_lbl = rgb_to_label(target.unsqueeze(0).expand(output_batch.shape[0], -1, -1, -1))
        loss = torch.abs(compute_boundary_length(out_lbl) - compute_boundary_length(tgt_lbl))

    elif config["target_space"] == "average_cluster_size":
        out_lbl = rgb_to_label(output_batch)
        tgt_lbl = rgb_to_label(target.unsqueeze(0).expand(output_batch.shape[0], -1, -1, -1))
        loss = torch.abs(
            compute_average_cluster_size(out_lbl) - compute_average_cluster_size(tgt_lbl)
        )

    elif config["target_space"] == "moran_I":
        out_lbl = rgb_to_label(output_batch)
        tgt_lbl = rgb_to_label(target.unsqueeze(0).expand(output_batch.shape[0], -1, -1, -1))
        loss = torch.abs(compute_moran_I(out_lbl) - compute_moran_I(tgt_lbl))

    else:
        raise ValueError("fitness must be either pixel or embedding")
    return loss


def _make_target(config):
    # Target image provided
    if config["target_image_path"] is not None:
        if config["target_image_path"].endswith(".npy"):
            raise ValueError("npy files not supported yet, need to return H,W,3 in unit8 0-255")

        # Target image as png
        elif config["target_image_path"].endswith(".png"):
            if (
                config["system_name"] == "reaction_diffusion"
                or config["system_name"] == "schelling"
            ):
                target_image = load_image_as_tensor(
                    config["target_image_path"],
                    resize=None,
                    reverse_image=False,
                    minmax_target_image=False,
                    color=False,
                    swap_channels=True,
                    negative=config["negative"],
                )
            elif config["system_name"] == "blastocyst":
                target_image = read_png(config["target_image_path"])
            else:
                raise ValueError("System name not recognized")
            assert target_image.shape[-1] == 3, "target image must have 3 channels"

    # Make target image
    else:
        if config["system_name"] == "gray_scott":
            population_params = config["params_gray_scott"]
        elif config["system_name"] == "schelling":
            population_params = config["want_similar"]
        elif config["system_name"] == "blastocyst":
            population_params = config["params_blastocyst"]
        else:
            raise ValueError("System name not recognized")

        if config["system_name"] == "gray_scott" or config["system_name"] == "schelling":
            target_image = run_model_with_population_parameters([population_params], config)
            target_image = (target_image).to(torch.uint8)
            target_image = target_image[0]
        elif config["system_name"] == "blastocyst":
            run_morpheus_blastocyst(
                params=population_params,
                xml_path="src/models/Blastocyst/Mammalian_Embryo_Development.xml",
                outdir=f"{config['run_folder_path']}/temp/target_pattern",
            )
            target_image = read_png(f"{config['run_folder_path']}/temp/target_pattern", last=True)
        else:
            raise ValueError("System name not recognized")

    # Save target image
    pathlib.Path(config["run_folder_path"]).mkdir(parents=True, exist_ok=True)
    make_image(
        target_image,
        filename="target",
        folder_path=config["run_folder_path"],
        system_name=config["system_name"],
    )

    if config["target_space"] == "embedding":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_image = target_image.to(device)

        CLIP_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        if config["visual_embedding"] == "clip":
            CLIP_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )
            CLIP_model = CLIP_model.to(device)
            target_image = make_embedding_clip(
                images=target_image,
                do_rescale=config["do_rescale"],
                processor=CLIP_processor,
                model=CLIP_model,
                device=device,
            )

        else:
            raise ValueError("visual_embedding must be clip")

        target_image = target_image[0]

    return target_image


def save_experiment_data(
    config,
    best_solution,
    best_loss,
    gen_solutions,
    gen_losses,
):
    np.save(f'{config["run_folder_path"]}/best_solution.npy', np.array(best_solution))
    np.save(f'{config["run_folder_path"]}/best_loss.npy', np.array(best_loss))

    np.save(f'{config["run_folder_path"]}/gen_solutions.npy', np.array(gen_solutions))
    np.save(f'{config["run_folder_path"]}/gen_losses.npy', np.array(gen_losses))

    wandb.save(f'{config["run_folder_path"]}/best_solution.npy')
    wandb.save(f'{config["run_folder_path"]}/best_loss.npy')


def plots_and_logs(
    config,
    best_solution,
    best_loss,
    gen_solutions,
    gen_losses,
    best_last_gen_sol,
    best_last_gen_loss,
):

    system_name = config["system_name"]
    run_folder = config["run_folder_path"]
    last_frame = run_model_with_population_parameters([best_solution], config)
    make_image(
        last_frame.squeeze(),
        "best_historical_solution",
        folder_path=run_folder,
        system_name=system_name,
    )

    last_frame = run_model_with_population_parameters([best_last_gen_sol], config)
    make_image(
        last_frame.squeeze(),
        "best_lastgen_solution",
        folder_path=run_folder,
        system_name=system_name,
    )

    # Log image(s)
    target_image = plt.imread(f"{run_folder}/target.png")
    best_historical_solution = plt.imread(f"{run_folder}/best_historical_solution.png")
    best_lastgen_solution = plt.imread(f"{run_folder}/best_lastgen_solution.png")

    log_dict = {
        "Target": [wandb.Image(target_image, caption="target")],
        "Result Hist": [wandb.Image(best_historical_solution, caption=config["target_space"])],
        "Result Last": [wandb.Image(best_lastgen_solution, caption=config["target_space"])],
        "Video": [],
        "Grid": [],
    }

    video_path = f"{run_folder}/best_solution.mp4"
    if os.path.isfile(video_path):
        log_dict["Video"].append(wandb.Video(video_path, caption=config["target_space"]))

    grid_path = f"{run_folder}/unique_solutions_grid.png"
    if os.path.isfile(grid_path):
        log_dict["Grid"].append(wandb.Image(grid_path, caption=f"Grid of target"))

    wandb.log(log_dict)


def optimize_parameters_cmaes(config):

    # Generate target
    target = _make_target(config)

    # Train
    (
        best_historical_solution,
        best_historical_loss,
        gen_solutions,
        gen_losses,
        best_last_gen_sol,
        best_last_gen_loss,
        # ath_solutions,
        # ath_losses,
    ) = train(config, target, run_model_with_population_parameters)

    # Save results
    save_experiment_data(
        config,
        best_historical_solution,
        best_historical_loss,
        gen_solutions,
        gen_losses,
        # best_last_gen_sol,
        # best_last_gen_loss,
    )

    # Evaluate best solution and save output
    plots_and_logs(
        config,
        best_historical_solution,
        best_historical_loss,
        gen_solutions,
        gen_losses,
        best_last_gen_sol,
        best_last_gen_loss,
    )


if __name__ == "__main__":
    import wandb

    wandb.init(mode="disabled")

    config_blastocyst = {
        "system_name": "blastocyst",
        "search_space": "parameters",
        "params_blastocyst": {
            "vsg1": 1.202,
            "vsg2": 1,
            "vsn1": 0.856,
            "vsn2": 1,
            "vsfr1": 2.8,
            "vsfr2": 2.8,
        },
        "target_image_path": "test_blastocyst/plot_03000.png",
        # "target_image_path": None, #!
        "lower_bounds": [0, 0, 0, 0, 0, 0],
        "upper_bounds": [2, 2, 2, 2, 2, 2],
    }

    config_RD = {
        "system_name": "gray_scott",
        "search_space": "parameters",
        # "params_gray_scott": [1.0, 0.343817, 0.052043, 0.063162],
        # "initial_state_seed_type": "random",
        # "params_gray_scott": [1.0, 0.1, 0.028, 0.062],
        # "initial_state_seed_type": "random",
        "params_gray_scott": [0.8, 0.25, 0.03, 0.065],
        "initial_state_seed_type": "random",

        "output_grid_size": [100, 100],
        "initial_state_seed_radius": 10,
        "target_image_path": None,
        "lower_bounds": [0, 0, 0, 0],
        "upper_bounds": [1, 1, 0.1, 0.1],
        "pad_mode": "circular",
    }

    config_schelling = {
        "system_name": "schelling",
        "search_space": "parameters",
        "want_similar": 0.7,
        "output_grid_size": [100, 100],
        "initial_state_seed_type": None,
        "initial_state_seed_radius": None,
        "lower_bounds": [0, 0, 0, 0],
        "target_image_path": None,
        "lower_bounds": [0],
        "upper_bounds": [1],
        "pad_mode": "zeros",
    }

    config_shared = {
        # "target_space": "pixel",
        # "target_space": "embedding",
        # RD-specific losses
        # "target_space": "spectral_entropy",
        # "target_space": "dominant_wavelength",
        # "target_space": "spectral_radial",
        # "target_space": "edge_density",
        # "target_space": "local_contrast",
        # "target_space": "frequency_band_energy",
        # "target_space": "skeleton_difference",
        # "target_space": "distance_transform",
        # "target_space": "orientation_variance",
        "target_space": "fwd",
        "fwd_wave": "haar",
        "fwd_level": 2,
        "fwd_log": True,
        # Schelling-specific losses
        # "target_space": "dissimilarity_index",  # works well with schelling and fails on RD
        # "target_space": "boundary_length",
        # "target_space": "average_cluster_size",
        # "target_space": "moran_I",
        #
        "update_steps": 1000,
        "visual_embedding": "clip",
        "popsize": 64,
        "generations": 32,
        "sigma_init": 0.25,
        "anisotropic": False,
        "minmax_RD_output": False,
        "minmax_target_image": False,
        "early_stop": None,  # [15, 0.04],
        "disable_cma_bounds": False,
        "negative": False,
        "do_rescale": True,  # needed
    }


    config = {**config_shared, **config_RD}
    # config = {**config_shared, **config_schelling}
    # config = {**config_shared, **config_blastocyst}

    config["run_folder_path"] = f"tests/test_{config['system_name']}_{config['target_space']}"

    # use pathlib to create experiment folder
    pathlib.Path(config["run_folder_path"]).mkdir(parents=True, exist_ok=True)
    # save config as YAML
    with open(f"{config['run_folder_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    optimize_parameters_cmaes(config)

    print(f"\nResults saved in {config['run_folder_path']}\n")
