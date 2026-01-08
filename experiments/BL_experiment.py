import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import json
import time

from experiments.utils_experiment import run_cmaes
from transformers import CLIPModel, CLIPImageProcessor, AutoModel, AutoImageProcessor

# Download CLIP model and processor
CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Download NOMIC model and processor
AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", use_fast=True)
AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

print("Models downloaded and cached successfully!")

if __name__ == "__main__":

    EXPERIMENT_NAME = "blastocyst_experiment"
    WANDB_MODE = "offline"  # offline, online, disabled
    DEV_MODE = WANDB_MODE == "offline" or WANDB_MODE == "disabled"

    TIMESTEAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
    POP_SIZE = 16
    GENERATIONS = 20
    RUNS = 10
    TARGET_SPACES = [
        "embedding",
        "fwd",
        "pixel",
        "edge_density",
        "orientation_variance",
        "dissimilarity_index",
    ]

    parameters_blastocyst = {
        "q": 4,
        "v": 4,
        "w": 4,
        "z": 4,
    }

    # parameters as array:
    parameters_blastocyst_array = np.array([v for v in parameters_blastocyst.values()])

    lower_bounds = parameters_blastocyst_array - 1
    lower_bounds = np.clip(lower_bounds, 0, None)

    upper_bounds = parameters_blastocyst_array + 1

    config = {
        "wandb_mode": WANDB_MODE,
        "notes experiment": f"{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "experiment_folder_path": f"experiments_results/BL/{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "system_name": "blastocyst",
        "lower_bounds": lower_bounds.tolist(),
        "upper_bounds": upper_bounds.tolist(),
        "disable_cma_bounds": False,
        "output_grid_size": [100, 100],
        "update_steps": 1000,
        "popsize": POP_SIZE,
        "generations": GENERATIONS,
        "sigma_init": 0.2,
        "do_rescale": True,
        "normalise": False,
        "custom_embedding_processor": False,
        "visual_embedding": None,  # This will be set appropriately for each target_space
        "target_space": None,  # This will be overridden in the loop
        "params_blastocyst": parameters_blastocyst,
        "target_image_path": "data/blastocyst_instances/3000/3000_1.png",
    }

    config["dev_mode"] = DEV_MODE

    EARLY_STOPS = [None]  # Only applied for embedding target_space
    t0 = time.time()

    for run in range(RUNS):
        for target_space in TARGET_SPACES:

            if target_space == "embedding":
                # Embedding space: test both visual embeddings with early stopping
                visual_embeddings = ["clip", "nomic"]
                early_stops_to_test = EARLY_STOPS

            else:
                visual_embeddings = [None]
                early_stops_to_test = [None]

            for visual_embedding in visual_embeddings:
                for es in early_stops_to_test:
                    run_config = copy.deepcopy(config)
                    run_config["target_space"] = target_space
                    run_config["visual_embedding"] = visual_embedding
                    run_config["early_stop"] = es
                    run_config["early_stop_flag"] = not es is None

                    print(
                        f"Running: run={run+1}/{RUNS}, BL_config={run_config['params_blastocyst']}, target_space={target_space}, visual_embedding={visual_embedding}, early_stop={es}"
                    )
                    run_cmaes(run_config)
    print(f"Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))}")
