import os
import sys
import copy
import datetime
import pathlib
import time

import git
import numpy as np
import petname
import yaml

import wandb
from src.trainers.trainer import optimize_parameters_cmaes
from src.utils.utils import seed_python_numpy_torch_cuda

BASE_CONFIG = {
    "wandb_mode": None,
    "project_name": None,
    "group": None,
    "entity": None,
    "log_freq": 10,
    "disable_cma_bounds": False,
    "target_space": "embedding",
    "search_space": "parameters",
    "minmax_RD_output": False,
    "visual_embedding": "clip",
    "target_image_path": None,
    "anisotropic": False,
}


def _clean_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: _clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_yaml(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _initialize_wandb(config):
    wandb.init(
        project=config["project_name"],
        config=config,
        reinit=True,
        allow_val_change=True,
        mode=config["wandb_mode"],
        name=config["run_name"],
        entity=config["entity"],
        resume="allow",
        id=wandb.util.generate_id(),
    )


def _check_git_status(config):
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty() and not config["dev_mode"]:
        print("\n" + 20 * "*")
        sys.exit(
            "\nExperiment aborted\nRepository is dirty. Commit or discard code changes before running experiment.\n"
        )
    config["hash_code"] = repo.head.object.hexsha
    config["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["machine"] = os.uname()[1]


def run_cmaes(user_config: dict):
    # _check_git_status(user_config) #TODO this must be in the experiment not in each run
    config = copy.deepcopy(BASE_CONFIG)
    config.update(user_config)

    config["seed"] = seed_python_numpy_torch_cuda(config.get("seed"))
    config["run_name"] = petname.Generate(3)

    print(f"Run name: {config['run_name']}")

    _initialize_wandb(config)

    run_path = os.path.join(config["experiment_folder_path"], config["run_name"])
    config["run_folder_path"] = run_path
    pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)

    tic = time.time()
    optimize_parameters_cmaes(copy.deepcopy(config))

    run_time = time.time() - tic
    config["run_time"] = run_time

    with open(os.path.join(run_path, "config.yml"), "w") as file:
        yaml.dump(_clean_for_yaml(config), file)

    wandb.finish()
    return config["run_name"]


if __name__ == "__main__":
    pass
