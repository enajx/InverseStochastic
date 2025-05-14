import copy
import time
import numpy as np

from experiments.utils_experiment import run_cmaes

if __name__ == "__main__":

    EXPERIMENT_NAME = "SH"
    WANDB_MODE = "disabled"  # offline, online, disabled
    DEV_MODE = WANDB_MODE == "offline" or WANDB_MODE == "disabled"

    TIMESTEAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
    POP_SIZE = 16
    GENERATIONS = 25
    RUNS = 10

    parameters = np.linspace(0.1, 0.8, 8)

    config = {
        "wandb_mode": WANDB_MODE,
        "notes experiment": f"{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "experiment_folder_path": f"experiments_results/SH/{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "system_name": "schelling",
        "lower_bounds": [0],
        "upper_bounds": [1],
        "disable_cma_bounds": True,
        "output_grid_size": [50, 50],
        "update_steps": 1000,
        "popsize": POP_SIZE,
        "generations": GENERATIONS,
        "sigma_init": 0.1,
        "do_rescale": True,
        "normalise": False,
        "custom_embedding_processor": True,
    }

    config["dev_mode"] = DEV_MODE

    t0 = time.time()
    for run in range(RUNS):
        # for target_space in ['embedding', 'pixel']:
        for target_space in ["embedding"]:
            for param in parameters:
                run_config = copy.deepcopy(config)

                run_config["target_space"] = target_space
                run_config["want_similar"] = float(param)
                run_name = run_cmaes(run_config)

    print(f"Total time elapsed: {time.time() - t0} seconds")
