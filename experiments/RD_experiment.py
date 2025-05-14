import copy
import json
import time

from experiments.utils_experiment import run_cmaes

if __name__ == "__main__":

    EXPERIMENT_NAME = "RD"
    WANDB_MODE = "disabled"  # offline, online, disabled
    TIMESTEAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
    POP_SIZE = 128
    GENERATIONS = 32
    RUNS = 10

    with open("data/RD_parameters.json") as f:
        RD_SELECTION = json.load(f)

    config = {
        "wandb_mode": WANDB_MODE,
        "notes experiment": f"{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "experiment_folder_path": f"experiments_results/RD/{EXPERIMENT_NAME}-{TIMESTEAMP}",
        "system_name": "gray_scott",
        "lower_bounds": [0, 0, 0, 0],
        "upper_bounds": [1, 1, 0.1, 0.1],
        "disable_cma_bounds": False,
        "output_grid_size": [100, 100],
        "update_steps": 1000,
        "popsize": POP_SIZE,
        "generations": GENERATIONS,
        "sigma_init": 0.25,
        "do_rescale": True,
        "normalise": False,
        "custom_embedding_processor": True,
    }

    EARLY_STOPS = [[15, 0.04]]
    t0 = time.time()
    for run in range(RUNS):
        for c in RD_SELECTION:
            for es in EARLY_STOPS:
                run_config = copy.deepcopy(config)

                run_config.update(c)
                run_config["early_stop"] = es
                run_config["early_stop_flag"] = not es is None
                run_cmaes(run_config)

    print(f"Total time elapsed: {time.time() - t0} seconds")
