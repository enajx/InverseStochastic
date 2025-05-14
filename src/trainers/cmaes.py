import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import time
import warnings

import cma
import numpy as np
import psutil
import wandb
from transformers import CLIPImageProcessor, CLIPModel, AutoModel

# warnings.filterwarnings("ignore") No bueno

# numpy print options 4 digits and no scientific notation
np.set_printoptions(precision=4, suppress=True)

# If using torch anywhere in the pipeline
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def x0_bounded(nb_params, bounds):
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    if lower_bounds.shape != upper_bounds.shape:
        raise ValueError("Bounds arrays must have the same shape.")

    if len(lower_bounds) != nb_params:
        raise ValueError("Number of bounds must match n_params.")

    return np.random.uniform(low=lower_bounds, high=upper_bounds, size=nb_params)


def train(
    config: dict,
    target: np.ndarray,
    run_model_with_population_parameters: callable,
):

    from trainers.trainer import compute_fitness

    if config["system_name"] == "gray_scott" or config["system_name"] == "gray_scott":
        if config["search_space"] == "parameters":
            nb_parameters = 4
            if config["anisotropic"]:
                nb_parameters = 6
        else:
            raise ValueError("search_space must be either parameters or initial_state")

    elif config["system_name"] == "schelling":
        nb_parameters = 1
    else:
        raise ValueError("system_name must be either gray_scott or schelling")

    # Initilise the embedding models and processors
    if config["target_space"] == "embedding":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        CLIP_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        if config["visual_embedding"] == "clip":
            CLIP_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", local_files_only=True
            )
            embedding_model = CLIP_model.to(device)
        else:
            raise ValueError("visual_embedding must be either clip")

    bounds = [config["lower_bounds"], config["upper_bounds"]]

    def initialize_optimizer():
        x0 = x0_bounded(nb_params=nb_parameters, bounds=bounds)
        opti = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=config["sigma_init"],
            options={
                "verb_disp": 1,
                "popsize": config["popsize"],
                "maxiter": config["generations"],
                "bounds": (
                    None
                    if config["disable_cma_bounds"] or config["system_name"] == "schelling"
                    else bounds
                ),
                # "bounds": bounds if config["system_name"] != "schelling" else None,
                "CMA_mu": config["popsize"] // 2,
                "CMA_diagonal": 0,
                "verbose": -9,  # Set verbosity level to suppress most output
                # "diagonal": 1,
                # More options: https://github.com/CMA-ES/pycma/blob/ac4a913b4f2c0ff61c18ee29695bbe732a3828d3/cma/evolution_strategy.py#L418
            },
        )
        return opti

    # Initialize the optimizer
    es = initialize_optimizer()

    print("\n.......................................................")
    print("\nInitilisating CMA-ES with", nb_parameters, "trainable parameters \n")
    print("\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n")

    tic = time.time()

    if "early_stop" in config and config["early_stop"] is not None:
        early_stop_gen, early_stop_value = config["early_stop"]
    else:
        early_stop_gen, early_stop_value = None, None
    print(f"Early stop: {early_stop_gen}, {early_stop_value} ✋🛑\n")

    best_loss_so_far = np.inf
    best_solution_so_far = None

    gen_losses = []  # Generation best losses:      best loss of each generation
    ath_losses = []  # All time best losses:        best loss so far
    gen_solutions = []  # Generation best solutions:    best solution of each generation
    ath_solutions = []  # All time best solutions:      best solution so far
    gen = 0
    restarts = 0
    max_restarts = config.get("max_restarts", 5)  # Get from config or default to 5

    # Optimisation loop
    while not es.stop() or gen < config["generations"]:
        try:
            # Check if we need to restart based on early stopping criteria
            if (
                early_stop_gen is not None
                and early_stop_value is not None
                and gen >= early_stop_gen
                and best_loss_so_far > early_stop_value
                and restarts < max_restarts
            ):
                print(
                    f"\nRestarting optimization at generation {gen}. Best loss so far: {best_loss_so_far:.4f}"
                )
                print(f"Target early stop value: {early_stop_value}. Restarts so far: {restarts}\n")

                # Reinitialize CMA-ES with a new random starting point
                es = initialize_optimizer()

                # Reset generation counter but keep best solution and loss
                gen_losses = []
                ath_losses = []
                gen_solutions = []
                ath_solutions = []
                gen = 0
                restarts += 1

                # Log restart if wandb is enabled
                if wandb.run is not None:
                    wandb.log({"restart": restarts, "restart_best_loss": best_loss_so_far})

                continue

            X = es.ask()  # Generate candidate solutions

            # clip the parameters to the bounds
            # X = np.clip(X, config["lower_bounds"], config["upper_bounds"])

            # Run the model with the candidate solutions
            output = run_model_with_population_parameters(np.array(X), config)

            fitvals = compute_fitness(
                output_batch=output,
                target=target,
                config=config,
                processor=(
                    None
                    if (
                        config["custom_embedding_processor"]
                        or config["target_space"] != "embedding"
                    )
                    else CLIP_processor
                ),
                model=embedding_model if config["target_space"] == "embedding" else None,
            )

            if isinstance(fitvals, torch.Tensor):
                fitvals = fitvals.detach().cpu().numpy()
            es.tell(X, fitvals)  # Update the evolution strategy

            gen_best_loss = np.min(fitvals)  # Best loss for the current generation
            gen_best_solution = X[np.argmin(fitvals)]  # Best solution for the current generation

            # Update all-time best if the current generation's best is better
            if gen_best_loss <= best_loss_so_far:
                best_loss_so_far = gen_best_loss  # Update historical best loss
                best_solution_so_far = es.best.x  # Update historical best solution

            # Track the all
            gen_losses.append(gen_best_loss)  # Track the best loss of the current generation
            gen_solutions.append(
                gen_best_solution
            )  # Track the best solution of the current generation
            ath_losses.append(best_loss_so_far)  # Historical best loss so far
            ath_solutions.append(best_solution_so_far)  # Historical best solution so far

            # centroid = es.result.xbest
            # TRACK_CENTROID = False
            # if TRACK_CENTROID:
            #     output = run_model_with_population_parameters("X", config)
            #     loss_centroid = compute_fitness(
            #         output_batch=output,
            #         target=target,
            #         config=config,
            #         processor=CLIP_processor if config["target_space"] == "embedding" else None,
            #         model=CLIP_model if config["target_space"] == "embedding" else None,
            #     )
            #     print(f"Centroid loss: {loss_centroid}")

            # display(es)
            print(
                f"\nGen: {gen} | Historical best - {best_solution_so_far} : {best_loss_so_far:.4f} || Generation best: {gen_best_solution} : {gen_best_loss:.4f} "
            )

            # check is wandb is enabled
            if wandb.run is None:
                pass
            else:
                wandb.log(
                    {
                        "his_best_loss": best_loss_so_far,
                        "gen_best_loss": gen_best_loss,
                        "gen": gen,
                    }
                )

                if best_solution_so_far is not None:
                    wandb.log({"parameter_histogram": wandb.Histogram(best_solution_so_far)})
                    wandb.log({f"param_{i}": val for i, val in enumerate(best_solution_so_far)})
            gen += 1

        # Allows to interrupt optimation with Ctrl+C
        except KeyboardInterrupt:  # Only works with python mp
            print("\n" + 20 * "*")
            print(f"\nCaught Ctrl+C!\nStopping evolution\n")
            print(20 * "*" + "\n")
            break

    toc = time.time()
    print(f"\nEvolution took: {int(toc - tic)} seconds")
    if restarts > 0:
        print(f"Optimization was restarted {restarts} times")

    best_historical_solution = np.array(best_solution_so_far)
    best_historical_loss = np.array(best_loss_so_far)
    gen_losses = np.array(gen_losses)
    ath_losses = np.array(ath_losses)
    # gen_solutions = np.array(gen_solutions)
    # ath_solutions = np.array(ath_solutions)

    # best_last_gen_sol = es.pop_sorted[0]
    best_last_gen_sol = gen_best_solution
    best_last_gen_loss = gen_best_loss

    print(f"\n\nBest historical: {best_historical_solution} | Loss: {best_historical_loss:.4f}")
    print(f"Best last gen  : {best_last_gen_sol} | Loss: {best_last_gen_loss:.4f}")

    if config["system_name"] == "gray_scott":
        print(f"\nTarget parameter: {config["params_gray_scott"]}")
    elif config["system_name"] == "schelling":
        print(f"\nTarget parameter: {config["want_similar"]}")

    return (
        best_historical_solution,
        best_historical_loss,
        gen_solutions,
        gen_losses,
        best_last_gen_sol,
        best_last_gen_loss,
        # ath_solutions,
        # ath_losses,
    )


if __name__ == "__main__":
    pass
