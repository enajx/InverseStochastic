# Inverse Stochastic

Codebase for `Solving Inverse Problems in Stochastic Self-Organising Systems through Invariant Representations` paper. The codebase contains:

**Main directories:**
- `src/models/`: Models for reaction-diffusion and Schelling systems.
- `src/trainers/`: Training and optimization routines, including evolutionary strategies.
- `src/vision_models/`: Tools for creating embedding representations using CLIP.
- `src/visualisations/`: Visualization utilities for results and experiments.
- `src/scrips_analysis/`: Scripts for analyzing and plotting experiment data.
- `src/utils/`: General utility functions.


---

## Installation

Install dependencies using [uv](https://docs.astral.sh/uv/) and the provided `pyproject.toml` or `uv.lock`. 

First [install uv](https://docs.astral.sh/uv/getting-started/installation/), then simply type `uv sync`.

---

## Usage

### Running Reaction-Diffusion Experiments

To reproduce the reported reaction-diffusion experiment, run:

```bash
python experiments/RD_experiment.py
```

This will optimize parameters for the Gray-Scott reaction-diffusion system to match target patterns, saving results in the `experiments_results/RD/` directory.

### Running Schelling Model Experiments

To reproduce the reported  Schelling segregation model experiment, run:

```bash
python experiments/SH_experiment.py
```

Results will be saved in the `experiments_results/SH/` directory.

### Running Blastocyst Experiments

Running the blastocyst experiment requires installing [Morpheus](https://morpheus.gitlab.io) and setting up the environment as per the instructions on their site, then run:

```bash
python experiments/BL_experiment.py
```


---

### Running specific targets

To train for specific target parameters, use the trainer module directly:

- See `src/trainers/trainer.py` and run it with your desired settings.
- Edit the configuration at the top of the file to set your target system and parameters.

Run:

```bash
python src/trainers/trainer.py
```

Results and outputs will be saved in the specified run folder.

