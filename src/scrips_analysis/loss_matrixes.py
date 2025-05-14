import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy as clopy
import json
import os
from copy import deepcopy

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from transformers import AutoModel, CLIPImageProcessor, CLIPModel

from models.RD.RDBatch4Params import RD_GPU
from models.Schelling.schelling import run_schelling
from utils.utils import load_image_as_tensor
from vision_models.clip_run import make_embedding_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(
    device
)
CLIP_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", local_files_only=True
)


def build_color_mapping(mode="cmap", values=None, cmap_name="tab20c"):
    def to_rgb(color):
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) == 6:
                return [int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
            else:
                return list(mcolors.to_rgb(color))  # handles named colors
        return color  # already an RGB list

    if mode == "cmap":
        cmap = plt.get_cmap(cmap_name)
        mapping = torch.tensor(
            [
                cmap(0.0)[:3],
                cmap(0.5)[:3],
                cmap(1.0)[:3],
            ],
            dtype=torch.float32,
        )
    elif mode == "manual":
        if values is None:
            raise ValueError("For manual mode, you must provide a list of colors.")
        rgb_colors = [to_rgb(v) for v in values]
        mapping = torch.tensor(rgb_colors, dtype=torch.float32)
    else:
        raise ValueError("Mode must be 'cmap' or 'manual'.")

    return mapping


# COLOR_MAPPING = build_color_mapping(mode="manual", values=[
#        [1.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0],
#        [0.0, 0.0, 1.0]])

COLOR_MAPPING = build_color_mapping(mode="manual", values=["#a9a9a9", "#ffffff", "#3182bd"])

################ VISUALIZATION ################


def make_image(
    frame: np.ndarray, filename: str, folder_path=None, system_name=None, scale_factor=2
) -> None:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    assert np.min(frame) >= 0, "Negative values detected in data array."
    assert np.max(frame) <= 255, "Values above 255 detected in data array."
    assert frame.shape[-1] == 3, "Expected RGB image with shape [..., 3]"
    channel = frame[..., -1]
    if channel.max() > 1:
        channel = channel / 255.0
    # Scale
    h, w = channel.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    channel = cv2.resize(channel, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Smooth
    channel = gaussian_filter(channel, sigma=1)
    cmap = "jet"
    plt.figure(figsize=(new_w / 100, new_h / 100), dpi=300)
    plt.imshow(channel, cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        f"{folder_path}/{filename}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()


################ XY ################


def get_invariant_XY():
    a = {
        "key_name": "Circle",
        "initial_state_seed_type": "circle",
        "initial_state_seed_radius": 10,
        "params_gray_scott": [1.0, 0.5, 0.055, 0.062],
        "output_grid_size": [100, 100],
        "update_steps": 500,
        "pad_mode": "circular",
    }

    b = {
        "key_name": "Circle Big",
        "initial_state_seed_type": "circle",
        "initial_state_seed_radius": 10,
        "params_gray_scott": [1.0, 0.5, 0.055, 0.062],
        "output_grid_size": [100, 100],
        "update_steps": 1200,
        "pad_mode": "circular",
    }

    c = {
        "key_name": "trebol",
        "initial_state_seed_type": "random_triangle",
        "initial_state_seed_radius": 10,
        "params_gray_scott": [0.9838, 0.4, 0.0722, 0.061],
        "output_grid_size": [100, 100],
        "update_steps": 1000,
        "pad_mode": "circular",
    }

    all = [a, b, c]

    Xs = []
    Ys = []
    for d in all:
        rd = RD_GPU(
            param_batch=np.array(
                [
                    d["params_gray_scott"],
                ],
                dtype=np.float64,
            ),
            seed=d["initial_state_seed_type"],
            seed_radius=d["initial_state_seed_radius"],
            grid_size=d["output_grid_size"],
            pad_mode=d["pad_mode"],
            anisotropic=False,
        )
        x, data = rd.run(
            d["update_steps"], save_all=False, show_progress=True, minimax_RD_output=False
        )
        y = clopy.deepcopy(x)

        Xs.append(x[0])
        Ys.append(y[0])

    # repeat last element of Xs and Ys
    Xs.append(torch.transpose(Xs[-1], 0, 1))
    Ys.append(torch.transpose(Ys[-1], 0, 1))

    Xs = torch.stack(Xs).float()
    Ys = torch.stack(Ys).float()

    return Xs, Ys


def trebols_images():

    GRID_SIZE = (100, 100)
    STEPS = 1000
    PAD_MODE = "circular"

    # da db f k

    for i in range(10):
        parameters = np.array(
            [
                [0.9838, 0.4, 0.0722, 0.061],
            ],
            dtype=np.float64,
        )

        rd = RD_GPU(
            param_batch=parameters,
            seed="random_triangle",
            seed_radius=10,
            grid_size=GRID_SIZE,
            pad_mode=PAD_MODE,
            anisotropic=False,
        )

        last_frame, data = rd.run(
            STEPS, save_all=False, show_progress=True, minimax_RD_output=False
        )

        print(last_frame.shape)

        lf = last_frame[0]
        print(lf.shape)
        make_image(frame=lf, filename=f"{i}", folder_path="visuals_paper/trebols", system_name=None)


def get_RD_XY():
    json_path = "data/RD_parameters.json"
    with open(json_path, "r") as f:
        selected = json.load(f)

    print(len(selected))

    for i, d in enumerate(selected):
        d["key_name"] = i

    # save selected to json
    with open("selected.json", "w") as f:
        json.dump(selected, f, indent=4)

    Xs = []
    Ys = []
    for d in selected:
        rd = RD_GPU(
            param_batch=np.array([d["params_gray_scott"],],dtype=np.float64,),
            seed=d["initial_state_seed_type"],
            seed_radius=d["initial_state_seed_radius"],
            grid_size=d["output_grid_size"],
            pad_mode=d["pad_mode"],
            anisotropic=False,
        )
        x, _ = rd.run(d["update_steps"], save_all=False, show_progress=True, minimax_RD_output=False)

        rd = RD_GPU(
            param_batch=np.array(
                [
                    d["params_gray_scott"],
                ],
                dtype=np.float64,
            ),
            seed=d["initial_state_seed_type"],
            seed_radius=d["initial_state_seed_radius"],
            grid_size=d["output_grid_size"],
            pad_mode=d["pad_mode"],
            anisotropic=False,
        )
        y, _ = rd.run(
            d["update_steps"], save_all=False, show_progress=True, minimax_RD_output=False
        )

        Xs.append(x[0])
        Ys.append(y[0])

    Xs = torch.stack(Xs).float()
    Ys = torch.stack(Ys).float()

    return Xs, Ys


def get_SH_XY():

    def run_sch(want_similar, grid_size):
        print(
            f"Running Schelling with want_similar={want_similar}, grid_size={grid_size}x{grid_size}"
        )
        params = {
            "want_similar": want_similar,
            "n_groups": 2,
            "density": 0.9,
            "size": grid_size,
            "max_run_duration": 100,
        }
        return run_schelling(params)

    grid_size = (100, 100)
    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # parameters = np.array([0.3, 0.5, 0.7])

    X = [run_sch(p, grid_size[0]) for p in parameters]
    Y = [run_sch(p, grid_size[0]) for p in parameters]

    Xs = torch.stack(X).float()
    Ys = torch.stack(Y).float()
    return Xs, Ys


def distance_matrix(system, run=None):

    if system == "RD":
        Xs, Ys = get_RD_XY()
    elif system == "SH":
        Xs, Ys = get_SH_XY()
    elif system == "INV":
        Xs, Ys = get_invariant_XY()
    else:
        raise ValueError("Unknown system")

    #### Embeddings
    ex = make_embedding_clip(
        images=Xs.float(),
        model=CLIP_model,
        processor=None,
        normalise=False,
        do_rescale=True,
        device=device,
    )
    ey = make_embedding_clip(
        images=Ys.float(),
        model=CLIP_model,
        processor=None,
        normalise=False,
        do_rescale=True,
        device=device,
    )

    print(f"Embedding shape: {ex.shape}")
    print(f"Embedding shape: {ey.shape}")

    ### Losses
    n = len(Xs)
    clip_loss = np.zeros((n, n))
    print(f"Computing {n}x{n} losses")
    for i in range(n):
        for j in range(n):
            clip_loss[i, j] = 1 - np.dot(ex[i], ey[j])

    ### Plotting
    fig, axs = plt.subplots(
        n + 1, n + 1, figsize=(n + 1, n + 1), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    for ax in axs.flat:
        ax.axis("off")

    if system == "RD" or system == "INV":
        for j in range(n):
            axs[0, j + 1].imshow(Ys[j][..., -1].cpu().numpy().astype(np.uint8), cmap="jet")
        for i in range(n):
            axs[i + 1, 0].imshow(Xs[i][..., -1].cpu().numpy().astype(np.uint8), cmap="jet")

    elif system == "SH":
        for j in range(n):
            img_tensor = Ys[j].cpu().float()  # (H, W, 3)
            img_tensor = img_tensor / img_tensor.max()  # normalize to [0, 1]
            img_rgb = (img_tensor @ COLOR_MAPPING).clamp(0.0, 1.0)
            axs[0, j + 1].imshow(img_rgb.numpy())

        for i in range(n):
            img_tensor = Xs[i].cpu().float()
            img_tensor = img_tensor / img_tensor.max()
            img_rgb = (img_tensor @ COLOR_MAPPING).clamp(0.0, 1.0)
            axs[i + 1, 0].imshow(img_rgb.numpy())

    norm = mcolors.Normalize(vmin=clip_loss.min(), vmax=clip_loss.max())

    for i in range(n):
        for j in range(n):
            color = plt.get_cmap(LOSS_CMAP)(norm(clip_loss[i, j]))[:3]
            axs[i + 1, j + 1].imshow(np.full((10, 10, 3), color))
            axs[i + 1, j + 1].text(
                0.5,
                0.5,
                f"{clip_loss[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                transform=axs[i + 1, j + 1].transAxes,
                color="white",
            )

    plt.tight_layout()
    base_path = (
        f"visuals_paper/loss_matrixes/{system}_{run}"
        if run
        else f"visuals_paper/loss_matrixes/{system}"
    )
    plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    OUT_FOLDER = "visuals_paper/loss_matrixes"
    os.makedirs(OUT_FOLDER, exist_ok=True)
    LOSS_CMAP = "viridis_r"
    
    for i in range(1):
        tag = f"{i}"
        distance_matrix(system = 'RD', run = tag)
        distance_matrix(system = 'SH', run = tag)
        distance_matrix(system="INV", run=tag)
