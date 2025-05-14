import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPImageProcessor, CLIPModel

from models.RD.RDBatch4Params import RD_GPU
from models.Schelling.schelling import run_schelling
from utils.utils import seed_python_numpy_torch_cuda
from vision_models.clip_run import make_embedding_clip

from utils.utils import load_image_as_tensor

# Add your src path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from models.RD.RDBatch4Params import RD_GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
CLIP_model = CLIP_model.to(device)
CLIP_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", local_files_only=True
)


def compute_loss(x, y):
    cosine_similarity = np.dot(x, y.T)
    loss = 1 - cosine_similarity
    return loss.item() if isinstance(loss, np.ndarray) else float(loss)


def plot_param_grid(target_image, runs_frames, losses, output_path, exp_name):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from scipy.ndimage import zoom
    import torch

    def resize_array(arr, factor):
        return zoom(arr, factor, order=1)

    N_RUNS = len(runs_frames)
    resize_factor = 2

    os.makedirs(output_path, exist_ok=True)

    # Ensure target_image is NumPy float
    if isinstance(target_image, torch.Tensor):
        target_image = target_image.cpu().numpy()
    if target_image.dtype != np.float32 and target_image.dtype != np.float64:
        target_image = target_image.astype(np.float32) / 255.0

    # Ensure runs_frames is NumPy float
    if isinstance(runs_frames, torch.Tensor):
        runs_frames = runs_frames.to(torch.float32).cpu().numpy() / 255.0

    # Grayscale conversion
    target_image_gray = np.mean(target_image, axis=-1)
    target_image_gray = resize_array(target_image_gray, resize_factor)

    runs_frames_gray = runs_frames.mean(axis=-1)
    runs_frames_gray = np.stack([resize_array(f, resize_factor) for f in runs_frames_gray])

    # Save target image
    plt.imshow(target_image_gray, cmap="Greys_r")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_path, f"{exp_name}_target.pdf"), bbox_inches="tight", pad_inches=0
    )
    plt.close()

    # Save each run independently
    for i in range(N_RUNS):
        plt.imshow(runs_frames_gray[i], cmap="Greys_r")
        plt.axis("off")
        plt.savefig(
            os.path.join(output_path, f"{exp_name}_{i}.pdf"), bbox_inches="tight", pad_inches=0
        )
        plt.close()


def losses_computation(data, images_folder, output_folder, n_runs=10):
    print(f'Processing {data["key_name"]}')
    os.makedirs(output_folder, exist_ok=True)

    image_path = os.path.join(images_folder, data["image_filename"])
    target_image = load_image_as_tensor(
        image_path,
        resize=None,
        normalize=False,
        color=False,
        minmax_target_image=False,
        reverse_image=False,
        swap_channels=True,
    )
    assert target_image.shape[-1] == 3

    target_image_embedding = make_embedding_clip(
        images=target_image.float(),
        model=CLIP_model,
        processor=None,
        normalise=False,
        do_rescale=True,
        device=device,
    )

    parameters = np.repeat(data["parameters"], n_runs, axis=0)

    rd = RD_GPU(
        parameters,
        grid_size=data["grid_size"],
        seed="random",
        seed_radius=5,
        pad_mode="circular",
        anisotropic=data["anisotropic"],
    )

    last_frames, _ = rd.run(1000, save_all=False, show_progress=True, minimax_RD_output=True)

    runs_embeddings = make_embedding_clip(
        images=last_frames.float(),
        model=CLIP_model,
        processor=None,
        normalise=False,
        do_rescale=True,
        device=device,
    )

    losses = []
    for i in range(n_runs):
        loss = compute_loss(target_image_embedding, runs_embeddings[i])
        losses.append(loss)

    losses = np.array(losses)
    loss_data = {
        "losses": losses,
        "mean loss": losses.mean(),
        "std loss": losses.std(),
        "sem loss": losses.std() / np.sqrt(n_runs),
        "similarities": 1 - losses,
        "mean similarities": (1 - losses).mean(),
        "std similarities": losses.std(),  # same as losses
        "sem similarities": losses.std() / np.sqrt(n_runs),
    }

    print(f"Target {data['key_name']}, N runs: {n_runs}:")

    print(f"    1 - cosine_similarity                       |   cosine_similarity")
    print(
        f"    Mean loss: {loss_data['mean loss']:.4f}     |   {loss_data['mean similarities']:.4f}"
    )
    print(
        f"    Std loss:  {loss_data['std loss']:.4f}      |   {loss_data['std similarities']:.4f}"
    )
    print(
        f"    SEM loss:  {loss_data['sem loss']:.4f}      |   {loss_data['sem similarities']:.4f}"
    )

    np.save(os.path.join(output_folder, f"{data['key_name']}_losses.npy"), loss_data)

    plot_param_grid(
        target_image=target_image,
        runs_frames=last_frames,
        losses=losses,
        output_path=os.path.join(output_folder, f"{data['key_name']}_runs"),
        exp_name=data["key_name"],
    )


if __name__ == "__main__":

    data = [
        {
            "key_name": "zebra",
            "image_filename": "zebra.png",
            "parameters": np.array([[0.7491, 0.8091, 0.2424, 0.923, 0.079, 0.0296],],dtype=np.float64,),
            "grid_size": [50, 50],
            "anisotropic": True,
        },
        {
            "key_name": "gecco",
            "image_filename": "lizard.png",
            "parameters": np.array(
                [
                    [0.9272, 0.1412, 0.0257, 0.0184],
                ],
                dtype=np.float64,
            ),
            "grid_size": [50, 50],
            "anisotropic": False,
        },
    ]

    images_folder = "data/natural_targets"

    for d in data:
        losses_computation(
            data=d,
            images_folder=images_folder,
            output_folder="visuals_paper/animal_target",
            n_runs=100,
        )
