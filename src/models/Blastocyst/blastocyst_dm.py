from PIL import Image
import torch
import numpy as np
from typing import Callable
import torch.nn.functional as F
from torchvision import transforms
import warnings
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


from transformers import CLIPImageProcessor, CLIPModel
from transformers import AutoModel, AutoImageProcessor
from src.trainers.fwd import frechet_wavelet_distance


def plot_distance_matrix(distance_matrix, images, output_path):

    n = len(images)
    fig, axs = plt.subplots(
        n + 1, n + 1, figsize=(n + 1, n + 1), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    for ax in axs.flat:
        ax.axis("off")

    for i in range(n):
        axs[0, i + 1].imshow(images[i])
        axs[i + 1, 0].imshow(images[i])

    norm = mcolors.Normalize(vmin=distance_matrix.min(), vmax=distance_matrix.max())
    cmap = plt.get_cmap("viridis")

    for i in range(n):
        for j in range(n):
            color = cmap(norm(distance_matrix[i, j]))[:3]
            axs[i + 1, j + 1].imshow(np.full((10, 10, 3), color))
            axs[i + 1, j + 1].text(
                0.5,
                0.5,
                f"{distance_matrix[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                transform=axs[i + 1, j + 1].transAxes,
                color="white",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # embedding_model = "clip"
    # embedding_model = "nomic"
    embedding_model = "fwd"
    fwd_wave = "haar"
    fwd_level = 2
    fwd_log = True

    # Load all png images in folder path
    folder_path = "blastocyst_instances/"
    # folder_path = "blastocyst_instances/500/"
    # folder_path = "blastocyst_instances/1000/"
    # folder_path = "blastocyst_instances/1500/"
    # folder_path = "blastocyst_instances/2000/"
    # folder_path = "blastocyst_instances/3000/"
    # folder_path = "flowers"

    images = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort(key=lambda x: int(x))
        print(root)
        images.extend(os.path.join(root, f) for f in files if f.endswith(".png"))
        # images.extend(os.path.join(root, f) for f in files if f.endswith(".jpg"))

    # Load all images
    images = [Image.open(f) for f in images]

    # Resize images to 224x224
    images = [image.resize((224, 224)) for image in images]

    # convert images to RGB
    # images = [image.convert("RGB") for image in images]

    # Convert images to numpy arrays
    images = [np.array(image) for image in images]

    if embedding_model == "clip":

        # Load pre-trained CLIP model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )

        # Prepare the image for the model
        inputs = processor(images=images, return_tensors="pt")

        # Generate image embedding
        with torch.no_grad():
            print("Generating image embeddings")
            image_embeddings = model.get_image_features(inputs["pixel_values"])

        # Convert to numpy for easier handling
        image_embeddings = image_embeddings.numpy()
        # Normalise the image embedding
        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarity between all images pairs
        distance_matrix = 1 - np.dot(image_embeddings, image_embeddings.T)

        print(f"Distance matrix shape: {distance_matrix.shape}")

    elif embedding_model == "nomic":

        processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", local_files_only=False, use_fast=True
        )  # same as clip's processor

        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )

        inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            print("Generating image embeddings")
            image_embeddings = model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0]

        # Normalise the image embedding
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        image_embeddings = image_embeddings.numpy()

        distance_matrix = 1 - np.dot(image_embeddings, image_embeddings.T)

    elif embedding_model == "fwd":

        # Convert images to tensors
        images = [torch.tensor(image) for image in images]
        images = torch.stack(images).float() / 255.0

        # Compute FWD distance between all images pairs
        distance_matrix = np.zeros((len(images), len(images)))
        for i, image in enumerate(images):
            pairwise_distance = frechet_wavelet_distance(
                images, image, fwd_wave, fwd_level, fwd_log
            )
            distance_matrix[i, :] = pairwise_distance.cpu().numpy()

        print(f"Distance matrix shape: {distance_matrix.shape}")

    # Plot figure with images and similarity matrix values
    # Clean the folder path for filename (remove trailing slash and replace slashes with underscores)
    clean_folder_name = folder_path.rstrip("/").replace("/", "_")
    if embedding_model == "fwd":
        embedding_model = f"{embedding_model}_{fwd_wave}_{fwd_level}_{fwd_log}"
    os.makedirs(f"temp/{embedding_model}", exist_ok=True)
    plot_distance_matrix(
        distance_matrix,
        images,
        f"temp/{embedding_model}/blastocyst_dm_{clean_folder_name}.png",
    )
