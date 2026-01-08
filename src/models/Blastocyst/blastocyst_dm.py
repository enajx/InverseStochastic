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


def plot_cosine_distances_circle(target_embedding, batch_embeddings, output_path):
    """
    Plot cosine distances between embeddings on a unit circle.
    
    Args:
        target_embedding: numpy array of shape (embedding_dim,) or None
        batch_embeddings: numpy array of shape (n_embeddings, embedding_dim)
        output_path: str, path to save the output figure
    """
    # Ensure inputs are numpy arrays
    batch_embeddings = np.asarray(batch_embeddings)
    
    # Normalize vectors if they aren't already
    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    if target_embedding is not None:
        # Single target mode: compute distances from target to each embedding in batch
        target_embedding = np.asarray(target_embedding)
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        
        # Compute cosine distances from target to each embedding in batch
        cosine_similarities = np.dot(batch_embeddings, target_embedding)
        cosine_distances = 1 - cosine_similarities
        
        # Compute cosine distance of target to itself (should be 0)
        target_distance = 1 - np.dot(target_embedding, target_embedding)
        target_angle = target_distance  # angle = cosine_distance
        target_x = np.cos(target_angle)
        target_y = np.sin(target_angle)
        
        # Plot target embedding at computed position
        ax.scatter(target_x, target_y, color='red', s=100, label='Target', zorder=5, alpha=0.7)
        
        # Plot batch embeddings on unit circle
        # Cosine distance is the ANGLE, radius is always 1 (unit circle)
        # For cosine distance d, the angle is d (in radians)
        angles = cosine_distances
        
        # Position points ON the unit circle at the computed angles
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        
        # Create scatter plot with color coding by distance
        scatter = ax.scatter(x_coords, y_coords, c=cosine_distances, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=4)
        
        # Add legend
        ax.legend()
        
    else:
        # Pairwise mode: compute all pairwise cosine distances
        n_embeddings = len(batch_embeddings)
        
        # Compute all pairwise cosine distances
        cosine_distances = []
        angles = []
        
        for i in range(n_embeddings):
            for j in range(i+1, n_embeddings):  # Only upper triangle to avoid duplicates
                cosine_sim = np.dot(batch_embeddings[i], batch_embeddings[j])
                cosine_dist = 1 - cosine_sim
                cosine_distances.append(cosine_dist)
                angles.append(cosine_dist)  # angle = cosine_distance
        
        # Position points ON the unit circle at the computed angles
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        
        # Create scatter plot with color coding by distance
        scatter = ax.scatter(x_coords, y_coords, c=cosine_distances, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=4)
        
        # Update title for pairwise mode
        ax.set_title('Pairwise Cosine Distances (Unit Circle)', fontsize=14, pad=20)
    scatter = ax.scatter(x_coords, y_coords, c=cosine_distances, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Cosine Distance', fontsize=12)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Cosine Distances from Target Embedding (Unit Circle)', fontsize=14, pad=20)
    
    # Add legend
    ax.legend()
    
    # Add statistics text
    stats_text = f'Mean Cosine Distance: {np.mean(cosine_distances):.3f}\n'
    stats_text += f'Min Cosine Distance: {np.min(cosine_distances):.3f}\n'
    stats_text += f'Max Cosine Distance: {np.max(cosine_distances):.3f}\n'
    stats_text += f'Std Cosine Distance: {np.std(cosine_distances):.3f}'
    
    # Position text in upper right corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Cosine distance circle plot saved to: {output_path}")
    print(f"Distance statistics - Mean: {np.mean(cosine_distances):.3f}, "
          f"Min: {np.min(cosine_distances):.3f}, Max: {np.max(cosine_distances):.3f}")


if __name__ == "__main__":

    embedding_model = "clip"
    # embedding_model = "nomic"
    # embedding_model = "fwd"
    fwd_wave = "haar"
    fwd_level = 2
    fwd_log = True

    # Load all png images in folder path
    # folder_path = "temp_circle/"
    # folder_path = "data/blastocyst_instances/"
    folder_path = "data/blastocyst_noise/0.0"
    # folder_path = "data/blastocyst_instances/500/"
    # folder_path = "data/blastocyst_instances/1000/"
    # folder_path = "data/blastocyst_instances/1500/"
    # folder_path = "data/blastocyst_instances/2000/"
    # folder_path = "data/blastocyst_instances/3000/"
    # folder_path = "data/flowers"

    images = []
    for root, dirs, files in os.walk(folder_path):
        dirs.sort(key=lambda x: float(x))
        # dirs.sort(key=lambda x: int(x))
        print(root)
        images.extend(os.path.join(root, f) for f in files if f.endswith(".png"))
        # images.extend(os.path.join(root, f) for f in files if f.endswith(".jpg"))
    
    clean_folder_name = folder_path.rstrip("/").replace("/", "_")

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

        print(f"Embedding size: {image_embeddings.shape}")
        print(f"Distance matrix shape: {distance_matrix.shape}")

        # Plot cosine distances circle for CLIP
        os.makedirs(f"temp/{embedding_model}", exist_ok=True)
        target_emb = image_embeddings[0]
        batch_embs = image_embeddings[1:]

        
        plot_cosine_distances_circle(
            target_emb,
            # None,
            batch_embs,
            f"temp/{embedding_model}/cosine_distances_circle_{clean_folder_name}.png"
        )

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

        print(f"Embedding size: {image_embeddings.shape}")
        print(f"Distance matrix shape: {distance_matrix.shape}")

        # Plot cosine distances circle for Nomic
        os.makedirs(f"temp/{embedding_model}", exist_ok=True)
        target_emb = image_embeddings[0]
        batch_embs = image_embeddings[1:]
        
        plot_cosine_distances_circle(
            target_emb,
            batch_embs,
            f"temp/{embedding_model}/cosine_distances_circle_{clean_folder_name}.png"
        )

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
    if embedding_model == "fwd":
        embedding_model = f"{embedding_model}_{fwd_wave}_{fwd_level}_{fwd_log}"
    os.makedirs(f"temp/{embedding_model}", exist_ok=True)
    plot_distance_matrix(
        distance_matrix,
        images,
        f"temp/{embedding_model}/blastocyst_dm_{clean_folder_name}.png",
    )
