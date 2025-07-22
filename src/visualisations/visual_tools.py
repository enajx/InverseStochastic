import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from models.RD.RDBatch4Params import RD_GPU

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)


# FIX: verify this works with all systems
def make_image(frame: np.ndarray, filename: str, folder_path=None, system_name=None) -> None:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()

    # if frame.ndim == 4 and frame.shape[0] == 1:
    # frame = frame[0]

    assert np.min(frame) >= 0, "Negative values detected in data array."
    assert np.max(frame) <= 255, "Values above 255 detected in data array."
    assert frame.shape[-1] == 3, "Expected RGB image with shape [..., 3]"

    # # Check if all channels are (approximately) the same
    # is_grayscale = np.allclose(frame[..., 0], frame[..., 1], atol=1e-3) and np.allclose(
    #     frame[..., 1], frame[..., 2], atol=1e-3
    # )

    # cmap = "jet"

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(frame, cmap=None)
    # plt.imshow(frame[..., -1], cmap=None)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        f"{folder_path}/{filename}.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()


def make_video(
    data: np.ndarray,
    filename: str,
    cmap: str = "viridis",
    fps=60,
    reduction_rate=0.0,
    folder_path=None,
) -> None:

    # Handle reduction_rate
    if 0 < reduction_rate < 1:
        n_frames = len(data)
        frames_to_keep = max(
            int(n_frames * (1 - reduction_rate)), 1
        )  # Ensure at least 1 frame is kept
        step = max(n_frames // frames_to_keep, 1)  # Avoid step becoming zero
        indices = list(range(0, n_frames, step))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        data = [data[i] for i in indices]

    elif reduction_rate >= 1:
        raise ValueError("Reduction rate should be a fraction between 0 and 1 (e.g., 0.1 for 10%).")

    frame_size = (512, 512)  # Default frame size
    # frame_size = (64, 64) # Default frame size
    writer = imageio.get_writer(f"{folder_path}/{filename}.mp4", fps=fps)  # Explicit fps

    # Prepare figure for visualization
    original_shape = data[0].shape
    aspect_ratio = original_shape[1] / original_shape[0]
    fig_width = 5
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    img_plot = ax.imshow(data[0], cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    print("Creating video...")
    for array in tqdm(data):
        img_plot.set_data(array)
        fig.canvas.draw()

        # Capture canvas content
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Convert RGBA to RGB

        # Resize to consistent size
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(frame_size, Image.LANCZOS)
        img_resized = np.array(pil_img, dtype=np.uint8)

        # Append frame to video
        writer.append_data(img_resized)

    writer.close()
    plt.close(fig)
    return


def run_and_make_visuals(
    params: np.array,
    output_grid_size: tuple,
    update_steps: int,
    initial_state_seed_type: str,
    initial_state_seed_radius: int,
    pad_mode: str,
    filename: str,
    folder_path: str,
    video: bool,
    image: bool,
) -> None:

    if not image and not video:
        raise ValueError("At least one of make_image or make_video must be True.")

    if params.ndim == 1:
        params = params.reshape(1, -1)  # Reshape to a 2D array

    rd = RD_GPU(
        param_batch=params,
        grid_size=output_grid_size,
        seed=initial_state_seed_type,
        seed_radius=initial_state_seed_radius,
        pad_mode=pad_mode,
    )

    last_frame, all_frames = rd.run(update_steps, save_all=video, show_progress=False)

    if image:
        make_image(last_frame, filename, folder_path=folder_path)
    if video:
        make_video(
            list(all_frames[:, 0, :, :]), filename, folder_path=folder_path, reduction_rate=0.9
        )

    return


def make_grid(
    data: np.ndarray, losses: np.ndarray, generations: np.ndarray, filename: str, folder_path: str
) -> None:

    n_images = data.shape[0]

    # Special case: if there's only one image, make a standalone image
    if n_images == 1:
        make_image(data, filename, folder_path)
        return

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

    # Ensure ax is always a 2D array
    if n_rows * n_cols > 1:
        ax = np.array(ax).reshape(n_rows, n_cols)
    else:
        ax = np.array([[ax]])  # Single subplot case

    # Iterate over the grid to plot images
    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        ax[row, col].imshow(data[i], cmap="viridis")
        ax[row, col].set_title(f"Gen: {generations[i]}\nLoss: {losses[i]:.4f}", fontsize=8)

    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax[row, col].axis("off")

    # Turn off axes for all subplots
    for a in ax.flatten():
        a.axis("off")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return


def _apply_colormap(frame: np.ndarray, cmap: str) -> np.ndarray:
    """
    Apply a matplotlib colormap to a grayscale frame.

    Parameters:
    - frame: 2D numpy array representing a single frame.
    - cmap: String name of the matplotlib colormap.

    Returns:
    - RGB image as a numpy array with shape (height, width, 3).
    """
    import matplotlib.cm as cm

    colormap = cm.get_cmap(cmap)
    colored_frame = colormap(frame)
    # Convert from RGBA to RGB by ignoring the alpha channel
    colored_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)
    return colored_frame


def make_grid_video(
    data: np.ndarray,
    filename: str,
    cmap: str = "viridis",
    fps: int = 60,
    reduction_rate: float = 0.0,
    folder_path: str = "media",
    frame_size: tuple = (256, 256),
) -> None:
    """
    Create a grid video from multiple video data arrays.

    Parameters:
    - data: 5D numpy array with shape (frames, videos, height, width, channels).
    - filename: Output video filename (without extension).
    - cmap: Colormap for the video frames.
    - fps: Frames per second for the output video.
    - reduction_rate: Fraction to reduce the number of frames (0 <= reduction_rate < 1).
    - folder_path: Directory to save the video.
    - frame_size: Tuple indicating the size (width, height) to resize each video frame.
    """
    # Validate input data
    if data.ndim != 5 or data.shape[-1] != 3:
        raise ValueError(
            f"Data must be a 5D numpy array with RGB channels, but got shape {data.shape}."
        )

    total_frames, num_videos, height, width, _ = data.shape
    print(
        f"Data shape: (frames={total_frames}, videos={num_videos}, height={height}, width={width}, channels=3)"
    )

    if reduction_rate < 0 or reduction_rate >= 1:
        raise ValueError("Reduction rate must be in the range [0, 1).")

    # Apply frame reduction if needed
    if 0 < reduction_rate < 1:
        frames_to_keep = max(int(total_frames * (1 - reduction_rate)), 1)
        step = max(total_frames // frames_to_keep, 1)
        indices = list(range(0, total_frames, step))
        if indices[-1] != total_frames - 1:
            indices.append(total_frames - 1)
        data = data[indices]
        total_frames = data.shape[0]
        print(f"Reduced frames to {total_frames} using reduction_rate={reduction_rate}")

    import math
    import os

    import imageio
    from PIL import Image
    from tqdm import tqdm

    # Compute grid size (rows and columns) based on the number of videos
    grid_cols = math.ceil(math.sqrt(num_videos))
    grid_rows = math.ceil(num_videos / grid_cols)
    print(f"Computed grid size: {grid_rows} rows x {grid_cols} columns")

    # Initialize the video writer
    os.makedirs(folder_path, exist_ok=True)
    video_path = os.path.join(folder_path, f"{filename}.mp4")
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    # Preprocess all videos: resize frames
    print("Preprocessing videos...")
    processed_videos = []
    for vid in tqdm(range(num_videos), desc="Processing Videos"):
        video_frames = []
        for frm in range(total_frames):
            frame = data[frm, vid]  # Extract RGB frame
            pil_img = Image.fromarray(frame.cpu().numpy())  # No colormap needed
            pil_img = pil_img.resize(frame_size, Image.LANCZOS)
            video_frames.append(np.array(pil_img))
        processed_videos.append(video_frames)

    # Determine the size of the grid frame
    grid_width = frame_size[0] * grid_cols
    grid_height = frame_size[1] * grid_rows

    print("Assembling and writing video frames...")
    for frm_idx in tqdm(range(total_frames), desc="Writing Frames"):
        # Create a blank canvas for the grid
        grid_image = Image.new("RGB", (grid_width, grid_height), color=(0, 0, 0))

        for vid_idx in range(num_videos):
            row = vid_idx // grid_cols
            col = vid_idx % grid_cols
            top_left_x = col * frame_size[0]
            top_left_y = row * frame_size[1]
            frame = Image.fromarray(processed_videos[vid_idx][frm_idx])
            grid_image.paste(frame, (top_left_x, top_left_y))

        # If there are empty grid slots, they remain black
        writer.append_data(np.array(grid_image))

    writer.close()
    print(f"Grid video saved as {video_path}")


def visualize_volume(final_state: torch.Tensor) -> None:
    grid = pv.ImageData()
    grid.dimensions = final_state.shape  # (50, 50, 50)

    grid.origin = (0, 0, 0)  # optional
    grid.spacing = (1, 1, 1)  # optional
    grid.point_data["values"] = final_state.flatten(order="C")

    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="values",
        cmap="inferno",  # choose any colormap
        opacity="sigmoid",  # "sigmoid" "linear"
    )
    plotter.show()


def visualize_volume_video(
    data: torch.Tensor,
    output_filename: str = "rd3d_evolution.mp4",
    fps: int = 60,
    reduction_rate: float = 0.5,
) -> None:
    pv.start_xvfb()
    """
    Generates a video visualizing the evolution of a 3D volume over time.

    Args:
        data (torch.Tensor): The tensor of shape (n_steps, H, W, L) representing the evolution of the 3D reaction-diffusion system.
        output_filename (str): The name of the output video file.
        fps (int): Frames per second for the output video.
    """
    total_frames = data.shape[0]
    if 0 < reduction_rate < 1:
        frames_to_keep = max(int(total_frames * (1 - reduction_rate)), 1)
        step = max(total_frames // frames_to_keep, 1)
        indices = list(range(0, total_frames, step))
        if indices[-1] != total_frames - 1:
            indices.append(total_frames - 1)
        data = data[indices]
        total_frames = data.shape[0]
        print(f"Reduced frames to {total_frames} using reduction_rate={reduction_rate}")

    n_steps, H, W, L = data.shape

    # Create a plotter
    plotter = pv.Plotter(off_screen=True)
    grid = pv.ImageData()
    grid.dimensions = (H, W, L)
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)

    # Video writer
    writer = imageio.get_writer(output_filename, fps=fps)

    print(f"Rendering {n_steps} frames...")

    plotter.set_background("black")  # Set background to black
    plotter.hide_axes()  # Hide text and axes
    plotter.window_size = [512, 512]

    for t in tqdm(range(n_steps)):
        # Update volume data
        grid.point_data["values"] = data[t].flatten(order="C")

        plotter.clear()  # Clear previous frame
        plotter.add_volume(
            grid,
            show_scalar_bar=False,
            # scalars="values",
            cmap="inferno",  # Choose colormap
            opacity="sigmoid",  # Opacity function
        )

        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

    writer.close()
    print(f"Video saved to {output_filename}")


if __name__ == "__main__":
    pass
