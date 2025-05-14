import os
import sys
import random
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_folder(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return


def seed_python_numpy_torch_cuda(seed: int):
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(2**32, size=1)[0])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeeded with {seed}")
    return seed


def load_image_as_tensor(
    image_path,
    resize=None,
    normalize=False,
    color=False,
    minmax_target_image=False,
    reverse_image=False,
    swap_channels=False,
    negative=False,
):
    """
    Load an image from a file and convert it to a PyTorch tensor.

    Args:
        image_path (str): Path to the image file
        normalize (bool): If True, normalize values to [0, 1] range
        resize (tuple, optional): If provided, resize the image to (width, height)

    Returns:
        torch.Tensor: The image as a tensor of shape (3, height, width) or (1, height, width)
    """

    if color:
        raise NotImplementedError("Color loading not implemented")
    if minmax_target_image:
        raise NotImplementedError("Minmax on loaded image not implemented")
    if reverse_image:
        raise NotImplementedError("Reverse image not implemented")

    # Read image using PIL
    img = Image.open(image_path)

    # Resize if requested
    if resize is not None:
        img = img.resize(resize, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        # Add channel dimension
        img_array = img_array[:, :, np.newaxis]
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        # Remove alpha channel
        img_array = img_array[:, :, :3]

    # Transpose dimensions from (H, W, C) to (C, H, W)
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

    # Covert to negative
    if negative:
        img_tensor = 255 - img_tensor

    # Convert to float
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float()
        if normalize:
            img_tensor /= 255.0

    if swap_channels:
        img_tensor = img_tensor.permute(1, 2, 0)

    return img_tensor


if __name__ == "__main__":
    pass
