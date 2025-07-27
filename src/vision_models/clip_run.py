from PIL import Image
import torch
import numpy as np
from typing import Callable
from transformers import CLIPImageProcessor, CLIPModel
import torch.nn.functional as F
from torchvision import transforms
import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)


def make_embedding_clip(
    images: np.array,
    model: Callable,
    processor: Callable,
    do_rescale: bool,
    device: str,
) -> np.array:

    # if images.max() <= 1:
    # print(f"You might be re-scaling images already in 0,1 range")

    if processor is not None:
        inputs = processor(images=images, return_tensors="pt", do_rescale=do_rescale)

    # Generate images embedding
    with torch.no_grad():
        image_embeddings = model.get_image_features(inputs["pixel_values"].to(device))

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

    image_embedding = image_embeddings.cpu().numpy()  # Move back to CPU before converting to numpy

    return image_embedding


if __name__ == "__main__":

    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True
    )

    # Load the image
    image_path = "blastocyst_instances/3000_5.png"
    # image_path = "data/image.png"
    image = Image.open(image_path)

    # Convert image to numpy array
    image = image.convert("RGB")
    image = np.array(image)

    # Create a list of batch aimges
    images = [image] * 5

    # Prepare the image for the model
    inputs = processor(images=images, return_tensors="pt")

    # Generate image embedding
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    # Convert to numpy for easier handling
    image_embedding = image_embeddings.numpy()

    # Print the embedding
    print(image_embedding)
    print(image_embedding.shape)
