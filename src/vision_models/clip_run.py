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
    normalise: bool,
    do_rescale: bool,
    device: str,
) -> np.array:


    if processor is not None:
        inputs = processor(images=images, return_tensors="pt", do_rescale=do_rescale)
    elif images.shape[-1] == 3 and len(images.shape) == 3:
        inputs = images.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        # if inputs.dtype == torch.uint8:
        # inputs = inputs.float() / 255.0
        if inputs.shape[-1] != 224:
            inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear", align_corners=False)

    elif images.shape[-1] == 3 and len(images.shape) == 4:
        inputs = images.permute(0, 3, 1, 2)  # (B, 3, H, W)
        # if inputs.dtype == torch.uint8:
        # inputs = inputs.float() / 255.0
        if inputs.shape[-1] != 224:
            inputs = F.interpolate(inputs, size=(224, 224), mode="bilinear", align_corners=False)

    else:
        raise ValueError("images must be either 3 (unbatched) or 4 (batched) dimensional")

    inputs = inputs.to(device)

    if do_rescale:        
        inputs = inputs / 255
    
    if normalise:
        if inputs.max() > 1:
            raise ValueError("Images have to be in range (0,1) if to be normalised")
        OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        normalise_func = transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
        inputs = normalise_func(inputs)

    # Generate images embedding
    with torch.no_grad():
        image_embeddings = model.get_image_features(inputs)

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

    image_embedding = image_embeddings.cpu().numpy()  # Move back to CPU before converting to numpy

    return image_embedding


# def make_embedding_clip(
#     images: np.array,
#     do_rescale: bool,
#     device: str = "cpu",
#     processor: Callable = None,
#     model: Callable = None,
# ) -> np.array:

#     # Prepare the image for the model
#     inputs = processor(images=images, return_tensors="pt", do_rescale=do_rescale)
#     inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU if available

#     # Generate image embedding
#     with torch.no_grad():
#         image_embeddings = model.get_image_features(**inputs)

#     # Convert to numpy for easier handling
#     image_embedding = image_embeddings.cpu().numpy()  # Move back to CPU before converting to numpy

#     image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)

#     return image_embedding


# def _gpu_experiment():
#     import time
#     import matplotlib.pyplot as plt

#     batches = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

#     t_cpu = []
#     t_gpu = []
#     for batch in batches:
#         images = np.random.random((batch, 3, 224, 224))
#         start = time.time()
#         image_embedding = make_embedding_clip(images, do_rescale=False, gpu=True)
#         end = time.time()
#         t_gpu.append(end - start)
#         print(f"Batch size: {batch}, GPU time: {end - start}")

#         start = time.time()
#         image_embedding = make_embedding_clip(images, do_rescale=False, gpu=False)
#         end = time.time()
#         t_cpu.append(end - start)
#         print(f"Batch size: {batch}, CPU time: {end - start}")

#     print("GPU times:", t_gpu)
#     print("CPU times:", t_cpu)

#     plt.plot(batches, t_gpu, label="GPU")
#     plt.plot(batches, t_cpu, label="CPU")
#     plt.xlabel("Batch size")
#     plt.ylabel("Time (s)")
#     plt.legend()
#     plt.savefig("gpu_vs_cpu.png")


if __name__ == "__main__":

    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True
    )

    # Load the image
    image_path = "data/image.png"
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
