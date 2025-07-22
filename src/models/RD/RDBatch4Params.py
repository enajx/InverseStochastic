import os
import sys

from scipy.ndimage import zoom

# ad this folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@torch.jit.script
def laplace_2d_gpu(arr: torch.Tensor, kernel: torch.Tensor, pad_mode: str) -> torch.Tensor:
    batch_size, H, W = arr.shape
    arr_padded = torch.nn.functional.pad(arr, (1, 1, 1, 1), mode=pad_mode)
    out = torch.zeros_like(arr)
    for b in range(batch_size):
        grid = arr_padded[b].unsqueeze(0).unsqueeze(0)  # shape [1,1,H+2,W+2]
        out[b] = torch.nn.functional.conv2d(grid, kernel)[0, 0]
    return out


@torch.jit.script
def rd_step_gpu(
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor,
    kernel: torch.Tensor,
    pad_mode: str,
) -> None:

    lap_A = laplace_2d_gpu(A, kernel, pad_mode)
    lap_B = laplace_2d_gpu(B, kernel, pad_mode)

    # Reshape dA, dB, f, and k to match the shape of A and B
    dA = dA.view(-1, 1, 1).expand_as(A)
    dB = dB.view(-1, 1, 1).expand_as(B)
    f = f.view(-1, 1, 1).expand_as(A)
    k = k.view(-1, 1, 1).expand_as(B)

    # Perform the reaction-diffusion step
    A += dA * lap_A - A * B * B + f * (1.0 - A)
    A.clamp_(0, 1)
    B += dB * lap_B + A * B * B - (k + f) * B
    B.clamp_(0, 1)

    # detect if nan or ing is present
    if torch.isnan(A).any() or torch.isnan(B).any() or torch.isinf(A).any() or torch.isinf(B).any():
        print("NAN or INF detected")
        print(torch.max(A), torch.min(A), torch.max(B), torch.min(B))


@torch.jit.script
def rd_step_gpu_anisotropic(
    A: torch.Tensor,
    B: torch.Tensor,
    dAx: torch.Tensor,
    dAy: torch.Tensor,
    dBx: torch.Tensor,
    dBy: torch.Tensor,
    f: torch.Tensor,
    k: torch.Tensor,
    kernelx: torch.Tensor,
    kernely: torch.Tensor,
    pad_mode: str,
) -> None:

    lap_Ax = laplace_2d_gpu(A, kernelx, pad_mode)
    lap_Ay = laplace_2d_gpu(A, kernely, pad_mode)
    lap_Bx = laplace_2d_gpu(B, kernelx, pad_mode)
    lap_By = laplace_2d_gpu(B, kernely, pad_mode)

    # Reshape dA, dB, f, and k to match the shape of A and B
    dAx = dAx.view(-1, 1, 1).expand_as(A)
    dAy = dAy.view(-1, 1, 1).expand_as(A)
    dBx = dBx.view(-1, 1, 1).expand_as(B)
    dBy = dBy.view(-1, 1, 1).expand_as(B)
    f = f.view(-1, 1, 1).expand_as(A)
    k = k.view(-1, 1, 1).expand_as(B)

    # Perform the reaction-diffusion step
    A += dAx * lap_Ax + dAy * lap_Ay - A * B * B + f * (1.0 - A)
    A.clamp_(0, 1)
    B += dBx * lap_Bx + dBy * lap_By + A * B * B - (k + f) * B
    B.clamp_(0, 1)

    # detect if nan or ing is present
    if torch.isnan(A).any() or torch.isnan(B).any() or torch.isinf(A).any() or torch.isinf(B).any():
        print("NAN or INF detected")
        print(torch.max(A), torch.min(A), torch.max(B), torch.min(B))


class RD_GPU:
    def __init__(
        self,
        param_batch: np.array,
        grid_size: tuple,
        seed: str,
        seed_radius: int,
        pad_mode: str,
        anisotropic: bool,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if pad_mode is None:
            raise ValueError(
                "pad_mode should be one of ['constant', 'reflect', 'replicate', 'circular']."
            )

        self.pad_mode = pad_mode
        self.anisotropic = anisotropic

        # anisotropic components
        if anisotropic:
            self.Laplacian_x = (
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.2, -0.4, 0.2], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # shape (1,1,3,3)

            self.Laplacian_y = (
                torch.tensor(
                    [[0.0, 0.2, 0.0], [0.0, -0.4, 0.0], [0.0, 0.2, 0.0]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # shape (1,1,3,3)
        else:
            self.Laplacian = (
                torch.tensor(
                    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # shape (1,1,3,3)

        self.grid_size = grid_size
        if param_batch is not None:
            self.reset(param_batch, seed, seed_radius, anisotropic)
        else:
            raise ValueError("Must provide a parameter batch to initialize the RD process")

    def reset(
        self,
        param_batch: np.array,
        seed: str,
        seed_radius: int,
        anisotropic: bool,
    ):
        """
        Reset all state variables for a new batch of parameters
        without having to reinitialize the class.
        """
        if seed not in [
            "single",
            "triple",
            "circle",
            "random",
            "noise",
            "random_oval",
            "random_rectangle",
            "random_circle",
            "random_circle_small",
            "random_cross",
            "cross",
            "random_thin_stripes",
            "random_thick_stripes",
            "random_square",
            "random_triangle",
        ]:
            raise ValueError(
                "seed should be one of ['single', 'circle', 'triple', 'random', 'noise', 'random_circle', 'random_oval', 'random_rectagle', 'random_circle_small']."
            )

        # Convert parameters to torch tensors and move to GPU
        if anisotropic:
            self.dAsx = torch.tensor(param_batch[:, 0:1], dtype=torch.float32, device=self.device)
            self.dAsy = torch.tensor(param_batch[:, 1:2], dtype=torch.float32, device=self.device)
            self.dBsx = torch.tensor(param_batch[:, 2:3], dtype=torch.float32, device=self.device)
            self.dBsy = torch.tensor(param_batch[:, 3:4], dtype=torch.float32, device=self.device)
            self.fs = torch.tensor(param_batch[:, 4:5], dtype=torch.float32, device=self.device)
            self.ks = torch.tensor(param_batch[:, 5:6], dtype=torch.float32, device=self.device)
        else:
            self.dAs = torch.tensor(param_batch[:, 0:1], dtype=torch.float32, device=self.device)
            self.dBs = torch.tensor(param_batch[:, 1:2], dtype=torch.float32, device=self.device)
            self.fs = torch.tensor(param_batch[:, 2:3], dtype=torch.float32, device=self.device)
            self.ks = torch.tensor(param_batch[:, 3:4], dtype=torch.float32, device=self.device)

        # Initialize grids on GPU
        batch_size = param_batch.shape[0]
        H, W = self.grid_size
        self.A = torch.ones((batch_size, H, W), dtype=torch.float32, device=self.device)
        self.B = torch.zeros((batch_size, H, W), dtype=torch.float32, device=self.device)

        if seed == "single":  # Initialize B with one circle
            self.B[:, H // 2, W // 2] = 1.0

        elif seed == "circle":  # Initialize B with a circle
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )
            mask = (y - cy) ** 2 + (x - cx) ** 2 < seed_radius**2
            self.B[:, mask] = 1.0

        elif seed == "triple":  # Initialize B with three circles
            spacing = W // 4
            centers_x = [spacing, 2 * spacing, 3 * spacing]
            cy = H // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )
            for cx in centers_x:
                mask = (y - cy) ** 2 + (x - cx) ** 2 < seed_radius**2
                self.B[:, mask] = 1.0

        elif seed == "random":
            gen = torch.Generator(device=self.device)
            gen.seed()  # Generates a random seed automatically
            sensibility = 0.8  # 0.8
            rx = torch.sqrt(torch.tensor(self.B.shape[1], dtype=torch.float32)).to(torch.int32)
            ry = torch.sqrt(torch.tensor(self.B.shape[2], dtype=torch.float32)).to(torch.int32)
            random_seed = (
                (torch.rand((batch_size, rx, ry), generator=gen, device=self.device) > sensibility)
                .float()
                .cpu()
                .numpy()
            )

            scale_factors = (
                1,  # Keep batch dimension unchanged
                self.B.shape[1] / random_seed.shape[1],  # Scale height
                self.B.shape[2] / random_seed.shape[2],  # Scale width
            )

            # Expand using scipy.ndimage.zoom
            expanded_states = np.clip(zoom(random_seed, zoom=scale_factors, order=3), 0, 1)

            if False:
                for i in range(batch_size):
                    # imshow
                    plt.imshow(expanded_states[i])
                    plt.show()

            self.B[:] = torch.tensor(expanded_states, device=self.device, dtype=self.B.dtype)

            # import matplotlib.pyplot as plt
            # plt.imshow(self.B[0].to(torch.float32).cpu().numpy(), cmap='Greys')
            ##remove axis
            # plt.axis('off')
            # plt.savefig('media/noise.png', bbox_inches='tight', pad_inches=0)

        elif seed == "noise":
            print("Generating Perlin noise")
            from perlin_noise import PerlinNoise

            N = PerlinNoise(octaves=3, seed=np.random.randint(0, 1000))
            x, y = self.B.shape[1], self.B.shape[2]
            noise = [[N([i / x, j / y]) for j in range(y)] for i in range(x)]
            noise = np.array(noise)
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            noise = np.clip(noise, 0, 1)
            noise = np.expand_dims(noise, axis=0)
            noise = np.repeat(noise, batch_size, axis=0)

            self.B[:] = torch.tensor(noise, device=self.device, dtype=self.B.dtype)

            import matplotlib.pyplot as plt

            plt.imshow(self.B[0].to(torch.float32).cpu().numpy(), cmap="Greys")
            # remove axis
            plt.axis("off")
            plt.savefig("media/noise.png", bbox_inches="tight", pad_inches=0)

        elif seed == "random_oval":  # Initialize B with a random oval
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )

            # Random ratios for width and height
            gen = torch.Generator(device=self.device)
            gen.seed()

            rx_ratio = torch.empty(1, device=self.device).uniform_(0.5, 1.5).item()
            ry_ratio = torch.empty(1, device=self.device).uniform_(0.5, 1.5).item()

            rx = seed_radius * rx_ratio
            ry = seed_radius * ry_ratio

            # Equation of an ellipse: ((y-cy)/ry)^2 + ((x-cx)/rx)^2 < 1
            mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1
            self.B[:, mask] = 1.0

        elif (
            seed == "random_rectangle"
        ):  # Initialize B with a rectangle centered, random axis ratio
            gen = torch.Generator(device=self.device)
            gen.seed()

            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )

            # Center of the field
            cy, cx = H // 2, W // 2

            # Random width and height based on seed_radius and random ratios
            width_ratio = torch.empty(1, device=self.device).uniform_(0.5, 1.5).item()
            height_ratio = torch.empty(1, device=self.device).uniform_(0.5, 1.5).item()
            half_width = int(seed_radius * width_ratio)
            half_height = int(seed_radius * height_ratio)

            # Define rectangle bounds
            y_min = max(cy - half_height, 0)
            y_max = min(cy + half_height, H)
            x_min = max(cx - half_width, 0)
            x_max = min(cx + half_width, W)

            # Fill the centered rectangle
            self.B[:, y_min:y_max, x_min:x_max] = 1.0

        elif seed == "random_circle":  # Initialize B with a randomly-sized circle
            gen = torch.Generator(device=self.device)
            gen.seed()

            # Generate random radius between 0.1 and 10 times seed_radius
            radius_scale = torch.empty(1, device=self.device).uniform_(1, 5.0).item()
            rand_radius = seed_radius * radius_scale

            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )

            mask = (y - cy) ** 2 + (x - cx) ** 2 < rand_radius**2
            self.B[:, mask] = 1.0

        elif seed == "random_circle_small":
            gen = torch.Generator(device=self.device)
            gen.seed()

            # Generate random radius between 0.1 and 10 times seed_radius
            radius_scale = torch.empty(1, device=self.device).uniform_(1, 2.0).item()
            print(radius_scale)
            rand_radius = seed_radius * radius_scale

            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )

            mask = (y - cy) ** 2 + (x - cx) ** 2 < rand_radius**2
            self.B[:, mask] = 1.0

        elif seed == "random_cross":  # Initialize B with a rotated, randomly-sized cross
            gen = torch.Generator(device=self.device)
            gen.seed()

            import math

            # Random size factors
            arm_length = seed_radius * torch.empty(1, device=self.device).uniform_(20, 20.1).item()
            arm_width = seed_radius * torch.empty(1, device=self.device).uniform_(0.2, 0.21).item()
            # arm_width = seed_radius * torch.empty(1, device=self.device).uniform_(0.1, 0.15).item()

            # Create coordinate grid
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device) - cy,
                torch.arange(W, device=self.device) - cx,
                indexing="ij",
            )

            # Random rotation angle in radians
            angle = torch.empty(1, device=self.device).uniform_(0, 1 * math.pi).item()
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Rotate coordinates
            x_rot = cos_a * x + sin_a * y
            y_rot = -sin_a * x + cos_a * y

            # Define cross arms in rotated space
            mask = ((torch.abs(x_rot) < arm_width / 2) & (torch.abs(y_rot) < arm_length / 2)) | (
                (torch.abs(y_rot) < arm_width / 2) & (torch.abs(x_rot) < arm_length / 2)
            )

            self.B[:, mask] = 1.0

        elif seed == "random_triangle":
            import math

            gen = torch.Generator(device=self.device)
            gen.seed()

            cy, cx = H // 2, W // 2
            scale = seed_radius * torch.empty(1, device=self.device).uniform_(1.0, 2.0).item()

            # Equilateral triangle vertices before rotation
            vertices = torch.tensor(
                [
                    [0, -scale],
                    [-scale * math.sin(math.pi / 3), scale / 2],
                    [scale * math.sin(math.pi / 3), scale / 2],
                ],
                device=self.device,
            )

            # Random rotation
            angle = torch.empty(1, device=self.device).uniform_(0, 2 * math.pi).item()
            rot = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ],
                device=self.device,
            )
            vertices = (rot @ vertices.T).T

            # Shift to center
            vertices += torch.tensor([cx, cy], device=self.device)

            # Create mask using barycentric coordinates
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )
            p = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)

            def edge_fn(v0, v1, p):
                return (p[:, 0] - v0[0]) * (v1[1] - v0[1]) - (p[:, 1] - v0[1]) * (v1[0] - v0[0])

            w0 = edge_fn(vertices[1], vertices[2], p)
            w1 = edge_fn(vertices[2], vertices[0], p)
            w2 = edge_fn(vertices[0], vertices[1], p)

            mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            mask = mask.reshape(H, W)
            self.B[:, mask] = 1.0

        elif seed == "random_square":
            import math

            gen = torch.Generator(device=self.device)
            gen.seed()

            cy, cx = H // 2, W // 2
            side = seed_radius * torch.empty(1, device=self.device).uniform_(1.0, 2.0).item()
            half = side / 2

            # Coordinates before rotation
            coords = torch.tensor(
                [
                    [-half, -half],
                    [-half, half],
                    [half, half],
                    [half, -half],
                ],
                device=self.device,
            )

            # Random rotation
            angle = torch.empty(1, device=self.device).uniform_(0, 2 * math.pi).item()
            rot = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ],
                device=self.device,
            )
            coords = (rot @ coords.T).T + torch.tensor([cx, cy], device=self.device)

            # Bounding box rasterization
            from matplotlib.path import Path

            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing="ij",
            )
            coords_grid = torch.stack([x_grid, y_grid], dim=-1).cpu().numpy().reshape(-1, 2)
            square_path = Path(coords.cpu().numpy())
            mask = square_path.contains_points(coords_grid).reshape(H, W)
            self.B[:, torch.tensor(mask, device=self.device)] = 1.0
        elif seed == "random_thin_stripes" or seed == "random_thick_stripes":
            import math

            gen = torch.Generator(device=self.device)
            gen.seed()

            # Adjust stripe width based on seed
            if seed == "random_thin_stripes":
                stripe_width = seed_radius * 1.0
            else:  # "random_thick_stripes"
                stripe_width = seed_radius * 4.0

            # Random angle
            angle = torch.empty(1, device=self.device).uniform_(0, 2 * math.pi).item()
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # Grid centered
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(
                torch.arange(H, device=self.device) - cy,
                torch.arange(W, device=self.device) - cx,
                indexing="ij",
            )

            # Rotate grid
            x_rot = cos_a * x + sin_a * y

            # Create stripes
            mask = ((x_rot / stripe_width) % 2) < 1
            self.B[:, mask] = 1.0

    def step(self, anisotropic):
        """
        Perform a single step of the reaction-diffusion process on the GPU.
        """
        if anisotropic:
            rd_step_gpu_anisotropic(
                self.A,
                self.B,
                self.dAsx,
                self.dAsy,
                self.dBsx,
                self.dBsy,
                self.fs,
                self.ks,
                self.Laplacian_x,
                self.Laplacian_y,
                self.pad_mode,
            )
        else:
            rd_step_gpu(
                self.A, self.B, self.dAs, self.dBs, self.fs, self.ks, self.Laplacian, self.pad_mode
            )

    #        print(f"A out: {float(self.A.max())}")
    #        print(f"ID A: {id(self.A)}")
    #        print(f"B out: {float(self.B.max())}")
    #        print(f"ID B: {id(self.B)}")
    #        print("\n")

    def run(
        self,
        n_steps: int,
        minimax_RD_output: bool,
        save_all=False,
        show_progress=False,
    ):
        """
        Run the reaction-diffusion process for a given number of steps.
        """
        data = [self.B.clone()] if save_all else None

        if show_progress:
            for _ in tqdm(range(n_steps)):
                self.step(anisotropic=self.anisotropic)
                if save_all:
                    data.append(self.B.clone())
        else:
            for _ in range(n_steps):
                self.step(anisotropic=self.anisotropic)
                if save_all:
                    data.append(self.B.clone())

        # Post-process
        if minimax_RD_output:
            B_max = self.B.amax(dim=(1, 2), keepdim=True)
            B_min = self.B.amin(dim=(1, 2), keepdim=True)
            self.B = (self.B - B_min) / (B_max - B_min + 1e-8)

        # Convert to RGB uint8
        self.B = torch.unsqueeze(self.B, dim=-1)  # Add channel dimension
        self.B = (torch.repeat_interleave(self.B, repeats=3, dim=-1) * 255).to(torch.uint8)

        # Apply normalization to stacked data if enabled
        stacked_data = torch.unsqueeze(torch.stack(data), dim=-1) if save_all else None

        if stacked_data is not None and minimax_RD_output:
            stacked_data_max = stacked_data.amax(dim=(1, 2), keepdim=True)
            stacked_data_min = stacked_data.amin(dim=(1, 2), keepdim=True)
            stacked_data = (stacked_data - stacked_data_min) / (
                stacked_data_max - stacked_data_min + 1e-8
            )

        if stacked_data is not None:
            stacked_data = torch.repeat_interleave(stacked_data, repeats=3, dim=-1) * 255
            stacked_data = (
                torch.clamp(stacked_data, 0, 255).to(torch.uint8)
                if stacked_data is not None
                else None
            )

        return self.B, stacked_data


def selected_params():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from visualisations.visual_tools import make_grid_video

    GRID_SIZE = (100, 100)
    STEPS = 2000
    SEED = "random"
    SEED_RADIUS = 5
    PAD_MODE = "circular"

    parameters = np.array(
        [
            [1.0, 0.5, 0.055, 0.062],
            [1.0, 0.3, 0.029, 0.057],
            [0.5903167655090454, 0.23283340087519355, 0.0651899268206337, 0.03765790956696743],
            # [0.4804, 0.0058, 0.1323, 0.0004],
            [1.0, 0.39, 0.03, 0.06],
            [1.0, 0.5, 0.02, 0.05],
            [1.0, 0.5, 0.023, 0.055],
            [0.9, 0.31, 0.04, 0.060],
            [1.0, 0.5, 0.04, 0.065],
            [1.0, 0.5, 0.04001, 0.065002],
            # [0.3266, 0.9991, 0.3001, 0.4669],
            # [0.0292, 0.0072, 0.128,  0.0005],
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            # [0.3151, 0.1127, 0.0111, 0.0442], # BZ ???
            [0.2294, 0.2925, 0.2067, 0.0018],  # Big
            [0.8913302510365129, 0.3018495344749094, 0.06370998930656666, 0.05438057014787519],
            [0.9521489674548775, 0.00521321009526779, 5.473165367852129e-06, 0.0030694849813222213],
            [0.979040, 0.249930, 0.039302, 0.059738],
            [1.0, 0.1, 0.028, 0.062],
            [0.955216, 0.158981, 0.041567, 0.063682],
            [1.000000, 0.343817, 0.052043, 0.063162],
        ],
        dtype=np.float64,
    )
    parameters2 = np.array(
        [
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
        ],
        dtype=np.float64,
    )

    rd = RD_GPU(
        parameters,
        grid_size=GRID_SIZE,
        seed=SEED,
        seed_radius=SEED_RADIUS,
        pad_mode=PAD_MODE,
        anisotropic=False,
    )
    last_frame, data = rd.run(STEPS, save_all=True, show_progress=True)
    make_grid_video(
        data=data,
        filename="my favorite tunes",
        cmap="viridis",
        fps=120,
        reduction_rate=0.5,
        folder_path="media",
        frame_size=(256, 256),
    )


def replicate_nice_results():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from visualisations.visual_tools import make_grid_video

    GRID_SIZE = (100, 100)
    STEPS = 1000
    SEED = "random"
    SEED_RADIUS = 5
    PAD_MODE = "circular"

    parameters = np.array(
        [
            # [0.4176305749353404, 0.05927126517290736, 0.024480070195420568, 0.0002900052920793725], # good gecco
            [0.9953, 0.0028, 0.1804, 0.6891, 0.0331, 0.0625],  # some zebra
            [
                0.9952916672544577,
                0.0028354643341353346,
                0.18042228330553545,
                0.6890920964915667,
                0.03312062438646195,
                0.062496750905907145,
            ],
            [
                0.8715738112775989,
                0.18272911964287777,
                0.12085826326549523,
                0.3483329030312582,
                0.022317162298573882,
                0.016134824889468484,
            ],  # Zebra
            [
                0.6403485941901343,
                0.1772305695430292,
                0.03308397197889,
                0.02786666445146474,
            ],  # gecko
            [
                0.7283757618153087,
                0.1867743493915589,
                0.06941691296235403,
                0.02751044585930677,
            ],  # gecko
            [
                0.31912433040360616,
                0.11260253256185057,
                0.12063423354042015,
                0.00027876343873553213,
            ],  # gecko
            [
                0.583348325083343,
                0.24885376416971522,
                0.09326780020878378,
                0.01496979603918121,
            ],  # Leopard 1
            [
                0.5984694985228396,
                0.2142295817266098,
                0.06266365532339285,
                0.01841777636541918,
            ],  # Leopard1
            [1.0, 0.5, 0.055, 0.062],
            # [0.48991791658836625, 0.23525082680244488, 0.09375742826767684, 0.017276873514912496],#Leopard1
            # [0.6619348353980437, 0.2156411419120617, 0.07556440527025164, 0.0415121401020364], #Leopard1
            # [0.7527125870417704, 0.23770421738900707, 0.0578506381564788, 0.0498999100404883], #Leopard2
        ],
        dtype=np.float64,
    )

    from visualisations.visual_tools import make_image

    for i in range(len(parameters)):
        anisotropic = True if len(parameters[i]) == 6 else False
        rd = RD_GPU(
            np.array([parameters[i]]),
            grid_size=GRID_SIZE,
            seed=SEED,
            seed_radius=SEED_RADIUS,
            pad_mode=PAD_MODE,
            anisotropic=anisotropic,  # Make true for zebra
        )

        last_frame, data = rd.run(STEPS, save_all=True, show_progress=True, minimax_RD_output=True)

        filename = (f"frame{i}_{PAD_MODE}_{SEED}_{np.random.randint(0, 1000)}",)
        # filename = "heyyyyyy"
        make_image(
            last_frame[0],
            filename=str(filename),
            folder_path="media",
        )

        make_grid_video(
            data=data.squeeze(0),
            filename=str(filename),
            fps=120,
            reduction_rate=0.5,
            folder_path="media",
            frame_size=GRID_SIZE,
        )

    return
    make_grid_video(
        data=data,
        filename="Animals",
        fps=120,
        reduction_rate=0.5,
        folder_path="media",
        frame_size=(256, 256),
    )


def make_image_examples():

    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from visualisations.visual_tools import make_image
    from visualisations.visual_tools import make_grid_video

    GRID_SIZE = (100, 100)
    STEPS = 1000
    SEED = "random"
    SEED_RADIUS = 5
    PAD_MODE = "circular"

    parameters = np.array(
        [
            [1.0, 0.5, 0.055, 0.062],
            [1.0, 0.3, 0.029, 0.057],
            [0.5903167655090454, 0.23283340087519355, 0.0651899268206337, 0.03765790956696743],
            [1.0, 0.39, 0.03, 0.06],
            # [1.0, 0.5, 0.02, 0.05],
            # [1.0, 0.5, 0.023, 0.055],
            # [0.9, 0.31, 0.04, 0.060],
            # [1.0, 0.5, 0.04, 0.065],
            # [1.0, 0.5, 0.04001, 0.065002],
            # [0.3266, 0.9991, 0.3001, 0.4669],
            # [0.0292, 0.0072, 0.128,  0.0005],
            [0.9695, 0.018, 0.0081, 0.0088],  # cool hexagonal lizard seed
            # [0.3151, 0.1127, 0.0111, 0.0442], # BZ ???
            # [0.2294, 0.2925, 0.2067, 0.0018],  # Big
            [0.8913302510365129, 0.3018495344749094, 0.06370998930656666, 0.05438057014787519],
            [0.9521489674548775, 0.00521321009526779, 5.473165367852129e-06, 0.0030694849813222213],
            # [0.979040, 0.249930, 0.039302, 0.059738],
            # [1.0, 0.1, 0.028, 0.062],
            [0.955216, 0.158981, 0.041567, 0.063682],
            # [1.000000, 0.343817, 0.052043, 0.063162],
            #
        ],
        dtype=np.float64,
    )

    # Create figure with multiple columns (1 row)
    fig, axes = plt.subplots(
        1, len(parameters), figsize=(len(parameters) * 4, 4)
    )  # Adjust width dynamically

    if len(parameters) == 1:
        axes = [axes]  # Ensure axes is iterable when only one subplot exists

    for i in range(len(parameters)):
        anisotropic = True if len(parameters[i]) == 6 else False
        rd = RD_GPU(
            np.array([parameters[i]]),
            grid_size=GRID_SIZE,
            seed=SEED,
            seed_radius=SEED_RADIUS,
            pad_mode=PAD_MODE,
            anisotropic=anisotropic,
        )

        last_frame, data = rd.run(STEPS, save_all=True, show_progress=True, minimax_RD_output=True)
        # lastframe to cpu
        last_frame = last_frame[0].to(torch.float32).cpu().numpy() / 255.00
        # Plot the last frame in the corresponding subplot
        axes[i].imshow(last_frame, cmap="gray")  # Change cmap if needed
        axes[i].axis("off")
        # axes[i].set_title(f"Run {i+1}")

    # Save the figure as a PDF
    plt.tight_layout()
    plt.savefig("rd_runs.pdf", format="pdf", bbox_inches="tight")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    # Parameters from:  https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/#:~:text=The%20Gray%2DScott%20system%20is,used%20up%2C%20while%20is%20produced.
    # and from:         https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Introduction_to_the_Modeling_and_Analysis_of_Complex_Systems_(Sayama)/13%3A_Continuous_Field_Models_I__Modeling/13.06%3A_Reaction-Diffusion_Systems

    # replicate_nice_results()
    # import sys

    # sys.exit()

    # sys.exit()
    GRID_SIZE = (50, 50)
    STEPS = 1000
    SEED = "random"
    SEED_RADIUS = 5
    PAD_MODE = "circular"

    ## # da db f k
    # parameters = np.array(
    #     [
    #         # duly-cool-ghoul
    #         # [1, 0.5, 0.055, 0.062],
    #         # [0.9995, 0.4805, 0.0745, 0.0619],
    #         # [0.9995, 0.4805, 0.0745, 0.0619],
    #         # [0.9995, 0.4805, 0.0745, 0.0619],
    #         # yearly-heroic-sponge
    #         # [0.97904, 0.24993, 0.039302, 0.059738],
    #         # [0.9996, 0.2584, 0.0791, 0.0625],
    #         # [0.9996, 0.2584, 0.0791, 0.0625],
    #         # [0.9996, 0.2584, 0.0791, 0.0625],
    #         # hardly-model-weevil
    #         # [1, 0.1, 0.028, 0.062],
    #         # [0.9903, 0.1113, 0.0285, 0.0662],
    #         # [0.9903, 0.1113, 0.0285, 0.0662],
    #         # [0.9903, 0.1113, 0.0285, 0.0662],
    #         #
    #         # [0.6763, 0.187, 0.068, 0.0293],
    #         # [0.6743, 0.1865, 0.0675, 0.0288],
    #         [0.8914, 0.411, 0.0743, 0.0307],
    #     ],
    #     dtype=np.float64,
    # )
    parameters = np.array(
        [
            [1.0, 0.5, 0.055, 0.062],
            [1.0, 0.3, 0.029, 0.057],
            # [1.0, 0.1, 0.028, 0.062],
            # [1.0, 0.39, 0.03, 0.06],
            # [1.0, 0.5, 0.02, 0.05],
            # [1.0, 0.5, 0.023, 0.055],
            # [0.9, 0.31, 0.04, 0.060],
            # [1.0, 0.5, 0.04, 0.065],
            # [0.979040, 0.249930, 0.039302, 0.059738],
            # [0.955216, 0.158981, 0.041567, 0.063682],
            # [1.000000, 0.343817, 0.052043, 0.063162],
        ],
        dtype=np.float64,
    )

    # # dax day dbx dby f k
    # parameters = np.array(
    #     [
    #         [1.0, 0.1, 0.1, 0.4, 0.055, 0.062],
    #         [1.0, 0.1, 0.3, 0.3, 0.029, 0.057],
    #         [1.0, 0.1, 0.1, 0.3, 0.028, 0.062],
    #     ],
    #     dtype=np.float64,
    # )

    ANISOTROPIC = True if parameters.shape[1] == 6 else False
    print(f"Anisotropic: {ANISOTROPIC}")

    rd = RD_GPU(
        parameters,
        grid_size=GRID_SIZE,
        seed=SEED,
        seed_radius=SEED_RADIUS,
        pad_mode=PAD_MODE,
        anisotropic=ANISOTROPIC,
    )
    last_frame, data = rd.run(STEPS, save_all=False, show_progress=True, minimax_RD_output=True)

    print(f"\nLast frame max: {last_frame.max()}")
    print(last_frame.min())

    fig, ax = plt.subplots(nrows=1, ncols=len(last_frame))
    for i in range(len(last_frame)):
        ax[i].imshow(last_frame[i], cmap="inferno")
        ax[i].set_title(f"Frame {i}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.show()

    # # save each frame individually
    # for i in range(len(last_frame)):
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     ax.imshow(last_frame[i], cmap="inferno")
    #     # disable ticks and labels
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     plt.savefig(f"results/frame{i}.png", bbox_inches="tight", pad_inches=0)
