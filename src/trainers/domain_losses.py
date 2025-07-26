import torch
import torch.fft
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
import numpy as np
from scipy.linalg import sqrtm
import pywt


def to_grayscale(img):
    return img.float().mean(dim=-1)  # [B, H, W]


def compute_radial_power_spectrum(img):
    # Range: ≥ 0, unbounded
    # Sensitive to both scale and overall spectral shape differences.
    # Typical values depend on image intensity scale; with uint8, expect ~10–500 range depending on how different the patterns are.
    B, H, W = img.shape
    fft = torch.fft.fft2(img)
    power = torch.abs(fft) ** 2
    power = torch.fft.fftshift(power)

    y = torch.arange(-H // 2, H // 2).view(-1, 1).expand(H, W)
    x = torch.arange(-W // 2, W // 2).view(1, -1).expand(H, W)
    r = torch.sqrt(x**2 + y**2).to(img.device)
    r = r.view(1, H, W).expand(B, -1, -1)

    bins = torch.floor(r).long().clamp(max=min(H, W) // 2 - 1)
    spectrum = torch.zeros(B, min(H, W) // 2, device=img.device)
    counts = torch.zeros_like(spectrum)

    spectrum.scatter_add_(1, bins.view(B, -1), power.view(B, -1))
    counts.scatter_add_(1, bins.view(B, -1), torch.ones_like(power).view(B, -1))
    spectrum /= counts + 1e-8
    return spectrum


def spectral_entropy(spectrum):
    # Range: [0, ln(N)], where N ≈ 32 (radial bins) → max ≈ 3.47
    # Values are non-negative, closer to 0 means more ordered, higher means more complex.
    prob = spectrum / (spectrum.sum(dim=1, keepdim=True) + 1e-8)
    logp = torch.log(prob + 1e-8)
    return -(prob * logp).sum(dim=1)


def dominant_wavelength(spectrum):
    # Range: [0, N_bins], here N_bins ≈ 32
    # Measures shift in dominant frequency. 0 = perfect match.
    # Typical values: 0–10, depending on spatial scale differences.
    freqs = torch.arange(spectrum.shape[1], device=spectrum.device).float()
    peak = (spectrum * freqs).sum(dim=1) / (spectrum.sum(dim=1) + 1e-8)
    return peak  # in frequency units


def rgb_to_label(img):  # [B, H, W, 3] → [B, H, W]
    return img.argmax(dim=-1)


def compute_dissimilarity_index(label_map, num_groups=3, window=5):
    B, H, W = label_map.shape
    label_onehot = (
        F.one_hot(label_map, num_classes=num_groups).permute(0, 3, 1, 2).float()
    )  # [B, C, H, W]
    kernel = torch.ones(num_groups, 1, window, window, device=label_map.device)
    local_counts = F.conv2d(label_onehot, kernel, padding=window // 2, groups=num_groups)
    local_props = local_counts / (window * window + 1e-8)  # [B, C, H, W]
    global_props = label_onehot.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    return torch.sum(torch.abs(local_props - global_props), dim=1).mean(dim=(1, 2))  # [B]


def compute_boundary_length(label_map):
    B, H, W = label_map.shape
    pad = F.pad(label_map.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate")  # [B,1,H+2,W+2]
    diffs = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = pad[:, :, 1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
        diffs += (shifted != label_map.unsqueeze(1)).float()
    return diffs.mean(dim=(1, 2, 3))  # [B]


def compute_average_cluster_size(label_map, num_groups=3):
    out = []
    for i in range(label_map.shape[0]):
        sizes = []
        for cls in range(num_groups):
            mask = (label_map[i].cpu().numpy() == cls).astype(np.int32)
            lbl, n = scipy_label(mask)
            if n > 0:
                sizes.append((mask.sum() / n))
        out.append(np.mean(sizes) if sizes else 0.0)
    return torch.tensor(out, dtype=torch.float32, device=label_map.device)  # [B]


def compute_moran_I(label_map):
    B, H, W = label_map.shape
    flat = label_map.view(B, -1).float()
    z = flat - flat.mean(dim=1, keepdim=True)
    W_mat = (
        torch.eye(H * W, device=label_map.device).roll(1, dims=0)
        + torch.eye(H * W, device=label_map.device).roll(-1, dims=0)
        + torch.eye(H * W, device=label_map.device).roll(H, dims=0)
        + torch.eye(H * W, device=label_map.device).roll(-H, dims=0)
    )
    W_sum = W_mat.sum()
    num = (z @ W_mat.to(z.device) * z).sum(dim=1)
    denom = (z**2).sum(dim=1)
    return (H * W / W_sum) * (num / (denom + 1e-8))  # [B]


def edge_density(img):  # [B, H, W]
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    img = img.unsqueeze(1)  # [B, 1, H, W]
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    return grad_mag.mean(dim=(1, 2, 3))  # [B]


def local_contrast(img, kernel_size=7):  # [B, H, W]
    pad = kernel_size // 2
    img = img.unsqueeze(1)  # [B, 1, H, W]
    mu = F.avg_pool2d(img, kernel_size, stride=1, padding=pad)
    mu_sq = F.avg_pool2d(img**2, kernel_size, stride=1, padding=pad)
    std = torch.sqrt(mu_sq - mu**2 + 1e-8)
    return std.mean(dim=(1, 2, 3))  # [B]


def band_energy(spectrum, low=5, high=15):  # [B, R]
    band = spectrum[:, low:high]
    return band.sum(dim=1) / (spectrum.sum(dim=1) + 1e-8)  # [B]


# === Additional RD Loss Functions (Topology-aware) ===

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from torch.nn.functional import avg_pool2d


def binarize(img, threshold=0.5):
    return (img > threshold).float()


def skeleton_difference(img1, img2):  # [B, H, W] float
    out = []
    for i in range(img1.shape[0]):
        skel1 = skeletonize(img1[i].cpu().numpy() > 0.5)
        skel2 = skeletonize(img2[i].cpu().numpy() > 0.5)
        diff = np.abs(skel1.astype(np.float32) - skel2.astype(np.float32))
        out.append(diff.mean())
    return torch.tensor(out, device=img1.device)


def distance_transform_loss(img1, img2):  # [B, H, W]
    out = []
    for i in range(img1.shape[0]):
        bin1 = (img1[i].cpu().numpy() > 0.5).astype(np.uint8)
        bin2 = (img2[i].cpu().numpy() > 0.5).astype(np.uint8)
        dt1 = distance_transform_edt(1 - bin1)
        dt2 = distance_transform_edt(1 - bin2)
        out.append(np.mean((dt1 - dt2) ** 2))
    return torch.tensor(out, device=img1.device)


def local_orientation_variance(img, kernel_size=5):  # [B, H, W]
    gx = (
        F.pad(img, (1, 1), mode="replicate")[:, :, 2:]
        - F.pad(img, (1, 1), mode="replicate")[:, :, :-2]
    )
    gy = (
        F.pad(img, (0, 0, 1, 1), mode="replicate")[:, 2:, :]
        - F.pad(img, (0, 0, 1, 1), mode="replicate")[:, :-2, :]
    )
    orientation = torch.atan2(gy, gx)
    orientation = orientation.unsqueeze(1)
    var = (
        avg_pool2d(orientation**2, kernel_size, stride=1, padding=kernel_size // 2)
        - avg_pool2d(orientation, kernel_size, stride=1, padding=kernel_size // 2) ** 2
    )
    return var.mean(dim=(1, 2, 3))  # [B]


# === FWD Utility Functions ===

# ---------------------------  FWD FUNCTIONS  ---------------------------
# #

# import torch
# from pytorch_wavelets import DWTForward  # standard 2-D discrete wavelet (Haar, etc.)


# def torch_cov(x, eps=1e-5):  # x:[N,D]
#     xc = x - x.mean(0, keepdim=True)
#     cov = (xc.T @ xc) / (x.shape[0] - 1)
#     cov = torch.nan_to_num(cov)  # scrub NaN/Inf
#     eye = eps * torch.eye(cov.size(0), device=x.device, dtype=x.dtype)
#     return cov + eye  # keep on GPU


# def sqrtm_psd(mat, eps=1e-5, jitter=1e-4):
#     mat = torch.nan_to_num(mat)  # scrub NaN/Inf
#     mat = 0.5 * (mat + mat.T)  # symmetrise
#     eye = jitter * torch.eye(mat.size(0), device=mat.device, dtype=mat.dtype)
#     u, s, v = torch.linalg.svd(mat + eye)  # all-GPU SVD
#     return u @ torch.diag(torch.sqrt(torch.clamp(s, min=eps))) @ u.T


# def frechet_distance_torch(mu1, cov1, mu2, cov2):
#     diff = mu1 - mu2
#     cov_sqrt = sqrtm_psd(cov1 @ cov2)
#     return diff @ diff + torch.trace(cov1 + cov2 - 2 * cov_sqrt)


# def dwt_flat_features(imgs, wave, level):
#     """
#     imgs : tensor [B, C, H, W] in [0,1] float32/float16, CUDA OK
#     returns tensor [B, D] where D = (level+1)*C*H*W /(4^level)
#     """
#     B, C, H, W = imgs.shape
#     dwt = DWTForward(J=level, wave=wave, mode="zero").to(imgs.device)
#     Yl, Yh = dwt(imgs)  # Yl: [B,C,H/2^L,W/2^L],  Yh: list length L
#     feats = [Yl.flatten(1)]  # low-low

#     for j in range(level):
#         # Yh[j] is [B, 3, C, H/2^(j+1), W/2^(j+1)] => reshape & append
#         band = Yh[j].permute(0, 2, 1, 3, 4)  # [B,C,3,h,w]
#         feats.append(band.flatten(1))

#     return torch.cat(feats, dim=1)  # [B,D]


# def compute_fwd_torch(real_imgs, gen_imgs, wave, level):
#     """
#     real_imgs / gen_imgs : [B,C,H,W] float in [0,1], same batch size
#     returns scalar FWD (torch tensor, device = input device)
#     """
#     Fr = dwt_flat_features(real_imgs, wave, level)  # [B,D]
#     Fg = dwt_flat_features(gen_imgs, wave, level)

#     mu_r, mu_g = Fr.mean(0), Fg.mean(0)
#     cov_r = torch_cov(Fr)
#     cov_g = torch_cov(Fg)

#     return frechet_distance_torch(mu_r, cov_r, mu_g, cov_g)
