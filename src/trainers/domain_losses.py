import torch
import torch.fft
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
import numpy as np

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

    y = torch.arange(-H//2, H//2).view(-1, 1).expand(H, W)
    x = torch.arange(-W//2, W//2).view(1, -1).expand(H, W)
    r = torch.sqrt(x**2 + y**2).to(img.device)
    r = r.view(1, H, W).expand(B, -1, -1)

    bins = torch.floor(r).long().clamp(max=min(H, W)//2 - 1)
    spectrum = torch.zeros(B, min(H, W)//2, device=img.device)
    counts = torch.zeros_like(spectrum)

    spectrum.scatter_add_(1, bins.view(B, -1), power.view(B, -1))
    counts.scatter_add_(1, bins.view(B, -1), torch.ones_like(power).view(B, -1))
    spectrum /= (counts + 1e-8)
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
    label_onehot = F.one_hot(label_map, num_classes=num_groups).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    kernel = torch.ones(num_groups, 1, window, window, device=label_map.device)
    local_counts = F.conv2d(label_onehot, kernel, padding=window//2, groups=num_groups)
    local_props = local_counts / (window * window + 1e-8)  # [B, C, H, W]
    global_props = label_onehot.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    return torch.sum(torch.abs(local_props - global_props), dim=1).mean(dim=(1, 2))  # [B]

def compute_boundary_length(label_map):
    B, H, W = label_map.shape
    pad = F.pad(label_map.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate')  # [B,1,H+2,W+2]
    diffs = 0
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        shifted = pad[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
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
    W_mat = torch.eye(H*W, device=label_map.device).roll(1, dims=0) + \
            torch.eye(H*W, device=label_map.device).roll(-1, dims=0) + \
            torch.eye(H*W, device=label_map.device).roll(H, dims=0) + \
            torch.eye(H*W, device=label_map.device).roll(-H, dims=0)
    W_sum = W_mat.sum()
    num = (z @ W_mat.to(z.device) * z).sum(dim=1)
    denom = (z ** 2).sum(dim=1)
    return (H * W / W_sum) * (num / (denom + 1e-8))  # [B]