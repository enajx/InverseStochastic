# frechet_wavelet_distance_functions.py
import torch
import ptwt, pywt
import numpy as np
from scipy import linalg


def _wp_feats(img, wave, level, log):
    """Wavelet-packet features for one image tensor [C,H,W] → 1-D vector."""
    packets = ptwt.WaveletPacket2D(img, pywt.Wavelet(wave), maxlevel=level)
    vec = [
        torch.log1p(packet.abs()) if log else packet  # log scale
        for node in packets.get_natural_order(level)
        for packet in (packets[node],)
    ]
    # mean over spatial dims → [packets] then cat → [P]
    return torch.cat([v.mean((-2, -1)).flatten() for v in vec])


def _mean_cov(mat):
    """µ and Σ of features; mat = [N,D]."""
    mu = mat.mean(0)
    x = mat - mu
    # unbiased covariance
    sigma = x.T @ x / (mat.shape[0] - 1)
    return mu, sigma


def _frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2

    # --- NEW: handle zero-covariance case fast ---
    if torch.count_nonzero(s1) == 0 and torch.count_nonzero(s2) == 0:
        return diff.dot(diff).item()  # Σ terms vanish

    # existing path
    covmean, _ = linalg.sqrtm((s1 @ s2).cpu().numpy(), disp=False)
    if not np.isfinite(covmean).all():
        eye = torch.eye(s1.shape[0], device=mu1.device) * eps
        covmean = linalg.sqrtm(((s1 + eye) @ (s2 + eye)).cpu().numpy())
    covmean = torch.from_numpy(covmean.real).to(mu1.device)

    return (diff.dot(diff) + torch.trace(s1 + s2 - 2 * covmean)).item()


# def frechet_wavelet_distance(output_batch, target, wave, level, log):
#     """
#     Per-sample Fréchet Wavelet Distance.
#     Args
#     ----
#     output_batch : tensor [B,H,W,C]   – generated images
#     target       : tensor [H,W,C]     – reference image
#     Returns
#     -------
#     Tensor [B]   – one FWD value per image in the batch
#     """
#     # broadcast target
#     if target.dim() == 3:
#         target = target.unsqueeze(0).expand(output_batch.shape[0], *target.shape)

#     # NHWC → NCHW, float64 for numeric stability
#     gen = output_batch.permute(0, 3, 1, 2).to(torch.float64)
#     real = target.permute(0, 3, 1, 2).to(torch.float64)

#     # extract features
#     feats_gen = [_wp_feats(img, wave, level, log) for img in gen]
#     feat_t = _wp_feats(real[0], wave, level, log)  # single target vector

#     # zero covariance for single-sample “distribution”
#     zeros = torch.zeros((feat_t.numel(), feat_t.numel()), dtype=feat_t.dtype, device=feat_t.device)

#     # compute per-image FWD
#     losses = torch.tensor(
#         [_frechet(f, zeros, feat_t, zeros) for f in feats_gen],
#         dtype=feat_t.dtype,
#         device=feat_t.device,
#     )
#     return losses


def frechet_wavelet_distance(output_batch, target, wave, level, log):
    """
    Fréchet Wavelet Distance (FWD).
    Supports:
    - Per-sample: target shape [H, W, C]
    - Batch-to-batch: target shape [B, H, W, C]

    Args
    ----
    output_batch : tensor [B, H, W, C] – generated images
    target       : tensor [H, W, C] or [B, H, W, C] – real images
    Returns
    -------
    Tensor [B] if per-sample mode, else scalar
    """
    gen = output_batch.permute(0, 3, 1, 2).to(torch.float64)

    if target.dim() == 3:
        # Pairwise mode
        real = (
            target.unsqueeze(0)
            .expand(gen.shape[0], *target.shape)
            .permute(0, 3, 1, 2)
            .to(torch.float64)
        )
        feats_gen = [_wp_feats(img, wave, level, log) for img in gen]
        feat_t = _wp_feats(real[0], wave, level, log)

        zeros = torch.zeros(
            (feat_t.numel(), feat_t.numel()), dtype=feat_t.dtype, device=feat_t.device
        )
        losses = torch.tensor(
            [_frechet(f, zeros, feat_t, zeros) for f in feats_gen],
            dtype=feat_t.dtype,
            device=feat_t.device,
        )
        return losses
    else:
        # Batch-to-batch mode
        real = target.permute(0, 3, 1, 2).to(torch.float64)
        feats_gen = torch.stack([_wp_feats(img, wave, level, log) for img in gen])
        feats_real = torch.stack([_wp_feats(img, wave, level, log) for img in real])
        mu_gen, cov_gen = _mean_cov(feats_gen)
        mu_real, cov_real = _mean_cov(feats_real)
        return torch.tensor(
            _frechet(mu_gen, cov_gen, mu_real, cov_real),
            dtype=mu_gen.dtype,
            device=mu_gen.device,
        )
