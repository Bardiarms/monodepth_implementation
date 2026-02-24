import torch
import torch.nn.functional as F


# SSIM
def ssim(x: torch.Tensor, y: torch.Tensor,
         C1: float = 0.0001,C2: float = 0.0009) -> torch.Tensor:
    
    mu_x = F.avg_pool2d(x, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, 3, stride=1, padding=1)
    
    # Elementwise squares and cross-term of the local means needed by SSIM formula.
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    
    sigma_x = F.avg_pool2d(x * x, 3, stride=1, padding=1) - mu_x_sq     # E[x^2] - (E[x])^2
    sigma_y = F.avg_pool2d(y * y, 3, stride=1, padding=1) - mu_y_sq     # E[y^2] - (E[y])^2
    sigma_xy = F.avg_pool2d(x * y, 3, stride=1, padding=1) - mu_xy      # E[(x*y)^2] - (E[x*y])^2
    
    # SSIM formula.
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-7)
    
    # Convert similarity to a loss in [0,1]
    loss_map = torch.clamp((1 - ssim_map) / 2, 0, 1)
    
    return loss_map


# Computes horizontal gradient
def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :, :-1] - img[:, :, :, 1:]


# Computes vertical gradient
def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :-1, :] - img[:, :, 1:, :]


# Disparity Smoothness Loss
def smoothness_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    # Mean-normalize disparity to reduce scale shrinkage bias
    disp = disp / (disp.mean(dim=[2, 3], keepdim=True) + 1e-7)

    disp_grad_x = torch.abs(gradient_x(disp))
    disp_grad_y = torch.abs(gradient_y(disp))

    img_grad_x = torch.mean(torch.abs(gradient_x(img)), 1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(gradient_y(img)), 1, keepdim=True)

    weight_x = torch.exp(-img_grad_x)
    weight_y = torch.exp(-img_grad_y)

    smooth_x = disp_grad_x * weight_x
    smooth_y = disp_grad_y * weight_y

    return smooth_x.mean() + smooth_y.mean()

# Photometric Loss (SSIM + L1)
# Returns scalar loss.
def photometric_loss(recon: torch.Tensor,
                     target: torch.Tensor,
                     ssim_weight: float=0.85) -> float:
    
    ssim_map = ssim(recon, target)
    ssim_loss = ssim_map.mean()
    l1_loss = torch.abs(recon - target).mean()
    
    return ssim_weight * ssim_loss + (1.0 - ssim_weight) * l1_loss

def multiscale_loss(outputs: dict,
                    left: torch.Tensor,
                    right: torch.Tensor,
                    warp_fn,
                    scales=(0, 1, 2, 3),
                    smooth_weight=1e-3,
                    ssim_weight=0.85):
    """
    Stereo-only Monodepth2-style:
      outputs contains disp_{s} at multiple scales.
    Key differences vs your current loss:
      - no disp_r, no LR-consistency
      - photometric computed at FULL resolution for every scale
      - mean-normalized disparity smoothness
    """
    total = left.new_tensor(0.0)
    photometric_total = left.new_tensor(0.0)
    smooth_total = left.new_tensor(0.0)

    B, _, H, W = left.shape
    scale_weights = {s: 1.0 / (2 ** s) for s in scales}

    for s in scales:
        disp_s = outputs.get(f"disp_{s}", None)
        if disp_s is None:
            continue

        # Upsample disparity to full resolution (Monodepth2)
        disp_full = F.interpolate(disp_s, size=(H, W), mode="bilinear", align_corners=True)

        # Reconstruct left from right at full resolution
        recon_left = warp_fn(right, disp_full)

        photo = photometric_loss(recon_left, left, ssim_weight=ssim_weight)
        photometric_scale = photo * scale_weights[s]
        photometric_total += photometric_scale

        # Smoothness: use disparity at its own scale with corresponding image scale
        h_s, w_s = disp_s.shape[2], disp_s.shape[3]
        left_s = F.interpolate(left, size=(h_s, w_s), mode="bilinear", align_corners=True)

        smooth = smoothness_loss(disp_s, left_s)
        smooth_scale = smooth_weight * smooth * scale_weights[s]
        smooth_total += smooth_scale

        total = total + photometric_scale + smooth_scale

    return total, {"total": total, "photometric": photometric_total, "smooth": smooth_total}