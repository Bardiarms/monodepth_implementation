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
def smoothness_loss(disp: torch.Tensor,
                    img: torch.Tensor) -> torch.Tensor:
    
    # Absolute spatial gradients
    disp_grad_x = torch.abs(gradient_x(disp))
    disp_grad_y = torch.abs(gradient_y(disp))
    
    # Compute per-pixel image gradient magnitude and then average across channels
    img_grad_x = torch.mean(torch.abs(gradient_x(img)), 1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(gradient_y(img)), 1, keepdim=True)
    
    # Exponential weights
    weight_x = torch.exp(-img_grad_x)
    weight_y = torch.exp(-img_grad_y)
    
    # Multiply disparity gradient by weights
    smooth_x = disp_grad_x * weight_x
    smooth_y = disp_grad_y * weight_y
    
    return smooth_x.mean() + smooth_y.mean()


# Left-Right Consistency Loss
def lr_consistency_loss(disp_l: torch.Tensor,
                        disp_r: torch.Tensor,
                        warp_fn) -> torch.Tensor:
    
    # warp disp_r into left coordinate using disp_l
    disp_r_warped = warp_fn(disp_r, disp_l)
    return torch.mean(torch.abs(disp_l - disp_r_warped))


# Photometric Loss (SSIM + L1)
# Returns scalar loss.
def photometric_loss(recon: torch.Tensor,
                     target: torch.Tensor,
                     ssim_weight: float=0.85) -> float:
    
    ssim_map = ssim(recon, target)
    ssim_loss = ssim_map.mean()
    l1_loss = torch.abs(recon - target).mean()
    
    return ssim_weight * ssim_loss + (1.0 - ssim_weight) * l1_loss

# Aggregate photometric, smoothness, and left-right consistency losses across multiple scales.
def multiscale_loss(outputs: dict,
                    left: torch.Tensor,
                    right: torch.Tensor,
                    warp_fn,
                    scales=[0,1,2,3],
                    smooth_weight=0.1, # Bigger weight compared to the one used in the paper
                    lr_weight=1.0,
                    ssim_weight=0.85) -> tuple:
    """
    outputs: dict that should contain:
        - 'disp_{s}'  (left disparities)
        - optionally 'disp_r_{s}' (right disparities) provided by training script
    """
    total_loss = 0.0
    photometric_total = 0.0
    smooth_total = 0.0
    lr_total = 0.0

    scale_weights = dict()
    for s in scales:
        scale_weights[s] = 1.0 / (2 ** s)
        
    B, _, H, W = left.shape

    for s in scales:
        disp_l = outputs.get(f'disp_{s}', None)
        if disp_l is None:
            continue
        # derive native size
        h_s, w_s = disp_l.shape[2], disp_l.shape[3]
        left_s = F.interpolate(left, size=(h_s, w_s), mode='bilinear', align_corners=True)
        right_s = F.interpolate(right, size=(h_s, w_s), mode='bilinear', align_corners=True)

        # left photometric (reconstruct left from right via disp_l)
        recon_left = warp_fn(right_s, disp_l)
        photo_l = photometric_loss(recon_left, left_s, ssim_weight=ssim_weight)

        # right photometric (if right disparities present)
        photo_r = 0.0
        disp_r = outputs.get(f'disp_r_{s}', None)
        if disp_r is not None:
            recon_right = warp_fn(left_s, disp_r)
            photo_r = photometric_loss(recon_right, right_s, ssim_weight=ssim_weight)

        photometric_scale = (photo_l + photo_r) * scale_weights[s]
        photometric_total += photometric_scale

        # smoothness (both left and right if both available)
        smooth_l = smoothness_loss(disp_l, left_s)
        smooth_scale = smooth_l * (smooth_weight * scale_weights[s])
        smooth_total += smooth_scale

        if disp_r is not None:
            smooth_r = smoothness_loss(disp_r, right_s)
            smooth_total += smooth_r * (smooth_weight * scale_weights[s])

        # left-right consistency (both directions if possible)
        lr_scale = 0.0
        if disp_r is not None:
            # warp right disparity to left and compare
            try:
                lr_l = lr_consistency_loss(disp_l, disp_r, warp_fn)
                lr_r = lr_consistency_loss(disp_r, disp_l, warp_fn)
                lr_scale = (lr_l + lr_r) * (lr_weight * scale_weights[s])
                lr_total += lr_scale
            except Exception:
                lr_scale = 0.0

        total_loss = total_loss + photometric_scale + smooth_scale + lr_scale

    losses = {
        'total': total_loss,
        'photometric': photometric_total,
        'smooth': smooth_total,
        'lr': lr_total
    }
    return total_loss, losses