from typing import Union
import torch
import torch.nn.functional as F


#
def make_base_grid(B: int, H: int, W: int,
                   device: torch.device,
                   dtype: torch.dtype,
                   align_corners: bool):
    """
    Create base normalized grid of shape (B, H, W, 2) with values in [-1,1].
    """
    if align_corners:
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)  # length W
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)  # length H
    else:
        # sample at pixel centers
        xs = (torch.arange(0, W, device=device, dtype=dtype) + 0.5) / W * 2.0 - 1.0
        ys = (torch.arange(0, H, device=device, dtype=dtype) + 0.5) / H * 2.0 - 1.0

    # grid_x should be H x W where each row is xs
    grid_x = xs.unsqueeze(0).repeat(H, 1)       # (H, W)
    # grid_y should be H x W where each column is ys
    grid_y = ys.unsqueeze(1).repeat(1, W)       # (H, W)

    grid = torch.stack((grid_x, grid_y), dim=2)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # (B, H, W, 2)
    return grid


def warp(image: torch.Tensor,
         disp: torch.Tensor,
         padding_mode: str = "border",
         align_corners: bool = True) -> torch.Tensor:
    
    MODE = "bilinear"

    # Ensure input is of the shape (B, C, H, W)
    if image.ndim != 4:
        raise ValueError(f"image must be 4D tensor (B,C,H,W). Got shape: {image.shape}")
        
    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype
    
    # Normilize disparity tensor shape to (B, H, W)
    if disp.ndim == 4 and disp.shape[1] == 1:
        disp_ = disp[:, 0, :, :]
    elif disp.ndim == 3:
        disp_ = disp
    else:
        raise ValueError("disp must have shape (B,1,H,W) or (B,H,W)")
    
    # check disparity shapes with image shapes
    if disp_.shape[0] != B or disp_.shape[1] != H or disp_.shape[2] != W:
        raise ValueError(f"disp spatial dims must match image. image (H,W)=({H},{W}), disp {tuple(disp_.shape)}")
    
    # Create base grid
    base_grid = make_base_grid(B, H, W, device=device, dtype=dtype, align_corners=align_corners)
    
    # Convert disparoty to normalized units
    if align_corners:
        denom = W - 1
    else: denom = W
    
    norm_shift = (2.0 * disp_.to(dtype=dtype) / float(denom)).unsqueeze(-1)  # B,H,W,1
    
    grid = base_grid.clone()
    grid[..., 0:1] = grid[..., 0:1] - norm_shift 
    warped = F.grid_sample(image, grid, mode=MODE,
                        padding_mode=padding_mode,
                        align_corners=align_corners)
    return warped