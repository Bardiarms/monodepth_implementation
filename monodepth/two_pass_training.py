#!/usr/bin/env python3
"""
Two-pass training script (left & right) matching Godard et al. 2017.

Usage (from project root):
 python scripts/two_pass_training.py \
  --list data/lists/kitti_train.txt \
  --out ~//Users/bardiarms/Documents/CS/University_Courses/UT/Thesis/monodepth_implementation/monodepth/outputs \
  --n 999999 \
  --batch_size 4 \
  --height 256 --width 512 \
  --iters 400 \
  --print_freq 5 \ 
  --save_freq 200 \
  --num_workers 0

"""

#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from datasets import KITTIDataset
from model import DepthNet
from warp import warp
from losses import multiscale_loss
import time


def select_device():
    return torch.device('cpu')


def build_subset_dataset(list_path, height, width, n_samples):
    DATA_ROOT = "/Users/bardiarms/Documents/CS/University_Courses/UT/Thesis/monodepth_implementation/monodepth/data/kitti_raw"
    ds = KITTIDataset(list_path, DATA_ROOT, height=height, width=width, training=True)
    if n_samples >= len(ds):
        return ds
    return Subset(ds, list(range(n_samples)))


def main(args):
    device = select_device()
    print("Using device:", device)

    ds_sub = build_subset_dataset(args.list, args.height, args.width, args.n)
    dl = DataLoader(ds_sub, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)

    model = DepthNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    def warp_fn(src, disp):
        return warp(src, disp, padding_mode='border', align_corners=True)

    step = 0
    save_path = None

    start = time.time()

    while step < args.iters:
        for batch in dl:
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            
            
            # --- LR decay schedule ---
            if step == 6000:
                for g in optim.param_groups:
                    g["lr"] *= 0.1
                print(f"[LR] step {step}: decayed to {optim.param_groups[0]['lr']:.2e}")

            if step == 10000:
                for g in optim.param_groups:
                    g["lr"] *= 0.1
                print(f"[LR] step {step}: decayed to {optim.param_groups[0]['lr']:.2e}")
                        

            # --- single forward pass on LEFT only ---
            outs = model(left)  # disp_l_s and disp_r_s
            
            

            total_loss, losses = multiscale_loss(
                outs, left, right, warp_fn,
                scales=(0, 1, 2, 3),
                smooth_weight=args.smooth_weight,
                lr_weight=args.lr_weight,
                ssim_weight=args.ssim_weight
            )
            
            
            # # Sanity check
            # if step == 0:
            #     dl0 = outs["disp_l_0"]
            #     dr0 = outs["disp_r_0"]
            #     print("disp_l_0:", dl0.shape, dl0.min().item(), dl0.mean().item(), dl0.max().item())
            #     print("disp_r_0:", dr0.shape, dr0.min().item(), dr0.mean().item(), dr0.max().item())
            #     print("losses:", {k: float(v.item()) for k,v in losses.items()})
                
            #     with torch.no_grad():
            #         disp_l0 = outs["disp_l_0"]
            #         left0 = left
            #         right0 = right
            #         recon_left0 = warp_fn(right0, disp_l0)
            #         base_err = (right0 - left0).abs().mean().item()
            #         recon_err = (recon_left0 - left0).abs().mean().item()
            #         print("base L1 (right-left):", base_err)
            #         print("recon L1 (warp(right, disp_l)):", recon_err)
            #         step = args.iters
            #         break
        
            

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            if step % args.print_freq == 0:
                lr_now = optim.param_groups[0]["lr"]
                print(f"step {step:04d} lr={lr_now:.2e} "
                    f"total={losses['total'].item():.6f} "
                    f"phot={losses['photometric'].item():.6f} "
                    f"smooth={losses['smooth'].item():.8f} "
                    f"lrcons={losses['lr'].item():.6f}")
                        
            
            if step % args.save_freq == 0:
                ckpt = {
                    'step': step,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict()
                }
                os.makedirs(args.out, exist_ok=True)
                save_path = os.path.join(args.out, f"ckpt_{step:06d}.pth")
                torch.save(ckpt, save_path)
               
                        
               
            # Debug snippets   
            # if step % 50 == 0:
            #     with torch.no_grad():
            #         recon_left  = warp_fn(right, outs["disp_l_0"])
            #         recon_right = warp_fn(left,  -outs["disp_r_0"])
            #         base_l1 = (right - left).abs().mean().item()
            #         l_recon_l1 = (recon_left - left).abs().mean().item()
            #         r_recon_l1 = (recon_right - right).abs().mean().item()

            #     dl0 = outs["disp_l_0"]
            #     print(f"step {step} baseL1={base_l1:.4f} LreconL1={l_recon_l1:.4f} RreconL1={r_recon_l1:.4f} "
            #         f"dispL(mean={dl0.mean().item():.2f}, max={dl0.max().item():.2f})")
            #     dr0 = outs["disp_r_0"]
            #     print(f"dispR(mean={dr0.mean().item():.2f}, max={dr0.max().item():.2f})")

            # Another check
            # if step % 200 == 0:
            #     dl = outs["disp_l_0"]
            #     print("disp_l_0 mean/max:", dl.mean().item(), dl.max().item())

            step += 1
            if step >= args.iters:
                break

    end = time.time()
    print("Training done. Last saved checkpoint:", save_path)
    print(f"Total training time: {((end - start) // 60)} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', required=True)
    parser.add_argument('--out', default='out_overfit_single_pass')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)

    # expose loss weights
    parser.add_argument('--smooth_weight', type=float, default=1e-3)
    parser.add_argument('--lr_weight', type=float, default=1e-3)
    parser.add_argument('--ssim_weight', type=float, default=0.85)

    args = parser.parse_args()
    main(args)