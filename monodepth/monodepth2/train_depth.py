
"""
python train_depth.py \
  --list /Users/bardiarms/Documents/CS/University_Courses/UT/Thesis/monodepth_implementation/monodepth/data/splits_eigen/train_files.txt \
  --out /Users/bardiarms/Documents/CS/University_Courses/UT/Thesis/monodepth_implementation/monodepth/monodepth2/outputs/monodepth2_stereo/train_with_15000_iters \
  --n 999999 \
  --batch_size 4 \
  --height 256 --width 512 \
  --lr 1e-4 \
  --iters 15001 \
  --print_freq 100 \
  --save_freq 1500 \
  --num_workers 0 \
  --smooth_weight 1e-3 \
  --ssim_weight 0.85
  """





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
            
            # Normalized
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            
            # Raw(Used in Loss) 
            left_raw = batch["left_raw"].to(device)  # [0,1]
            right_raw = batch["right_raw"].to(device)
            
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
            outs = model(left)  # disp_0 ... disp_3
            
            

            total_loss, losses = multiscale_loss(
                outs, left_raw, right_raw, warp_fn,
                scales=(0, 1, 2, 3),
                smooth_weight=args.smooth_weight,
                ssim_weight=args.ssim_weight
            )
            

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            if step % args.print_freq == 0:
                lr_now = optim.param_groups[0]["lr"]
                print(f"step {step:04d} lr={lr_now:.2e} "
                    f"total={losses['total'].item():.6f} "
                    f"phot={losses['photometric'].item():.6f} "
                    f"smooth={losses['smooth'].item():.8f}")
                        
            
            if step % args.save_freq == 0:
                ckpt = {
                    'step': step,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict()
                }
                os.makedirs(args.out, exist_ok=True)
                save_path = os.path.join(args.out, f"ckpt_{step:06d}.pth")
                torch.save(ckpt, save_path)
               
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
    parser.add_argument('--ssim_weight', type=float, default=0.85)

    args = parser.parse_args()
    main(args)