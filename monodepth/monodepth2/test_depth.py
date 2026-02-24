
import os
import argparse
import numpy as np
import csv
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from model import DepthNet


# ---------------- VISUALS ---------------- #

def colorize_map(x, vmin=None, vmax=None, cmap=cv2.COLORMAP_MAGMA):
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1e-12

    xn = (x - vmin) / (vmax - vmin)
    xn = np.clip(xn, 0, 1)
    xn = np.nan_to_num(xn, nan=0.0, posinf=1.0, neginf=0.0)
    xn_u8 = (xn * 255.0).astype(np.uint8)
    
    return cv2.applyColorMap(xn_u8, cmap)

def save_vis_triplet(out_dir, idx, rgb_pil, disp, depth=None, mask=None):
    """
    Saves:
      - RGB
      - colorized disparity
      - colorized depth (optional)
    """
    os.makedirs(out_dir, exist_ok=True)

    # RGB (convert PIL RGB -> BGR for cv2)
    rgb = np.array(rgb_pil)[:, :, ::-1]
    cv2.imwrite(os.path.join(out_dir, f"rgb_{idx:06d}.png"), rgb)

    # Disparity color (use robust percentiles for nicer contrast)
    d_lo, d_hi = np.percentile(disp, 5), np.percentile(disp, 95)
    disp_color = colorize_map(disp, vmin=d_lo, vmax=d_hi, cmap=cv2.COLORMAP_MAGMA)
    cv2.imwrite(os.path.join(out_dir, f"disp_color_{idx:06d}.png"), disp_color)

    if depth is not None:
        depth_vis = depth.copy()
        if mask is not None:
            # hide invalid pixels for nicer visualization
            depth_vis = np.where(mask, depth_vis, np.nan)

        # Common depth viz range: 0..80m, clamp for display
        depth_vis = np.clip(depth_vis, 0, 80)
        z_lo, z_hi = np.nanpercentile(depth_vis, 5), np.nanpercentile(depth_vis, 95)
        depth_color = colorize_map(depth_vis, vmin=z_lo, vmax=z_hi, cmap=cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, f"depth_color_{idx:06d}.png"), depth_color)



# ---------------- METRICS ---------------- #

def compute_errors(gt, pred):
    if gt.size == 0:
        return None

    pred = np.clip(pred, 1e-6, None)

    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "abs_rel": np.mean(np.abs(gt - pred) / gt),
        "sq_rel": np.mean(((gt - pred) ** 2) / gt),
        "rmse": np.sqrt(np.mean((gt - pred) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
        "a1": (thresh < 1.25).mean(),
        "a2": (thresh < 1.25 ** 2).mean(),
        "a3": (thresh < 1.25 ** 3).mean(),
    }


# ---------------- LIST PARSER ---------------- #

def parse_list(path):
    samples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            left, right = parts[0], parts[1]
            bf = float(parts[2]) if len(parts) >= 3 else None
            gt = parts[3] if len(parts) >= 4 else None

            samples.append((left, right, bf, gt))
    return samples


# ---------------- UTILS ---------------- #

def abs_path(p, root):
    return p if os.path.isabs(p) else os.path.join(root, p)


# ---------------- MAIN ---------------- #

def main(args):
    device = torch.device("cpu")
    print("Using device:", device)

    model = DepthNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = parse_list(args.list)
    print("Test samples:", len(samples))

    os.makedirs(args.out, exist_ok=True)
    pred_dir = os.path.join(args.out, "preds")
    os.makedirs(pred_dir, exist_ok=True)
    vis_dir = os.path.join(args.out, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    metrics = []

    for i, (l, r, bf, gt) in enumerate(samples):
        
        if (i + 1) % args.print_freq == 0:
            print(f"[{i+1}/{len(samples)}] metrics computed: {len(metrics)}")
        
        left = abs_path(l, args.data_root)
        #right = abs_path(r, args.data_root)

        left_img = TF.resize(Image.open(left).convert("RGB"), (args.height, args.width))
        #right_img = TF.resize(Image.open(right).convert("RGB"), (args.height, args.width))

        left_t = TF.to_tensor(left_img)
        left_t = TF.normalize(left_t, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        left_t = left_t.unsqueeze(0).to(device)

        with torch.no_grad():
            disp = model(left_t)["disp_0"]
            disp = F.interpolate(disp, (args.height, args.width),
                                 mode="bilinear", align_corners=True)
            disp = disp[0, 0].cpu().numpy()

        np.save(os.path.join(pred_dir, f"{i:06d}.npy"), disp.astype(np.float32))
        # Optionally save quick visuals even without GT (disp only)
        if args.save_vis and (i % args.vis_every == 0):
            save_vis_triplet(vis_dir, i, left_img, disp, depth=None)

        # ---- METRICS ----
        if gt is None:
            continue

        gt_depth = np.load(abs_path(gt, args.data_root))  # should be (375,1242)
        H_gt, W_gt = gt_depth.shape

        # disp is currently predicted at (args.height,args.width) = (256,512)
        # 1) resize disparity spatially to GT size
        disp_gt = cv2.resize(disp, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)

        # 2) scale disparity VALUES to account for width change
        disp_gt *= (W_gt / float(args.width))

        disp_gt[disp_gt <= 1e-6] = 1e-6

        # 3) convert to depth (bf doesn't matter after median scaling, but keep it)
        bf_eff = bf if bf is not None else 1.0
        pred_depth = bf_eff / disp_gt
        
        # 4) Eigen crop + depth cap mask
        mask = (gt_depth > 0) & (gt_depth < 80.0)
        
        # Save depth visualization too (only if GT exists, because we compute pred_depth here)
        if args.save_vis and (i % args.vis_every == 0):
            save_vis_triplet(vis_dir, i, left_img, disp, depth=pred_depth, mask=mask)


        y1, y2 = int(0.40810811 * H_gt), int(0.99189189 * H_gt)
        x1, x2 = int(0.03594771 * W_gt), int(0.96405229 * W_gt)
        crop_mask = np.zeros_like(mask, dtype=bool)
        crop_mask[y1:y2, x1:x2] = True

        mask = mask & crop_mask
        if mask.sum() == 0:
            continue

        # 5) median scaling (monocular protocol) using the SAME mask
        pred_depth *= np.median(gt_depth[mask]) / np.median(pred_depth[mask])

        # 6) metrics on masked pixels
        errs = compute_errors(gt_depth[mask], pred_depth[mask])
        if errs:
            metrics.append(errs)
            
    # ---- CSV ----
    if metrics:
        out_csv = os.path.join(args.out, "metrics.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            keys = metrics[0].keys()
            writer.writerow(["AVERAGE"] + list(keys))
            avg = {k: np.mean([m[k] for m in metrics]) for k in keys}
            writer.writerow([""] + [avg[k] for k in keys])

        print("Saved metrics to", out_csv)
        print("AVERAGE:", avg)
    else:
        print("No metrics computed (missing bf or gt).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", default=".")
    p.add_argument("--out", default="out_test")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--save_vis", action="store_true", default=True, help="save RGB/disp/depth visualizations")
    p.add_argument("--vis_every", type=int, default=50, help="save visuals every N samples")
    args = p.parse_args()
    main(args)
