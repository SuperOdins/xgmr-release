"""
XGMR 데모 스크립트.

한국어: 폴더에 있는 RGB/열 영상이나 랜덤 텐서로 모델 추론을 실행하고 시각화를 저장한다.
English: Runs the model on folder-based RGB/thermal pairs or random tensors and saves visualisations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from xgmr.config import load_config
from xgmr.models import XGMR
from xgmr.data.transforms import build_default_transforms
from xgmr.utils import get_logger
from xgmr.utils.viz import visualize_matches, visualize_quality


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        return np.random.rand(480, 640, 3).astype(np.float32)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--rgb_dir", type=str, default=None)
    parser.add_argument("--t_dir", type=str, default=None)
    parser.add_argument("--save_viz", type=str, default="viz_out")
    parser.add_argument("--ckpt", type=str, default="outputs/epoch_5.ckpt", help="Checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger = get_logger("demo")

    model = XGMR(
        cfg.model.backbone,
        cfg.model.mba,
        cfg.model.matcher,
        cfg.model.selfcalib,
        cfg.model.qfusion,
        use_mba=cfg.model.use_mba,
        use_self_calib=cfg.model.use_self_calib,
        use_qfusion=cfg.model.use_qfusion,
    )
    
    if args.ckpt and Path(args.ckpt).exists():
        logger.info(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        logger.warning("No checkpoint found or provided. Using random initialization.")

    model.to(device)
    model.eval()

    transform = build_default_transforms((480, 640), augment=False)

    rgb_paths: List[Path]
    t_paths: List[Path]
    if args.rgb_dir and args.t_dir:
        rgb_paths = sorted(Path(args.rgb_dir).glob("*.png")) + sorted(Path(args.rgb_dir).glob("*.jpg"))
        t_paths = sorted(Path(args.t_dir).glob("*.png")) + sorted(Path(args.t_dir).glob("*.jpg"))
        
        # Shuffle together
        import random
        combined = list(zip(rgb_paths, t_paths))
        random.shuffle(combined)
        rgb_paths, t_paths = zip(*combined)
        rgb_paths = list(rgb_paths)
        t_paths = list(t_paths)
    else:
        rgb_paths = []
        t_paths = []

    save_dir = Path(args.save_viz)
    save_dir.mkdir(exist_ok=True, parents=True)

    samples = max(len(rgb_paths), 1)
    # Limit samples for demo
    if samples > 10:
        samples = 10
        
    for idx in range(samples):
        if rgb_paths and t_paths:
            rgb_np = load_image(rgb_paths[idx])
            t_np = load_image(t_paths[idx])
        else:
            rgb_np = np.random.rand(480, 640, 3).astype(np.float32)
            t_np = np.random.rand(480, 640, 3).astype(np.float32)

        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0)
        t_tensor = torch.from_numpy(t_np.transpose(2, 0, 1)).unsqueeze(0)
        
        # Ensure thermal is 1 channel for transform if needed, but model handles 1 or 3
        if t_tensor.shape[1] == 3:
            t_tensor = t_tensor[:, :1, :, :] # Take first channel if 3
            
        sample = {"rgb": rgb_tensor.squeeze(0), "thermal": t_tensor.squeeze(0)}
        sample = transform(sample)
        rgb_tensor = sample["rgb"].unsqueeze(0).to(device)
        t_tensor = sample["thermal"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(rgb_tensor, t_tensor)

        # Use the resized tensors for visualization to ensure shape consistency
        # rgb_tensor: (1, 3, 480, 640)
        rgb_viz = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # (480, 640, 3)
        t_viz = t_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()     # (480, 640, 1) or (480, 640, 3)
        if t_viz.shape[2] == 1:
            t_viz = np.repeat(t_viz, 3, axis=2)

        # 1. Visualize Matches
        matches_c_np = {}
        for key, value in out["matches_c"].items():
            if torch.is_tensor(value):
                matches_c_np[key] = value.squeeze(0).detach().cpu().numpy()
            else:
                matches_c_np[key] = value
        
        # Scale coordinates from feature grid to pixels (480x640)
        # Assuming stride 16 (LightBackbone has 1 initial stride=2 + 3 layers of stride=2 = 16)
        stride = 16.0
        offset = stride / 2.0
        
        if "coords0" in matches_c_np:
            # Add offset to center the feature coordinates
            matches_c_np["coords0"] = matches_c_np["coords0"] * stride + offset
            matches_c_np["coords1"] = matches_c_np["coords1"] * stride + offset
            
            # Analyze Shift
            shift = matches_c_np["coords1"] - matches_c_np["coords0"]
            mean_shift = np.mean(shift, axis=0)
            print(f"Sample {idx} Mean Match Shift (px): {mean_shift} (Feature Space: {mean_shift/stride})")
            
            # Filter matches for RANSAC
            mask = matches_c_np["scores"] > 0.2 # Threshold
            mkpts0 = matches_c_np["coords0"][mask]
            mkpts1 = matches_c_np["coords1"][mask]
            
            if len(mkpts0) > 4:
                H_ransac, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                print(f"Sample {idx} RANSAC H:\n{H_ransac}")
            else:
                H_ransac = np.eye(3)
                print(f"Sample {idx} Not enough matches for RANSAC")
            
        # Filter by confidence for cleaner visualization
        if "scores" in matches_c_np:
            mask = matches_c_np["scores"] > 0.2
            matches_c_np["coords0"] = matches_c_np["coords0"][mask]
            matches_c_np["coords1"] = matches_c_np["coords1"][mask]
            matches_c_np["scores"] = matches_c_np["scores"][mask]
            
        viz_match = visualize_matches(rgb_viz, t_viz, matches_c_np)
        
        # 2. Visualize Quality
        Qmap = out["Qmap"].detach().cpu().numpy()
        wmap = out["wmap"].detach().cpu().numpy()
        viz_quality = visualize_quality(Qmap, wmap)
        
        # 3. Visualize Feature Overlay
        if "warped_t" in out:
            feat = out["warped_t"].squeeze(0).detach().cpu().numpy()
            feat_img = np.mean(feat, axis=0)
            feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min() + 1e-6)
            feat_img_resized = cv2.resize(feat_img, (640, 480)) # cv2 uses (W, H)
            feat_img_color = np.stack([feat_img_resized]*3, axis=-1)
            
            overlay = (rgb_viz * 0.5 + feat_img_color * 0.5)
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"overlay_feat_{idx}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 4. Visualize Real Image Alignment (Warped Thermal Image)
        
        # A. Original Overlay (Identity)
        # Green-Magenta
        rgb_green = rgb_viz.copy()
        rgb_green[:, :, 0] = 0
        rgb_green[:, :, 2] = 0
        t_magenta = t_viz.copy()
        if t_magenta.shape[2] == 1:
            t_magenta = np.repeat(t_magenta, 3, axis=2)
        t_magenta[:, :, 1] = 0
        
        overlay_orig = (rgb_green * 0.5 + t_magenta * 0.5)
        overlay_orig = np.clip(overlay_orig * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(save_dir / f"overlay_original_{idx}.png"), cv2.cvtColor(overlay_orig, cv2.COLOR_RGB2BGR))

        # B. RANSAC Overlay (Matcher)
        if "coords0" in matches_c_np and len(matches_c_np["coords0"]) > 4:
            src_pts = matches_c_np["coords0"]
            dst_pts = matches_c_np["coords1"]
            
            try:
                H_ransac, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H_ransac is not None:
                    print(f"Sample {idx} RANSAC H:\n{H_ransac}")
                    
                    import kornia
                    H_ransac_torch = torch.from_numpy(H_ransac).unsqueeze(0).to(device).float()
                    
                    t_tensor_warp_ransac = kornia.geometry.transform.warp_perspective(
                        t_tensor, H_ransac_torch, dsize=(480, 640)
                    )
                    t_warp_ransac_np = t_tensor_warp_ransac.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    if t_warp_ransac_np.shape[2] == 1:
                        t_warp_ransac_np = np.repeat(t_warp_ransac_np, 3, axis=2)
                    
                    t_magenta_ransac = t_warp_ransac_np.copy()
                    t_magenta_ransac[:, :, 1] = 0
                    
                    overlay_ransac = (rgb_green * 0.5 + t_magenta_ransac * 0.5)
                    overlay_ransac = np.clip(overlay_ransac * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(save_dir / f"overlay_ransac_{idx}.png"), cv2.cvtColor(overlay_ransac, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"RANSAC failed: {e}")

        # C. Predicted Overlay (Head)
        H_pred = out.get("H")
        if H_pred is not None:
            # Scale H from Feature Space (stride 16, as trained) to Image Space
            from xgmr.utils.geom import scale_homography
            
            # Model predicts RGB -> Thermal (based on Equivariance Loss).
            # We want to warp Thermal -> RGB.
            # So we need the Inverse.
            
            H_pred_img = scale_homography(H_pred, stride=16.0, inverse=True)
            H_pred_img_inv = torch.linalg.inv(H_pred_img)
            
            print(f"Sample {idx} Predicted H (Feat):\n{H_pred.squeeze(0).cpu().numpy()}")
            print(f"Sample {idx} Predicted H (Img, RGB->T):\n{H_pred_img.squeeze(0).cpu().numpy()}")
            print(f"Sample {idx} Predicted H (Img, T->RGB):\n{H_pred_img_inv.squeeze(0).cpu().numpy()}")
            
            import kornia
            t_tensor_warp = kornia.geometry.transform.warp_perspective(
                t_tensor, H_pred_img_inv, dsize=(480, 640)
            )
            t_warp_np = t_tensor_warp.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            if t_warp_np.shape[2] == 1:
                t_warp_np = np.repeat(t_warp_np, 3, axis=2)

            t_magenta_pred = t_warp_np.copy()
            t_magenta_pred[:, :, 1] = 0
            
            overlay_pred = (rgb_green * 0.5 + t_magenta_pred * 0.5)
            overlay_pred = np.clip(overlay_pred * 255, 0, 255).astype(np.uint8)
            
            cv2.imwrite(str(save_dir / f"overlay_pred_{idx}.png"), cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))
            
            # Blend Overlay
            overlay_blend = (rgb_viz * 0.5 + t_warp_np * 0.5)
            overlay_blend = np.clip(overlay_blend * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"overlay_real_blend_{idx}.png"), cv2.cvtColor(overlay_blend, cv2.COLOR_RGB2BGR))

        cv2.imwrite(str(save_dir / f"matches_{idx}.png"), cv2.cvtColor(viz_match, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"quality_{idx}.png"), cv2.cvtColor(viz_quality, cv2.COLOR_RGB2BGR))
        logger.info("Saved visualisations for sample %d to %s", idx, save_dir)

if __name__ == "__main__":
    main()
