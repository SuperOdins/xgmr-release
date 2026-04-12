"""
XGMR 평가 스크립트.

한국어: 저장된 체크포인트를 불러와 간단한 지표를 계산한다.
English: Loads a checkpoint and computes basic evaluation metrics.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from xgmr.config import load_config
from xgmr.data import RGBThermalPairDataset, build_default_transforms, paired_collate
from xgmr.models import XGMR
from xgmr.utils import get_logger, load_checkpoint


def _infer_grid_resolution(coords: torch.Tensor) -> tuple[int, int]:
    """Estimate (H, W) of the matcher grid from coarse coordinates."""

    if not coords.numel():
        return 1, 1
    xmax = torch.max(coords[:, 0]).item()
    ymax = torch.max(coords[:, 1]).item()
    width = max(int(round(xmax)) + 1, 1)
    height = max(int(round(ymax)) + 1, 1)
    return height, width


def _coords_to_pixels(
    coords: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Scale matcher coordinates onto the (H, W) pixel grid."""

    height, width = grid_hw
    img_h, img_w = image_hw

    scale_x = 0.0 if width <= 1 else (img_w - 1) / float(width - 1)
    scale_y = 0.0 if height <= 1 else (img_h - 1) / float(height - 1)

    scaled = coords.clone()
    scaled[:, 0] = scaled[:, 0] * scale_x
    scaled[:, 1] = scaled[:, 1] * scale_y
    return scaled


def _reprojection_error_px(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute per-match reprojection error in pixel space."""

    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    src_h = torch.cat([src, ones], dim=-1)
    proj = src_h @ h.transpose(0, 1)
    proj_xy = proj[:, :2] / proj[:, 2:].clamp(min=1e-6)
    return torch.norm(proj_xy - dst, dim=-1)


def _log_metrics(logger, summary: Dict[str, float], count: int) -> None:
    logger.info("Evaluated %d samples", count)
    for key, value in summary.items():
        logger.info("%s: %.4f", key, value)


def _mutual_precision(P: torch.Tensor) -> float:
    if P.numel() == 0:
        return 0.0
    row_best = torch.argmax(P, dim=-1)
    col_best = torch.argmax(P, dim=-2)
    rows = torch.arange(P.size(0), device=P.device)
    mutual = col_best[row_best] == rows
    return mutual.float().mean().item()


def evaluate(
    config_path: str,
    ckpt_path: str,
    limit: int | None = None,
    batch_size: int | None = None,
    eval_with_gt: bool = False,
) -> Tuple[Dict[str, float], int]:
    """Run evaluation and return aggregated metrics and sample count."""

    cfg = load_config(config_path)
    cfg.set_determinism()

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    transform = build_default_transforms((480, 640), augment=False)
    dataset = RGBThermalPairDataset(
        cfg.data.rgb_dir,
        cfg.data.thermal_dir,
        transform=transform,
        manifest=cfg.data.manifest,
    )
    eval_batch_size = batch_size or cfg.data.batch_size
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=paired_collate,
    )

    model = XGMR(cfg.model.backbone, cfg.model.mba, cfg.model.matcher, cfg.model.selfcalib, cfg.model.qfusion)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    metrics: Dict[str, List[float]] = defaultdict(list)
    num_samples = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit is not None and step >= limit:
                break

            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            homography_gt = None
            if eval_with_gt:
                homography_gt = batch.get("homography")
                if homography_gt is not None:
                    homography_gt = homography_gt.to(device)

            out = model(rgb, thermal)

            P = out["P"]
            matches_c = out["matches_c"]
            matches_f = out["matches_f"]
            Qmap = out.get("Qmap")
            H_pred = out.get("H")

            batch_size_tensor = rgb.size(0)
            for b in range(batch_size_tensor):
                rgb_hw = (rgb[b].shape[-2], rgb[b].shape[-1])
                coarse = matches_c["coords0"][b].detach()
                fine_src = matches_f["coords0"][b].detach()
                fine_dst = matches_f["coords1"][b].detach()
                grid_hw = _infer_grid_resolution(coarse)
                src_px = _coords_to_pixels(fine_src, grid_hw, rgb_hw)
                dst_px = _coords_to_pixels(fine_dst, grid_hw, rgb_hw)

                if H_pred is not None:
                    h_pred = H_pred[b]
                    err_pred = _reprojection_error_px(h_pred, src_px, dst_px)
                    metrics["pred_reproj_px"].append(err_pred.mean().item())

                if homography_gt is not None:
                    h_gt = homography_gt[b]
                    err_gt = _reprojection_error_px(h_gt, src_px, dst_px)
                    metrics["gt_reproj_px"].append(err_gt.mean().item())
                    if H_pred is not None:
                        diff = torch.norm(H_pred[b] - h_gt, p="fro").item()
                        metrics["homography_fro"].append(diff)

                row_max = P[b].max(dim=-1).values.mean().item()
                entropy = -(P[b].clamp(min=1e-9) * P[b].clamp(min=1e-9).log()).sum(dim=-1).mean().item()
                metrics["match_row_max"].append(row_max)
                metrics["match_entropy"].append(entropy)
                metrics["match_mutual"].append(_mutual_precision(P[b]))

                if Qmap is not None:
                    q = Qmap[b]
                    metrics["q_mean"].append(q.mean().item())
                    metrics["q_std"].append(q.std(unbiased=False).item())

                num_samples += 1

    if not metrics:
        return {}, num_samples

    summary = {key: sum(values) / len(values) for key, values in metrics.items() if values}
    return summary, num_samples


def run(
    *,
    config: str = "configs/default.yaml",
    ckpt: str,
    batch_size: int | None = None,
    limit: int | None = None,
    logger_name: str = "eval",
    eval_with_gt: bool = False,
) -> Tuple[Dict[str, float], int]:
    """Convenience wrapper for notebooks; returns metrics without argparse."""

    logger = get_logger(logger_name)
    summary, num_samples = evaluate(
        config,
        ckpt,
        limit=limit,
        batch_size=batch_size,
        eval_with_gt=eval_with_gt,
    )

    if summary:
        _log_metrics(logger, summary, num_samples)
    else:
        logger.warning("No samples were evaluated; please check dataset configuration.")
    return summary, num_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (defaults to cfg.data.batch_size)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of batches processed (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--eval_with_gt",
        action="store_true",
        help="Include ground-truth metrics when dataset provides homographies",
    )
    return parser.parse_args()


def _main_impl(parsed: argparse.Namespace) -> None:
    run(
        config=parsed.config,
        ckpt=parsed.ckpt,
        batch_size=parsed.batch_size,
        limit=parsed.limit,
        logger_name="eval",
        eval_with_gt=parsed.eval_with_gt,
    )


def main(args: argparse.Namespace | None = None, /, **kwargs) -> None:
    """Entry point for CLI and programmatic use."""

    if args is not None and kwargs:
        raise ValueError("Provide either args namespace or keyword overrides, not both")

    if args is None:
        if kwargs:
            args = argparse.Namespace(**kwargs)
        else:
            args = parse_args()
    elif not isinstance(args, argparse.Namespace):
        if hasattr(args, "__dict__"):
            args = argparse.Namespace(**vars(args))
        else:
            args = argparse.Namespace(**dict(args))

    _main_impl(args)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
