"""
XGMR 학습 스크립트 (Self-supervised 지원).

한국어: 동적 호모그래피를 GT 없이 학습하기 위한 Self-Supervised 트레이너를 포함한다.
English: Provides entry points for both baseline and self-supervised training of XGMR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from xgmr.config import XGMRConfig, load_config
from xgmr.data import RGBThermalPairDataset, build_default_transforms, paired_collate
from xgmr.losses import matching_nll, qmap_smoothness
from xgmr.models import XGMR
from xgmr.utils import get_logger
from xgmr.utils.ablation import set_ablation_config

from trainer_selfsup import SelfSupTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Base config YAML path")
    parser.add_argument("--selfsup", type=str, default=None, help="Self-supervised config YAML path")
    parser.add_argument("--data", type=str, default=None, help="Optional data config to merge")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--override", nargs="*", default=[], help="key=value pairs to override config")
    return parser.parse_args()


class SupervisedTrainer:
    """Minimal supervised trainer kept for regression testing / ablation."""

    def __init__(self, model: XGMR, cfg: XGMRConfig, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and torch.cuda.is_available())

    def train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running = {"loss": 0.0}
        for batch in loader:
            rgb = batch["rgb"].to(self.device)
            thermal = batch["thermal"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.train.amp and torch.cuda.is_available()):
                outputs = self.model(rgb, thermal)
                P = outputs["P"]
                target = torch.ones_like(P) / P.size(-1)
                loss_match = matching_nll(P, target)
                
                # Qmap이 있으면 smoothness loss 계산, 없으면 0
                if "Qmap" in outputs and outputs["Qmap"] is not None:
                    loss_q = qmap_smoothness(outputs["Qmap"])
                else:
                    loss_q = torch.tensor(0.0, device=P.device)
                
                loss = loss_match + self.cfg.loss.lambda_q * loss_q

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running["loss"] += loss.detach().item()

        for key in running:
            running[key] /= max(len(loader), 1)
        return running

    def save_checkpoint(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)


def main() -> None:
    args = parse_args()

    # Load base config
    base_cfg = OmegaConf.load(args.config)
    
    # Merge with optional YAMLs
    if args.selfsup:
        selfsup_cfg = OmegaConf.load(args.selfsup)
        base_cfg = OmegaConf.merge(base_cfg, selfsup_cfg)
    if args.data:
        data_cfg = OmegaConf.load(args.data)
        base_cfg = OmegaConf.merge(base_cfg, data_cfg)
        
    # Apply CLI overrides
    if args.override:
        cli_cfg = OmegaConf.from_dotlist(args.override)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    # Merge into structured config schema
    schema = OmegaConf.structured(XGMRConfig)
    cfg_obj = OmegaConf.merge(schema, base_cfg)
    cfg: XGMRConfig = OmegaConf.to_object(cfg_obj)
    cfg.set_determinism()
    set_ablation_config(cfg)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger = get_logger("train")
    logger.info("Output directory: %s", cfg.train.output_dir)

    transform = build_default_transforms(cfg.data.img_size)
    dataset = RGBThermalPairDataset(
        cfg.data.rgb_dir,
        cfg.data.thermal_dir,
        manifest_path=cfg.data.manifest,
        transform=transform,
    )
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

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    
    if cfg.data.limit is not None:
        indices = list(range(min(cfg.data.limit, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info("Dataset limited to %d samples.", len(dataset))
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=paired_collate,
        pin_memory=True,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
    )

    if not cfg.data.manifest:
        logger.info("No manifest provided, using self-supervised training.")
        trainer = SelfSupTrainer(model, cfg, device)
        # Self-supervised trainer handles the loop and logging internally
        trainer.fit(loader, loader)
    else:
        trainer = SupervisedTrainer(model, cfg, device)
        # Supervised trainer (legacy) still uses manual loop
        for epoch in range(cfg.train.epochs):
            logs = trainer.train_one_epoch(loader)
            log_str = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            logger.info("Epoch %d/%d - %s", epoch + 1, cfg.train.epochs, log_str)

            save_dir = Path(cfg.train.output_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            ckpt_path = save_dir / f"epoch_{epoch+1}.ckpt"
            trainer.save_checkpoint(ckpt_path)
            logger.info("Epoch %d saved to %s", epoch + 1, ckpt_path)


if __name__ == "__main__":
    main()
