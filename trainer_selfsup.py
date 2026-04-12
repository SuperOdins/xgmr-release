"""Self-supervised trainer for XGMR."""

from __future__ import annotations

import copy
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm import 추가

from xgmr.config import XGMRConfig
from xgmr.data.augs_geom import random_homography
from xgmr.losses.selfsup import (
    loss_equivariance,
    loss_geo,
    loss_entropy,
)
from xgmr.losses import qmap_smoothness, det_regularization, identity_regularization
from xgmr.utils.ablation import get_ablation_slug


def _apply_h(img: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """순수 PyTorch로 구현한 호모그래피 워프 (Kornia 의존성 제거)."""
    B, C, h, w = img.shape
    device, dtype = img.device, img.dtype
    
    # 정규화 좌표 그리드 생성 [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing='ij'
    )
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid.view(B, h * w, 3)
    
    # H를 정규화 좌표계로 변환
    S = torch.tensor([[2.0/w, 0, -1], [0, 2.0/h, -1], [0, 0, 1]], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    S_inv = torch.tensor([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1]], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    H_norm = torch.bmm(torch.bmm(S, H), S_inv)
    
    # 역 호모그래피 적용
    H_inv = torch.inverse(H_norm)
    src = torch.bmm(grid, H_inv.transpose(-2, -1))
    src = src[..., :2] / (src[..., 2:3] + 1e-8)
    src = src.view(B, h, w, 2)
    
    return F.grid_sample(img, src, mode='bilinear', align_corners=False)


class SelfSupTrainer:
    def __init__(self, model: torch.nn.Module, cfg: XGMRConfig, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(cfg.train.output_dir)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and torch.cuda.is_available())
        self.global_step = 0
        self.current_epoch = 0

        # CSV Logger Setup
        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = get_ablation_slug(self.cfg.ablation)
        self.log_file = self.output_dir / f"training_log_{slug}.csv"
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                # 확장된 로그 헤더
                f.write("epoch,loss,eq,geo,det,id,ent,q,direct,w_eq,w_id,w_direct,matches,inlier_ratio,lr,grad_norm,time\n")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, on_epoch_end: Callable | None = None) -> None:
        import time  # 시간 측정용
        self.model.train()
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_stats = {
                "eq": 0.0, "det": 0.0, "geo": 0.0, "id": 0.0, "ent": 0.0, "q": 0.0, "direct": 0.0,
                "num_matches": 0.0, "inlier_ratio": 0.0, "grad_norm": 0.0
            }
            steps = 0
            last_batch = None  # 에폭의 마지막 배치를 시각화에 사용
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.train.epochs}")
            for batch in pbar:
                last_batch = batch
                logs = self.step(batch)
                
                # Update stats
                epoch_loss += logs["total"]
                epoch_stats["eq"] += logs.get("eq", 0.0)
                epoch_stats["det"] += logs.get("det", 0.0)
                epoch_stats["geo"] += logs.get("geo", 0.0)
                epoch_stats["id"] += logs.get("id", 0.0)
                epoch_stats["ent"] += logs.get("ent", 0.0)
                epoch_stats["q"] += logs.get("q", 0.0)
                epoch_stats["direct"] += logs.get("direct", 0.0)
                epoch_stats["num_matches"] += logs.get("num_matches", 0.0)
                epoch_stats["inlier_ratio"] += logs.get("inlier_ratio", 0.0)
                epoch_stats["grad_norm"] += logs.get("grad_norm", 0.0)
                steps += 1
                
                pbar.set_postfix({"loss": f"{logs['total']:.4f}"})

            # Average stats
            avg_loss = epoch_loss / steps
            avg_eq = epoch_stats["eq"] / steps
            avg_det = epoch_stats["det"] / steps
            avg_geo = epoch_stats["geo"] / steps
            avg_id = epoch_stats["id"] / steps
            avg_ent = epoch_stats["ent"] / steps
            avg_q = epoch_stats["q"] / steps
            avg_matches = epoch_stats["num_matches"] / steps
            avg_inliers = epoch_stats["inlier_ratio"] / steps
            avg_grad_norm = epoch_stats["grad_norm"] / steps
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            warmup_epochs = getattr(self.cfg.train, "warmup_epochs", 0)
            in_warmup = epoch <= warmup_epochs
            w_eq, w_id, w_direct = self._current_loss_weights()

            # Console Log (확장)
            print(
                f"Epoch {epoch} | Loss: {avg_loss:.4f} | Eq: {avg_eq:.4f} | Geo: {avg_geo:.4f} | "
                f"Det: {avg_det:.4f} | ID: {avg_id:.4f} | Ent: {avg_ent:.4f} | Q: {avg_q:.4f} | "
                f"Matches: {avg_matches:.1f} | GradNorm: {avg_grad_norm:.2f} | Time: {epoch_time:.1f}s | "
                f"Warmup: {in_warmup} | w_eq={w_eq:.3f} w_id={w_id:.3f} w_direct={w_direct:.3f}"
            )

            # CSV Log
            self.log_to_csv(
                epoch,
                {
                    "loss": avg_loss,
                    "eq": avg_eq,
                    "geo": avg_geo,
                    "det": avg_det,
                    "id": avg_id,
                    "ent": avg_ent,
                    "q": avg_q,
                    "direct": epoch_stats.get("direct", 0.0) / steps,
                    "w_eq": w_eq,
                    "w_id": w_id,
                    "w_direct": w_direct,
                    "matches": avg_matches,
                    "inlier_ratio": avg_inliers,
                    "lr": current_lr,
                    "grad_norm": avg_grad_norm,
                    "time": epoch_time,
                },
            )

            # Checkpoint
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg,
            }, self.output_dir / f"epoch_{epoch}.ckpt")
            print(f"✅ Saved checkpoint to {self.output_dir / f'epoch_{epoch}.ckpt'}")
            
            # [Epoch Visualization] 매 에폭 Input/Output 시각화 저장
            if last_batch is not None:
                viz_path = self.output_dir / f"viz_epoch_{epoch:03d}.png"
                self.visualize_prediction(last_batch, viz_path)
                print(f"📸 Saved epoch {epoch} visualization to {viz_path}")
            
            # [Epoch Backup] Mirror checkpoint to Drive
            self._mirror_to_drive(self.output_dir / f"epoch_{epoch}.ckpt")
            
            if on_epoch_end is not None:
                on_epoch_end(epoch, self.output_dir / f"epoch_{epoch}.ckpt", self.log_file)


    def log_to_csv(self, step_or_epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics to CSV file."""
        if not self.log_file.exists():
             with open(self.log_file, "w") as f:
                f.write("epoch,loss,eq,geo,det,id,ent,q,direct,w_eq,w_id,w_direct,matches,inlier_ratio,lr,grad_norm,time\n")

        with open(self.log_file, "a") as f:
            f.write(
                f"{step_or_epoch},"
                f"{metrics.get('loss', 0.0):.4f},"
                f"{metrics.get('eq', 0.0):.4f},"
                f"{metrics.get('geo', 0.0):.4f},"
                f"{metrics.get('det', 0.0):.4f},"
                f"{metrics.get('id', 0.0):.4f},"
                f"{metrics.get('ent', 0.0):.4f},"
                f"{metrics.get('q', 0.0):.4f},"
                f"{metrics.get('direct', 0.0):.4f},"
                f"{metrics.get('w_eq', 0.0):.4f},"
                f"{metrics.get('w_id', 0.0):.4f},"
                f"{metrics.get('w_direct', 0.0):.4f},"
                f"{metrics.get('matches', 0.0):.1f},"
                f"{metrics.get('inlier_ratio', 0.0):.4f},"
                f"{metrics.get('lr', 0.0):.6f},"
                f"{metrics.get('grad_norm', 0.0):.4f},"
                f"{metrics.get('time', 0.0):.1f}\n"
            )
        
        # [Epoch Backup] Mirror log file to Drive
        self._mirror_to_drive(self.log_file)

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        logs: Dict[str, list[float]] = defaultdict(list)
        for batch in loader:
            step_logs = self.step(batch)
            for key, value in step_logs.items():
                logs[key].append(value)
        return {k: float(sum(v) / max(len(v), 1)) for k, v in logs.items()}

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        rgb = batch["rgb"].to(self.device)
        thermal = batch["thermal"].to(self.device)

        # Only need strong augmentation for student
        rgb_strong, thermal_strong = self._augment(rgb, thermal, strength="strong")

        synth_H = torch.stack(
            [
                random_homography(
                    self.cfg.selfsup.geom.h_bounds_px,
                    self.cfg.selfsup.geom.rot_deg,
                    self.cfg.selfsup.geom.scale,
                    device=self.device,
                )
                for _ in range(rgb.size(0))
            ],
            dim=0,
        )

        rgb_synth = _apply_h(rgb_strong, synth_H)

        # Prepare ground-truth homography if available
        H_orig = batch.get("homography", None)
        if H_orig is not None:
            H_orig = H_orig.to(self.device).to(torch.float32)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda", enabled=self.cfg.train.amp and torch.cuda.is_available()
        ):
            # 1. Student on original (strong aug)
            # If we have H_orig (GT alignment), use it for bias
            student_out = self.model(rgb_strong, thermal_strong, H0=H_orig)
            
            # 2. Student on synthetic warped
            # Relative homography between rgb_synth and thermal_strong is synth_H @ H0_original
            if H_orig is not None:
                H_rel = torch.bmm(synth_H.to(torch.float32), H_orig)
            else:
                H_rel = synth_H.to(torch.float32)
                
            synth_out = self.model(rgb_synth, thermal_strong, H0=H_rel)

        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        matches_c = student_out.get("matches_c")
        coords0 = matches_c.get("coords0") if isinstance(matches_c, dict) else None
        epi_bias_mask = None

        if "H" in student_out and "H" in synth_out:
            H_student = student_out["H"].to(torch.float32)
            H_synth = synth_out["H"].to(torch.float32)
            
            # Scale augmentation H from Image Space to Feature Space
            # Model predicts H in feature grid coordinates
            h_img, w_img = rgb.shape[-2], rgb.shape[-1]
            
            # [전문가 수정] matches_c["coords0"]의 형상이 아닌, 모델이 직접 보고한 실제 특징 맵 해상도 사용
            h_feat, w_feat = student_out["feat_hw"]
            
            # 실제 해상비를 사용하여 정밀한 스케일링 수행
            sx, sy = w_img / w_feat, h_img / h_feat
            synth_H_feat = self._scale_homography_nonuniform(synth_H, sx, sy)

            if coords0 is not None:
                epi_bias_mask = self._build_epi_bias_mask(H_student, coords0)

            # Weight Transition: [Warmup -> Transition -> Self-Sup]
            w_eq, w_id, w_direct = self._current_loss_weights()

            # 1. Equivariance Loss
            loss_eq = loss_equivariance(H_student, H_synth, synth_H_feat)
            losses["eq"] = loss_eq
            total_loss = total_loss + w_eq * loss_eq.to(total_loss.dtype)

            # 2. Direct Supervision (Synthetic Pair)
            loss_direct = F.l1_loss(H_synth, synth_H_feat)
            losses["direct"] = loss_direct
            total_loss = total_loss + w_direct * loss_direct.to(total_loss.dtype)

            # 3. Geometric Regularization (Determinant & Identity)
            loss_det = det_regularization(H_student)
            losses["det"] = loss_det
            total_loss = total_loss + getattr(self.cfg.loss, "lambda_det", 1.0) * loss_det.to(total_loss.dtype)
            
            loss_id = identity_regularization(H_student)
            losses["id"] = loss_id
            total_loss = total_loss + w_id * loss_id.to(total_loss.dtype)

            with torch.no_grad():
                if self.global_step % 10 == 0:
                    # 디버깅: H_student와 H_synth의 실제 값 차이 확인
                    diff = (H_student - H_synth).abs().mean().item()
                    print(f"[Step {self.global_step}] Det Loss: {loss_det.item():.4f}, "
                          f"H_student-H_synth diff={diff:.4f}")

            # Photometric and Cycle losses removed as they are invalid/trivial for this setup.

        if "P" in student_out:
            P_student = student_out["P"].to(torch.float32)
            loss_ent = loss_entropy(P_student)
            losses["ent"] = loss_ent
            total_loss = total_loss + self.cfg.loss.lambda_ent * loss_ent.to(total_loss.dtype)

            attn_logits = torch.log(P_student.clamp(min=1e-9))
            mask = epi_bias_mask
            if mask is not None:
                mask = mask.to(attn_logits.dtype)
            loss_geo_term = loss_geo(attn_logits, mask)
            losses["geo"] = loss_geo_term
            total_loss = total_loss + self.cfg.loss.lambda_geo * loss_geo_term.to(total_loss.dtype)

            # Teacher-Student loss removed

        if "Qmap" in student_out and student_out["Qmap"] is not None:
            loss_q = qmap_smoothness(student_out["Qmap"].to(torch.float32))
            losses["q"] = loss_q
            total_loss = total_loss + self.cfg.loss.lambda_q * loss_q.to(total_loss.dtype)

        self.scaler.scale(total_loss).backward()
        
        # Gradient norm 계산 (clip 전에 측정)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # EMA update removed

        self.global_step += 1

        logs = {k: v.detach().item() for k, v in losses.items()}
        logs["total"] = total_loss.detach().item()
        logs["grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        
        # Add match stats for logging
        if "num_matches" in student_out:
            logs["num_matches"] = float(student_out["num_matches"])
        if "inlier_ratio" in student_out:
            logs["inlier_ratio"] = float(student_out["inlier_ratio"])
            
        return logs

    def _augment(self, rgb: torch.Tensor, thermal: torch.Tensor, strength: str) -> tuple[torch.Tensor, torch.Tensor]:
        """데이터 증강 (순수 PyTorch, Kornia 제거)."""
        rgb_aug = rgb.clone()
        thermal_aug = thermal.clone()
        
        # config에서 augment 설정을 안전하게 가져오기
        augment_cfg = getattr(self.cfg.selfsup, 'augment', None)
        
        if augment_cfg is not None:
            if strength == "weak" and getattr(augment_cfg, 'weak_color', False):
                noise = torch.randn_like(rgb_aug) * 0.01
                rgb_aug = torch.clamp(rgb_aug + noise, 0.0, 1.0)
            if strength == "strong" and getattr(augment_cfg, 'strong_color', False):
                scale = (1.0 + 0.1 * torch.randn(rgb_aug.size(0), 1, 1, 1, device=rgb_aug.device))
                rgb_aug = torch.clamp(rgb_aug * scale, 0.0, 1.0)

            if getattr(augment_cfg, 'noise_blur', False) and strength == "strong":
                # 순수 PyTorch box blur (avg_pool2d 사용)
                blur = F.avg_pool2d(rgb_aug, kernel_size=3, stride=1, padding=1)
                rgb_aug = (rgb_aug + blur) * 0.5

        return rgb_aug, thermal_aug

    def _scale_homography_nonuniform(self, H: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
        """Scale homography with potentially different x and y strides."""
        H_feat = H.clone().to(torch.float32)
        H_feat[:, 0, 0] *= 1.0
        H_feat[:, 0, 1] *= (sy / sx)
        H_feat[:, 0, 2] /= sx
        H_feat[:, 1, 0] *= (sx / sy)
        H_feat[:, 1, 1] *= 1.0
        H_feat[:, 1, 2] /= sy
        H_feat[:, 2, 0] *= sx
        H_feat[:, 2, 1] *= sy
        H_feat[:, 2, 2] *= 1.0
        return H_feat

    def _build_epi_bias_mask(self, H: torch.Tensor, coords: torch.Tensor) -> torch.Tensor | None:
        if coords is None:
            return None

        coords = coords.to(torch.float32)
        ones = torch.ones_like(coords[..., :1])
        coords_h = torch.cat([coords, ones], dim=-1)

        H_inv = torch.linalg.pinv(H.to(torch.float32))
        mapped = coords_h @ H_inv.transpose(1, 2)
        mapped_xy = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-6)

        mapped_xy = mapped_xy.unsqueeze(2)
        target = coords.unsqueeze(1)
        dists = ((mapped_xy - target) ** 2).sum(dim=-1)

        # [전문가 조언] Dynamic Sigma: 학습 초기에 모델이 서투를 때는 넓은 범위를 주어 매칭을 유도합니다.
        warmup_epochs = getattr(self.cfg.train, "warmup_epochs", 0)
        transition_epochs = getattr(self.cfg.train, "transition_epochs", 0)
        
        sigma_target = float(getattr(self.cfg.selfsup, "bias_sigma_px", 4.0))
        sigma_init = float(getattr(self.cfg.selfsup, "sigma_init", 8.0))
        
        if self.current_epoch <= warmup_epochs:
            sigma_val = sigma_init
        elif self.current_epoch <= (warmup_epochs + transition_epochs):
            progress = (self.current_epoch - warmup_epochs) / max(transition_epochs, 1)
            sigma_val = sigma_init + (sigma_target - sigma_init) * progress
        else:
            sigma_val = sigma_target
            
        sigma = torch.as_tensor(sigma_val, device=coords.device, dtype=coords.dtype)
        sigma_sq = (sigma.clamp(min=1e-3) ** 2)
        mask = torch.exp(-0.5 * dists / sigma_sq)
        return mask.detach()

    def _current_loss_weights(self) -> tuple[float, float, float]:
        warmup_epochs = getattr(self.cfg.train, "warmup_epochs", 0)
        transition_epochs = getattr(self.cfg.train, "transition_epochs", 0)
        
        # Target Weights
        target_eq = float(getattr(self.cfg.loss, "lambda_eq", 0.0))
        target_id = float(getattr(self.cfg.loss, "lambda_identity", 0.0))
        target_direct = float(getattr(self.cfg.loss, "lambda_direct", 1.0))
        warmup_direct = float(getattr(self.cfg.loss, "lambda_direct_warmup", 10.0))

        # A. Warmup Phase (Fully Supervised)
        if self.current_epoch <= warmup_epochs:
            return 0.0, 0.0, warmup_direct
            
        # B. Transition Phase (Linear Interpolation)
        if self.current_epoch <= (warmup_epochs + transition_epochs):
            # Calculate progress within transition [0, 1]
            progress = (self.current_epoch - warmup_epochs) / max(transition_epochs, 1)
            
            w_eq = target_eq * progress
            w_id = target_id * progress
            # direct: warmup_direct -> target_direct
            w_direct = warmup_direct + (target_direct - warmup_direct) * progress
            return w_eq, w_id, w_direct
            
        # C. Full Self-Supervised Phase
        return target_eq, target_id, target_direct

    def _mirror_to_drive(self, local_path: Path | str) -> None:
        """지정된 로컬 파일을 구글 드라이브 미러 경로에 실시간으로 복사합니다."""
        if not self.cfg.train.drive_output_dir:
            return
            
        local_path = Path(local_path)
        if not local_path.exists():
            return
            
        try:
            drive_base = Path(self.cfg.train.drive_output_dir)
            drive_base.mkdir(parents=True, exist_ok=True)
            
            # 파일 이름만 추출하여 드라이브 기본 경로에 저장 (디렉토리 구조 유지 원할 시 수정 가능)
            target_path = drive_base / local_path.name
            
            # shutil.copy2를 사용하여 메타데이터 유지하며 복사
            shutil.copy2(str(local_path), str(target_path))
            print(f"☁️ [Drive Sync] {local_path.name} backed up to Drive.")
        except Exception as e:
            # 학습 루프를 멈추지 않도록 예외 처리만 수행
            print(f"⚠️ [Drive Mirror Error] {e}")

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            self.global_step = ckpt["step"]
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
    @torch.no_grad()
    def visualize_prediction(self, batch: Dict[str, torch.Tensor], save_path: str | Path) -> None:
        """Dual-Mode Visualization: Identity Consistency (Row 1) & Active Rectification (Row 2)."""
        import matplotlib.pyplot as plt
        import numpy as np

        self.model.eval()
        rgb = batch["rgb"][:1].to(self.device).to(torch.float32)
        thermal = batch["thermal"][:1].to(self.device).to(torch.float32)
        h_img, w_img = rgb.shape[-2], rgb.shape[-1]
        
        # --- Case A: Original (Identity Check) ---
        out_orig = self.model(rgb, thermal)
        
        # --- Case B: Synthetic (Rectification Check) ---
        synth_H = random_homography(
            self.cfg.selfsup.geom.h_bounds_px,
            self.cfg.selfsup.geom.rot_deg,
            self.cfg.selfsup.geom.scale,
            device=self.device,
        ).unsqueeze(0)
        rgb_synth = _apply_h(rgb, synth_H)
        out_synth = self.model(rgb_synth, thermal)

        # Plotting Setup
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row_idx, (out, rgb_in, name) in enumerate([
            (out_orig, rgb, "Status: Original (Identity Consistency)"),
            (out_synth, rgb_synth, "Status: Synthetic (Active Rectification)")
        ]):
            # 1. Prepare Images
            rgb_np = rgb_in[0].permute(1, 2, 0).cpu().numpy()
            t_np = thermal[0].permute(1, 2, 0).cpu().numpy()
            if t_np.shape[2] == 1: t_np = np.repeat(t_np, 3, axis=2)
            
            # 2. Get Matches & Scaling
            h_feat, w_feat = out["feat_hw"]
            sx, sy = w_img / w_feat, h_img / h_feat
            
            matches = out["matches_c"]
            c0 = matches["coords0"].cpu().numpy()
            c1 = matches["coords1"].cpu().numpy()
            s = matches["scores"].cpu().numpy()
            
            mask = s > 0.1
            c0, c1 = c0[mask], c1[mask]
            if len(c0) > 100:
                idx = np.argsort(s[mask])[-100:]
                c0, c1 = c0[idx], c1[idx]
            
            c0_img = c0 * np.array([sx, sy]) + (np.array([sx, sy]) / 2.0)
            c1_img = c1 * np.array([sx, sy]) + (np.array([sx, sy]) / 2.0)

            # 3. Warp
            H_img = out["H"].clone().to(torch.float32)
            H_img[:, 0, 2] *= sx; H_img[:, 1, 2] *= sy
            H_img[:, 2, 0] /= sx; H_img[:, 2, 1] /= sy
            
            t_warp = _apply_h(thermal, H_img)
            t_warp_np = t_warp[0].permute(1, 2, 0).cpu().numpy()
            if t_warp_np.shape[2] == 1: t_warp_np = np.repeat(t_warp_np, 3, axis=2)

            # 4. Plot
            # Column 1: Matches
            vis_matches = np.concatenate([rgb_np, t_np], axis=1)
            axes[row_idx, 0].imshow(vis_matches)
            axes[row_idx, 0].set_title(f"{name}\nMatches Count: {len(c0)}")
            for i in range(len(c0_img)):
                axes[row_idx, 0].plot([c0_img[i, 0], c1_img[i, 0] + w_img], [c0_img[i, 1], c1_img[i, 1]], 'g-', alpha=0.5, linewidth=0.5)
                axes[row_idx, 0].plot(c0_img[i, 0], c0_img[i, 1], 'r.', markersize=2)
                axes[row_idx, 0].plot(c1_img[i, 0] + w_img, c1_img[i, 1], 'r.', markersize=2)

            # Column 2: Initial Overlay
            axes[row_idx, 1].imshow(rgb_np * 0.5 + t_np * 0.5)
            axes[row_idx, 1].set_title("Input Alignment")

            # Column 3: Model Output Alignment
            axes[row_idx, 2].imshow(rgb_np * 0.5 + t_warp_np * 0.5)
            axes[row_idx, 2].set_title("Model Output Alignment")

        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        self._mirror_to_drive(save_path)
        self.model.train()
