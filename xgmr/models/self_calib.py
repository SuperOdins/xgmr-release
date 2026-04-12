"""
Self-Calibrating Head.

한국어: 호모그래피와 TPS 파라미터를 공동 추정하여 열 영상을 RGB에 정렬한다.
English: Jointly estimates homography and TPS parameters to align thermal to RGB.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xgmr.utils.ablation import ablation_entry




class MatchEncoder(nn.Module):
    """
    정합점 인코더 (Match Encoder).
    Korean: 기하학적 대응관계(Matches)를 글로벌 임베딩으로 인코딩하여 예측 헤드의 가이드로 사용한다.
    English: Encodes geometric match correspondences into a global embedding to guide the prediction head.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, 4) [coords0_norm, coords1_norm]
        # Korean: 각 정합점의 좌표 정보를 특징 벡터로 변환한다.
        # English: Transform the coordinate information of each match into a feature vector.
        feat = self.net(x)
        
        # Global Max Pooling
        # Korean: 모든 정합점의 특징 중 가장 강한 신호를 추출하여 글로벌 정합 정보를 생성한다. (B, out_dim)
        # English: Extract the strongest signal among all matches to create global matching information.
        return torch.max(feat, dim=1)[0]


class SelfCalibratingHead(nn.Module):
    """
    Self-Calib 헤드 (자가 보정 헤드).
    Korean: 호모그래피(H)와 TPS(Thin Plate Spline) 파라미터를 공동 추정하여 열 영상을 RGB에 정렬한다.
    English: Jointly estimates homography (H) and TPS parameters to align thermal imagery to RGB.
    """

    def __init__(
        self,
        dim: int = 256,
        tps_grid: int = 5,
        iters: int = 2,
        match_k: int = 128,
        match_dim: int = 128,
        h_res_scale: float = 1e-4,
        tps_res_scale: float = 0.02,
    ) -> None:
        """
        Args:
            dim: 입력 특징 차원 (Input feature dimension)
            tps_grid: TPS 제어점 그리드 크기 (TPS control point grid size, e.g., 5x5)
            iters: 반복 보정 횟수 (Number of iterative refinement steps)
            match_k: 활용할 상위 K개 정합점 수 (Number of top-K matches to use)
            match_dim: 정합점 임베딩 차원 (Match embedding dimension)
            h_res_scale: 호모그래피 잔차 스케일 (Residual scale for homography)
            tps_res_scale: TPS 잔차 스케일 (Residual scale for TPS)
        """
        super().__init__()
        self.tps_grid = tps_grid
        self.iters = iters
        self.match_k = match_k
        self.match_dim = match_dim
        self.h_res_scale = h_res_scale
        self.tps_res_scale = tps_res_scale
        
        # 특징 맵 투영 계층
        # Korean: RGB와 Thermal 특징 맵을 보정 작업에 맞게 가공한다.
        # English: Process RGB and Thermal feature maps for the calibration task.
        self.rgb_proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.t_proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        # Match Encoder 초기화
        self.match_encoder = MatchEncoder(input_dim=4, out_dim=self.match_dim)
        
        latent_dim = dim
        # 파라미터 예측 네트워크 (Regression Network)
        # Korean: RGB, Thermal, Match 임베딩을 결합하여 H(8개)와 TPS(2*G^2개) 파라미터를 예측한다.
        # English: Predict H (8 params) and TPS (2*G^2 params) by combining RGB, Thermal, and Match embeddings.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2 + self.match_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8 + 2 * tps_grid * tps_grid),
        )
        
        # 항등 변환으로 초기화 (Initialize to identity)
        # Korean: 초기 예측값이 변화 없는 상태(Zero delta)에서 시작하도록 가중치를 0으로 설정한다.
        # English: Set weights to zero so that initial predictions start from an identity transformation.
        nn.init.constant_(self.fc[-1].weight, 0.0)
        nn.init.constant_(self.fc[-1].bias, 0.0)

    @ablation_entry("no_selfcalib")
    def forward(self, features_rgb: Tensor, features_t: Tensor, matches: Dict) -> Dict:
        """
        자가 보정 수행 (Forward Pass).
        Korean: 정합점 정보와 특징 맵을 결합하여 반복적으로 변환 파라미터를 정밀화한다.
        English: Refine transformation parameters iteratively by combining matches and feature maps.
        """
        B, D, H_feat, W_feat = features_rgb.shape
        device = features_rgb.device
        dtype = features_rgb.dtype

        # 1. 정적 특징 처리 (Process Static Anchors)
        # Korean: 변하지 않는 타겟 영상(RGB)과 정합점 정보를 인코딩한다.
        # English: Encode the fixed target image (RGB) and match information.
        rgb_latent = self.rgb_proj(features_rgb)
        rgb_pool = F.adaptive_avg_pool2d(rgb_latent, 1).flatten(1)

        match_emb = torch.zeros(B, self.match_dim, device=device, dtype=dtype)
        if "matches_c" in matches:
            m = matches["matches_c"]
            coords0, coords1, scores = m["coords0"], m["coords1"], m["scores"]
            
            if coords0.dim() == 2:
                coords0 = coords0.unsqueeze(0).expand(B, -1, -1)
            
            # 상위 K개 정합점 추출 (Match-Aware Guidance)
            # Korean: 신뢰도가 높은 정합점들을 선별하여 기하학적 힌트를 제공한다.
            # English: Select high-confidence matches to provide geometric hints.
            if scores.size(1) > self.match_k:
                _, idx = torch.topk(scores, self.match_k, dim=1)
                c0 = torch.gather(coords0, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
                c1 = torch.gather(coords1, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
            else:
                pad_size = self.match_k - scores.size(1)
                c0 = F.pad(coords0, (0, 0, 0, pad_size))
                c1 = F.pad(coords1, (0, 0, 0, pad_size))
            
            # 좌표 정규화 및 인코딩
            scale = torch.tensor([W_feat, H_feat], device=device, dtype=dtype).view(1, 1, 2)
            c0_norm = (c0 / scale) * 2.0 - 1.0
            c1_norm = (c1 / scale) * 2.0 - 1.0
            match_input = torch.cat([c0_norm, c1_norm], dim=-1)
            match_emb = self.match_encoder(match_input)

        # 2. Iterative Refinement Loop (반복 보정 루프)
        # Korean: 현재 변환 상태(H_total, TPS_total)를 누적하며 반복적으로 정밀하게 보정한다.
        # English: Refine the alignment accurately by accumulating the current transformation state.
        H_total = torch.eye(3, device=device,                                                                                                                                                                                dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        TPS_total_offsets = torch.zeros(B, self.tps_grid * self.tps_grid, 2, device=device, dtype=dtype)
        
        current_t = features_t # 매 반복마다 워핑된 Thermal 특징맵
        
        # 샘플링용 기초 그리드 생성
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device, dtype=dtype),
            torch.linspace(-1, 1, W_feat, device=device, dtype=dtype),
            indexing='ij'
        )
        grid_base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        grid_homo_base = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        grid_homo_base = grid_homo_base.view(B, H_feat * W_feat, 3)

        # 좌표계 변환 행렬 (Normalized to Pixel coordinate mapping)
        S = torch.tensor([
            [2.0 / max(W_feat - 1, 1), 0.0, -1.0],
            [0.0, 2.0 / max(H_feat - 1, 1), -1.0],
            [0.0, 0.0, 1.0]
        ], device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        S_inv = torch.inverse(S.to(torch.float32)).to(dtype)

        for i in range(self.iters):
            # A. 특징 추출 (Feature Pooling)
            # Korean: 현재 워핑된 Thermal 특징 맵으로부터 글로벌 정보를 추출한다.
            # English: Extract global information from the currently warped thermal feature map.
            t_latent = self.t_proj(current_t)
            t_pool = F.adaptive_avg_pool2d(t_latent, 1).flatten(1)

            # B. 잔차 변환 예측 (Predict Residual Transformation)
            # Korean: RGB, Thermal, Match 임베딩을 결합하여 이전 단계와의 보정 차이(Residual)를 예측한다.
            # English: Combine embeddings to predict the adjustment (residual) from the previous step.
            pooled = torch.cat([rgb_pool, t_pool, match_emb], dim=1)
            logits = self.fc(pooled)

            # 호모그래피 잔차 (Delta H)
            delta_h = logits[:, :8].view(B, 8)
            one = torch.ones_like(delta_h[:, 0])
            H_res = torch.stack([
                torch.stack([one + delta_h[:, 0], delta_h[:, 1], delta_h[:, 2]], dim=-1),
                torch.stack([delta_h[:, 3], one + delta_h[:, 4], delta_h[:, 5]], dim=-1),
                torch.stack([delta_h[:, 6] * self.h_res_scale, delta_h[:, 7] * self.h_res_scale, one], dim=-1),
            ], dim=1)
            
            # TPS 오프셋 잔차 (TPS Residuals)
            tps_params = logits[:, 8:]
            res_offsets = tps_params.view(B, self.tps_grid * self.tps_grid, 2) * self.tps_res_scale

            # C. 글로벌 상태 업데이트 (Update Global State)
            # Korean: 예측된 잔차 행렬을 기존 전체 변환에 행렬곱/덧셈으로 합성한다.
            # English: Compose the predicted residual into the overall transformation using matrix multiplication/addition.
            H_total = torch.bmm(H_res, H_total)
            TPS_total_offsets = TPS_total_offsets + res_offsets

            # D. 다음 반복을 위한 워핑 (Warp for next iteration)
            if i < self.iters:
                # 호모그래피 워핑 (Homography Warp)
                H_normalized = H_total / (H_total[:, 2:3, 2:3].abs().clamp(min=1e-8))
                H_norm = torch.bmm(torch.bmm(S, H_normalized), S_inv)
                H_norm_inv = torch.inverse(H_norm.to(torch.float32)).to(dtype)
                
                src_coords = torch.bmm(grid_homo_base, H_norm_inv.transpose(-2, -1))
                src_coords = src_coords[..., :2] / (src_coords[..., 2:3].clamp(min=1e-8))
                src_coords = src_coords.view(B, H_feat, W_feat, 2)
                warped = F.grid_sample(features_t, src_coords, mode='bilinear', align_corners=True)
                
                # TPS 워핑 (Non-rigid refinement)
                # Korean: 호모그래피로 해결 불가능한 미세한 국소적 왜곡을 TPS로 추가 보정한다.
                # English: Correct fine local distortions that cannot be resolved by homography using TPS.
                ctrl_lin = torch.linspace(-1, 1, self.tps_grid, device=device, dtype=dtype)
                ctrl = torch.stack(torch.meshgrid(ctrl_lin, ctrl_lin, indexing="ij"), dim=-1).reshape(1, -1, 2)
                
                ctrl_offsets = TPS_total_offsets.view(B, self.tps_grid, self.tps_grid, 2).permute(0, 3, 1, 2)
                dense_offsets = F.interpolate(ctrl_offsets, size=(H_feat, W_feat), mode='bilinear', align_corners=True)
                dense_offsets = dense_offsets.permute(0, 2, 3, 1)
                
                sampling_grid = grid_base + dense_offsets
                current_t = F.grid_sample(warped, sampling_grid, mode='bilinear', align_corners=True)

        return {
            "H": H_total,                                       # 추정된 최종 호모그래피
            "tps_params": {                                     # 추정된 TPS 파라미터
                "ctrl": ctrl.expand(B, -1, -1), 
                "offsets": TPS_total_offsets
            },
            "warped_t": current_t,                              # 정렬이 완료된 Thermal 특징 맵
        }

    def forward_ablation(self, features_rgb: Tensor, features_t: Tensor, matches: Dict) -> Dict:
        """
        Korean: 자가 보정을 비활성화하고 항등 변환(Identity)과 원본 특징을 반환한다.
        English: Disables self-calibration and returns identity transformation and original features.
        """
        B = features_rgb.shape[0]
        device = features_rgb.device
        dtype = features_rgb.dtype
        
        # Identity Homography
        H_identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        
        # Identity TPS (Zero offsets)
        ctrl_lin = torch.linspace(-1, 1, self.tps_grid, device=device, dtype=dtype)
        ctrl = torch.stack(torch.meshgrid(ctrl_lin, ctrl_lin, indexing="ij"), dim=-1).reshape(1, -1, 2)
        TPS_identity_offsets = torch.zeros(B, self.tps_grid * self.tps_grid, 2, device=device, dtype=dtype)
        
        return {
            "H": H_identity,
            "tps_params": {
                "ctrl": ctrl.expand(B, -1, -1),
                "offsets": TPS_identity_offsets
            },
            "warped_t": features_t,
        }
