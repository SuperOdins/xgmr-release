"""
XGMR 모델 어셈블리.

한국어: MBA → EBA → Coarse→Fine 매칭 → Self-Calib → Q-Fusion 파이프라인을 구성한다.
English: Assembles the full pipeline with MBA, EBA, coarse-to-fine matcher, self-calibration, and quality fusion.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .backbones import build_backbone
from .mba import ModalityBridgingAdapter
from .loftr_like import CoarseFineMatcher
from .self_calib import SelfCalibratingHead
from .qfusion import QFusion
from xgmr.geometry.epipolar import homography_bias_mask


class XGMR(nn.Module):
    """Full XGMR assembly."""

    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        mba_cfg: Dict[str, Any],
        matcher_cfg: Dict[str, Any],
        selfcalib_cfg: Dict[str, Any],
        qfusion_cfg: Dict[str, Any],
        use_mba: bool = True,
        use_self_calib: bool = True,
        use_qfusion: bool = True,
    ) -> None:
        """
        XGMR 모델 초기화.
        Korean: 각 모듈(Backbone, MBA, Matcher, Self-Calib, Q-Fusion)을 설정에 따라 초기화한다.
        English: Initializes each module (Backbone, MBA, Matcher, Self-Calib, Q-Fusion) based on the configuration.
        """
        super().__init__()
        self.use_mba = use_mba
        self.use_self_calib = use_self_calib
        self.use_qfusion = use_qfusion

        # MBA 사용 여부에 따른 입력 채널 설정
        # Korean: MBA를 사용할 경우 어댑터 출력 차원(256)을 사용하고, 아니면 원본 채널(RGB:3, Thermal:1)을 사용한다.
        # English: If MBA is used, adapter output dimension (256) is used; otherwise, raw channels (RGB:3, Thermal:1) are used.
        if use_mba:
            rgb_in = mba_cfg.get("out_dim", 256)
            t_in = mba_cfg.get("out_dim", 256)
        else:
            rgb_in = 3
            t_in = 1 

        # 특징 추출 백본 빌드 (Modality-specific backbones)
        # Korean: RGB와 Thermal 각각에 대해 독립적인 백본을 생성하여 모달리티별 특징을 추출한다.
        # English: Build independent backbones for RGB and Thermal to extract modality-specific features.
        self.rgb_backbone, feat_dim = build_backbone({**backbone_cfg, "in_channels": rgb_in})
        self.t_backbone, _ = build_backbone({**backbone_cfg, "in_channels": t_in})

        # 핵심 모듈 정의
        # Korean: MBA(어댑터), Matcher(대응점 검색), Self-Calib(자가 보정), Q-Fusion(품질 기반 융합) 모듈을 정의한다.
        # English: Define key modules: MBA (Adapter), Matcher (Correspondence searching), Self-Calib, and Q-Fusion.
        self.mba = ModalityBridgingAdapter(**mba_cfg) if use_mba else nn.Identity()
        self.matcher = CoarseFineMatcher(**matcher_cfg)
        self.self_calib = SelfCalibratingHead(**selfcalib_cfg) if use_self_calib else nn.Identity()
        self.qfusion = QFusion(dim=matcher_cfg.get("dim", 256), **qfusion_cfg) if use_qfusion else nn.Identity()
        
        # 특징 맵 차원 투영 (Project to matching dimension)
        # Korean: 백본에서 나온 특징 맵을 Matcher가 사용하는 공통 차원(예: 256)으로 투영한다.
        # English: Project feature maps from backbones to a common dimension (e.g., 256) used by the matcher.
        self.proj = nn.Conv2d(feat_dim, matcher_cfg.get("dim", 256), kernel_size=1)

    def forward(
        self,
        rgb: Tensor,
        thermal: Tensor,
        H0: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        XGMR 순전파 (Forward Pass).
        Korean: MBA → 특징 추출 → 기하학적 편향 생성(학습 시) → 매칭 → 자가 보정 → 품질 기반 융합 과정을 수행한다.
        English: Performs the pipeline: MBA → Feature Extraction → Geometric Bias (Training) → Matching → Self-Calib → Q-Fusion.
        """

        aux: Dict[str, Any] = {}
        
        # 1. Modality Bridging Adapter (MBA)
        # Korean: 서로 다른 모달리티(RGB, Thermal) 간의 표현 격차를 줄이기 위해 MBA를 적용한다.
        # English: Apply MBA to bridge the representation gap between disparate modalities (RGB and Thermal).
        if self.use_mba and isinstance(self.mba, ModalityBridgingAdapter):
            rgb_feat, t_feat, mba_aux = self.mba(rgb, thermal)
            aux.update(mba_aux)
        else:
            rgb_feat, t_feat = rgb, thermal

        # 2. Feature Extraction & Projection
        # Korean: 백본을 통해 특징을 추출하고 매칭 차원으로 투영한다.
        # English: Extract features via backbones and project them into the matching dimension.
        rgb_encoded = self.proj(self.rgb_backbone(rgb_feat))
        t_encoded = self.proj(self.t_backbone(t_feat))

        # 3. Generate Homography Bias Mask (Training only)
        # Korean: 학습 시, 자가 지도 학습을 위한 기하학적 가이드로서 호모그래피 기반 편향 마스크를 생성한다.
        # English: During training, generate a homography-based bias mask as a geometric guide for self-supervision.
        # [Technical Note] We use Homography-based bias (Point-to-Point) for self-supervised guidance.
        # Epipolar-based bias (K-matrix calibration) is removed to align with thesis architecture.
        epi_bias = None
        if self.training and H0 is not None:
            rgb_feat_h, rgb_feat_w = rgb_encoded.shape[-2], rgb_encoded.shape[-1]
            rgb_img_h, rgb_img_w = rgb.shape[-2], rgb.shape[-1]
            t_feat_h, t_feat_w = t_encoded.shape[-2], t_encoded.shape[-1]
            t_img_h, t_img_w = thermal.shape[-2], thermal.shape[-1]
            
            strip_width = getattr(self.matcher, "strip_width", 5.0)
            sigma = strip_width / 2.0

            # 특징 맵 좌표계로 호모그래피 변환 (Scaling H to feature map resolution)
            # Korean: 입력 영상 해상도 기준의 호모그래피(H0)를 특징 맵 해상도 기준(H_feat)으로 스케일링한다.
            # English: Scale the image-level homography (H0) to the feature-level homography (H_feat).
            H0_f32 = H0.to(torch.float32)
            stride_rgb_x, stride_rgb_y = rgb_img_w / rgb_feat_w, rgb_img_h / rgb_feat_h
            stride_t_x, stride_t_y = t_img_w / t_feat_w, t_img_h / t_feat_h
            
            S_rgb = torch.tensor([[1/stride_rgb_x, 0, 0], [0, 1/stride_rgb_y, 0], [0, 0, 1]], device=rgb.device, dtype=torch.float32)
            S_t_inv = torch.tensor([[stride_t_x, 0, 0], [0, stride_t_y, 0], [0, 0, 1]], device=rgb.device, dtype=torch.float32)
            
            H_feat = S_rgb @ H0_f32 @ S_t_inv
            
            # 편향 마스크 생성 (Generating the mask for Dual-Softmax)
            # Korean: H_feat을 사용하여 매칭 가능성이 높은 영역에 가중치를 부여하는 마스크를 생성한다.
            # English: Create a mask using H_feat to give higher weights to geometrically plausible matching regions.
            epi_bias = homography_bias_mask(
                H_feat,
                q_hw=(rgb_feat_h, rgb_feat_w),
                k_hw=(t_feat_h, t_feat_w),
                sigma=sigma
            )
                 
        # 4. Neural Matching (Coarse-to-Fine)
        # Korean: Dual-Softmax 확률 기반 정합을 통해 특징 맵 간의 대응관계를 찾는다.
        # English: Find correspondences between feature maps via dual-softmax probabilistic matching.
        # Inference: epi_bias remains None, using global attention
        matches = self.matcher(rgb_encoded, t_encoded, epi_bias=epi_bias)

        # 5. Self-Calibration (Homography Estimation)
        # Korean: 추정된 정합점을 기반으로 정교한 호모그래피(H)를 예측하고 영상을 워핑한다.
        # English: Predict a refined homography (H) based on estimated matches and warp the image.
        if self.use_self_calib and isinstance(self.self_calib, SelfCalibratingHead):
            calib = self.self_calib(rgb_encoded, t_encoded, matches)
            aux.update(calib)
            warped_t = calib["warped_t"]
        else:
            warped_t = t_encoded
            aux["H"] = torch.eye(3, device=rgb.device, dtype=rgb.dtype).unsqueeze(0).repeat(rgb.shape[0], 1, 1)
            aux["tps_params"] = {}

        # 6. Geometric Reprojection Error Calculation (Statistics)
        # Korean: 예측된 H를 사용하여 재투영 오차를 실시간으로 모니터링한다. (픽셀 단위)
        # English: Monitor the reprojection error in real-time using the predicted H. (Unit: Pixels)
        # H를 coords1에 적용 후 coords0과의 거리를 픽셀 단위로 계산
        coords0 = matches["matches_c"]["coords0"]  # (B, N, 2) - feature 좌표
        coords1 = matches["matches_c"]["coords1"]  # (B, N, 2) - feature 좌표
        scores = matches["matches_c"]["scores"]    # (B, N)
        
        # 예측된 H 가져오기 (feature 좌표계)
        H_pred = aux.get("H", torch.eye(3, device=rgb.device, dtype=rgb.dtype).unsqueeze(0).repeat(rgb.shape[0], 1, 1))
        
        # coords1에 H 적용하여 RGB 좌표계로 재투영
        b, n, _ = coords1.shape
        ones = torch.ones(b, n, 1, device=coords1.device, dtype=coords1.dtype)
        coords1_homo = torch.cat([coords1, ones], dim=-1)  # (B, N, 3)
        
        # H @ coords1 (역변환)
        coords1_warped = torch.bmm(coords1_homo, H_pred.transpose(-2, -1))  # (B, N, 3)
        coords1_warped = coords1_warped[..., :2] / (coords1_warped[..., 2:3].clamp(min=1e-8))  # (B, N, 2)
        
        # feature 좌표에서 차이 계산
        diff = coords0 - coords1_warped  # (B, N, 2)
        
        # x/y 개별 스케일링 후 L2 norm (non-uniform scaling 대응)
        feat_h, feat_w = rgb_encoded.shape[-2], rgb_encoded.shape[-1]
        img_h, img_w = rgb.shape[-2], rgb.shape[-1]
        
        scale_x = img_w / feat_w
        scale_y = img_h / feat_h
        
        # 픽셀 단위로 변환
        diff_pixels = diff.clone()
        diff_pixels[..., 0] *= scale_x  # x 방향
        diff_pixels[..., 1] *= scale_y  # y 방향
        
        # L2 norm
        reproj_error = diff_pixels.norm(dim=-1)  # (B, N)
        
        # 가중 평균 (매칭 신뢰도 반영)
        weighted_reproj = (reproj_error * scores).sum(dim=-1) / (scores.sum(dim=-1) + 1e-8)
        reproj_mean_pixels = weighted_reproj.mean().item()
        
        stats = {
            "num_matches": matches["P"].sum().item(),
            "inlier_ratio": matches["P"].max(dim=-1).values.mean().item(),
            "prob_mean": matches["P"].mean().item(),
            "reproj_mean": reproj_mean_pixels,  # 픽셀 단위 reprojection error
        }

        # 7. Quality-aware Fusion (Q-Fusion)
        # Korean: 특징 맵의 국소적 정합 품질을 예측하여 신뢰할 수 있는 영역 위주로 특징을 융합한다.
        # English: Predict local matching quality and fuse features by prioritizing reliable regions.
        if self.use_qfusion and isinstance(self.qfusion, QFusion):
            fused, q_aux = self.qfusion(rgb_encoded, warped_t, stats)
            aux.update({"Qmap": q_aux["Q"], "wmap": q_aux["w"], "fused": fused, "quality_scalar": q_aux["quality_scalar"]})
        else:
            fused = (rgb_encoded + warped_t) / 2.0
            aux["Qmap"] = torch.zeros_like(fused[:, :1])

        # 8. Update Results Dict
        # Korean: 모든 중간 결과와 통계치를 결과 딕셔너리에 합친다.
        # English: Aggregate all intermediate results and statistics into the output dictionary.
        aux.update(stats)
        aux.update(matches)
        
        # [전문가 조치] Trainer가 정확한 스케일을 계산할 수 있도록 특징 맵 해상도 포함
        # Useful for downstream tasks requiring scale-aware coordinate calculations.
        aux["feat_hw"] = (rgb_encoded.shape[-2], rgb_encoded.shape[-1])
        
        return aux
