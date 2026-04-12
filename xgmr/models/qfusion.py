"""
품질 인지 융합 모듈.

한국어: 매칭 품질 지표를 사용하여 타일 단위 게이팅을 수행한다.
English: Performs tile-wise gating using match quality indicators.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xgmr.utils.ablation import ablation_entry


def _sanitize_stat(value, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    통계치 정제 (Stat Sanitization).
    Korean: 입력 통계치를 텐서로 변환하고 NaN 또는 Inf 값을 0으로 대체하여 수치적 안정성을 확보한다.
    English: Convert raw statistics to a tensor and replace NaN/Inf values with zeros to ensure numerical stability.
    """

    if not torch.is_tensor(value):
        tensor = torch.tensor([value], device=device, dtype=dtype)
    else:
        tensor = value.to(device=device, dtype=dtype).reshape(-1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor




class QFusion(nn.Module):
    """
    품질 기반 융합 모듈 (Quality-aware Fusion).
    Korean: 매칭 품질 지표(Matching Quality)와 특징 정보를 결합하여 신뢰할 수 있는 영역 위주로 특징을 융합한다.
    $F_{fused} = w \odot F_{thermal} + (1-w) \odot F_{rgb}$
    English: Fuses features by prioritizing reliable regions using matching quality indicators and feature maps.
    $F_{fused} = w \odot F_{thermal} + (1-w) \odot F_{rgb}$
    """

    def __init__(
        self,
        dim: int = 64,
        tile: int = 32,
        k: float = 8.0,
        tau: float = 0.5,
        num_matches_scale: float = 100.0,
    ) -> None:
        """
        Args:
            dim: 피처 채널 차원 (Feature channel dimension)
            tile: 품질 추정 타일 크기 (Tile size for regional quality estimation)
            k: 게이팅 시그모이드 샤프니스 (Sharpness of the gating sigmoid)
            tau: 게이팅 임계값 (Threshold for the gating signal)
            num_matches_scale: 매칭 수 정규화 스케일 (Normalization scale for num_matches)
        """
        super().__init__()
        self.tile = tile
        self.k = k
        self.tau = tau
        self.num_matches_scale = num_matches_scale
        
        # 공간적 품질 추정기 (Spatial Quality Estimator, Ψ_qual)
        # Korean: RGB와 워핑된 Thermal 특징을 입력받아 공간적인 품질 가중치 맵 $Q_{map}$을 생성한다.
        # $Q_{map} = \Psi_{qual}([\Phi_{rgb} ; \Phi_{thermal}])$
        # English: Generates a spatial quality weight map $Q_{map}$ from RGB and warped Thermal features.
        # $Q_{map} = \Psi_{qual}([\Phi_{rgb} ; \Phi_{thermal}])$
        self.quality_estimator = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 통계 임베딩 (Statistical Embedding, E_stat)
        # Korean: 정합 과정에서 추출된 4가지 글로벌 통계치를 공간적 편향(Bias) 정보 $S_{bias}$로 변환한다.
        # $S_{bias} = E_{stat}(V_{stats})$
        # English: Transform 4 global matching statistics into spatial bias information $S_{bias}$.
        # $S_{bias} = E_{stat}(V_{stats})$
        self.stat_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, dim),
        )
        
        # 공간적 스무딩 (Spatial Smoothing)
        # Korean: 추정된 품질 맵의 노이즈를 억제하고 부드러운 변화를 유도한다.
        # English: Suppress noise in the estimated quality map and induce smooth transitions.
        self.smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.smooth.weight, 1.0 / 9.0)

    @ablation_entry("no_qfusion")
    def forward(
        self,
        F_rgb: Tensor,
        F_t_warped: Tensor,
        match_stats: Dict[str, Tensor | float | int],
    ) -> Tuple[Tensor, Dict]:
        """
        융합 프로세스 수행 (Forward Pass).
        Korean: 전역 통계와 국소 피처를 결합하여 공간적 품질을 예측하고 특징을 융합한다.
        English: Predict spatial quality and fuse features by combining global stats and local features.
        """

        batch, c, h, w = F_rgb.shape
        device = F_rgb.device
        dtype = F_rgb.dtype
        
        # 1. 통계 추출 및 정규화 (Stat Normalization & Vector Construction)
        # Korean: 매칭 수($N$), 인라이어 비율($R_{in}$), 평균 확률($P_{mean}$), 재투영 오차($E_{rep}$)를 정제하여 통계 벡터 $V_{stats}$를 구성한다.
        # $V_{stats} = [\tanh(N/S), R_{in}, P_{mean}, \exp(-E_{rep})]^T$
        # English: Construct the statistical vector $V_{stats}$ by sanitizing $N$ matches, $R_{in}$ ratio, $P_{mean}$ probability, and $E_{rep}$ error.
        # $V_{stats} = [\tanh(N/S), R_{in}, P_{mean}, \exp(-E_{rep})]^T$
        num_matches = _sanitize_stat(match_stats.get("num_matches", 0), device, dtype)
        inlier_ratio = _sanitize_stat(match_stats.get("inlier_ratio", 0.0), device, dtype)
        prob_mean = _sanitize_stat(match_stats.get("prob_mean", 0.0), device, dtype)
        reproj_mean = _sanitize_stat(match_stats.get("reproj_mean", 1.0), device, dtype)
        
        # 전역 통계 벡터 생성 (B, 4)
        stats_vec = torch.stack([
            torch.tanh(num_matches / self.num_matches_scale),  # Normalization
            inlier_ratio,                                      # Range [0,1]
            prob_mean,                                         # Range [0,1]
            torch.exp(-reproj_mean),                          # Error inverse (High error = Low quality)
        ], dim=-1).expand(batch, -1)  # (B, 4)
        
        # 2. 통계 임베딩 반영 (Feature Bias Projection)
        # Korean: 전역 통계 정보를 모든 픽셀에 공간적 편향 $S_{bias}$로 투영하여 특징을 보강한다.
        # $\Phi' = \Phi + S_{bias}$ where $S_{bias} = E_{stat}(V_{stats})$
        # English: Project global stats as spatial bias $S_{bias}$ to reinforce features across all pixels.
        # $\Phi' = \Phi + S_{bias}$ where $S_{bias} = E_{stat}(V_{stats})$
        stat_bias = self.stat_embed(stats_vec)  # (B, dim)
        stat_bias = stat_bias.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # (B, dim, H, W)
        
        # RGB와 워핑된 Thermal 피처 결합 및 Bias 반영
        concat_feat = torch.cat([F_rgb, F_t_warped], dim=1)  # (B, 2C, H, W)
        concat_feat[:, :c, :, :] = concat_feat[:, :c, :, :] + stat_bias  # RGB Reinforcement
        concat_feat[:, c:, :, :] = concat_feat[:, c:, :, :] + stat_bias  # Thermal Reinforcement
        
        # 3. 타일 단위 품질 추정 (Tile-wise Quality Gating)
        # Korean: 국소적 노이즈에 강건하도록 타일 단위($T \times T$)로 정보를 집약(Pooling)하여 품질을 추정한다.
        # $Q_{tile} = \Psi_{qual}(\text{AvgPool}(\Phi', T))$
        # English: Aggregate information on a tile-wise basis ($T \times T$) via pooling for robust quality estimation.
        # $Q_{tile} = \Psi_{qual}(\text{AvgPool}(\Phi', T))$
        if self.tile > 1:
            # 타일 크기에 맞춰 해상도 축소
            th, tw = max(h // self.tile, 2 if h >= 2 else 1), max(w // self.tile, 2 if w >= 2 else 1)
            tile_feat = F.adaptive_avg_pool2d(concat_feat, (th, tw))
            
            # 타일 단위 품질 예측 및 원래 해상도 복원 (Bilinear upsampling)
            Qtile = self.quality_estimator(tile_feat) 
            Qmap_raw = F.interpolate(Qtile, size=(h, w), mode='bilinear', align_corners=True)
        else:
            Qmap_raw = self.quality_estimator(concat_feat)  # Per-pixel mode
        
        # 4. 최종 게이팅 및 융합 (Gating & Soft Fusion)
        # Korean: 품질 마스크 $Q$를 기반으로 게이팅 신호 $w$를 생성하여 가중 융합을 수행한다.
        # $w = \sigma(k \cdot (Q - \tau))$, $F_{fused} = w \cdot F_t + (1-w) \cdot F_{rgb}$
        # English: Perform weighted fusion by creating a gating signal $w$ based on the quality map $Q$.
        # $w = \sigma(k \cdot (Q - \tau))$, $F_{fused} = w \cdot F_t + (1-w) \cdot F_{rgb}$
        Qmap = self.smooth(Qmap_raw)
        wmap = torch.sigmoid(self.k * (Qmap - self.tau)) # [0: High RGB, 1: High Thermal]

        # 가중 융합 수행
        fused = wmap * F_t_warped + (1.0 - wmap) * F_rgb
        
        # 로깅용 품질 스칼라값
        quality_scalar = Qmap.mean().item()
        
        aux = {"Q": Qmap, "w": wmap, "quality_scalar": quality_scalar}
        return fused, aux

    def forward_ablation(
        self,
        F_rgb: Tensor,
        F_t_warped: Tensor,
        match_stats: Dict[str, Tensor | float | int],
    ) -> Tuple[Tensor, Dict]:
        """
        Korean: 품질 기반 융합을 비활성화하고 단순 평균 융합을 수행한다.
        English: Disables quality-aware fusion and performs simple average fusion.
        """
        fused = (F_rgb + F_t_warped) / 2.0
        aux = {
            "Q": torch.zeros_like(fused[:, :1]),
            "w": torch.zeros_like(fused[:, :1]),
            "quality_scalar": 0.0,
        }
        return fused, aux

