"""
Modality Bridging Adapter (MBA).

한국어: RGB와 열화상 간의 격차를 줄이기 위해 사전 처리 및 채널 주의 메커니즘을 적용한다.
English: Applies preprocessing and channel attention to bridge RGB and thermal modalities.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sobel_filter(channels: int) -> nn.Conv2d:
    """
    내부 Sobel 엣지 필터 (Internal Sobel Filter).
    Korean: 열화상 이미지에서 기하학적 엣지 정보를 추출하여 특징 맵을 보강한다.
    $I_{edge} = \sqrt{(\mathbf{K}_x * I)^2 + (\mathbf{K}_y * I)^2}$
    English: Extract geometric edge information from thermal images to reinforce feature maps.
    $I_{edge} = \sqrt{(\mathbf{K}_x * I)^2 + (\mathbf{K}_y * I)^2}$
    """

    kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    weight = torch.stack([kernel_x, kernel_y], dim=0).unsqueeze(1)
    conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, padding=1, bias=False, groups=channels)
    weight = weight.repeat(channels, 1, 1, 1)
    conv.weight.data = weight
    for p in conv.parameters():
        p.requires_grad = False
    return conv


class EAEFBlock(nn.Module):
    """
    명시적 주의 집중 융합 블록 (Explicit Attention-Enhanced Fusion).
    Korean: 채널 주의 집중 메커니즘을 통해 RGB와 Thermal 모달리티 간의 상대적 가중치를 명시적으로 계산하여 특징을 강화한다.
    $\alpha = \text{Softmax}([A_{rgb} ; A_{t}])$, $F'_{rgb} = \alpha_{rgb} \odot F_{rgb}$, $F'_{t} = \alpha_{t} \odot F_{t}$
    English: Explicitly computes relative weights between RGB and Thermal modalities via channel attention to enhance feature maps.
    $\alpha = \text{Softmax}([A_{rgb} ; A_{t}])$, $F'_{rgb} = \alpha_{rgb} \odot F_{rgb}$, $F'_{t} = \alpha_{t} \odot F_{t}$
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(dim // 4, 1)
        # Average Importance Block (AIB): $A_{avg} = \text{MLP}(\text{GAP}(F))$
        self.aib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )
        # Attentive Contrastive Block (ACB): $A_{max} = \text{MLP}(\text{GMP}(F))$
        self.acb = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb: Tensor, thermal: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 모달리티 채널 가중치 계산 (Modality Channel Weight Calculation)
        # Korean: 각 모달리티의 글로벌 맥락 정보를 결합하여 채널별 중요도를 산출한다.
        # English: Combine global context of each modality to compute channel-wise importance.
        att_rgb = self.aib(rgb) + self.acb(rgb)
        att_t = self.aib(thermal) + self.acb(thermal)
        
        # Softmax 경쟁 기반 통합 (Competency-based Fusion)
        # Korean: 두 모달리티 간의 상대적 정보를 비교하여 통합 가중치를 결정한다.
        # English: Determine integration weights by comparing relative information between modalities.
        weights = torch.stack([att_rgb, att_t], dim=1)
        weights = self.softmax(weights)
        
        # 특징 강조 (Feature Enhancement)
        # Korean: 계산된 가중치를 원본 특징 맵에 적용한다.
        # English: Apply the calculated weights to the original feature maps.
        fused_rgb = weights[:, 0] * rgb
        fused_t = weights[:, 1] * thermal
        return fused_rgb, fused_t, weights


class ModalityBridgingAdapter(nn.Module):
    """
    RGB와 열화상 간 모달리티 차이를 줄이는 어댑터.

    한국어: 채널 증강, 공유 임베딩, 선택적 EAEF 블록을 포함한다.
    English: Includes channel augmentation, shared embedding, and optional EAEF block.
    """

    def __init__(
        self,
        in_ch_rgb: int = 3,
        in_ch_t: int = 1,
        out_dim: int = 256,
        use_eaef: bool = True,
        use_sobel: bool = True,
    ) -> None:
        super().__init__()
        self.use_eaef = use_eaef
        self.use_sobel = use_sobel
        mid_dim = max(out_dim // 2, 16)

        self.rgb_embed = nn.Sequential(
            nn.Conv2d(in_ch_rgb, mid_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dim, affine=True),
        )
        self.thermal_pre = nn.Conv2d(in_ch_t, mid_dim, kernel_size=3, padding=1)
        self.edge_filter = _sobel_filter(in_ch_t)
        self.thermal_embed = nn.Sequential(
            nn.InstanceNorm2d(mid_dim + 2 * in_ch_t, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim + 2 * in_ch_t, out_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dim, affine=True),
        )
        self.ln = nn.InstanceNorm2d(out_dim, affine=True)
        self.eaef = EAEFBlock(out_dim) if use_eaef else None

    def forward(self, rgb: Tensor, thermal: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        """
        모달리티 브릿징 수행 (Forward Pass).
        Korean: 입력 RGB와 Thermal 데이터를 백본 처리에 적합한 공통 공간으로 투영한다.
        English: Project input RGB and Thermal data into a common space suitable for backbone processing.
        """
        
        # 1. RGB 특징 임베딩 (RGB Embedding)
        # Korean: 학습 가능한 컨볼루션 계층을 통해 RGB 특징을 추상화한다.
        # English: Abstract RGB features through learnable convolutional layers.
        rgb_feat = self.rgb_embed(rgb)
        
        # 2. 열화상 기하 특징 추출 (Thermal Edge Extraction)
        # Korean: Sobel 필터를 사용하여 열화상의 경계선 정보를 추출한다. (사용 안 할 경우 0으로 채움)
        # English: Extract boundary information from thermal images using Sobel filters. (Fill with zeros if disabled)
        if self.use_sobel:
            edges = self.edge_filter(thermal) # (B, 2*Ct, H, W)
        else:
            edges = torch.zeros(
                thermal.shape[0], 2 * thermal.shape[1], thermal.shape[2], thermal.shape[3],
                device=thermal.device, dtype=thermal.dtype
            )
        
        # 3. 열화상 보강 임베딩 (Thermal Enhancement)
        # Korean: 원본 열화상 정보와 엣지 특징을 결합하여 모달리티 고유 정보를 보존한다.
        # English: Preserve modality-specific information by combining raw thermal data with edge features.
        thermal_pre = self.thermal_pre(thermal)
        thermal_feat = torch.cat([thermal_pre, edges], dim=1)
        thermal_feat = self.thermal_embed(thermal_feat)

        # 4. 정규화 계층 (Normalization Phase)
        # Korean: Instance Normalization을 통해 모달리티 간의 스타일 격차를 최소화한다.
        # English: Minimize the style gap between modalities via Instance Normalization.
        rgb_feat = self.ln(rgb_feat)
        thermal_feat = self.ln(thermal_feat)

        aux: Dict[str, Tensor] = {}
        # 5. 교차 모달리티 주의 집중 (Cross-modal Interaction)
        # Korean: EAEF 블록을 통해 두 모달리티 간의 상호작용 및 중요도 조절을 수행한다.
        # English: Perform interaction and importance adjustment between modalities via the EAEF block.
        if self.use_eaef and self.eaef is not None:
            rgb_feat, thermal_feat, weights = self.eaef(rgb_feat, thermal_feat)
            aux["eaef_weights"] = weights

        # 통계적 정렬 지표 (Statistical Alignment Metric)
        # Korean: 특징 공간상의 평균 차이를 모니터링한다 (학습 보조용).
        # English: Monitor the mean difference in feature space (for training assistance).
        aux["stat_l2"] = (rgb_feat.mean() - thermal_feat.mean()).abs()

        return rgb_feat, thermal_feat, aux
