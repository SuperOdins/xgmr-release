"""
LoFTR 유사 Coarse→Fine 매처.

한국어: coarse 격자 토큰과 fine 윈도 탐색을 수행하고 dual-softmax 또는 Sinkhorn OT를 지원한다.
English: Implements a lightweight LoFTR-like matcher with coarse and fine refinement.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .attention import EpipolarBiasedCrossAttention
from .matcher import dual_softmax_match


def _coords_grid(height: int, width: int, device: torch.device) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=device),
        torch.linspace(0, width - 1, width, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y], dim=-1)
    return coords.view(-1, 2)


class PositionEncodingSine(nn.Module):
    """
    2D 사인파 위치 인코딩 (2D Sinusoidal Positional Encoding).
    Korean: 특징 맵의 개별 토큰에 2D 공간 정보를 주입하여 정합 시 상대적 위치 관계를 고려할 수 있게 한다.
    $PE(pos, 2i) = \sin(pos/10000^{2i/d})$, $PE(pos, 2i+1) = \cos(pos/10000^{2i/d})$
    English: Inject 2D spatial information into individual tokens of the feature map to consider relative positional relationships during matching.
    $PE(pos, 2i) = \sin(pos/10000^{2i/d})$, $PE(pos, 2i+1) = \cos(pos/10000^{2i/d})$
    """
    def __init__(self, d_model, max_shape=(256, 256), temp=10000.0):
        super().__init__()
        pe = torch.zeros((d_model, *max_shape))
        y, x = torch.meshgrid(torch.arange(max_shape[0]), torch.arange(max_shape[1]), indexing="ij")
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-torch.log(torch.tensor(temp)) / (d_model // 2)))
        
        pe_x = torch.zeros((d_model // 2, *max_shape))
        pe_y = torch.zeros((d_model // 2, *max_shape))

        pe_x[0::2, :, :] = torch.sin(x.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe_x[1::2, :, :] = torch.cos(x.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        pe_y[0::2, :, :] = torch.sin(y.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe_y[1::2, :, :] = torch.cos(y.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        pe = torch.cat([pe_x, pe_y], dim=0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, C, H, W)"""
        return x + self.pe[:, :x.size(2), :x.size(3)]


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2) # (B, C, H, W)


class CoarseFineMatcher(nn.Module):
    """
    Coarse→Fine 매칭 모듈 (Coarse-to-Fine Matching Module).
    Korean: 격자 단위의 대략적인 정합(Coarse) 후, 선택된 후보점에 대해 국소적 정밀 보정(Fine)을 수행한다.
    English: Performs coarse grid-level matching followed by local fine-grained refinement on selected candidates.
    """

    def __init__(
        self,
        dim: int = 256,
        layers: Tuple[int, int] = (2, 1),
        temp: float = 0.1,
        n_fine_matches: int = 512,
        strip_width: float = 5.0,
        num_heads: int = 4,
        pe_max_shape: Tuple[int, int] = (256, 256),
        pe_temp: float = 10000.0,
        bias_strength: float = 2.0,
        use_pe: bool = True,
        use_epi_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temp = temp
        self.n_fine_matches = n_fine_matches
        self.strip_width = strip_width
        self.use_pe = use_pe
        self.use_epi_bias = use_epi_bias
        self.pos_encoding = PositionEncodingSine(dim, max_shape=pe_max_shape, temp=pe_temp)
        self.coarse_layers = nn.ModuleList(
            [EpipolarBiasedCrossAttention(dim, num_heads=num_heads, bias_strength=bias_strength) for _ in range(layers[0])]
        )
        self.fine_layers = nn.ModuleList(
            [EpipolarBiasedCrossAttention(dim, num_heads=num_heads, bias_strength=bias_strength) for _ in range(layers[1])]
        )
        self.norm = LayerNorm2d(dim)
        if layers[1] > 0:
            self.refine_head = nn.Linear(dim * 2, 2)

    def forward(
        self,
        Frgb: Tensor,
        Ft: Tensor,
        epi_bias: Tensor | None = None,
    ) -> Dict:
        """
        Coarse-to-Fine 매칭 프로세스 (Forward Pass).
        Korean: 에피폴라 기하 기반의 주의 집중 기법을 통해 정합을 수행하고 정밀한 하위 픽셀 오차를 보정한다.
        English: Performs matching via epipolar geometry-based attention and refines sub-pixel offsets.
        """

        # 1. 위치 정보 주입 및 에피폴라 주의 집중 (Positional Encoding & Attention)
        # $F' = \text{Attention}(F + PE_{2D}, B_{epi})$
        rgb = self.pos_encoding(Frgb) if self.use_pe else Frgb
        t = self.pos_encoding(Ft) if self.use_pe else Ft
        bias = epi_bias if self.use_epi_bias else None
        for layer in self.coarse_layers:
            rgb = rgb + layer(self.norm(rgb), self.norm(t), self.norm(t), bias)
            t = t + layer(self.norm(t), self.norm(rgb), self.norm(rgb), bias)

        # 2. Coarse 단계 대응 확률 계산 (Coarse Assignment)
        # $P = \text{DualSoftmax}(S/\tau)$, where $S_{i,j} = \frac{f_i \cdot f_j^T}{\sqrt{d}}$
        b, d, h, w = rgb.shape
        rgb_tokens = self.norm(rgb).permute(0, 2, 3, 1).reshape(b, h * w, d)
        t_tokens = self.norm(t).permute(0, 2, 3, 1).reshape(b, h * w, d)

        scores = torch.matmul(rgb_tokens, t_tokens.transpose(-2, -1)) / (d ** 0.5)
        assignment = dual_softmax_match(scores, temp=self.temp)

        # 그리드 좌표 및 최대 확률 기반 선택
        coords = _coords_grid(h, w, device=rgb.device)
        coords = coords.unsqueeze(0).repeat(b, 1, 1)
        scores_max, idx = assignment.max(dim=-1)
        match_coords1 = torch.gather(coords, 1, idx.unsqueeze(-1).repeat(1, 1, 2))

        matches_c = {
            "coords0": coords,
            "coords1": match_coords1,
            "scores": scores_max,
            "scores_raw": scores,
        }

        # Fine refinement using actual fine_layers
        # Extract local windows around coarse matches and apply fine cross-attention
        
        if len(self.fine_layers) > 0:
            # 3. Fine 단계 후보 선택 (Top-K Selection)
            # Korean: 신뢰도가 높은 상위 K개의 정합점을 정밀 보정 대상으로 선정한다.
            # English: Select top-K highly confident matches for fine-grained refinement.
            k = min(self.n_fine_matches, h * w)
            topk_scores, topk_idx = torch.topk(scores_max, k, dim=1)
            
            topk_coords0 = torch.gather(coords, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
            topk_coords1 = torch.gather(match_coords1, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))
            
            # 4. 국소 특징 추출 (Local Feature Sampling)
            # Korean: `grid_sample`을 사용하여 정합점 위치의 정밀 특징 벡터를 샘플링한다.
            # English: Sample high-resolution feature vectors at match locations using `grid_sample`.
            grid0 = topk_coords0.clone()
            grid0[..., 0] = (grid0[..., 0] / (w - 1)) * 2 - 1
            grid0[..., 1] = (grid0[..., 1] / (h - 1)) * 2 - 1
            grid0 = grid0.unsqueeze(2)  # (B, K, 1, 2)
            
            grid1 = topk_coords1.clone()
            grid1[..., 0] = (grid1[..., 0] / (w - 1)) * 2 - 1
            grid1[..., 1] = (grid1[..., 1] / (h - 1)) * 2 - 1
            grid1 = grid1.unsqueeze(2)  # (B, K, 1, 2)
            
            feat0 = torch.nn.functional.grid_sample(
                rgb, grid0, mode='bilinear', align_corners=True
            ).squeeze(-1).permute(0, 2, 1)  # (B, K, D)
            
            feat1 = torch.nn.functional.grid_sample(
                t, grid1, mode='bilinear', align_corners=True
            ).squeeze(-1).permute(0, 2, 1)  # (B, K, D)
            
            # 5. Fine 단계 교차 어텐션 (Fine Cross-Attention)
            # Korean: 국소 특징 간의 상호작용을 통해 하위 픽셀 변위를 예측하기 위한 정보를 강화한다.
            # English: Enhance information for sub-pixel displacement prediction via local feature interaction.
            for fine_layer in self.fine_layers:
                feat0_in = self.norm(feat0.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
                feat1_in = self.norm(feat1.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
                feat0 = feat0 + fine_layer(feat0_in, feat1_in, feat1_in, None)
                
                feat0_in_updated = self.norm(feat0.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
                feat1 = feat1 + fine_layer(feat1_in, feat0_in_updated, feat0_in_updated, None)
            
            # 6. 하위 픽셀 오차 보정 (Sub-pixel Offset Regression)
            # Korean: 강화된 특징을 기반으로 최종적인 좌표 미세 조정값 $\Delta$를 예측한다.
            # $\Delta = \tanh(\text{MLP}([F_{rgb}^f ; F_{thermal}^f])) \times 0.5$
            # English: Predict the final coordinate fine-tuning value $\Delta$ based on the enhanced features.
            # $\Delta = \tanh(\text{MLP}([F_{rgb}^f ; F_{thermal}^f])) \times 0.5$
            feat0_norm = torch.nn.functional.normalize(feat0, dim=-1)
            feat1_norm = torch.nn.functional.normalize(feat1, dim=-1)
            
            feat_combined = torch.cat([feat0, feat1], dim=-1) 
            fine_offsets = torch.tanh(self.refine_head(feat_combined)) * 0.5 # (B, K, 2)
            
            fine_scores = (feat0_norm * feat1_norm).sum(dim=-1) 
            refined_coords1 = topk_coords1 + fine_offsets
            
            matches_f = {
                "coords0": topk_coords0,
                "coords1": refined_coords1,
                "scores": fine_scores,
            }
        else:
            # Fallback: use coarse matches without refinement
            matches_f = {
                "coords0": coords,
                "coords1": match_coords1,
                "scores": scores_max,
            }

        return {
            "matches_c": matches_c,
            "matches_f": matches_f,
            "P": assignment,
        }
