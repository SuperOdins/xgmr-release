"""
Epipolar-Biased Attention blocks.

한국어: 에피폴라 기하를 이용하여 크로스 어텐션 점수에 기하학적 편향을 추가한다.
English: Injects geometric priors into cross-attention logits via epipolar bias masks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xgmr.utils.ablation import ablation_entry


def _flatten_tokens(t: Tensor) -> Tensor:
    if t.dim() == 4:
        b, c, h, w = t.shape
        return t.permute(0, 2, 3, 1).reshape(b, h * w, c)
    return t


def _reshape_back(t: Tensor, ref: Tensor) -> Tensor:
    if ref.dim() == 4:
        b, c, h, w = ref.shape
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2)
    return t


class EpipolarBiasedCrossAttention(nn.Module):
    """
    에피폴라 편향 크로스 어텐션 (Epipolar-Biased Cross-Attention).
    Korean: 두 모달리티의 토큰 간 상관관계를 계산할 때 에피폴라 기하 제약 조건을 편향(Bias)으로 주입하여 정합의 기하학적 정확도를 높인다.
    $Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \lambda B_{epi}\right)V$
    English: Enhances matching geometric accuracy by injecting epipolar geometry constraints as a bias when computing correlation between tokens of two modalities.
    $Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \lambda B_{epi}\right)V$
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias_strength: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.bias_strength = bias_strength

    @ablation_entry("no_eba")
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        epi_bias: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        어텐션 연산 수행 (Forward Pass).
        Korean: 전사(Projection)된 쿼리, 키, 값을 이용하여 어텐션 스코어를 계산하고 에피폴라 편향을 더한다.
        English: Compute attention scores using projected Q, K, and V, then add epipolar bias.
        """

        # 1. 선형 투영 (Linear Projections)
        # $Q = W_q f_q, K = W_k f_k, V = W_v f_v$
        q_ = self.q_proj(_flatten_tokens(q))
        k_ = self.k_proj(_flatten_tokens(k))
        v_ = self.v_proj(_flatten_tokens(v))

        b, nq, _ = q_.shape
        nk = k_.shape[1]

        # 멀티헤드 분할 (Multi-head Splitting)
        q_ = q_.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = k_.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention 계산
        # $Score = \frac{QK^T}{\sqrt{head\_dim}}$
        import math
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q_, k_.transpose(-2, -1)) / scale
        
        # 3. 에피폴라 기하 편향 주입 (Epipolar Bias Injection)
        # $Score' = Score + \lambda B_{epi}$
        if epi_bias is not None:
            if epi_bias.dim() == 5:
                # 해상도 기반 마스크를 2D 행렬 형태로 변형
                epi_bias = epi_bias.reshape(b, nq, nk)
            epi_bias = epi_bias.unsqueeze(1)  # 헤드 차원 확장
            attn = attn + self.bias_strength * epi_bias

        # 추가적인 어텐션 마스크 처리 (Optional Masking)
        if attn_mask is not None:
            attn = attn + attn_mask

        # 4. 가중치 정규화 및 특징 집계 (Softmax & Aggregation)
        # $Out = \text{Softmax}(Score')V$
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_)
        
        # 5. 출력 투영 및 복원 (Final Projection & Reshape)
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        out = self.out_proj(out)
        return _reshape_back(out, q)

    def forward_ablation(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        epi_bias: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Korean: 에피폴라 편향(epi_bias)을 무시하고 일반 어텐션을 수행한다.
        English: Performs standard attention by ignoring the epipolar bias (epi_bias).
        """
        return self.forward(q, k, v, epi_bias=None, attn_mask=attn_mask)
