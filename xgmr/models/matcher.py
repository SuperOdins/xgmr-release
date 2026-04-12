"""
매칭 유틸리티.

한국어: Dual-Softmax 매칭 함수를 제공한다.
English: Provides dual-softmax matching helpers.
"""

from __future__ import annotations

import torch
from torch import Tensor
from xgmr.utils.ablation import ablation_entry


@ablation_entry("no_dsm")
def dual_softmax_match(S: Tensor, temp: float = 0.1) -> Tensor:
    """
    이중 소프트맥스 정합 (Dual-Softmax Matching).
    Korean: 행 방향과 열 방향으로 각각 소프트맥스를 적용하여 양방향 일관성을 가진 확률 행렬 $P$를 생성한다.
    $P_{i,j} = \text{Softmax}(S_{i,\cdot} / \tau)_j \times \text{Softmax}(S_{\cdot,j} / \tau)_i$
    English: Applies bidirectional softmax to produce a matching probability matrix $P$ with bidirectional consistency.
    $P_{i,j} = \text{Softmax}(S_{i,\cdot} / \tau)_j \times \text{Softmax}(S_{\cdot,j} / \tau)_i$
    """

    # 행 단위 소프트맥스 (Row-wise Softmax): $sm_{row} = \text{Softmax}(S / \tau, \text{dim}=-1)$
    sm_row = torch.softmax(S / temp, dim=-1)
    # 열 단위 소프트맥스 (Column-wise Softmax): $sm_{col} = \text{Softmax}(S / \tau, \text{dim}=-2)$
    sm_col = torch.softmax(S / temp, dim=-2)
    # 성분별 곱셈 (Component-wise Multiplication)
    return sm_row * sm_col

def dual_softmax_match_ablation(S: Tensor, temp: float = 0.1) -> Tensor:
    """
    이중 소프트맥스 비활성화 (Single-Softmax Matching).
    Korean: 행 방향 소프트맥스만을 사용하여 단방향 정합을 수행한다.
    English: Performs unidirectional matching using only row-wise softmax.
    """
    return torch.softmax(S / temp, dim=-1)
