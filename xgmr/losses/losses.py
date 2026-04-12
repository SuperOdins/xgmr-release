"""
손실 함수 모음.

한국어: 매칭, 재투영, TPS 정규화, 순환 일관성, Q 맵 스무딩 손실을 정의한다.
English: Defines matching, reprojection, TPS regularization, cycle consistency, and Q-map smoothing losses.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def matching_nll(P: Tensor, P_gt: Tensor) -> Tensor:
    """
    매칭 음의 로그우도 손실 (Matching Negative Log-Likelihood Loss).
    Korean: 예측된 확률 행렬 $P$와 정답 행렬 $P_{gt}$ 사이의 교차 엔트로피를 계산하여 정합 품질을 최적화한다.
    $\mathcal{L}_{nll} = -\sum_{i,j} P_{gt}(i,j) \log(P(i,j) + \epsilon)$
    English: Optimizes matching quality by computing the cross-entropy between the predicted probability matrix $P$ and the ground truth $P_{gt}$.
    $\mathcal{L}_{nll} = -\sum_{i,j} P_{gt}(i,j) \log(P(i,j) + \epsilon)$
    """

    eps = 1e-6
    loss = -(P_gt * torch.log(P + eps)).sum(dim=(-2, -1))
    return loss.mean()


def qmap_smoothness(Q: Tensor) -> Tensor:
    """
    품질 맵 스무딩 손실 (Q-map Smoothness Loss / Total Variation).
    Korean: 품질 맵 $Q$의 인접 픽셀 간 차이를 최소화하여 공간적으로 부드러운 변화를 유도한다.
    $\mathcal{L}_{smooth} = \sum_{h,w} |Q_{h+1,w} - Q_{h,w}| + |Q_{h,w+1} - Q_{h,w}|$
    English: Induces spatially smooth transitions by minimizing the difference between adjacent pixels in the quality map $Q$.
    $\mathcal{L}_{smooth} = \sum_{h,w} |Q_{h+1,w} - Q_{h,w}| + |Q_{h,w+1} - Q_{h,w}|$
    """

    tv_h = torch.abs(Q[:, :, 1:, :] - Q[:, :, :-1, :]).mean()
    tv_w = torch.abs(Q[:, :, :, 1:] - Q[:, :, :, :-1]).mean()
    return tv_h + tv_w


def det_regularization(H: Tensor) -> Tensor:
    """
    행렬식 정규화 (Determinant Regularization).
    Korean: 호모그래피 행렬의 행렬식이 1에 가깝도록 강제하여 이미지의 과도한 축소나 뒤집힘을 방지한다.
    $\mathcal{L}_{det} = |\det(\mathbf{H}) - 1|$
    English: Prevents excessive shrinking or flipping of the image by forcing the determinant of the homography matrix to be close to 1.
    $\mathcal{L}_{det} = |\det(\mathbf{H}) - 1|$
    """
    # 수치적 안정성을 위해 행렬식 계산 시 미세값 추가 (Numerical stability guard)
    det = torch.linalg.det(H + torch.eye(3, device=H.device).unsqueeze(0) * 1e-6)
    return torch.abs(det - 1.0).mean()


def identity_regularization(H: Tensor) -> Tensor:
    """
    항등 행렬 정규화 (Identity Regularization).
    Korean: 호모그래피 $\mathbf{H}$가 단위 행렬 $\mathbf{I}$ 근처에 머물도록 하여 학습 초기 단계를 안정화한다.
    $\mathcal{L}_{id} = \|\mathbf{H} - \mathbf{I}\|_F$
    English: Stabilizes the early training phase by encouraging the homography $\mathbf{H}$ to remain near the identity matrix $\mathbf{I}$.
    $\mathcal{L}_{id} = \|\mathbf{H} - \mathbf{I}\|_F$
    """
    B = H.shape[0]
    I = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
    return torch.norm(H - I, p='fro', dim=(-2, -1)).mean()

