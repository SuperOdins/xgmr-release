"""
에피폴라 기하 유틸리티.

한국어: 카메라 행렬과 상대 자세에서 에피폴라 기반 주의 마스크를 생성한다.
English: Generates epipolar bias masks from camera intrinsics and relative pose.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F





def homography_bias_mask(
    H: Tensor,
    q_hw: Tuple[int, int],
    k_hw: Tuple[int, int],
    sigma: float = 2.0,
) -> Tensor:
    """
    호모그래피 기반 편향 마스크 생성 (Homography-based Bias Mask Generation).
    Korean: 주어진 호모그래피 $H$를 사용하여 키(Key) 포인트를 쿼리(Query) 공간으로 투영하고, 거리 기반 가우시안 마스크를 생성한다.
    $M_{ij} = \exp\left(-\frac{\|\mathbf{x}_{q,i} - \text{Proj}(\mathbf{H}, \mathbf{x}_{k,j})\|^2}{2\sigma^2}\right)$
    English: Projects key points into the query space using the given homography $H$ and generates a distance-based Gaussian mask.
    $M_{ij} = \exp\left(-\frac{\|\mathbf{x}_{q,i} - \text{Proj}(\mathbf{H}, \mathbf{x}_{k,j})\|^2}{2\sigma^2}\right)$
    """
    b = H.shape[0]
    device = H.device
    dtype = H.dtype
    
    qh, qw = q_hw
    kh, kw = k_hw
    
    # 1. 픽셀 좌표 그리드 생성 (Pixel Coordinate Grid Generation)
    # Query points: $\mathbf{x}_q \in \mathbb{R}^{H_q \times W_q \times 2}$
    y_q, x_q = torch.meshgrid(
        torch.arange(qh, device=device, dtype=dtype),
        torch.arange(qw, device=device, dtype=dtype),
        indexing='ij'
    )
    pts_q = torch.stack([x_q.reshape(-1), y_q.reshape(-1)], dim=-1)  # (nq, 2)
    
    # Key points: $\mathbf{x}_k \in \mathbb{R}^{H_k \times W_k \times 2}$
    y_k, x_k = torch.meshgrid(
        torch.arange(kh, device=device, dtype=dtype),
        torch.arange(kw, device=device, dtype=dtype),
        indexing='ij'
    )
    pts_k = torch.stack([x_k.reshape(-1), y_k.reshape(-1)], dim=-1)  # (nk, 2)
    
    # 2. 키 포인트를 쿼리 공간으로 워핑 (Warping Key Points to Query Space)
    # $\mathbf{x}_{k,h} = [\mathbf{x}_k ; 1]$, $\mathbf{x}'_k = \mathbf{H} \mathbf{x}_{k,h}$
    ones = torch.ones(kh * kw, 1, device=device, dtype=dtype)
    pts_k_h = torch.cat([pts_k, ones], dim=-1) # (nk, 3)
    
    # 투영 변환 (Projection transformation)
    pts_k_warped = torch.bmm(pts_k_h.unsqueeze(0).expand(b, -1, -1), H.transpose(-2, -1)) # (B, nk, 3)
    pts_k_warped = pts_k_warped[..., :2] / (pts_k_warped[..., 2:3].clamp(min=1e-8)) # (B, nk, 2)
    
    # 3. 쌍별 유클리드 거리 및 마스크 산출 (Pairwise Distance & Masking)
    # $D_{ij} = \|\mathbf{x}_{q,i} - \mathbf{x}'_{k,j}\|^2 = \|\mathbf{x}_{q,i}\|^2 + \|\mathbf{x}'_{k,j}\|^2 - 2\langle\mathbf{x}_{q,i}, \mathbf{x}'_{k,j}\rangle$
    a = pts_q.unsqueeze(0).expand(b, -1, -1) # (B, nq, 2)
    b_pts = pts_k_warped # (B, nk, 2)
    
    a2 = (a**2).sum(dim=-1, keepdim=True) # (B, nq, 1)
    b2 = (b_pts**2).sum(dim=-1, keepdim=True).transpose(-2, -1) # (B, 1, nk)
    ab = torch.bmm(a, b_pts.transpose(-2, -1)) # (B, nq, nk)
    
    d2 = a2 + b2 - 2 * ab
    
    # 가우시안 분포 적용 (Gaussian falloff application)
    mask = -d2 / (2 * sigma**2)
    return mask
