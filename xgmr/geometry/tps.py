"""
TPS(Thin-Plate Spline) 유틸리티.

한국어: TPS 그리드 생성과 와핑 및 정규화를 제공한다.
English: Provides TPS grid generation, warping, and regularization helpers.

Note: Kornia 의존성 제거됨 - 순수 PyTorch 구현
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_tps_grid(ctrl: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    TPS 기본 샘플링 그리드 구축 (TPS Base Sampling Grid Construction).
    Korean: $[-1, 1]$ 범위의 정규화된 2D 격자를 생성하여 와핑 연산의 기초 좌표계로 사용한다.
    English: Generates a normalized 2D grid in the range of $[-1, 1]$ to serve as the base coordinate system for warping.
    """
    h, w = size
    ys = torch.linspace(-1.0, 1.0, h, device=ctrl.device, dtype=ctrl.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=ctrl.device, dtype=ctrl.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return grid


def _radial_basis(r: Tensor) -> Tensor:
    """TPS radial basis function: r^2 * log(r)"""
    r2 = r ** 2
    # Avoid log(0) by clamping
    return r2 * torch.log(r2.clamp(min=1e-8)) * 0.5


def apply_tps(image: Tensor, ctrl_src: Tensor, ctrl_dst: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    TPS 변형 수행 (TPS Warping - Sparse-to-Dense).
    Korean: 소스 제어점과 대상 제어점 사이의 변위 벡터를 보간하여 이미지 전체에 대한 비선형 변형을 수행한다.
    $\mathbf{x}_{new} = \mathbf{x} + \sum w_i (\mathbf{p}_{dst,i} - \mathbf{p}_{src,i})$
    English: Performs non-linear deformation across the image by interpolating displacement vectors between source and destination control points.
    $\mathbf{x}_{new} = \mathbf{x} + \sum w_i (\mathbf{p}_{dst,i} - \mathbf{p}_{src,i})$
    """
    B, C, _, _ = image.shape
    H, W = size
    N = ctrl_src.shape[1]
    device, dtype = image.device, image.dtype
    
    # 출력 그리드 생성
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([xx, yy], dim=-1).view(1, H * W, 2).expand(B, -1, -1)
    
    # TPS 변환 계산 (간단한 선형 보간 근사)
    # 제어점 간 이동 벡터
    displacement = ctrl_dst - ctrl_src  # (B, N, 2)
    
    # 각 그리드 포인트에 대해 가장 가까운 제어점들의 가중 평균 사용
    # dist: (B, H*W, N)
    dist = torch.cdist(grid, ctrl_src)  # (B, H*W, N)
    weights = 1.0 / (dist + 1e-6)  # Inverse distance weighting
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
    
    # 각 포인트의 displacement 계산
    # (B, H*W, N) @ (B, N, 2) -> (B, H*W, 2)
    grid_displacement = torch.bmm(weights, displacement)
    
    # 새 그리드 위치
    new_grid = grid + grid_displacement
    new_grid = new_grid.view(B, H, W, 2)
    
    return F.grid_sample(image, new_grid, mode='bilinear', align_corners=False)


def tps_regularization(offsets: Tensor) -> Tensor:
    """
    TPS 곡률 정규화 (Bending Energy Regularization).
    Korean: 인접한 제어점 오프셋 간의 변동을 억제하여 변형이 과도하게 찌그러지지 않고 부드럽게 유지되도록 한다.
    $\mathcal{L}_{reg} = \sum_{i} \| \mathbf{o}_{i+1} - \mathbf{o}_i \|^2$
    English: Suppresses variations between adjacent control point offsets to ensure the deformation remains smooth and not excessively distorted.
    $\mathcal{L}_{reg} = \sum_{i} \| \mathbf{o}_{i+1} - \mathbf{o}_i \|^2$
    """
    diffs = offsets[:, 1:, :] - offsets[:, :-1, :]
    return (diffs**2).mean()
