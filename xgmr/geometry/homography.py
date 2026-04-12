"""
호모그래피 유틸리티.

한국어: DLT, 정규화, RANSAC 추정을 단순화한 함수들.
English: Provides simplified DLT, normalization, and RANSAC wrappers.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


def normalize_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    좌표 정규화 (Hartley Point Normalization).
    Korean: DLT 연산의 수치적 안정성을 위해 포인트들을 무게중심 $\mathbf{c}$로 이동시키고 평균 거리가 $\sqrt{2}$가 되도록 스케일링한다.
    $\mathbf{x}_{norm} = \text{diag}(s, s, 1) \cdot (\mathbf{x} - \mathbf{c})$
    English: Shifts points to centroid $\mathbf{c}$ and scales them so that the average distance to origin is $\sqrt{2}$ for DLT stability.
    $\mathbf{x}_{norm} = \text{diag}(s, s, 1) \cdot (\mathbf{x} - \mathbf{c})$
    """

    centroid = points.mean(dim=0, keepdim=True)
    shifted = points - centroid
    scale = torch.sqrt(2.0) / torch.sqrt((shifted**2).sum(dim=1).mean())
    norm_points = shifted * scale
    T = torch.eye(3, device=points.device, dtype=points.dtype)
    T[:2, :2] *= scale
    T[:2, 2] = -centroid.squeeze() * scale
    homog = torch.cat([norm_points, torch.ones(norm_points.size(0), 1, device=points.device, dtype=points.dtype)], dim=1)
    return homog, T


def dlt_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    직접 선형 변환 (Direct Linear Transform, DLT).
    Korean: 두 평면 사이의 대응점 쌍을 이용해 호모그래피 행렬 $H$를 직접 계산한다.
    $A \mathbf{h} = 0$, where $\mathbf{h} = \text{vec}(\mathbf{H})$
    English: Directly computes the homography matrix $H$ using pairs of corresponding points between two planes.
    $A \mathbf{h} = 0$, where $\mathbf{h} = \text{vec}(\mathbf{H})$
    """

    if src.size(0) < 4:
        raise ValueError("Need at least 4 points for homography.")
    src_norm, T_src = normalize_points(src)
    dst_norm, T_dst = normalize_points(dst)
    A = []
    for i in range(src_norm.size(0)):
        x, y, w = src_norm[i]
        xp, yp, wp = dst_norm[i]
        A.append(torch.tensor([0, 0, 0, -w * xp, -w * yp, -w * wp, y * xp, y * yp, y * wp], device=src.device))
        A.append(torch.tensor([w * xp, w * yp, w * wp, 0, 0, 0, -x * xp, -x * yp, -x * wp], device=src.device))
    A = torch.stack(A)
    _, _, vh = torch.linalg.svd(A)
    H_norm = vh[-1].reshape(3, 3)
    H = torch.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[-1, -1]


def ransac_homography(src: np.ndarray, dst: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 호모그래피 추정.
    """

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    if H is None:
        H = np.eye(3, dtype=np.float32)
        mask = np.zeros((src.shape[0], 1), dtype=np.uint8)
    return H.astype(np.float32), mask
