"""
평가 지표 구현.

한국어: 재투영 오차, 매칭 정밀도/재현율, SSIM/IoU를 계산한다.
English: Implements reprojection error, matching precision/recall, and SSIM/IoU metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor


def reprojection_error(H: Tensor, pts_src: Tensor, pts_dst: Tensor) -> Tensor:
    """
    재투영 오차 계산 (Reprojection Error Calculation).
    Korean: 소스 점 $\mathbf{x}_{src}$를 호모그래피 $\mathbf{H}$로 투영한 후, 타겟 점 $\mathbf{x}_{dst}$와의 유클리드 거리를 계산한다.
    $d = \| \text{Proj}(\mathbf{H}, \mathbf{x}_{src}) - \mathbf{x}_{dst} \|_2$
    English: Computes the Euclidean distance between the target points and source points projected via homography $\mathbf{H}$.
    $d = \| \text{Proj}(\mathbf{H}, \mathbf{x}_{src}) - \mathbf{x}_{dst} \|_2$
    """

    batch = H.size(0)
    # 동차 좌표계 변환 / Convert to homogeneous coordinates
    pts_src_h = torch.cat([pts_src, torch.ones(batch, pts_src.size(1), 1, device=H.device, dtype=H.dtype)], dim=-1)
    # 호모그래피 투영 수행 / Perform homography projection: $\mathbf{x}' = \mathbf{H} \mathbf{x}$
    proj = torch.bmm(pts_src_h, H.transpose(1, 2))
    # 일반 좌표계로 정규화 / Normalize to Cartesian coordinates
    proj = proj[..., :2] / proj[..., 2:].clamp(min=1e-6)
    # 오차 산출 / Compute error
    diff = proj - pts_dst
    return torch.sqrt((diff**2).sum(dim=-1) + 1e-6)


def matching_precision_recall(matches, gt, thresh_px: float = 3.0) -> Dict[str, float]:
    """
    매칭 정밀도 및 재현율 (Matching Precision & Recall / PCK).
    Korean: 임계값 $\tau$ 이내의 오차를 가진 정합점을 정답(True Positive)으로 간주하여 성능을 측정한다.
    $Precision = \frac{|TP|}{|Matches|}$, $Recall = \frac{|TP|}{|GT|}$
    English: Measures performance by considering matches with error within threshold $\tau$ as True Positives.
    $Precision = \frac{|TP|}{|Matches|}$, $Recall = \frac{|TP|}{|GT|}$
    """

    if len(matches) == 0 or len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0}
    matches = np.asarray(matches)
    gt = np.asarray(gt)
    # 유클리드 거리 기반 오차 측정 / Distance-based error measurement
    distances = np.linalg.norm(matches[:, :2] - gt[:, :2], axis=1)
    # 임계값 기준 정답 판별 (True Positive)
    tp = (distances < thresh_px).sum()
    precision = tp / max(len(matches), 1)
    recall = tp / max(len(gt), 1)
    return {"precision": float(precision), "recall": float(recall)}


def ssim_iou(rgb_warped: np.ndarray, rgb: np.ndarray) -> Tuple[float, float]:
    """
    구조적 유사도 및 교집합 비율 (SSIM & IoU Estimation).
    Korean: 이미지의 구조적 유사성(SSIM)과 영역 중첩도(IoU)를 계산하여 정렬 품질을 정량화한다.
    $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x + \sigma_y + c_2)}$
    English: Quantifies alignment quality by calculating structural similarity (SSIM) and area overlap (IoU).
    $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x + \sigma_y + c_2)}$
    """

    rgb_warped = rgb_warped.astype(np.float32)
    rgb = rgb.astype(np.float32)
    
    # 통계치 산출 / Compute statistics
    mu_x = rgb_warped.mean()
    mu_y = rgb.mean()
    sigma_x = rgb_warped.var()
    sigma_y = rgb.var()
    covariance = ((rgb_warped - mu_x) * (rgb - mu_y)).mean()
    
    # SSIM 상수 설정 / SSIM Constants
    c1 = 0.01**2
    c2 = 0.03**2
    
    # SSIM 수식 적용 / Apply SSIM formula
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    
    # 이진 마스크를 이용한 IoU 계산 (IoU via binary mask threshold)
    mask_x = rgb_warped > 0.5
    mask_y = rgb > 0.5
    intersection = np.logical_and(mask_x, mask_y).sum()
    union = np.logical_or(mask_x, mask_y).sum()
    # $IoU = \frac{|A \cap B|}{|A \cup B|}$
    iou = intersection / union if union > 0 else 0.0
    return float(ssim), float(iou)
