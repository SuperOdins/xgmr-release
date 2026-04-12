"""
Geometric utility functions.

한국어: 호모그래피 스케일링 및 기하 변환 유틸리티.
English: Homography scaling and geometric transformation utilities.

Note: Kornia 의존성 제거됨 - 순수 PyTorch 구현
"""

import torch
import torch.nn.functional as F

def scale_homography(H: torch.Tensor, stride: float, inverse: bool = False) -> torch.Tensor:
    """
    호모그래피 스케일 변환 (Homography Scaling).
    Korean: 이미지 공간과 특징 공간 사이의 해상도 차이를 보정하기 위해 호모그래피 행렬 $H$를 스케일링한다.
    $H_{scaled} = S \cdot H \cdot S^{-1}$ where $S = diag(1/s, 1/s, 1)$
    English: Scales the homography matrix $H$ to compensate for resolution differences between image and feature spaces.
    $H_{scaled} = S \cdot H \cdot S^{-1}$ where $S = diag(1/s, 1/s, 1)$
    """
    H_scaled = H.clone().to(torch.float32)
    
    if inverse:
        # 특징 공간 -> 이미지 공간 (Feature Space -> Image Space)
        # $t' = t \times s$, $p' = p / s$
        H_scaled[:, 0, 2] *= stride
        H_scaled[:, 1, 2] *= stride
        H_scaled[:, 2, 0] /= stride
        H_scaled[:, 2, 1] /= stride
    else:
        # 이미지 공간 -> 특징 공간 (Image Space -> Feature Space)
        # $t' = t / s$, $p' = p \times s$
        H_scaled[:, 0, 2] /= stride
        H_scaled[:, 1, 2] /= stride
        H_scaled[:, 2, 0] *= stride
        H_scaled[:, 2, 1] *= stride
        
    return H_scaled


def warp_perspective(img: torch.Tensor, H: torch.Tensor, dsize: tuple[int, int]) -> torch.Tensor:
    """
    투영 변환 수행 (Perspective Warping).
    Korean: 호모그래피 행렬 $H$를 기반으로 이미지 $I$를 대상 평면으로 변환한다.
    $x_{src} = H^{-1} x_{tgt}$
    English: Transforms image $I$ to the target plane based on the homography matrix $H$.
    $x_{src} = H^{-1} x_{tgt}$
    """
    B, C, _, _ = img.shape
    h_out, w_out = dsize
    device, dtype = img.device, img.dtype
    
    # 1. 정규화 좌표 그리드 생성 (Normalized Coordinate Grid)
    # Korean: $[-1, 1]$ 범위의 격자 좌표를 생성하여 `grid_sample` 연산에 활용한다.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h_out, device=device, dtype=dtype),
        torch.linspace(-1, 1, w_out, device=device, dtype=dtype),
        indexing='ij'
    )
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid.view(B, h_out * w_out, 3)
    
    # 2. 호모그래피 정규화 및 스케일 보정 (Homography Normalization)
    H_normalized = H / (H[:, 2:3, 2:3] + 1e-8)
    
    # 이미지 좌표와 정규화 좌표 간의 변환 행렬 $S$
    S = torch.tensor([[2.0/w_out, 0, -1], [0, 2.0/h_out, -1], [0, 0, 1]], 
                     device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    S_inv = torch.tensor([[w_out/2.0, 0, w_out/2.0], [0, h_out/2.0, h_out/2.0], [0, 0, 1]], 
                         device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    H_norm = torch.bmm(torch.bmm(S, H_normalized), S_inv)
    
    # 3. 역행렬 계산 및 샘플링 (Inverse Mapping & Sampling)
    # $x_{src} = H_{norm}^{-1} x_{grid}$
    H_norm_f32 = H_norm.to(torch.float32)
    H_inv = torch.inverse(H_norm_f32).to(dtype)
    src = torch.bmm(grid, H_inv.transpose(-2, -1))
    src = src[..., :2] / (src[..., 2:3] + 1e-8) # 원근 분모로 나누기
    src = src.view(B, h_out, w_out, 2)
    
    return F.grid_sample(img, src, mode='bilinear', align_corners=False)
