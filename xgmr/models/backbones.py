"""
백본 네트워크 정의.

한국어: ResNet 스타일의 경량 백본과 피처 피라미드를 제공한다.
English: Provides a light-weight ResNet-style backbone and feature pyramid output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    잔차 블록 (Residual Block).
    Korean: 입력 데이터 $x$에 대해 비선형 변환 $\mathcal{F}(x)$를 학습하고, 원본 입력을 더해주는 스킵 커넥션을 통해 기울기 소실 문제를 완화한다.
    $y = \text{ReLU}(\mathcal{F}(x) + x)$, where $\mathcal{F}(x) = GN(Conv(ReLU(GN(Conv(x)))))$
    English: Learns a non-linear transformation $\mathcal{F}(x)$ and mitigates the vanishing gradient problem via a skip-connection.
    $y = \text{ReLU}(\mathcal{F}(x) + x)$, where $\mathcal{F}(x) = GN(Conv(ReLU(GN(Conv(x)))))$
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 첫 번째 합성곱 계층 / First convolutional layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        # 두 번째 합성곱 계층 / Second convolutional layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # 특징 변환 수행 / Perform feature transformation
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        # 스킵 커넥션 적용 / Apply skip-connection
        out = out + residual
        return self.relu(out)


class LightBackbone(nn.Module):
    """
    경량 백본 네트워크 (Lightweight Backbone).
    Korean: 입력 이미지로부터 계층적으로 특징을 추출한다. 각 단계마다 해상도를 절반으로 줄이고 채널을 두 배로 확장한다.
    $H_{out} = H_{in} / 2^{depth+1}$, $C_{out} = C_{base} \times 2^{depth}$
    English: Hierarchically extracts features from an input image. Each step halves the resolution and doubles the channels.
    $H_{out} = H_{in} / 2^{depth+1}$, $C_{out} = C_{base} \times 2^{depth}$
    """

    def __init__(self, in_channels: int = 3, base_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        # 초기 스트라이드 합성곱 계층 (1/2 Downsampling)
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, base_dim),
            nn.ReLU(inplace=True),
        ]
        dim = base_dim
        # 깊이에 따른 점진적 해상도 축소 및 특징 추출 (Downsampling & Residual learning)
        for _ in range(depth):
            layers.append(nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.GroupNorm(8, dim * 2))
            layers.append(nn.ReLU(inplace=True))
            dim *= 2
            layers.append(ResidualBlock(dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 추출 수행.
        Korean: 입력 이미지를 인코딩하여 고수준 특징 맵을 생성한다.
        English: Encode the input image to generate a high-level feature map.
        """
        return self.encoder(x)


@dataclass
class BackboneConfig:
    """백본 설정 / Configuration holder."""

    name: str = "light"
    in_channels: int = 3
    base_dim: int = 64
    depth: int = 3


def build_backbone(cfg: Dict) -> Tuple[nn.Module, int]:
    """
    백본 네트워크 빌더 (Backbone Builder).
    Korean: 설정값에 따라 적절한 백본 모델을 인스턴스화하고 최종 출력 채널 수를 반환한다.
    $C_{out} = \text{base\_dim} \times 2^{\text{depth}}$
    English: Instantiates the appropriate backbone model based on configs and returns the output channel count.
    $C_{out} = \text{base\_dim} \times 2^{\text{depth}}$
    """

    name = cfg.get("name", "light")
    if name != "light":
        raise ValueError(f"Unsupported backbone '{name}'.")
    model = LightBackbone(
        in_channels=cfg.get("in_channels", 3),
        base_dim=cfg.get("base_dim", 64),
        depth=cfg.get("depth", 3),
    )
    # 출력 채널 계산 (Output dimension calculation)
    # Korean: 깊이에 따라 채널이 누적 확장된 최종 차원을 산출한다.
    out_dim = cfg.get("base_dim", 64) * (2 ** cfg.get("depth", 3))
    return model, out_dim
