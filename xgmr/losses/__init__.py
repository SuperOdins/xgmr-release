"""
손실 함수 초기화 / Loss package init.

한국어: 손실 함수와 가중 합산을 외부로 노출한다.
English: Re-export loss helpers for external modules.
"""

from .losses import (
    matching_nll,
    qmap_smoothness,
    det_regularization,
    identity_regularization,
)

__all__ = [
    "matching_nll",
    "qmap_smoothness",
    "det_regularization",
    "identity_regularization",
]
