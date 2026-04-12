"""
체크포인트 유틸리티.

한국어: 모델/옵티마이저 상태를 저장하고 불러온다.
English: Save and load model / optimizer checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(path: str | Path, state: Dict) -> None:
    """
    체크포인트 저장 / Persist checkpoint dictionary.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
    """
    체크포인트 로드 / Load checkpoint.
    """

    return torch.load(Path(path), map_location=map_location)
