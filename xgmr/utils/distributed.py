"""
분산 학습 유틸리티.

한국어: DDP 초기화를 단순화한다.
English: Simplifies distributed data parallel initialisation.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def setup_ddp_env(rank: int = 0, world_size: int = 1, backend: str = "nccl") -> None:
    """
    DDP 환경 설정 / Configure torch.distributed.
    """

    if world_size == 1 or torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
