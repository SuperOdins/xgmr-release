"""
로깅 유틸리티.

한국어: Python logging을 설정하고 컬러 포맷을 적용한다.
English: Configures Python logging with a concise format.
"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    시스템 로거 설정 (System Logger Configuration).
    Korean: 통합된 로깅 형식을 설정하고 실행 중의 상태 정보를 출력한다.
    English: Configures unified logging format and outputs runtime status information.
    """

    logger = logging.getLogger(name if name else "xgmr")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
