"""
XGMR package initialisation.

한국어: XGMR 패키지는 RGB-열화상 정합을 위한 핵심 모듈을 제공한다.
English: The XGMR package bundles core modules for RGB-thermal registration.
"""

from .config import XGMRConfig, load_config

__all__ = ["XGMRConfig", "load_config"]
