"""
Configuration utilities for XGMR.

한국어: Hydra/OmegaConf 기반 설정 로딩과 시드 제어를 담당한다.
English: Provides Hydra/OmegaConf based configuration loading and seed control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf


@dataclass
class TrainerConfig:
    """기본 학습 설정 / Basic training hyper-parameters."""

    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    transition_epochs: int = 0
    output_dir: str = "outputs"
    drive_output_dir: str | None = None
    seed: int = 2025


@dataclass
class DataConfig:
    """데이터 경로 및 변환 설정 / Data paths and transforms."""

    rgb_dir: str = "./data/rgb"
    thermal_dir: str = "./data/thermal"
    manifest: str | None = None
    batch_size: int = 2
    num_workers: int = 0
    img_size: Tuple[int, int] = (480, 640)
    limit: int | None = None


@dataclass
class ModelConfig:
    """모델 컴포넌트 구성 / Model component configuration."""

    backbone: Dict[str, Any] = field(
        default_factory=lambda: {"name": "light", "pretrained": False, "dim": 256}
    )
    mba: Dict[str, Any] = field(
        default_factory=lambda: {"in_ch_rgb": 3, "in_ch_t": 1, "out_dim": 256, "use_eaef": True, "use_sobel": True}
    )
    matcher: Dict[str, Any] = field(
        default_factory=lambda: {
            "dim": 256,
            "layers": (2, 1),
            "temp": 0.1,
            "strip_width": 5,
            "num_heads": 4,
            "pe_max_shape": (256, 256),
            "pe_temp": 10000.0,
            "bias_strength": 2.0,
            "use_pe": True,
            "use_epi_bias": True,
        }
    )
    selfcalib: Dict[str, Any] = field(
        default_factory=lambda: {
            "dim": 256,
            "tps_grid": 5,
            "iters": 2,
            "match_k": 128,
            "match_dim": 128,
            "h_res_scale": 1e-4,
            "tps_res_scale": 0.02,
        }
    )
    qfusion: Dict[str, Any] = field(
        default_factory=lambda: {"tile": 32, "k": 8.0, "tau": 0.5, "num_matches_scale": 100.0}
    )
    use_mba: bool = True
    use_self_calib: bool = True
    use_qfusion: bool = True


@dataclass
class LossConfig:
    """Loss weighting parameters."""

    lambda_q: float = 0.1
    lambda_eq: float = 10.0
    lambda_geo: float = 2.0
    lambda_ent: float = 0.5
    lambda_det: float = 5.0
    lambda_identity: float = 0.1
    lambda_direct: float = 1.0
    lambda_direct_warmup: float = 10.0


@dataclass
class GeomAugConfig:
    """Self-supervised geometric augmentation parameters."""

    h_bounds_px: Tuple[float, float] = (5.0, 15.0)
    rot_deg: Tuple[float, float] = (3.0, 8.0)
    scale: Tuple[float, float] = (0.95, 1.05)


@dataclass
class AugmentConfig:
    """Color/noise augmentation switches."""

    weak_color: bool = True
    strong_color: bool = True
    noise_blur: bool = True


@dataclass
class SelfSupConfig:
    """Self-supervision specific settings."""

    bias_sigma_px: float = 4.0
    sigma_init: float = 8.0  # Added: Initial wide sigma for unstable training start
    geom: GeomAugConfig = field(default_factory=GeomAugConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class AblationConfig:
    """Ablation study flags."""

    no_eba: bool = False
    no_dsm: bool = False
    no_lgeo: bool = False
    no_mba: bool = False
    no_qfusion: bool = False
    no_selfcalib: bool = False


@dataclass
class XGMRConfig:
    """전체 구성 / Complete configuration."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    selfsup: SelfSupConfig = field(default_factory=SelfSupConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def set_determinism(self) -> None:
        """시드 고정 및 CuDNN 설정 / Fix random seeds and configure CuDNN."""

        torch.manual_seed(self.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train.seed)
        torch.backends.cudnn.deterministic = not self.cudnn_benchmark
        torch.backends.cudnn.benchmark = self.cudnn_benchmark


def load_config(path: str | Path | None = None) -> XGMRConfig:
    """
    YAML 또는 기본값으로 설정을 로드한다.

    한국어: 경로가 지정되면 YAML을 OmegaConf로 읽어 dataclass에 병합한다.
    English: Load configuration from YAML when path is provided; otherwise use defaults.
    """

    cfg = XGMRConfig()
    if path is None:
        return cfg

    conf = OmegaConf.load(Path(path))
    merged = OmegaConf.merge(OmegaConf.structured(cfg), conf)
    return OmegaConf.to_object(merged)
