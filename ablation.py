"""
Ablation Study Runner for XGMR.

This script runs a series of training experiments to verify the contribution of each component:
1. Baseline (Full Model)
2. No MBA (Modality Bridging Adapter)
3. No Self-Calib (Self-Calibrating Head)
4. No Q-Fusion (Quality-guided Fusion)
5. No EBA (Epipolar-Biased Attention Loss)
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(name: str, overrides: list[str]) -> None:
    print(f"\n{'='*20}\nRunning Experiment: {name}\n{'='*20}")
    
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/default.yaml",
        "--selfsup", "configs/train/selfsup.yaml",  # Assuming this exists or using default
        "--override",
        f"train.epochs=1", # Fast check
        f"data.batch_size=4", # Small batch for safety
    ] + overrides
    
    # If selfsup config doesn't exist, we might need to handle it. 
    # But train.py handles optional selfsup.
    # Let's check if configs/train/selfsup.yaml exists. 
    # If not, we rely on default.yaml having selfsup config or passed via override.
    
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] Experiment {name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILURE] Experiment {name} failed with exit code {e.returncode}.")

def main() -> None:
    experiments = [
        (
            "Baseline",
            []
        ),
        (
            "No_MBA",
            ["model.use_mba=false"]
        ),
        (
            "No_SelfCalib",
            ["model.use_self_calib=false"]
        ),
        (
            "No_QFusion",
            ["model.use_qfusion=false"]
        ),
        (
            "No_EBA_Loss",
            ["loss.lambda_geo=0.0"]
        ),
    ]

    for name, overrides in experiments:
        run_experiment(name, overrides)

if __name__ == "__main__":
    main()
