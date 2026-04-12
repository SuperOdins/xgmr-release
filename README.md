# XGMR: Detector-Free RGB-Thermal Image Registration with Geometric Inductive Bias

Official implementation of **XGMR**, a detector-free framework for RGB-Thermal image registration with geometric inductive bias. XGMR extends the LoFTR dense-matching paradigm with five modules designed for cross-modal alignment:

- **MBA** (Modality Bridging Adapter) — bridges the domain gap between RGB and thermal features with optional EAEF channel attention.
- **EBA** (Epipolar-Biased Attention) — injects geometric priors into the coarse/fine cross-attention stages.
- **Dual-Softmax Matching** — optimal-transport-free assignment that produces reliable coarse-to-fine correspondences.
- **Self-Calibrating Head** — predicts homography + TPS deformation fields for robust spatial alignment.
- **Q-Fusion** (Quality-Aware Fusion) — tile-level gating that adaptively fuses aligned modalities based on match quality.

## Installation

```bash
git clone https://github.com/<your-org>/xgmr-release.git
cd xgmr-release
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python >= 3.9, PyTorch >= 2.0, CUDA recommended.

## Quick Start

### Training

```bash
# Self-supervised training (no GT annotations required)
python train.py --config configs/default.yaml

# Fast debug run
python train.py --config configs/default.yaml --override train.epochs=1 train.lr=0.0001
```

### Evaluation

```bash
python eval.py --config configs/default.yaml --ckpt outputs/best.ckpt
```

### Demo

```bash
# Run on image folders
python demo.py --rgb_dir ./data/rgb --t_dir ./data/thermal --save_viz ./viz_out

# Run with random tensors (no data needed)
python demo.py --save_viz ./viz_out
```

### Ablation Study

```bash
python ablation.py
```

## Data Setup

1. Prepare paired RGB and thermal images in separate directories:
   ```
   data/
     rgb/        # RGB images (.png or .jpg)
     thermal/    # Thermal images (.png or .jpg)
   ```
2. File names should match across directories (sorted alphabetically).
3. Configure paths in `configs/default.yaml` or override via CLI:
   ```bash
   python train.py --config configs/default.yaml \
     --override data.rgb_dir=./your/rgb data.thermal_dir=./your/thermal
   ```

> **Note:** If no images are found, the pipeline automatically generates random tensors for smoke-testing.

## Tests

```bash
pytest -q
```

## Project Structure

```
xgmr/
  data/        # Datasets, transforms, geometric augmentations, collate
  models/      # MBA, attention, LoFTR-like matcher, self-calib, Q-Fusion, XGMR assembly
  geometry/    # Homography, TPS, epipolar masks, warping helpers
  losses/      # Matching & regularisation losses, self-supervised objectives
  metrics/     # Reprojection error, precision/recall, SSIM, IoU
  utils/       # Logging, distributed helpers, checkpoints, visualisation
configs/       # Hydra/OmegaConf YAML configs (default, model, data, train)
tests/         # Pytest smoke & ablation tests
train.py       # Training entry point (supervised & self-supervised)
eval.py        # Evaluation script
demo.py        # Inference & visualisation
ablation.py    # Ablation study runner
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{park2026xgmr,
  title   = {XGMR: Detector-Free RGB-Thermal Image Registration with Geometric Inductive Bias},
  author  = {Park, Jong Il and Kim, Cheong},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
