# XGMR: Detector-Free RGB-Thermal Image Registration with Geometric Inductive Bias

Official implementation of **XGMR**, a detector-free framework for RGB-Thermal image registration with geometric inductive bias. XGMR extends the LoFTR dense-matching paradigm with five modules designed for cross-modal alignment:

- **MBA** (Modality Bridging Adapter) — bridges the domain gap between RGB and thermal features with optional EAEF channel attention.
- **EBA** (Epipolar-Biased Attention) — injects geometric priors into the coarse/fine cross-attention stages.
- **Dual-Softmax Matching** — optimal-transport-free assignment that produces reliable coarse-to-fine correspondences.
- **Self-Calibrating Head** — predicts homography + TPS deformation fields for robust spatial alignment.
- **Q-Fusion** (Quality-Aware Fusion) — tile-level gating that adaptively fuses aligned modalities based on match quality.

## Installation

```bash
git clone https://github.com/SuperOdins/xgmr-release.git
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

### Benchmark datasets (used in the paper)

The paper evaluates on three public RGB–thermal datasets. They are **not** redistributed
here; download them from their official sources and cite the original authors.

| Dataset | Reference | DOI | Source |
|---------|-----------|-----|--------|
| **NII-CU MAPD** | Speth et al., *Deep Learning with RGB and Thermal Images Onboard a Drone for Monitoring Operations*, J. Field Robotics 39(6):840–868, 2022 | `10.1002/rob.22082` | From the original authors (NII-CU Multispectral Aerial Person Detection) |
| **LLVIP** | Jia et al., *LLVIP: A Visible-Infrared Paired Dataset for Low-light Vision*, ICCVW 2021 | `10.1109/ICCVW54120.2021.00389` | https://github.com/bupt-ai-cz/LLVIP |
| **MSRS** | Tang et al., *PIAFusion: A Progressive Infrared and Visible Image Fusion Network Based on Illumination Aware*, Information Fusion 83–84:79–92, 2022 | `10.1016/j.inffus.2022.03.007` | https://github.com/Linfeng-Tang/MSRS |

Each dataset is governed by its own license/terms; review and comply before use.

### Data splits

`splits/` lists the exact train/validation/test partitions used in every experiment. Each
CSV gives the paired filenames in loader order (`rgb,thermal`), so the splits are fully
reproducible without redistributing the images.

| Split file | Dataset | Role | Pairs |
|------------|---------|------|------:|
| `splits/nii_train.csv`   | NII-CU MAPD | train | 4980 |
| `splits/llvip_train.csv` | LLVIP | train | 12025 |
| `splits/msrs_train.csv`  | MSRS | train | 1083 |
| `splits/nii_val.csv`     | NII-CU MAPD | validation (held-out eval) | 485 |
| `splits/llvip_test.csv`  | LLVIP | test (eval) | 3463 |
| `splits/msrs_test.csv`   | MSRS | test (eval) | 361 |

Training uses the **combined** train split of all three datasets
(4980 + 12025 + 1083 = **18,088 pairs**, 50 epochs), as reported in the paper. NII-CU MAPD has
no public test partition, so its **validation** split is the held-out evaluation set; LLVIP and
MSRS are evaluated on their official **test** partitions. Train and test partitions are disjoint
(zero filename overlap within each dataset).

To reconstruct a split, download the dataset and place the listed files under
`data/<dataset>/{rgb,thermal}/<split>/`. Filenames already match the originals, so no
renaming is needed. There is no ground-truth homography file: evaluation warps are generated
synthetically from a fixed seed, so code + seed + these splits fully determine the protocol.

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

## Pretrained Checkpoints

All trained weights are attached to the [`xgmr-v1.0` release](../../releases/tag/xgmr-v1.0).
They are PyTorch-Lightning `.ckpt` files (50 epochs, self-supervised). Download the one you
need and pass it via `--ckpt`:

```bash
python eval.py --config configs/default.yaml --ckpt XGMR_Full.ckpt
```

| File | Configuration | Reproduces |
|------|---------------|------------|
| `XGMR_Full.ckpt` | Full model (all modules) | Main results + qualitative figures |
| `XGMR_Full_seed42/7/123.ckpt` | Full model, 3 independent seeds | Seed variance / robustness |
| `ablation_NoMBA.ckpt` | MBA (cross-modal fusion) disabled | Ablation |
| `ablation_NoQFusion.ckpt` | Q-Fusion disabled | Ablation |
| `ablation_NoSelfCalib.ckpt` | Self-Calibrating Head disabled | Ablation |
| `ablation_NoDualSoftmax.ckpt` | Dual-softmax matching disabled | Ablation |
| `ablation_NoEpipolarBias.ckpt` | Epipolar-Biased Attention disabled | Ablation |
| `ablation_NoGeoLoss.ckpt` | Geometric loss `L_geo` disabled | Ablation |
| `ablation_Vanilla.ckpt` | All proposed components off | Ablation |
| `baseline_LoFTR_finetuned.ckpt` | LoFTR fine-tuned on the same data | Baseline |

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
