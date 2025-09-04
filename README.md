# Waveform → TX Parameters (CNN Regression)

**A professional, end-to-end toolkit for** generating synthetic waveform images, training a convolutional neural network to regress transmission-line parameters, and evaluating predictions.

---

## Project Overview

This repository demonstrates a full pipeline that converts waveform images into four transmission-line parameters using deep learning:

- **Frequency** (Hz) — modeled in log-space during training
- **Alpha** (attenuation)
- **Beta** (phase)
- **Dielectric** (permittivity)

The project includes:

- a reproducible dataset generator
- a compact, well-regularized CNN for regression
- training utilities with early stopping and LR scheduling
- robust prediction & evaluation code that handles numeric edge cases

This README is ready to use as-is — no edits required.

---

## Repository Layout

```
project-root/
├─ .ipynb_checkpoints/        # Jupyter checkpoint files
├─ dataset_large/             # Generated images + tx_parameters.csv
├─ dataset_sample/            # Small sample of images (10) for quick checks
├─ tx_env/                    # Optional virtual environment folder (if created)
├─ .gitignore
├─ cnnModel.ipynb             # Notebook for generate/train/evaluate
├─ waveform_cnn_best.pth      # Trained model weights (best by val loss)
├─ label_scaler.pkl           # Scaler used to transform labels during training
├─ README.md                  # This file
├─ requirements.txt           # Project dependencies
└─ scripts/                   # Optional (exported scripts: train.py, eval.py, etc.)
```

---

## Quick Start (3 steps)

1. **Create & activate environment** (recommended):

```bash
python -m venv tx_env
# macOS / Linux
source tx_env/bin/activate
# Windows (PowerShell)
./tx_env/Scripts/Activate.ps1
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Open the Jupyter notebook** and run the cells interactively:

```bash
jupyter lab cnnModel.ipynb
# or
jupyter notebook cnnModel.ipynb
```

---

## Main Components

### Dataset Generator
- Script generates `TOTAL_IMAGES` waveform PNGs and a CSV `tx_parameters.csv` with columns: `image, frequency, alpha, beta, dielectric`.
- Frequency is sampled across many decades (e.g., 1e6–1e10 Hz); training uses `log10(frequency)` for stability.

### Model & Training
- `WaveformCNN` — a compact CNN with conv → batchnorm → ReLU → pooling blocks and a small MLP head producing 4 outputs.
- Training features:
  - Mean Squared Error loss on scaled targets
  - `StandardScaler` on the 4-label vector (after frequency `log10` transform)
  - `ReduceLROnPlateau` scheduler and early stopping
  - Mixed-precision (AMP) when a CUDA GPU is available

### Prediction & Evaluation
- Predictions are inverse-transformed (including `10**pred_log_freq`) with clipping to prevent overflow.
- Robust relative-error computation uses `denom = max(|gt|, |pred|, 1e-9)` to avoid divide-by-zero or huge ratios.
- Utility scripts find the most-accurate predictions and optionally export `accurate_images.csv` (images satisfying a user-defined accuracy threshold).

---

## Usage Examples

### Train (via notebook)
- Run the training cells in `cnnModel.ipynb`. The script saves the best weights to `waveform_cnn_best.pth` and label scaler `label_scaler.pkl`.

### Predict single image (headless)
- Ensure model class is defined exactly the same as during training, then:

```python
from model import WaveformCNN  # or re-define class in cell
import torch, joblib
model = WaveformCNN().to(device)
state = torch.load('waveform_cnn_best.pth', map_location=device)
model.load_state_dict(state)
model.eval()
scaler = joblib.load('label_scaler.pkl')
# use predict_image helper in notebook/scripts
```

### Find the most accurate prediction
- Use the provided evaluation cell/script to iterate all images, compute mean relative error across the 4 parameters, and return the best (lowest mean error) image.
- The script clips predicted log-frequency before exponentiating and writes human-readable results to console and CSV.

---

## Recommended Configuration

- `IMG_SIZE = 64` for training — strikes a balance between information and compute.
- `BATCH_SIZE = 32` on a mid-range GPU (adjust down for smaller GPUs).
- `VAL_RATIO = 0.1`, `PATIENCE = 8`, `LR = 1e-4` are good starting hyperparameters.

---

## Troubleshooting & Tips

- **`torch.load` FutureWarning**: Safe to ignore, but to future-proof use `weights_only=True` when available.
- **Overflow when computing `10**x`**: The code clips the exponent to a safe range (e.g., 0–12) before exponentiation.
- **Large relative errors for tiny GT values**: The evaluation uses a robust denominator `max(|gt|,|pred|,1e-9)` to prevent misleading ratios.
- **GPU OOM**: reduce `BATCH_SIZE` or `IMG_SIZE`, or switch to CPU for testing.
- **Path errors**: confirm `DATA_DIR`, `MODEL_PATH`, and `SCALER_PATH` are set to the files in your project root.

---

## Reproducibility

To reproduce results exactly, pin versions in `requirements.txt`, set NumPy/PyTorch seeds, and keep `random_state` in any sampling functions fixed.

Suggested pinned versions (example):
```
numpy==1.24.0
pandas==2.1.0
matplotlib==3.7.1
Pillow==10.0.1
tqdm==4.65.0
scikit-learn==1.2.2
joblib==1.3.1
torch==2.2.0
torchvision==0.17.0
```

---

## Outputs

- `waveform_cnn_best.pth` — saved model weights (best by validation loss)
- `label_scaler.pkl` — scaler used to map model outputs back to physical quantities
- `accurate_images.csv` — optional export of images satisfying the accuracy threshold

---

## Contact & Attribution

If you use or extend this project, please attribute the original implementation. For questions or help adapting the pipeline, open an issue or contact the maintainer listed in your project metadata.

---

## License

Provided for educational and research use. Modify and redistribute as needed.

