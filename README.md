# Snow and Ice Detection — Model Training & Hyperparameter Tuning

Road surface snow/ice detection system using YOLO and RT-DETR series models, with systematic hyperparameter tuning and multi-model benchmarking.

---

## Performance Comparison (sorted by mAP50)

All models trained on the same dataset with identical settings: 300 epochs, AdamW, imgsz=640, batch auto-scale.

| Model | F1 | mAP50 | mAP50-95 | Params (M) | FPS |
|---|---|---|---|---|---|
| **rtdetr-x** | **0.8814** | **0.8548** | **0.6694** | 67.31 | 11.24 |
| rtdetr-l | 0.8743 | 0.8454 | 0.6385 | 32.81 | 14.96 |
| yolov8m | 0.5901 | 0.5467 | 0.3650 | 25.86 | 142.89 |
| yolov10m | 0.5697 | 0.5296 | 0.3467 | 16.49 | 96.32 |
| yolov13s | 0.5883 | 0.5238 | 0.3453 | 9.55 | 63.41 |
| yolov9m | 0.5711 | 0.5180 | 0.3471 | 20.16 | 94.89 |
| yolov12m | 0.4740 | 0.4114 | 0.2625 | 20.05 | 87.14 |
| yolov11m | 0.4115 | 0.3464 | 0.1880 | 19.71 | 139.76 |

> FPS measured at batch=1, imgsz=640 on a single GPU.

**Key findings:**
- RT-DETR-X achieves the highest accuracy (mAP50 **0.855**), ~60% higher than the best YOLO variant.
- RT-DETR-L offers a strong accuracy/efficiency trade-off at half the parameter count.
- Among YOLO variants, YOLOv8m leads on mAP50 with the highest FPS (143 FPS).
- YOLOv13s is the most lightweight option (9.55M params) with competitive accuracy.

---

## Detection Classes

Only two road surface hazard categories are used. Background and dry road classes were excluded to focus the model on ice/snow detection.

| Class | Hazard Level |
|---|---|
| snow | Moderate |
| ice | High |

---

## Project Structure

```
.
├── train.py                   # Hyperparameter grid search (cls × box)
├── compare_yolo_models.py     # YOLO series multi-model benchmark
├── compareDetrModels.py       # RT-DETR series multi-model benchmark
├── testEnvironment.py         # Environment and dependency check
├── yolov11m.yaml              # Dataset config (nc=2, train/val split paths)
├── hyperparameter_results_*/  # Grid search logs and best weights
│   ├── manual_runs/           # Per-config training results
│   │   └── config_NNN/
│   │       ├── weights/       # best.pt, last.pt
│   │       ├── results.csv
│   │       └── args.yaml
│   ├── detailed_results.json
│   ├── best_result.json
│   └── results_summary.csv
└── runs/compare/              # Multi-model comparison outputs
    ├── yolo_comparison_*/
    └── detr_comparison_*/
```

---

## Dataset

YOLO-format dataset with two classes. The dataset is auto-split 80/20 into train/val using Ultralytics' `autosplit`:

```python
from ultralytics.data.split import autosplit
autosplit(image_dir, weights=(0.8, 0.2, 0.0), annotated_only=True)
```

**Dataset config (`yolov11m.yaml`):**
```yaml
path: ./
train: autosplit_train.txt
val:   autosplit_val.txt
nc: 2
names: ['snow', 'ice']
```

---

## Environment Setup

**Requirements:** Python 3.8+, CUDA-capable GPU

```bash
pip install ultralytics torch pandas pillow
```

Verify environment:
```bash
python testEnvironment.py
```

---

## Usage

### 1. Run YOLO series comparison

Trains YOLOv8m, YOLOv9m, YOLOv10m, YOLOv11m, YOLOv12m, YOLOv13s sequentially and generates a comparison report:

```bash
python compare_yolo_models.py
```

### 2. Run RT-DETR series comparison

Trains RT-DETR-L and RT-DETR-X and generates a comparison report:

```bash
python compareDetrModels.py
```

### 3. Hyperparameter grid search

Grid search over `cls` (classification loss weight) and `box` (bounding box loss weight):

```bash
python train.py
```

Search space: `cls ∈ [0.3, 0.4, ..., 1.0]` × `box ∈ [3, 4, ..., 10]` = **64 configurations**.

Results are saved to `hyperparameter_results_<timestamp>/`:
- `results_summary.csv` — all configs ranked by recall
- `best_result.json` — best hyperparameter combination
- `detailed_results.json` — full training log per config

---

## Training Configuration

Common parameters used across all comparison experiments:

| Parameter | Value |
|---|---|
| epochs | 300 |
| imgsz | 640 |
| optimizer | AdamW |
| lr0 | 0.0005 |
| lrf | 0.005 |
| weight_decay | 0.0005 |
| warmup_epochs | 15 |
| mosaic | 1.0 |
| close_mosaic | 20 |
| patience | 30 |
| amp | True |
| box | 10 |
| cls | 1.0 |

RT-DETR uses a lower learning rate (`lr0=0.0001`) and smaller batch size (`batch=16`) as recommended for transformer-based detectors.

---

## Metrics

| Metric | Description |
|---|---|
| F1 | Harmonic mean of precision and recall at best epoch |
| mAP50 | Mean Average Precision @ IoU=0.50 |
| mAP50-95 | Mean Average Precision @ IoU=0.50:0.95 |
| Params (M) | Total model parameters in millions |
| FPS | Inference frames per second (batch=1, imgsz=640) |
