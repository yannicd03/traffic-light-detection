# TLD — Traffic Light Detection (Ultralytics YOLO)

Scaffold for the ML2 bonus TLD project. Train a detector on the ATLAS dataset,
then emit `predictions.csv` in the grader's format.

## Setup

```bash
cd tld
uv sync                 # installs ultralytics + torch into .venv
```

> For training you want a CUDA-enabled torch (or use Google Colab's free GPU).
> The default install is fine for CPU inference / smoke tests.

## 1. Point the config at your dataset

Edit `atlas.yaml` — set `path` to the extracted ATLAS root and check that
`train:`/`val:` match its `images/...` folders. The 25 class names are already
filled in (order matters — it defines the integer class indices in the CSV).

## 2. Train

```bash
uv run train.py                       # yolo26m, 100 epochs, imgsz 1280
uv run train.py --model yolo26l.pt --batch 8 --device 0
```

Best checkpoint: `runs/detect/atlas_yolo/weights/best.pt`.
Traffic lights are small → high `--imgsz` (1280) helps more than a bigger model.
Note: `fliplr` augmentation is disabled because left/right arrow classes are
direction-specific.

Default model is **YOLO26** (`yolo26m.pt`), the current Ultralytics SOTA
(Jan 2026): NMS-free end-to-end head and improved small-object accuracy
(ProgLoss + STAL) — both helpful for tiny traffic lights. Needs
`ultralytics>=8.4`. Swap to `--model yolo11s.pt` if you hit any issues.

### Model alternatives

Four training entry points, all sharing `predict.py` except RF-DETR:

| Script | Model | Dataset config | NMS | Notes |
|---|---|---|---|---|
| `train.py` | **YOLO26m** (default) | `atlas.yaml` | NMS-free | SOTA, best small-object; **start here** |
| `train_yolo12.py` | YOLOv12m | `atlas.yaml` | uses NMS (`--iou`) | attention-based YOLO comparison |
| `train_rtdetr.py` | RT-DETR-L (Ultralytics) | `atlas.yaml` | NMS-free | transformer, memory-hungry |
| `train_rfdetr.py` | RF-DETR (Roboflow) | **own layout** ⚠️ | NMS-free | separate `rfdetr` pkg + own CSV path |

```bash
uv run train.py                       # YOLO26m
uv run train_yolo12.py                # YOLOv12m
uv run train_rtdetr.py                # RT-DETR-L
uv run --extra rfdetr train_rfdetr.py --dataset-dir /path/to/atlas_rfdetr   # RF-DETR
```

**RF-DETR caveats** (read `train_rfdetr.py` header): it's a separate Roboflow
package — install with `uv sync --extra rfdetr`; it needs a different dataset
layout (`train/valid/test/{images,labels}` + `data.yaml`, passed as a dir, not
the `atlas.yaml`); `--resolution` must be divisible by 56; and its predictions
do **not** go through `predict.py` — use the CSV-export sketch at the bottom of
that script (and double-check the class-id mapping for an off-by-one).

## 3. Predict the test set → CSV

```bash
# --weights is auto-detected: picks the most recent runs/detect/*/weights/best.pt
# (works for both the YOLO26 and RT-DETR runs).
uv run predict.py --images /path/to/test_images
# tune the recall/precision trade-off:
uv run predict.py --images /path/to/test_images --conf 0.2
# or pin a specific run:
uv run predict.py --images /path/to/test_images \
    --weights runs/detect/atlas_rtdetr/weights/best.pt
```

Produces `predictions.csv`:

```
ImageName,xywh,Conf,Classification
1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.934,1
```

`xywh` = [center_x, center_y, width, height] in absolute pixels; one row per
box; images with no detections get no row.

## 4. Validate

Upload `predictions.csv` to <https://kit-ml2.streamlit.app/> to see your F1 /
bonus points before the Ilias submission.

## Scoring recap

F1 = 2·TP / (2·TP + FP + FN); TP needs IoU>0.5 **and** correct class.
F1 > 40 → 3 pts · 30–40 → 2 pts · 20–30 → 1 pt.
Confidence isn't scored, so tune `--conf` purely to maximize F1.
