# TLD Project Architecture

KIT ML2 bonus project. Train an object detector on the ATLAS traffic-light
dataset; produce `predictions.csv` for grader upload.

---

## Tech stack

- **Python** managed by `uv` (`pyproject.toml`, `uv.lock`)
- **Ultralytics >= 8.4** — YOLO26, YOLOv12, RT-DETR training + inference
- **rfdetr** (Roboflow, optional extra) — RF-DETR training + inference; kept
  isolated from the Ultralytics env due to conflicting torch pins
- **pandas**, **torch**, **torchvision** as direct deps

Install:
```
uv sync                        # Ultralytics stack only (default)
uv sync --extra rfdetr         # adds rfdetr package
```

---

## Dataset: ATLAS

Stored at `dataset/ATLAS/`. On-disk structure (YOLO label format, normalized
`<class> <cx> <cy> <w> <h>`):

```
ATLAS/
├── train/
│   ├── front_medium/{images,labels}/
│   ├── front_tele/{images,labels}/
│   └── front_wide/{images,labels}/
└── test/
    ├── front_medium/{images,labels}/
    ├── front_tele/{images,labels}/
    └── front_wide/{images,labels}/
```

**No validation split** — ATLAS only has train + test.

25 classes (integers 0–24): circle/arrow variants encoding color and direction
(e.g. `circle_green`=0, `arrow_straight_right_green`=24). Class order is fixed
by `atlas.yaml` and must be reproduced exactly in the submission CSV.

---

## Two model families — the key architectural split

```
┌─────────────────────────────────────────────────────────┐
│  Ultralytics stack (3 models, 1 shared predict path)    │
│                                                         │
│  atlas.yaml ──► train.py      → YOLO26m  ─┐            │
│                 train_yolo12.py → YOLOv12m ─┤           │
│                 train_rtdetr.py → RT-DETR-L ─┤          │
│                                              ▼           │
│                runs/detect/<name>/weights/best.pt        │
│                                              │           │
│                              predict.py ◄───┘           │
│                                   │                     │
│                          predictions.csv                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RF-DETR stack (fully separate)                         │
│                                                         │
│  reorg_for_rfdetr.py ──► dataset/ATLAS_rfdetr/          │
│                 (merges cameras + carves valid split)    │
│                               │                         │
│               train_rfdetr.py ▼                         │
│           runs/rfdetr/atlas_rfdetr/*.pth                 │
│                               │                         │
│            predict_rfdetr.py  ▼                         │
│                          predictions.csv                 │
└─────────────────────────────────────────────────────────┘
```

### Ultralytics models

All three share `atlas.yaml`, the same training hyperparameters (fliplr=0,
mosaic=1, imgsz=1280), and `predict.py`. Checkpoints land under
`runs/detect/`.

| Script | Model | NMS | Run dir |
|---|---|---|---|
| `train.py` | YOLO26m (default) | NMS-free | `runs/detect/atlas_yolo/` |
| `train_yolo12.py` | YOLOv12m | Classic NMS | `runs/detect/atlas_yolo12/` |
| `train_rtdetr.py` | RT-DETR-L | NMS-free | `runs/detect/atlas_rtdetr/` |

`predict.py` auto-selects the most recently modified `runs/detect/*/weights/best.pt`
when `--weights` is omitted. It outputs one CSV row per box; images with zero
detections produce no row.

### RF-DETR (Roboflow package)

Different in every dimension:

- **Package**: `rfdetr` (not `ultralytics`), installed via `uv sync --extra rfdetr`
- **Dataset layout**: requires `train/valid/test/{images,labels}` + `data.yaml`
  at a directory root (not `atlas.yaml`). `reorg_for_rfdetr.py` creates this
  non-destructively from the original ATLAS tree using hardlinks (default),
  symlinks, or copies. It merges the three camera folders and carves a
  validation split (default 10% of train, deterministic every-k-th selection).
- **Resolution constraint**: `--resolution` must be divisible by 56 (e.g. 1008)
- **Checkpoints**: `.pth` files under `runs/rfdetr/`; preferred filename is
  `checkpoint_best_ema.pth`
- **Inference**: `predict_rfdetr.py` (not `predict.py`). RF-DETR returns
  `supervision` Detections in xyxy pixel coords; the script converts to
  center-xywh. Has a `--class-offset` flag for potential off-by-one in the
  COCO-background-reserved index.

---

## Submission CSV format

```
ImageName,xywh,Conf,Classification
1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.934,1
```

- `xywh` = [center_x, center_y, width, height] in **absolute pixels**
- `Conf` = float confidence
- `Classification` = integer class index 0–24
- One row per detected box; **no row for images with zero detections**
- Validate at https://kit-ml2.streamlit.app/ before Ilias submission

Scoring: F1 = 2·TP / (2·TP + FP + FN); TP requires IoU > 0.5 AND correct class.
F1 > 40 → 3 pts, 30–40 → 2 pts, 20–30 → 1 pt. Confidence is not scored.

---

## Key design decisions

**`fliplr=0.0` in all trainers** — left/right arrow classes are directionally
distinct (e.g. `arrow_left_green` ≠ `arrow_right_green`), so horizontal flip
augmentation would corrupt labels.

**imgsz=1280** — traffic lights are small objects in wide-angle CoCar images;
high resolution matters more than a larger backbone.

**RF-DETR kept as an optional extra** — its `rfdetr` package pins its own torch
version, which conflicts with Ultralytics. Keeping it in a separate optional dep
group avoids resolver conflicts in the default env.

**reorg_for_rfdetr.py uses hardlinks by default** — no extra disk space; the
original ATLAS tree is never modified, so the three Ultralytics training scripts
continue to work unchanged against `atlas.yaml`.

---

## Usage quick-reference

See `README.md` for full usage. Short form:

```bash
# Edit atlas.yaml: set path: to your ATLAS root (and train:/val: subdirs)

# Train (pick one)
uv run train.py                                        # YOLO26m — start here
uv run train_yolo12.py                                 # YOLOv12m
uv run train_rtdetr.py                                 # RT-DETR-L

# Predict (Ultralytics models)
uv run predict.py --images /path/to/test_images

# RF-DETR path
uv run reorg_for_rfdetr.py --src dataset/ATLAS --dst dataset/ATLAS_rfdetr
uv sync --extra rfdetr
uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_rfdetr
uv run --extra rfdetr predict_rfdetr.py --images /path/to/test_images
```
