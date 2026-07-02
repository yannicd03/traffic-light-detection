# Traffic Light Detection — ATLAS (KIT ML2 Bonus, SS2026)

2D detection + classification of traffic lights for the KIT *Maschinelles Lernen
II* bonus (Project B). Fine-tunes **Ultralytics YOLO26** on the ATLAS dataset
(25 traffic-light classes) and emits `predictions.csv` in the grader's format.

The final submission is a **3-model Weighted Box Fusion (WBF) ensemble** of
YOLO26s checkpoints, evaluated with **SAHI-style tiled inference** (traffic
lights are tiny — down to ~8 px — so tiling + high resolution matters more than a
bigger backbone).

**Result: F1 = 71.27** on the 425-image test set — well above the `F1 > 40`
threshold, i.e. the full **3 bonus points**. (Validated at
<https://kit-ml2.streamlit.app/>.)

---

## Repository layout

```
tld-traffic-light-detection/
├── train.py            # YOLO26m training (+ tiled-dataset build, mixup, live val-F1)
├── train_yolo12.py     # YOLOv12m (attention-based) — comparison
├── train_rtdetr.py     # RT-DETR-L (Ultralytics transformer) — comparison
├── train_rfdetr.py     # RF-DETR (Roboflow, optional extra; own dataset layout)
├── predict.py          # shared single-model inference → CSV
├── predict_tiled.py    # SAHI-style tiled inference → CSV
├── predict_ensemble.py # N-model Weighted Box Fusion ensemble → CSV  (final path)
├── predict_rfdetr.py   # RF-DETR inference → CSV
├── tile_dataset.py     # builds the tiled + downscaled + full-frame training set
├── reorg_for_rfdetr.py # reshapes ATLAS into the RF-DETR train/valid/test layout
├── atlas.yaml          # dataset config (25 classes) — edit `path:` to your ATLAS root
├── csv_to_tb.py        # CSV → TensorBoard helper
├── *.sh                # training queue / monitoring helper scripts
├── predictions.csv     # final submission (committed)
├── deliverables/       # frozen Ilias submission (erklaerung.txt + predictions.csv + code/)
├── .agent/             # architecture / design-decision docs
├── pyproject.toml      # uv / ultralytics / torch (+ optional rfdetr extra)
│
├── dataset/            # (gitignored) ATLAS/, ATLAS_tiled_*/, ATLAS_rfdetr/, test_tld/
├── runs/               # (gitignored) trained checkpoints + TensorBoard  → Google Drive
└── *.pt                # (gitignored) Ultralytics base weights (auto-downloaded)
```

---

## Installation

Requires [`uv`](https://docs.astral.sh/uv/). A CUDA GPU (or Google Colab's free
GPU) is strongly recommended for training; CPU is fine for inference / smoke
tests.

```bash
git clone <your-repo-url> tld-traffic-light-detection
cd tld-traffic-light-detection
uv sync                    # ultralytics + torch into .venv
# uv sync --extra rfdetr   # only if you want the RF-DETR path (separate torch pin)
```

> For training you want a CUDA-enabled torch build. The default install works for
> CPU inference.

### Model weights (Google Drive)

Trained checkpoints (`runs/`) are too large for git and are hosted separately.

**Download:** `<GOOGLE_DRIVE_LINK_PLACEHOLDER>`

Unzip so the runs land under `runs/detect/`. The three checkpoints used by the
final ensemble:

```
runs/detect/atlas_yolo26s-2/weights/best.pt          # "incumbent"  — full images @1024 (precision)
runs/detect/atlas_yolo26s_sahi-2/weights/best.pt     # "SAHI-s"     — 640px tiles + full frames (recall)
runs/detect/atlas_yolo26s_sahi1024/weights/best.pt   # "ctx1024"    — 1024px tiles + full frames (context)
```

### Dataset

ATLAS is **not redistributed** here (course rules). Download it and place it (it
ships in YOLO format already) as:

```
dataset/
├── ATLAS/
│   ├── train/{front_medium,front_tele,front_wide}/{images,labels}/   # ~30k imgs
│   └── test/{front_medium,front_tele,front_wide}/{images,labels}/    # ~2.8k imgs (used as val)
└── test_tld/                                                          # 425 CoCar submission images
```

Then set `path:` in `atlas.yaml` to your absolute `dataset/ATLAS` path.

---

## Usage

### Reproduce the submission (the F1 = 71.27 ensemble)

With the three checkpoints under `runs/detect/` and the 425 test images in
`dataset/test_tld/`:

```bash
uv run predict_ensemble.py \
    --images dataset/test_tld \
    --weights runs/detect/atlas_yolo26s-2/weights/best.pt,runs/detect/atlas_yolo26s_sahi-2/weights/best.pt,runs/detect/atlas_yolo26s_sahi1024/weights/best.pt \
    --tile-size 1024 --out-conf 0.20 \
    --out predictions.csv
```

Upload `predictions.csv` to <https://kit-ml2.streamlit.app/> to see the F1.
Confidence isn't scored, so tune `--out-conf` purely to maximize F1.

### Single-model inference

```bash
# --weights auto-detects the most recent runs/detect/*/weights/best.pt
uv run predict.py --images dataset/test_tld
uv run predict.py --images dataset/test_tld --conf 0.2               # tune recall/precision
uv run predict_tiled.py --images dataset/test_tld --tile-size 1024  # SAHI tiled, one model
```

Output CSV columns: `ImageName,xywh,Conf,Classification` where
`xywh = [center_x, center_y, width, height]` in absolute pixels — one row per
box; images with no detections get no row.

### Train

```bash
uv run train.py                              # YOLO26m, imgsz 1280 — start here
uv run train.py --model yolo26s.pt --batch 8 --device 0
uv run train_yolo12.py                       # YOLOv12m
uv run train_rtdetr.py                       # RT-DETR-L
```

Best checkpoint lands at `runs/detect/<name>/weights/best.pt`.
`fliplr` augmentation is disabled (left/right arrow classes are direction-specific).

### Tiled training set (for the SAHI models)

```bash
# downscale to 1600px longest side, slice into overlapping tiles, KEEP full frames
uv run tile_dataset.py --src atlas.yaml --tile-size 1024 --downscale 1600
uv run train.py --data dataset/ATLAS_tiled_1024_full_ds1600/data.yaml
```

Monitor training: `uv run tensorboard --logdir runs/detect`.

### RF-DETR path (optional)

```bash
uv run reorg_for_rfdetr.py --src dataset/ATLAS --dst dataset/ATLAS_rfdetr
uv sync --extra rfdetr
uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_rfdetr
uv run --extra rfdetr predict_rfdetr.py --images dataset/test_tld
```

See `train_rfdetr.py`'s header and `.agent/project_architecture.md` for the
RF-DETR caveats (own dataset layout, `--resolution` divisible by 56, class-id
offset).

---

## Method notes

- **SAHI-style mixed training:** the tiled dataset keeps full frames alongside
  the tiles. Tiles teach detection of tiny lights at native resolution; the full
  frames preserve scene context and large objects. Training on tiles *alone*
  collapsed performance (following Akbas et al. 2022).
- **Weighted Box Fusion ensemble:** fuses the three models' detections — keeps
  boxes they agree on (precision) while recovering lights only one model found
  (recall).
- **`fliplr=0`:** left/right arrow classes are mirror-distinct
  (`arrow_left_green` ≠ `arrow_right_green`), so horizontal flip would corrupt
  labels. Rotation/shear/perspective are also off — they distort tiny lights.

## Scoring recap

F1 = 2·TP / (2·TP + FP + FN); TP needs IoU > 0.5 **and** correct class.
F1 `> 40` → 3 pts · `30–40` → 2 pts · `20–30` → 1 pt. Confidence is not scored.
