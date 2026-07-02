#!/usr/bin/env python3
"""Alternative: train an RF-DETR traffic-light detector on the ATLAS dataset.

RF-DETR is Roboflow's real-time DETR (ICLR 2026), a DIFFERENT model and package
from Ultralytics' RT-DETR. It is NMS-free and strong on small objects, but it
lives in its own `rfdetr` package with its own API and dataset layout.

  Install (separate from the Ultralytics env — heavy, own torch pin):
      uv sync --extra rfdetr          # uses the optional dep group in pyproject
      # or:  uv pip install rfdetr

IMPORTANT — dataset layout differs from atlas.yaml!
RF-DETR auto-detects YOLO format but expects this exact tree (note `valid`, and
per-split images/labels subfolders), passed as a DIRECTORY (not a .yaml):

    <dataset_dir>/
    ├── data.yaml                 # class names + nc
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/

Use reorg_for_rfdetr.py to build this tree non-destructively (it never touches
the source).

>>> TRAIN ON THE TILED DATASET, NOT THE RAW FULL-IMAGE ONE. <<<
RF-DETR resizes the WHOLE frame down to `resolution`. On raw ATLAS frames
(~2500x2000) or 1600x900 test frames, the ~8px lights shrink to ~3px and vanish
— DETRs are already weak on small objects. The fix is "SAHI-mixed" training:
train on 640px TILES (from frames downscaled to longest-side 1600, PLUS the full
downscaled frames for context). At resolution 1008 a 640 tile is resized UP =
MAGNIFICATION on the small lights, the opposite of what raw frames do.

    # one-time: convert our flat tiled set into the RF-DETR tree
    uv run reorg_for_rfdetr.py --layout flat \
        --src dataset/ATLAS_tiled_640_full_ds1600 --dst dataset/ATLAS_tiled_rfdetr
    # then train on it:
    uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_tiled_rfdetr

Plain usage (from the tld/ folder):

    uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_tiled_rfdetr
    uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_tiled_rfdetr \
        --model medium --resolution 1008 --batch 4 --grad-accum 4 --epochs 50

Notes:
- resolution MUST be divisible by 56 (e.g. 560, 728, 1008). Default 1008 = 56*18:
  640 tiles upscale to 1008 = magnification on small lights. Higher = more VRAM.
- Keep batch_size * grad_accum_steps ≈ 16 (e.g. A100: 16x1; T4: 4x4).
- early_stopping is ON and epochs are bounded (~50). DO NOT chase val mAP:
  val mAP is NOT predictive of test F1 (confirmed 3x — the highest-val-mAP
  checkpoint scored the WORST test F1). The only real verdict is TILED inference
  (predict_rfdetr.py, tiled by default) + a leaderboard upload.
- DOMAIN-GAP CEILING: training data is front-cameras only; the test set has
  side/rear cameras absent from training. The realistic F1 ceiling is mid-70s
  for ANY model — ~324 of 990 test lights are missed by every model. Do not
  expect 80%+.
- Geometric augmentations that distort tiny lights HURT (we keep them off for
  YOLO). RF-DETR's simple .train() API does not expose easy per-aug toggles; if
  you drop to the lower-level config, prefer keeping distorting geometric augs
  off so the ~3-8px lights are not smeared away.
- Checkpoints (incl. checkpoint_best_ema.pth) land in --output-dir.

Inference -> submission CSV is NOT handled by predict.py (that's Ultralytics):
use predict_rfdetr.py (TILED by default — the winning technique). RF-DETR
returns `supervision` Detections; see the CSV-export sketch at the bottom too.
"""
import argparse
from pathlib import Path

MODELS = ("nano", "small", "medium", "base", "large")
# Default output under this project's runs/rfdetr (matches predict_rfdetr.py auto-detect).
DEFAULT_OUT = str(Path(__file__).resolve().parent / "runs" / "rfdetr" / "atlas_rfdetr")


def main() -> None:
    p = argparse.ArgumentParser(description="Train RF-DETR on ATLAS traffic lights")
    p.add_argument("--dataset-dir", required=True,
                   help="dataset root (contains data.yaml + train/valid/test, see docstring). "
                        "RECOMMENDED: the flat-converted tiled set (ATLAS_tiled_rfdetr).")
    p.add_argument("--model", default="medium", choices=MODELS,
                   help="RF-DETR size variant")
    p.add_argument("--epochs", type=int, default=50,
                   help="bounded (~50); early_stopping stops sooner. Longer rarely helps — "
                        "val mAP is not predictive of test F1, so don't over-train chasing it.")
    p.add_argument("--resolution", type=int, default=1008,
                   help="input size, MUST be divisible by 56 (560/728/1008/...). 1008 upscales "
                        "640 tiles = magnification on small lights.")
    p.add_argument("--batch", type=int, default=4, help="batch size (tune to VRAM)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="grad accumulation; keep batch*grad_accum ~= 16")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", default=DEFAULT_OUT, help="checkpoint output dir")
    p.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    args = p.parse_args()

    if args.resolution % 56 != 0:
        raise SystemExit(f"--resolution must be divisible by 56, got {args.resolution}")

    # Import here so --help works without the (heavy) rfdetr install present.
    import rfdetr
    model_cls = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "base": rfdetr.RFDETRBase,
        "large": rfdetr.RFDETRLarge,
    }[args.model]

    model = model_cls()
    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=args.output_dir,
        early_stopping=True,
    )
    if args.resume:
        train_kwargs["resume"] = args.resume
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Inference -> submission CSV (RF-DETR variant). RF-DETR returns supervision
# Detections in xyxy (pixels); the grader wants center-xywh per box. Sketch:
#
#     import csv
#     from pathlib import Path
#     from rfdetr import RFDETRMedium
#
#     model = RFDETRMedium(pretrain_weights="runs/rfdetr/atlas_rfdetr/checkpoint_best_ema.pth")
#     with open("predictions.csv", "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["ImageName", "xywh", "Conf", "Classification"])
#         for img in sorted(Path("/path/to/test_images").glob("*.jpg")):
#             det = model.predict(str(img), threshold=0.25)   # supervision Detections
#             for (x1, y1, x2, y2), conf, cls in zip(det.xyxy, det.confidence, det.class_id):
#                 cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#                 box = [round(cx, 2), round(cy, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
#                 w.writerow([img.name, str(box), float(conf), int(cls)])
#
# NOTE: RF-DETR class ids may be offset by +1 if data.yaml/COCO reserves index 0
# for background — verify the mapping against the 25 ATLAS classes before
# submitting (a constant off-by-one will tank your F1 via class mismatch).
# ---------------------------------------------------------------------------
