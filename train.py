#!/usr/bin/env python3
"""Train a YOLO traffic-light detector on the ATLAS dataset.

Usage (from the tld/ folder):

    uv run train.py                          # defaults: yolo11s, 100 epochs, imgsz 1280
    uv run train.py --model yolo11m.pt --epochs 150 --imgsz 1280 --batch 8
    uv run train.py --data atlas.yaml --device 0

The traffic lights in CoCar images are small, so a large input resolution
(imgsz >= 1280) matters far more than a bigger backbone. Start here, watch the
val mAP50, and scale up only if you have GPU headroom.

Outputs land in runs/detect/<name>/ ; the best checkpoint is weights/best.pt,
which predict.py loads by default.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Pin runs to this project dir (Ultralytics' global runs_dir may point elsewhere),
# so predict.py's auto-detect finds the weights.
RUNS = str(Path(__file__).resolve().parent / "runs" / "detect")


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLO on ATLAS traffic lights")
    p.add_argument("--model", default="yolo26m.pt",
                   help="pretrained checkpoint or .yaml (e.g. yolo26n/s/m/l/x.pt, "
                        "yolo11s.pt, rtdetr-l.pt)")
    p.add_argument("--data", default="atlas.yaml", help="dataset config")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="train/val image size (lights are small)")
    p.add_argument("--batch", type=int, default=-1, help="batch size; -1 = auto (60%% GPU mem)")
    p.add_argument("--workers", type=int, default=8,
                   help="dataloader workers. LOWER this (e.g. 3) if host RAM is tight: "
                        "with --keep-full the full frames are large and many workers x "
                        "mosaic prefetch can trigger the system OOM killer.")
    p.add_argument("--device", default=None, help="cuda index e.g. 0 / 0,1, or cpu")
    p.add_argument("--name", default="atlas_yolo", help="run name under the project runs dir")
    p.add_argument("--project", default=RUNS, help="output parent dir for the run")
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    p.add_argument("--save-period", type=int, default=-1,
                   help="checkpoint every N epochs (-1 = only best+last); set >0 to keep "
                        "intermediate epochs so you can recover a less-overfit checkpoint")
    p.add_argument("--time", type=float, default=None,
                   help="max training time in HOURS; overrides --epochs and auto-fits the LR "
                        "schedule to the deadline (ideal for a fixed time budget)")
    p.add_argument("--tile", action="store_true",
                   help="TILED training: slice the dataset into tiles (cropping images + "
                        "remapping labels) and train on those, for small-object detection. "
                        "Pair with predict_tiled.py at inference.")
    p.add_argument("--tile-size", type=int, default=640, help="tile size (px) when --tile")
    p.add_argument("--tile-overlap", type=float, default=0.2, help="tile overlap when --tile")
    p.add_argument("--keep-full", action="store_true",
                   help="SAHI-style mixed training (with --tile): include the full-frame "
                        "images alongside the slices. Slices-only training regresses on "
                        "context + large objects (Akbas et al. 2022); this is the published "
                        "fix. Strongly recommended whenever you use --tile.")
    p.add_argument("--downscale", type=int, default=None,
                   help="with --tile: downscale source images to this longest-side (px) "
                        "before tiling, to match the test resolution (e.g. 1600). Aligns "
                        "train tile object-scale with the test set.")
    # Geometric augmentation (default 0 = unchanged) — simulates viewpoint variation
    # to help generalize to unseen camera angles. Keep fliplr off (arrows).
    p.add_argument("--degrees", type=float, default=0.0, help="rotation augmentation (deg)")
    p.add_argument("--shear", type=float, default=0.0, help="shear augmentation (deg)")
    p.add_argument("--perspective", type=float, default=0.0, help="perspective augmentation (0-0.001)")
    # Regularization knobs — to delay/raise the overfit peak (esp. for bigger models,
    # which overfit this domain faster). lr0/weight_decay only take effect when
    # --optimizer is NOT 'auto' (auto determines them itself).
    p.add_argument("--optimizer", default="auto",
                   help="optimizer: auto | SGD | AdamW | MuSGD. Set non-auto to honor --lr0/--weight-decay")
    p.add_argument("--lr0", type=float, default=0.01, help="initial LR (lower = gentler, flatter minima)")
    p.add_argument("--weight-decay", type=float, default=0.0005, help="L2 regularization strength")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="mixup augmentation prob (blends whole images; regularizes WITHOUT "
                        "distorting small objects, unlike geometric augs)")
    p.add_argument("--resume", action="store_true", help="resume from last.pt of --name run")
    args = p.parse_args()

    data = args.data
    if args.tile:
        # Build (once, cached) a tiled copy of the dataset and train on it.
        from tile_dataset import build_tiled_dataset
        data = str(build_tiled_dataset(args.data, tile_size=args.tile_size,
                                       overlap=args.tile_overlap,
                                       keep_full=args.keep_full,
                                       longest_side=args.downscale))
        print(f"Tiled training on: {data}")

    model = YOLO(args.model)

    # Surface validation F1 each epoch. Ultralytics logs precision/recall/mAP but
    # not F1 — which is exactly what the TLD grader scores. F1 = 2PR/(P+R) from the
    # validation precision & recall; printed every epoch and added to the metrics
    # dict (so it also reaches TensorBoard / results.csv where downstream logging
    # picks it up).
    def _log_val_f1(trainer):
        m = getattr(trainer, "metrics", {}) or {}
        p = float(m.get("metrics/precision(B)", 0.0))
        r = float(m.get("metrics/recall(B)", 0.0))
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        m["metrics/F1(B)"] = f1
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(f"   val F1={f1:.4f}  (P={p:.3f} R={r:.3f})")
        except Exception:
            print(f"val F1={f1:.4f} (P={p:.3f} R={r:.3f})")
    model.add_callback("on_fit_epoch_end", _log_val_f1)

    model.train(
        data=data,
        epochs=args.epochs,
        time=args.time,      # hours budget; None = use epochs (see --time)
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        save_period=args.save_period,
        resume=args.resume,
        # Light, sensible augmentation for traffic scenes. Defaults are a good
        # starting point per the assignment; tweak from the Ultralytics cfg docs.
        fliplr=0.0,          # do NOT mirror: left/right arrows are distinct classes!
        degrees=args.degrees,
        shear=args.shear,
        perspective=args.perspective,
        mosaic=1.0,
        close_mosaic=10,
        mixup=args.mixup,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        plots=True,
    )

    # Final val pass with the best weights so you see the metrics that map to F1.
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
