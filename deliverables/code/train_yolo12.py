#!/usr/bin/env python3
"""Alternative: train a YOLOv12 traffic-light detector on the ATLAS dataset.

YOLOv12 is an attention-centric YOLO (area-attention + R-ELAN) in Ultralytics.
It uses the exact same API, dataset config (atlas.yaml) and predict.py as the
YOLO26 path — only the backbone differs — so this is a drop-in comparison run.

Usage (from the tld/ folder):

    uv run train_yolo12.py                       # yolo12m, 100 epochs, imgsz 1280
    uv run train_yolo12.py --model yolo12l.pt --batch 8 --device 0

Note: YOLOv12 uses the classic one-to-many head, so NMS IS applied at inference
(unlike YOLO26's NMS-free default) — predict.py's --iou matters here.

Best checkpoint: runs/detect/atlas_yolo12/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Pin runs to this project dir so predict.py's auto-detect finds the weights.
RUNS = str(Path(__file__).resolve().parent / "runs" / "detect")


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv12 on ATLAS traffic lights")
    p.add_argument("--model", default="yolo12m.pt",
                   help="pretrained checkpoint or .yaml (yolo12n/s/m/l/x.pt)")
    p.add_argument("--data", default="atlas.yaml", help="dataset config")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="train/val image size (lights are small)")
    p.add_argument("--batch", type=int, default=-1, help="batch size; -1 = auto (60%% GPU mem)")
    p.add_argument("--device", default=None, help="cuda index e.g. 0 / 0,1, or cpu")
    p.add_argument("--name", default="atlas_yolo12", help="run name under the project runs dir")
    p.add_argument("--project", default=RUNS, help="output parent dir for the run")
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    p.add_argument("--resume", action="store_true", help="resume from last.pt of --name run")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        resume=args.resume,
        fliplr=0.0,          # left/right arrow classes are direction-specific
        mosaic=1.0,
        close_mosaic=10,
        optimizer="auto",
        plots=True,
    )

    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
