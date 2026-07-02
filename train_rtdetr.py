#!/usr/bin/env python3
"""Alternative: train an RT-DETR traffic-light detector on the ATLAS dataset.

RT-DETR is a transformer (DETR-family) detector. It is NMS-free by design and
often stronger on small / cluttered objects than CNN YOLOs, at the cost of
higher GPU memory and slower training. The assignment explicitly lists it as an
option. Same dataset config and the same predict.py work unchanged — only the
backbone differs.

Usage (from the tld/ folder):

    uv run train_rtdetr.py                       # rtdetr-l, 100 epochs, imgsz 1280
    uv run train_rtdetr.py --model rtdetr-x.pt --batch 4 --device 0

Heads-up vs YOLO:
- RT-DETR is memory-hungry at imgsz 1280; drop --batch (4-8) or --imgsz if you
  hit CUDA OOM.
- It has no objectness/NMS knobs; tune recall/precision purely via --conf at
  predict time.

Best checkpoint lands in runs/detect/atlas_rtdetr/weights/best.pt — pass it to
predict.py with --weights.
"""
import argparse
from pathlib import Path

from ultralytics import RTDETR

# Pin runs to this project dir so predict.py's auto-detect finds the weights.
RUNS = str(Path(__file__).resolve().parent / "runs" / "detect")


def main() -> None:
    p = argparse.ArgumentParser(description="Train RT-DETR on ATLAS traffic lights")
    p.add_argument("--model", default="rtdetr-l.pt",
                   help="pretrained checkpoint or .yaml (rtdetr-l.pt / rtdetr-x.pt)")
    p.add_argument("--data", default="atlas.yaml", help="dataset config")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280, help="train/val image size (lights are small)")
    p.add_argument("--batch", type=int, default=4,
                   help="batch size (RT-DETR is memory-hungry; raise if VRAM allows)")
    p.add_argument("--device", default=None, help="cuda index e.g. 0 / 0,1, or cpu")
    p.add_argument("--name", default="atlas_rtdetr", help="run name under the project runs dir")
    p.add_argument("--project", default=RUNS, help="output parent dir for the run")
    p.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    p.add_argument("--resume", action="store_true", help="resume from last.pt of --name run")
    args = p.parse_args()

    model = RTDETR(args.model)
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
