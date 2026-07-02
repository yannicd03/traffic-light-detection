#!/usr/bin/env python3
"""Run a trained YOLO model over the 425 TLD test images and write the
submission CSV in the exact format the grader expects.

    ImageName,xywh,Conf,Classification
    1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.934,1

- xywh: [center_x, center_y, width, height] in ABSOLUTE pixels.
- One row per predicted box. Images with no detections produce NO row.
- Classification: integer class index (0-24).

Usage (from the tld/ folder):

    uv run predict.py --images /path/to/test_images
    uv run predict.py --images /path/to/test_images \
        --weights runs/detect/atlas_yolo/weights/best.pt \
        --conf 0.25 --iou 0.5 --imgsz 1280 --out predictions.csv

Validate the resulting CSV at https://kit-ml2.streamlit.app/ before uploading.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Default runs dir = this project's runs/detect (matches train*.py --project).
DEFAULT_RUNS = str(Path(__file__).resolve().parent / "runs" / "detect")


def latest_best(runs_dir: str = DEFAULT_RUNS) -> Path:
    """Return the most recently modified weights/best.pt under runs_dir.

    Works across YOLO and RT-DETR runs (atlas_yolo/, atlas_rtdetr/, ...), so you
    don't have to pass --weights for whichever model you trained last.
    """
    candidates = list(Path(runs_dir).glob("*/weights/best.pt"))
    if not candidates:
        raise SystemExit(
            f"No checkpoints found under {runs_dir}/*/weights/best.pt — train a "
            "model first, or pass --weights explicitly."
        )
    return max(candidates, key=lambda q: q.stat().st_mtime)


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO inference -> TLD submission CSV")
    p.add_argument("--images", required=True, help="folder of the 425 test images")
    p.add_argument("--weights", default=None,
                   help="trained checkpoint; if omitted, picks the most recent "
                        "runs/detect/*/weights/best.pt")
    p.add_argument("--out", default="predictions.csv", help="output CSV path")
    p.add_argument("--imgsz", type=int, default=1280, help="inference image size (match training)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold; lower=more recall, higher=more precision")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold; ignored by YOLO26's default NMS-free end2end head")
    p.add_argument("--device", default=None, help="cuda index e.g. 0, or cpu")
    p.add_argument("--max-det", type=int, default=300, help="max detections per image")
    args = p.parse_args()

    img_dir = Path(args.images)
    images = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    print(f"Found {len(images)} test images")

    weights = Path(args.weights) if args.weights else latest_best()
    print(f"Using weights: {weights}")
    model = YOLO(str(weights))

    rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)  # auto-quotes the bracketed xywh field (it has commas)
        writer.writerow(["ImageName", "xywh", "Conf", "Classification"])

        # Process one image at a time. A list source with stream=True was
        # accumulating per-image GPU tensors and OOM'ing on the full set;
        # per-image inference keeps VRAM flat. We use img_path.name directly
        # (a list source also makes res.path generic "image0.jpg", which would
        # break the grader's filename matching).
        for img_path in images:
            res = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )[0]
            name = img_path.name
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue  # no detections -> no row for this image
            xywh = boxes.xywh.cpu().tolist()      # [cx, cy, w, h] in pixels
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.cpu().int().tolist()
            for (cx, cy, w, h), conf, cls in zip(xywh, confs, clss):
                box = [round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)]
                writer.writerow([name, str(box), conf, cls])
                rows += 1

    print(f"Wrote {rows} predictions to {args.out}")


if __name__ == "__main__":
    main()
