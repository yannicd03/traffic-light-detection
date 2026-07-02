#!/usr/bin/env python3
"""SAHI-style TILED inference for the TLD test set, writing the same submission
CSV as predict.py. Targets small/distant traffic lights that get destroyed when
a 1600x900 frame is letterboxed down to the network input.

How it works (manual SAHI — no extra dependency):
  1. Slide a `tile_size` window over each full-res image with `overlap`.
  2. Run the detector on each crop at full resolution (lights keep their pixels).
  3. Translate each crop's boxes back into full-image coordinates.
  4. (default) Also run one full-image pass so large/close lights and ones
     straddling tile seams are still caught.
  5. Merge everything with class-aware NMS to drop duplicates from the overlaps.

Trade-off vs predict.py: much better small-object recall, but tiling can raise
false positives (each tile fires independently). Since the grader scores F1,
tune --conf / --iou and compare against predict.py on the validator.

Usage (from tld/):
    uv run python predict_tiled.py --images .../test_tld --out predictions_tiled.csv
    uv run python predict_tiled.py --images .../test_tld \
        --tile-size 640 --overlap 0.2 --conf 0.25 --iou 0.5
"""
import argparse
import csv
from pathlib import Path

import cv2
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

# Reuse the proven helpers from predict.py (weight auto-detect, ext filter).
from predict import IMG_EXTS, latest_best


def tile_origins(extent: int, tile: int, step: int) -> list[int]:
    """Start coords covering [0, extent) with `tile`-wide windows of stride
    `step`; the last window is snapped to the edge so coverage is complete."""
    if extent <= tile:
        return [0]
    xs = list(range(0, extent - tile + 1, step))
    if xs[-1] + tile < extent:
        xs.append(extent - tile)
    return xs


def tiled_predict(model, img_path, *, tile_size, overlap, conf, iou,
                  imgsz_full, full_pass, device, max_det, agnostic=False,
                  tile_imgsz=None):
    """Return merged (xyxy, score, cls) detections in full-image pixel coords.

    tile_imgsz: network input size for each crop. Defaults to tile_size (crop fed
    at native size). Set it LARGER than tile_size to upsample a crop before the
    net — required to MATCH a model trained on magnified tiles (e.g. s@1024 trained
    640px tiles fed at imgsz 1024 -> infer with tile_size=640, tile_imgsz=1024)."""
    crop_imgsz = tile_imgsz or tile_size
    img = cv2.imread(str(img_path))  # BGR HxWx3 (Ultralytics treats numpy as BGR)
    if img is None:
        return []
    H, W = img.shape[:2]
    step = max(1, int(round(tile_size * (1.0 - overlap))))

    boxes, scores, classes = [], [], []

    def collect(res, ox, oy):
        b = res.boxes
        if b is None or len(b) == 0:
            return
        for (x1, y1, x2, y2), c, k in zip(b.xyxy.cpu().tolist(),
                                          b.conf.cpu().tolist(),
                                          b.cls.cpu().int().tolist()):
            boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
            scores.append(c)
            classes.append(k)

    for y0 in tile_origins(H, tile_size, step):
        for x0 in tile_origins(W, tile_size, step):
            x1, y1 = min(x0 + tile_size, W), min(y0 + tile_size, H)
            crop = img[y0:y1, x0:x1]
            res = model.predict(crop, imgsz=crop_imgsz, conf=conf, iou=iou,
                                device=device, max_det=max_det, verbose=False)[0]
            collect(res, x0, y0)

    if full_pass:
        res = model.predict(img, imgsz=imgsz_full, conf=conf, iou=iou,
                            device=device, max_det=max_det, verbose=False)[0]
        collect(res, 0, 0)

    if not boxes:
        return []
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(scores, dtype=torch.float32)
    c = torch.tensor(classes, dtype=torch.int64)
    # class-agnostic: one object detected as different classes across tiles/full
    # pass is deduped (keep highest conf). class-aware keeps all classes.
    idxs = torch.zeros_like(c) if agnostic else c
    keep = batched_nms(b, s, idxs, iou)
    return [(b[i].tolist(), float(s[i]), int(c[i])) for i in keep]


def main() -> None:
    p = argparse.ArgumentParser(description="Tiled (SAHI-style) TLD inference -> CSV")
    p.add_argument("--images", required=True, help="folder of test images")
    p.add_argument("--weights", default=None, help="checkpoint; default = newest under runs/detect/")
    p.add_argument("--out", default="predictions_tiled.csv", help="output CSV path")
    p.add_argument("--tile-size", type=int, default=640, help="square crop size (px) for slices")
    p.add_argument("--tile-imgsz", type=int, default=None,
                   help="network input size for each crop (default = tile-size). Set LARGER "
                        "to match a model trained on magnified tiles, e.g. s@1024: "
                        "--tile-size 640 --tile-imgsz 1024 (feeds 640px crops upsampled to 1024)")
    p.add_argument("--overlap", type=float, default=0.2, help="fractional overlap between tiles (0-1)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU for in-tile + cross-tile merge")
    p.add_argument("--imgsz-full", type=int, default=1024, help="imgsz for the full-image pass")
    p.add_argument("--no-full", action="store_true", help="skip the full-image pass (tiles only)")
    p.add_argument("--agnostic-nms", action="store_true",
                   help="merge overlapping boxes across classes (dedupes one light "
                        "detected as different classes by different tiles)")
    p.add_argument("--device", default=None, help="cuda index e.g. 0, or cpu")
    p.add_argument("--max-det", type=int, default=300, help="max detections per tile/image")
    p.add_argument("--class-offset", type=int, default=0, help="add to predicted class ids if shifted")
    args = p.parse_args()

    img_dir = Path(args.images)
    images = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    weights = Path(args.weights) if args.weights else latest_best()
    print(f"Found {len(images)} test images")
    print(f"Using weights: {weights}")
    print(f"Tiling: {args.tile_size}px @ {args.overlap:.0%} overlap, "
          f"full-image pass: {not args.no_full}")

    model = YOLO(str(weights))

    rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageName", "xywh", "Conf", "Classification"])
        for img_path in images:
            dets = tiled_predict(
                model, img_path,
                tile_size=args.tile_size, overlap=args.overlap,
                conf=args.conf, iou=args.iou, imgsz_full=args.imgsz_full,
                full_pass=not args.no_full, device=args.device, max_det=args.max_det,
                agnostic=args.agnostic_nms, tile_imgsz=args.tile_imgsz,
            )
            for (x1, y1, x2, y2), conf, cls in dets:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                box = [round(cx, 2), round(cy, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
                writer.writerow([img_path.name, str(box), conf, cls + args.class_offset])
                rows += 1

    print(f"Wrote {rows} predictions to {args.out}")


if __name__ == "__main__":
    main()
