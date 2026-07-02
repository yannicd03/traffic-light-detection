#!/usr/bin/env python3
"""Run a trained RF-DETR model over the TLD test images and write the
submission CSV in the grader's format (same contract as predict.py, which is
for the Ultralytics models).

    ImageName,xywh,Conf,Classification
    1708418258713137499_front_medium.jpg,"[260.25, 265.0, 37.0, 104.0]",0.93,1

- xywh = [center_x, center_y, width, height] in ABSOLUTE pixels.
- One row per predicted box; images with no detections produce NO row.

RF-DETR returns `supervision` Detections in xyxy (pixels); we convert to
center-xywh here.

>>> TILING IS THE WINNING TECHNIQUE — and RF-DETR needs it MORE than YOLO. <<<
RF-DETR resizes the WHOLE frame down to its network resolution (divisible by 56,
e.g. 1008). On a 1600x900 test frame the ~8px traffic lights shrink to ~3px and
DETRs are already weak on small objects, so a plain full-image pass badly
underperforms. The fix (ported from predict_tiled.py, our best YOLO config) is
SAHI-style tiled inference, which is ON BY DEFAULT here:

  1. Slide a `tile_size` window over each frame with `overlap`.
  2. Run model.predict() on each crop at full crop resolution (lights keep their
     pixels; a 640 tile fed to a 1008 net is MAGNIFIED).
  3. Translate each crop's boxes back into full-image coordinates.
  4. (default) Also run one full-image pass so large/close lights and ones
     straddling tile seams are still caught. Disable with --no-full.
  5. Merge everything with class-aware NMS (torchvision.ops.batched_nms) to drop
     duplicates produced by the tile overlaps.

Pass --no-tiled to fall back to the plain full-image path (kept as a fallback;
expect a much worse score on small lights).

>>> CLASS OFF-BY-ONE — READ THIS BEFORE YOU UPLOAD. <<<
RF-DETR's YOLO loader MAY reserve class index 0 for background, shifting every
predicted class id by +1 relative to the 25 ATLAS classes. The grader's F1
requires an EXACT class match, so a constant offset silently tanks the score
even with perfect boxes. ALWAYS sanity-check a handful of predictions against
known labels (e.g. eyeball a few rows vs. the ATLAS label files) BEFORE the
first leaderboard upload; if everything is shifted, pass --class-offset -1.

>>> DOMAIN-GAP CEILING (set expectations). <<<
Training data is front-cameras only; the test set contains side/rear cameras
absent from training. The realistic F1 ceiling is mid-70s for ANY model —
~324 of 990 test lights are missed by every model we tried. Do not expect 80%+,
and do not chase val mAP: val mAP is NOT predictive of test F1 (confirmed 3x —
the highest-val-mAP checkpoint scored the WORST test F1). Only tiled inference +
a leaderboard upload is the real verdict.

Install + usage (from the tld/ folder):

    uv sync --extra rfdetr
    # tiled (default, recommended):
    uv run --extra rfdetr predict_rfdetr.py --images /path/to/test_images
    uv run --extra rfdetr predict_rfdetr.py --images /path/to/test_images \
        --model medium --weights runs/rfdetr/atlas_rfdetr/checkpoint_best_ema.pth \
        --threshold 0.25 --resolution 1008 --tile-size 640 --overlap 0.2
    # plain full-image fallback (worse on small lights):
    uv run --extra rfdetr predict_rfdetr.py --images /path/to/test_images --no-tiled

Validate the resulting CSV at https://kit-ml2.streamlit.app/ before uploading.
"""
import argparse
import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MODELS = ("nano", "small", "medium", "base", "large")
# RF-DETR checkpoint names, best -> least preferred when auto-detecting.
CKPT_PREFERENCE = ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                   "checkpoint_best_total.pth", "checkpoint.pth")
# Default runs dir = this project's runs/rfdetr (matches train_rfdetr.py --output-dir).
DEFAULT_RUNS = str(Path(__file__).resolve().parent / "runs" / "rfdetr")


def latest_checkpoint(runs_dir: str = DEFAULT_RUNS) -> Path:
    """Find the best RF-DETR checkpoint under runs_dir.

    Prefers an EMA/best checkpoint; otherwise the most recently modified .pth.
    """
    root = Path(runs_dir)
    pths = list(root.rglob("*.pth"))
    if not pths:
        raise SystemExit(
            f"No .pth checkpoints found under {runs_dir}/ — train RF-DETR first "
            "(train_rfdetr.py), or pass --weights explicitly."
        )
    for preferred in CKPT_PREFERENCE:
        hits = [p for p in pths if p.name == preferred]
        if hits:
            return max(hits, key=lambda q: q.stat().st_mtime)
    return max(pths, key=lambda q: q.stat().st_mtime)


def tile_origins(extent: int, tile: int, step: int) -> list[int]:
    """Start coords covering [0, extent) with `tile`-wide windows of stride
    `step`; the last window is snapped to the edge so coverage is complete.

    (Same logic as predict_tiled.py.tile_origins.)
    """
    if extent <= tile:
        return [0]
    xs = list(range(0, extent - tile + 1, step))
    if xs[-1] + tile < extent:
        xs.append(extent - tile)
    return xs


def tiled_predict(model, img_path, *, tile_size, overlap, threshold, iou,
                  full_pass, agnostic=False):
    """SAHI-style tiled inference for one image.

    Slides a `tile_size` window with `overlap` over the full-res frame, runs
    model.predict() on each crop, translates the boxes back to full-image pixel
    coords, optionally adds a full-image pass, then merges everything with
    class-aware NMS (torchvision.ops.batched_nms) to dedupe cross-tile overlaps.

    RF-DETR's model.predict() accepts a numpy array (RGB) or path and returns a
    `supervision` Detections with .xyxy (pixels), .confidence, .class_id.

    Returns a list of ((x1, y1, x2, y2), score, cls) in full-image pixels.
    """
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import batched_nms

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # RF-DETR / supervision expect RGB
    H, W = img.shape[:2]
    step = max(1, int(round(tile_size * (1.0 - overlap))))

    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[int] = []

    def collect(det, ox: int, oy: int) -> None:
        if det is None or det.class_id is None or len(det) == 0:
            return
        for (x1, y1, x2, y2), conf, cls in zip(det.xyxy, det.confidence, det.class_id):
            boxes.append([float(x1) + ox, float(y1) + oy,
                          float(x2) + ox, float(y2) + oy])
            scores.append(float(conf))
            classes.append(int(cls))

    for y0 in tile_origins(H, tile_size, step):
        for x0 in tile_origins(W, tile_size, step):
            x1, y1 = min(x0 + tile_size, W), min(y0 + tile_size, H)
            crop = np.ascontiguousarray(img[y0:y1, x0:x1])
            det = model.predict(crop, threshold=threshold)
            collect(det, x0, y0)

    if full_pass:
        det = model.predict(np.ascontiguousarray(img), threshold=threshold)
        collect(det, 0, 0)

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
    p = argparse.ArgumentParser(description="RF-DETR inference -> TLD submission CSV")
    p.add_argument("--images", required=True, help="folder of the test images")
    p.add_argument("--weights", default=None,
                   help="RF-DETR checkpoint .pth; if omitted, auto-detects newest under runs/rfdetr/")
    p.add_argument("--model", default="medium", choices=MODELS,
                   help="RF-DETR size variant; MUST match the trained checkpoint")
    p.add_argument("--out", default="predictions.csv", help="output CSV path")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="confidence threshold; lower=more recall, higher=more precision")
    p.add_argument("--resolution", type=int, default=None,
                   help="inference resolution (divisible by 56); default uses the checkpoint's. "
                        "Tiles are fed at this resolution, so a 640 tile at 1008 is magnified.")
    # --- SAHI-style tiling (ON by default — our winning technique) ---
    tiling = p.add_mutually_exclusive_group()
    tiling.add_argument("--tiled", dest="tiled", action="store_true", default=True,
                        help="SAHI-style tiled inference (DEFAULT — strongly recommended)")
    tiling.add_argument("--no-tiled", dest="tiled", action="store_false",
                        help="fall back to a plain full-image pass (much worse on small lights)")
    p.add_argument("--tile-size", type=int, default=640,
                   help="square crop size (px) for slices (tiled mode)")
    p.add_argument("--overlap", type=float, default=0.2,
                   help="fractional overlap between tiles, 0-1 (tiled mode)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU for the cross-tile / full-pass merge (tiled mode)")
    p.add_argument("--no-full", action="store_true",
                   help="skip the extra full-image pass in tiled mode (tiles only)")
    p.add_argument("--agnostic-nms", action="store_true",
                   help="merge overlapping boxes across classes when deduping tiles "
                        "(dedupes one light detected as different classes by different tiles)")
    p.add_argument("--class-offset", type=int, default=0,
                   help="add to predicted class ids if your label mapping is shifted. "
                        "RF-DETR's YOLO loader MAY reserve index 0 for background, shifting "
                        "every class id by +1 vs. the 25 ATLAS classes — pass -1 then. The "
                        "grader needs an EXACT class match, so a constant offset silently "
                        "tanks F1. VERIFY a few predictions against known labels first!")
    args = p.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit(f"--overlap must be in [0, 1), got {args.overlap}")

    img_dir = Path(args.images)
    images = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    weights = Path(args.weights) if args.weights else latest_checkpoint()
    print(f"Found {len(images)} test images")
    print(f"Using weights: {weights}  (variant: {args.model})")
    if args.tiled:
        print(f"TILED inference: {args.tile_size}px @ {args.overlap:.0%} overlap, "
              f"full-image pass: {not args.no_full}, merge IoU: {args.iou}")
    else:
        print("PLAIN full-image inference (--no-tiled) — expect a worse score on small lights")
    if args.class_offset:
        print(f"!! Applying --class-offset {args.class_offset:+d} to every predicted class id")
    print("!! Reminder: verify class ids against the 25 ATLAS classes before uploading "
          "(off-by-one risk; see docstring).")

    # Import here so --help works without the (heavy) rfdetr install present.
    import rfdetr
    model_cls = {
        "nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase,
        "large": rfdetr.RFDETRLarge,
    }[args.model]

    init_kwargs = {"pretrain_weights": str(weights)}
    if args.resolution is not None:
        if args.resolution % 56 != 0:
            raise SystemExit(f"--resolution must be divisible by 56, got {args.resolution}")
        init_kwargs["resolution"] = args.resolution
    model = model_cls(**init_kwargs)

    rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)  # auto-quotes the bracketed xywh field (it has commas)
        writer.writerow(["ImageName", "xywh", "Conf", "Classification"])
        for img in images:
            if args.tiled:
                dets = tiled_predict(
                    model, img,
                    tile_size=args.tile_size, overlap=args.overlap,
                    threshold=args.threshold, iou=args.iou,
                    full_pass=not args.no_full, agnostic=args.agnostic_nms,
                )
                for (x1, y1, x2, y2), conf, cls in dets:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    # cast to python float so the CSV renders 37.0, not np.float64(37.0)
                    box = [round(float(cx), 2), round(float(cy), 2),
                           round(float(x2 - x1), 2), round(float(y2 - y1), 2)]
                    writer.writerow([img.name, str(box), float(conf),
                                     int(cls) + args.class_offset])
                    rows += 1
            else:
                det = model.predict(str(img), threshold=args.threshold)  # supervision Detections
                if det.class_id is None or len(det) == 0:
                    continue  # no detections -> no row
                for (x1, y1, x2, y2), conf, cls in zip(det.xyxy, det.confidence, det.class_id):
                    cx, cy = (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
                    box = [round(cx, 2), round(cy, 2),
                           round(float(x2 - x1), 2), round(float(y2 - y1), 2)]
                    writer.writerow([img.name, str(box), float(conf),
                                     int(cls) + args.class_offset])
                    rows += 1

    print(f"Wrote {rows} predictions to {args.out}")


if __name__ == "__main__":
    main()
