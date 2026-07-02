#!/usr/bin/env python3
"""Ensemble N trained detectors with Weighted Box Fusion (WBF) + tiled inference,
writing the submission CSV. This is how we got our best result (F1 69.94): the
incumbent s-model is high-precision but misses lights, the SAHI-s model is
high-recall but over-fires — fusing them keeps detections both agree on
(precision) while recovering lights only one found (recall). Generalized to N
models so the s@1024 magnification run (and any future model) can be added.

Pipeline per image:
  1. Run tiled inference (tiles + full pass) with EACH model -> per-model boxes.
  2. Normalize to [0,1], fuse across all models with weighted_boxes_fusion.
  3. Threshold the fused scores, convert to center-xywh, write CSV.

Feed the per-model passes at a LOW --conf so WBF has candidates to fuse, and
write at a LOW --out-conf, then sweep the threshold by filtering the output CSV
(that's how the 69.94 curve was built). NOTE the calibration lesson: raising the
threshold trades recall for precision unpredictably — find the optimum by
filtering + uploading, don't assume low-conf boxes are all false positives.

Usage (from tld/):
    # Default = the F1 69.94 pair (incumbent + SAHI-s):
    uv run python predict_ensemble.py --images .../test_tld --model-weights 2,1
    # Add the s@1024 run for a 3-way fusion:
    uv run python predict_ensemble.py --images .../test_tld \
        --weights runs/detect/atlas_yolo26s-2/weights/best.pt,\
runs/detect/atlas_yolo26s_sahi-2/weights/best.pt,\
runs/detect/atlas_yolo26s_sahi1024/weights/best.pt \
        --model-weights 2,1,1 --out-conf 0.10
"""
import argparse
import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

from predict_tiled import tiled_predict
from predict import IMG_EXTS

# The two models behind F1 69.94 (incumbent precision + SAHI-s recall). Append
# more checkpoints via --weights for a larger fusion.
DEFAULT_WEIGHTS = (
    "runs/detect/atlas_yolo26s-2/weights/best.pt,"
    "runs/detect/atlas_yolo26s_sahi-2/weights/best.pt"
)


def main() -> None:
    p = argparse.ArgumentParser(description="N-model WBF ensemble (tiled) -> TLD CSV")
    p.add_argument("--images", required=True, help="folder of test images")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="comma-separated list of 2+ checkpoints to fuse "
                        "(default = the incumbent + SAHI-s pair behind F1 69.94)")
    p.add_argument("--model-weights", default=None,
                   help="comma-separated WBF weight per model, e.g. '2,1' or '2,1,1'. "
                        "Default = equal. The 69.94 run used '2,1' (incumbent favored).")
    p.add_argument("--out", default="predictions_ensemble.csv")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--tile-imgsz", default=None,
                   help="per-model crop input size: comma-separated (one per model) or a single "
                        "value. Default = tile-size for all. Match each model's training scale, "
                        "e.g. '640,640,1024' for incumbent,SAHI-s,s@1024.")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.12,
                   help="per-model detection threshold feeding WBF (low = more candidates)")
    p.add_argument("--iou", type=float, default=0.5, help="per-model in/cross-tile NMS IoU")
    p.add_argument("--imgsz-full", type=int, default=1024)
    p.add_argument("--wbf-iou", type=float, default=0.55, help="WBF clustering IoU")
    p.add_argument("--skip-thr", type=float, default=0.0, help="WBF: drop input boxes below this")
    p.add_argument("--out-conf", type=float, default=0.10,
                   help="final threshold on fused scores (keep LOW; sweep by filtering the CSV)")
    p.add_argument("--device", default=None)
    p.add_argument("--max-det", type=int, default=300)
    args = p.parse_args()

    img_dir = Path(args.images)
    images = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images in {img_dir}")

    weight_paths = [w.strip() for w in args.weights.split(",") if w.strip()]
    if len(weight_paths) < 2:
        raise SystemExit("Need >= 2 models to ensemble (--weights a,b[,c...])")
    if args.model_weights:
        wbf_weights = [float(x) for x in args.model_weights.split(",")]
        if len(wbf_weights) != len(weight_paths):
            raise SystemExit(
                f"--model-weights has {len(wbf_weights)} values but {len(weight_paths)} models")
    else:
        wbf_weights = [1.0] * len(weight_paths)

    # Per-model crop input size (match each model's training scale).
    if args.tile_imgsz:
        tile_imgszs = [int(x) for x in args.tile_imgsz.split(",")]
        if len(tile_imgszs) == 1:
            tile_imgszs *= len(weight_paths)
        if len(tile_imgszs) != len(weight_paths):
            raise SystemExit(
                f"--tile-imgsz has {len(tile_imgszs)} values but {len(weight_paths)} models")
    else:
        tile_imgszs = [None] * len(weight_paths)

    print(f"Found {len(images)} images; fusing {len(weight_paths)} models:")
    for wp, ww, ti in zip(weight_paths, wbf_weights, tile_imgszs):
        print(f"  {wp}  (w={ww}, crop_imgsz={ti or args.tile_size})")
    models = [YOLO(wp) for wp in weight_paths]

    def dets(model, img, ti):  # tiled detections for one model at its crop scale
        return tiled_predict(model, img, tile_size=args.tile_size, overlap=args.overlap,
                             conf=args.conf, iou=args.iou, imgsz_full=args.imgsz_full,
                             full_pass=True, device=args.device, max_det=args.max_det,
                             agnostic=False, tile_imgsz=ti)

    rows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ImageName", "xywh", "Conf", "Classification"])
        for img in images:
            W, H = Image.open(img).size
            boxes_l, scores_l, labels_l = [], [], []
            for model, ti in zip(models, tile_imgszs):
                d = dets(model, img, ti)
                boxes_l.append([[max(0, x1) / W, max(0, y1) / H, min(W, x2) / W, min(H, y2) / H]
                                for (x1, y1, x2, y2), s, c in d])
                scores_l.append([s for _, s, _ in d])
                labels_l.append([c for _, _, c in d])
            if not any(boxes_l):
                continue  # no model detected anything
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_l, scores_l, labels_l, weights=wbf_weights,
                iou_thr=args.wbf_iou, skip_box_thr=args.skip_thr)
            for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, labels):
                if sc < args.out_conf:
                    continue
                X1, Y1, X2, Y2 = x1 * W, y1 * H, x2 * W, y2 * H
                cx, cy = (X1 + X2) / 2, (Y1 + Y2) / 2
                # cast np.float64 -> float so the CSV renders "[260.93, ...]" not "[np.float64(...)]"
                box = [round(float(cx), 2), round(float(cy), 2),
                       round(float(X2 - X1), 2), round(float(Y2 - Y1), 2)]
                w.writerow([img.name, str(box), float(sc), int(cl)])
                rows += 1

    print(f"Wrote {rows} predictions to {args.out}")


if __name__ == "__main__":
    main()
