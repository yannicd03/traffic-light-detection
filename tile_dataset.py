#!/usr/bin/env python3
"""Build a TILED (SAHI-style) copy of a YOLO dataset for training on small
objects. Each source image is sliced into overlapping tiles; every label box is
clipped to the tile, dropped if too little remains visible, and renormalized to
the tile's coordinates. Training on these tiles teaches the model to detect
lights at the same apparent scale that predict_tiled.py feeds it — removing the
train(full-image)/infer(tile) scale mismatch.

This is the dataset-prep half of tiled detection; pair it with predict_tiled.py.

Layout produced (Ultralytics YOLO format; labels derived via /images/->/labels/):
    <dst>/data.yaml
    <dst>/train/images/*.jpg   <dst>/train/labels/*.txt
    <dst>/val/images/*.jpg     <dst>/val/labels/*.txt

Standalone usage (from tld/):
    uv run python tile_dataset.py --src atlas.yaml --tile-size 640 --overlap 0.2
    uv run python tile_dataset.py --src atlas.yaml --limit 20 --dst /tmp/tiled_smoke  # quick test

Or call build_tiled_dataset() from train.py via its --tile flag.

Empty tiles (sky/road with no light) are subsampled (--neg-frac) so the dataset
isn't swamped by background; deterministic (every k-th), no RNG.
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def tile_origins(extent: int, tile: int, step: int) -> list[int]:
    """Window start coords covering [0, extent); last window snaps to the edge."""
    if extent <= tile:
        return [0]
    xs = list(range(0, extent - tile + 1, step))
    if xs[-1] + tile < extent:
        xs.append(extent - tile)
    return xs


def _load_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def _pairs(split_dir: Path):
    """Yield (image_path, label_path|None) for all images under split_dir,
    deriving labels via the /images/ -> /labels/ convention."""
    for img in sorted(split_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS or "/images/" not in img.as_posix() + "/":
            continue
        lbl = Path(img.as_posix().replace("/images/", "/labels/")).with_suffix(".txt")
        yield img, (lbl if lbl.exists() else None)


def _read_boxes(label_path):
    """Return list of (cls, cx, cy, w, h) normalized boxes, or [] for background."""
    if label_path is None:
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *(float(v) for v in p[1:5])))
    return out


def _tile_labels(boxes, W, H, tx0, ty0, tw, th, min_vis):
    """Clip full-image normalized boxes to a tile; return tile-normalized lines."""
    lines = []
    for cls, cx, cy, w, h in boxes:
        bx1, by1 = (cx - w / 2) * W, (cy - h / 2) * H
        bx2, by2 = (cx + w / 2) * W, (cy + h / 2) * H
        ix1, iy1 = max(bx1, tx0), max(by1, ty0)
        ix2, iy2 = min(bx2, tx0 + tw), min(by2, ty0 + th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        box_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        if (ix2 - ix1) * (iy2 - iy1) / box_area < min_vis:
            continue  # too little of the box survived the crop
        ncx = ((ix1 + ix2) / 2 - tx0) / tw
        ncy = ((iy1 + iy2) / 2 - ty0) / th
        nw, nh = (ix2 - ix1) / tw, (iy2 - iy1) / th
        lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _emit_full(img_path, im, boxes, out_img, out_lbl, write_resized):
    """SAHI-style: add the full-frame image + label to a tiled split so training
    keeps scene context and large-object scale (slices-only training regresses on
    both — Akbas et al. 2022). When the frame was downscaled (write_resized), the
    resized array is written so the full frame matches the test resolution; else
    the original is symlinked (no disk blowup). Labels are normalized, so the
    box coords are unchanged by the resize."""
    stem = img_path.stem + "_full"
    if write_resized:
        dst_img = out_img / f"{stem}.jpg"
        cv2.imwrite(str(dst_img), im)
    else:
        dst_img = out_img / f"{stem}{img_path.suffix.lower()}"
        if not dst_img.exists():
            try:
                os.symlink(img_path.resolve(), dst_img)
            except (OSError, NotImplementedError):
                shutil.copy2(img_path, dst_img)  # fallback if symlinks unsupported
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    (out_lbl / f"{stem}.txt").write_text("\n".join(lines))


def build_tiled_dataset(src_yaml, dst=None, *, tile_size=640, overlap=0.2,
                        min_vis=0.3, neg_frac=0.1, limit=None, force=False,
                        keep_full=False, longest_side=None):
    """Slice the dataset referenced by src_yaml into tiles. Returns the path to
    the generated data.yaml. Idempotent: skips generation if dst exists (unless
    force=True)."""
    src_yaml = Path(src_yaml).resolve()
    data = yaml.safe_load(src_yaml.read_text())
    root = Path(data["path"]).resolve()
    names = _load_names(data)
    splits = {"train": data.get("train"), "val": data.get("val")}

    if dst is None:
        suffix = (f"_tiled_{tile_size}" + ("_full" if keep_full else "")
                  + (f"_ds{longest_side}" if longest_side else ""))
        dst = root.parent / f"{root.name}{suffix}"
    dst = Path(dst).resolve()
    yaml_path = dst / "data.yaml"
    if yaml_path.exists() and not force:
        print(f"[tile_dataset] reusing existing tiled dataset at {dst}")
        return yaml_path

    step = max(1, int(round(tile_size * (1.0 - overlap))))
    print(f"[tile_dataset] building tiles {tile_size}px @ {overlap:.0%} -> {dst}")

    for split, rel in splits.items():
        if not rel:
            continue
        split_dir = (root / rel)
        out_img = dst / split / "images"
        out_lbl = dst / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        n_img = n_tiles = n_obj_tiles = neg_seen = n_full = 0
        for img_path, lbl_path in _pairs(split_dir):
            if limit and n_img >= limit:
                break
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            H, W = im.shape[:2]
            # Downscale to the test resolution BEFORE tiling so train tiles match
            # the apparent object scale of the test set (train ~2500px, test 1600px).
            # Boxes are normalized -> unchanged by the resize.
            did_resize = False
            if longest_side and max(H, W) > longest_side:
                s = longest_side / max(H, W)
                im = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
                H, W = im.shape[:2]
                did_resize = True
            boxes = _read_boxes(lbl_path)
            n_img += 1
            if keep_full:
                _emit_full(img_path, im, boxes, out_img, out_lbl, did_resize)
                n_full += 1
            for ty0 in tile_origins(H, tile_size, step):
                for tx0 in tile_origins(W, tile_size, step):
                    tw = min(tile_size, W - tx0)
                    th = min(tile_size, H - ty0)
                    lines = _tile_labels(boxes, W, H, tx0, ty0, tw, th, min_vis)
                    if not lines:
                        # background tile: keep a deterministic fraction
                        neg_seen += 1
                        if neg_frac <= 0 or (neg_seen % max(1, round(1 / neg_frac))) != 0:
                            continue
                    else:
                        n_obj_tiles += 1
                    stem = f"{img_path.stem}_t{tx0}_{ty0}"
                    cv2.imwrite(str(out_img / f"{stem}.jpg"), im[ty0:ty0 + th, tx0:tx0 + tw])
                    (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
                    n_tiles += 1
        print(f"[tile_dataset]   {split}: {n_img} imgs -> {n_tiles} tiles "
              f"({n_obj_tiles} with objects)"
              + (f" + {n_full} full frames" if keep_full else ""))

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    yaml_path.write_text(
        f"# Tiled dataset generated by tile_dataset.py from {src_yaml.name}\n"
        f"path: {dst}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(names)}\nnames:\n{names_block}\n"
    )
    print(f"[tile_dataset] wrote {yaml_path}")
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description="Slice a YOLO dataset into training tiles")
    p.add_argument("--src", default="atlas.yaml", help="source dataset yaml")
    p.add_argument("--dst", default=None, help="output dir (default: <dataset>_tiled_<size>)")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min-vis", type=float, default=0.3,
                   help="min fraction of a box that must remain in a tile to keep it")
    p.add_argument("--neg-frac", type=float, default=0.1,
                   help="fraction of empty (background) tiles to keep")
    p.add_argument("--limit", type=int, default=None, help="cap images per split (for testing)")
    p.add_argument("--force", action="store_true", help="regenerate even if dst exists")
    p.add_argument("--keep-full", action="store_true",
                   help="SAHI-style mixed set: also add the full-frame images "
                        "to each split (preserves scene context + large-object scale; "
                        "the canonical fix for slices-only training regressing on test)")
    p.add_argument("--downscale", type=int, default=None,
                   help="downscale each source image so its longest side = N px BEFORE "
                        "tiling (e.g. 1600 to match the 1600x900 test set, so train tiles "
                        "match test object scale). Labels are normalized -> unchanged.")
    args = p.parse_args()
    build_tiled_dataset(args.src, args.dst, tile_size=args.tile_size, overlap=args.overlap,
                        min_vis=args.min_vis, neg_frac=args.neg_frac, limit=args.limit,
                        force=args.force, keep_full=args.keep_full, longest_side=args.downscale)


if __name__ == "__main__":
    main()
