#!/usr/bin/env python3
"""Reorganize a source dataset into the layout RF-DETR expects, WITHOUT
touching the original (so train.py / train_yolo12.py / train_rtdetr.py, which
use atlas.yaml against the original ATLAS tree, keep working unchanged).

Two source layouts are supported via --layout:

--layout atlas  (default — the raw ATLAS per-camera tree)
    ATLAS ships as (split = train|test, camera = front_medium|front_tele|front_wide):

        ATLAS/<split>/<camera>/images/*.jpg
        ATLAS/<split>/<camera>/labels/*.txt    # filenames are camera-prefixed → no collisions
        ATLAS/ATLAS_classes.yaml

    Here we MERGE the three cameras into one images/labels per split and, since
    ATLAS has no validation split, carve one out of train (--val-frac).

--layout flat  (a pre-built FLAT tiled dataset, e.g. ATLAS_tiled_640_full_ds1600)
    Our best training set is SAHI-mixed: 640px tiles from frames downscaled to
    longest-side 1600, PLUS the full downscaled frames for context. It is FLAT
    (NOT per-camera):

        <src>/train/images/*.jpg
        <src>/train/labels/*.txt
        <src>/val/images/*.jpg
        <src>/val/labels/*.txt

    Here we map train -> train and val -> valid directly (no camera merge, no
    carve). >>> RECOMMENDED for RF-DETR: training on the tiled set and resizing
    the 640 tiles UP to 1008 magnifies the ~8px lights instead of shrinking them.

RF-DETR's YOLO loader requires (verified against rfdetr/datasets/yolo.py):

    <dst>/data.yaml                            # must contain `names`
    <dst>/train/{images,labels}/
    <dst>/valid/{images,labels}/               # REQUIRED (the split called "val" maps here)
    <dst>/test/{images,labels}/                # optional

This script only ever READS the source and creates a NEW dst tree via hardlinks
(default — no extra disk, source untouched), symlinks, or copies.

Usage (from the tld/ folder):

    # raw ATLAS (per-camera), carve a 10% valid split out of train:
    uv run reorg_for_rfdetr.py                                  # defaults below
    uv run reorg_for_rfdetr.py --src dataset/ATLAS --dst dataset/ATLAS_rfdetr \
        --val-frac 0.1 --mode hardlink
    uv run reorg_for_rfdetr.py --mode symlink --force

    # FLAT tiled dataset (RECOMMENDED) -> RF-DETR layout:
    uv run reorg_for_rfdetr.py --layout flat \
        --src dataset/ATLAS_tiled_640_full_ds1600 --dst dataset/ATLAS_tiled_rfdetr

Then train:  uv run --extra rfdetr train_rfdetr.py --dataset-dir dataset/ATLAS_rfdetr
"""
import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Canonical ATLAS classes (fallback if ATLAS_classes.yaml is absent).
ATLAS_NAMES = [
    "circle_green", "circle_red", "off", "circle_red_yellow", "arrow_left_green",
    "circle_yellow", "arrow_right_red", "arrow_left_red", "arrow_straight_red",
    "arrow_left_red_yellow", "arrow_left_yellow", "arrow_straight_yellow",
    "arrow_right_red_yellow", "arrow_right_green", "arrow_right_yellow",
    "arrow_straight_green", "arrow_straight_left_green", "arrow_straight_red_yellow",
    "arrow_straight_left_red", "arrow_straight_left_yellow",
    "arrow_straight_left_red_yellow", "arrow_straight_right_red",
    "arrow_straight_right_red_yellow", "arrow_straight_right_yellow",
    "arrow_straight_right_green",
]


def read_names(src: Path) -> list[str]:
    """Read class names from ATLAS_classes.yaml; fall back to the canonical list."""
    yml = src / "ATLAS_classes.yaml"
    if not yml.exists():
        return ATLAS_NAMES
    try:
        import yaml
        names = yaml.safe_load(yml.read_text()).get("names")
    except Exception:
        return ATLAS_NAMES
    if isinstance(names, dict):  # {0: name, 1: name, ...}
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return ATLAS_NAMES


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path | None]]:
    """All (image, label-or-None) pairs across every camera under a split dir."""
    pairs: list[tuple[Path, Path | None]] = []
    for cam in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        img_dir, lbl_dir = cam / "images", cam / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def collect_flat_pairs(split_dir: Path) -> list[tuple[Path, Path | None]]:
    """All (image, label-or-None) pairs from a FLAT split dir, i.e. one that has
    images/ and labels/ directly under it (no per-camera subfolders).

    Used for --layout flat (e.g. the tiled set ATLAS_tiled_640_full_ds1600,
    whose train/ and val/ each hold images/ + labels/ directly).
    """
    pairs: list[tuple[Path, Path | None]] = []
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"{img_dir} not found — expected a flat <split>/images dir.")
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def place(src_file: Path, dst_file: Path, mode: str) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if dst_file.exists() or dst_file.is_symlink():
        dst_file.unlink()
    if mode == "hardlink":
        os.link(src_file, dst_file)
    elif mode == "symlink":
        dst_file.symlink_to(src_file.resolve())
    else:  # copy
        shutil.copy2(src_file, dst_file)


def emit(pairs, dst_split: Path, mode: str) -> int:
    n = 0
    for img, lbl in pairs:
        place(img, dst_split / "images" / img.name, mode)
        if lbl is not None:
            place(lbl, dst_split / "labels" / (img.stem + ".txt"), mode)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Reorganize ATLAS for RF-DETR (non-destructive)")
    p.add_argument("--layout", default="atlas", choices=["atlas", "flat"],
                   help="source layout: 'atlas' = raw per-camera tree (default); "
                        "'flat' = pre-built FLAT tiled set with <split>/images + <split>/labels "
                        "and a train/ + val/ split (e.g. ATLAS_tiled_640_full_ds1600). See docstring.")
    p.add_argument("--src", default="dataset/ATLAS", help="source root (original, read-only here)")
    p.add_argument("--dst", default="dataset/ATLAS_rfdetr", help="output dataset root (created)")
    p.add_argument("--mode", default="hardlink", choices=["hardlink", "symlink", "copy"],
                   help="how to place files; hardlink=no extra disk, same filesystem (default)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="[atlas only] fraction of train held out as the required valid split "
                        "(0 disables: uses ATLAS test as valid). Ignored for --layout flat, "
                        "which already has a val/ split.")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --dst")
    args = p.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not (src / "train").is_dir():
        raise SystemExit(f"{src}/train not found — is --src the dataset root?")
    # Safety: never write into or over the source.
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit(f"--dst ({dst}) must be a separate directory outside --src ({src}).")
    if dst.exists() and any(dst.iterdir()) and not args.force:
        raise SystemExit(f"--dst {dst} is not empty; pass --force to overwrite.")

    names = read_names(src)
    print(f"{len(names)} classes; layout={args.layout}; mode={args.mode}; src={src}\n -> dst={dst}")

    if args.layout == "flat":
        # FLAT tiled set: <src>/train/{images,labels} + <src>/val/{images,labels}.
        # train -> train, val -> valid (the split RF-DETR requires). No camera
        # merge, no carve — the split already exists.
        if not (src / "val").is_dir():
            raise SystemExit(f"{src}/val not found — --layout flat expects train/ and val/ "
                             "subdirs each with images/ + labels/.")
        train_pairs = collect_flat_pairs(src / "train")
        val_pairs = collect_flat_pairs(src / "val")
        n_tr = emit(train_pairs, dst / "train", args.mode)
        n_va = emit(val_pairs, dst / "valid", args.mode)
        # RF-DETR treats test/ as optional. Reuse valid as test so the loader
        # always finds a non-empty test split (cheap via hardlinks; the eval is
        # only indicative anyway — val mAP is not predictive of test F1).
        n_te = emit(val_pairs, dst / "test", args.mode)
        print(f"train={n_tr}  valid={n_va} (= flat val/)  test={n_te} (= valid, reused)")
    else:
        train_pairs = collect_pairs(src / "train")
        test_pairs = collect_pairs(src / "test") if (src / "test").is_dir() else []

        # Build the required valid split.
        if args.val_frac and args.val_frac > 0:
            # Deterministic, reproducible hold-out: every k-th pair (no RNG).
            k = max(2, round(1 / args.val_frac))
            val_pairs = train_pairs[::k]
            val_set = set(id(x) for x in val_pairs)
            tr_pairs = [x for x in train_pairs if id(x) not in val_set]
            n_tr = emit(tr_pairs, dst / "train", args.mode)
            n_va = emit(val_pairs, dst / "valid", args.mode)
            n_te = emit(test_pairs, dst / "test", args.mode) if test_pairs else 0
            print(f"train={n_tr}  valid={n_va} (held out 1/{k} of train)  test={n_te}")
        else:
            # No carve: ATLAS test becomes the valid split RF-DETR requires.
            if not test_pairs:
                raise SystemExit("--val-frac 0 needs an ATLAS test split to use as valid, but none found.")
            n_tr = emit(train_pairs, dst / "train", args.mode)
            n_va = emit(test_pairs, dst / "valid", args.mode)
            print(f"train={n_tr}  valid={n_va} (= ATLAS test)  test=0")

    # data.yaml — RF-DETR only needs `names`; include nc + split hints for clarity.
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (dst / "data.yaml").write_text(
        f"# Generated by reorg_for_rfdetr.py for RF-DETR (YOLO format).\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames:\n{names_block}\n"
    )
    print(f"Wrote {dst/'data.yaml'}\nDone. Train with:\n"
          f"  uv run --extra rfdetr train_rfdetr.py --dataset-dir {dst}")


if __name__ == "__main__":
    main()
