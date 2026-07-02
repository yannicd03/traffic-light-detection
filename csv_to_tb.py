#!/usr/bin/env python3
"""Bridge an Ultralytics results.csv into live TensorBoard scalars.

Ultralytics' native TB logging was off for this run (settings tensorboard=false),
so there are no event files. This tails results.csv: it backfills every epoch
already written, then appends new epochs as training flushes them — giving a live
TensorBoard view without restarting the run.

Usage (from tld/):
    uv run python csv_to_tb.py runs/detect/atlas_yolo26s_sahi-2
Then point TensorBoard at the run's tb/ dir (or at runs/detect to see all runs).
Self-terminates once training has ended and the CSV is fully drained.
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def training_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "train.py"],
                          capture_output=True).returncode == 0


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1
               else "runs/detect/atlas_yolo26s_sahi-2")
    csv_path = run / "results.csv"
    writer = SummaryWriter(str(run / "tb"))
    print(f"[csv_to_tb] bridging {csv_path} -> {run / 'tb'}", flush=True)
    seen: set[int] = set()

    while True:
        new = 0
        if csv_path.exists():
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    row = {k.strip(): v for k, v in row.items()}
                    try:
                        ep = int(float(row.pop("epoch")))
                    except (KeyError, ValueError):
                        continue
                    if ep in seen:
                        continue
                    for k, v in row.items():
                        try:
                            writer.add_scalar(k, float(v), ep)
                        except (TypeError, ValueError):
                            pass
                    seen.add(ep)
                    new += 1
            if new:
                writer.flush()
                print(f"[csv_to_tb] wrote {new} new epoch(s); total {len(seen)}", flush=True)
        # Drain once more after training ends, then exit.
        if not training_alive():
            time.sleep(2)
            if csv_path.exists():
                with csv_path.open() as f:
                    total = sum(1 for _ in csv.reader(f)) - 1
                if len(seen) >= total:
                    print("[csv_to_tb] training ended and CSV drained — exiting", flush=True)
                    break
        time.sleep(30)
    writer.close()


if __name__ == "__main__":
    main()
