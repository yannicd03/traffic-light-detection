#!/bin/bash
# Wait for the current m+SAHI training to finish, then launch the s @ imgsz 1024
# resolution experiment on the same cached tiled dataset. Run as a FILE (not an
# inline command) so multi-line structure survives — an inline wait-loop got its
# `sleep` mangled into "Too many arguments" and never launched.
cd /home/yannic/code/tld-traffic-light-detection

# Block until no train.py is running (m+SAHI done).
while pgrep -f 'train.py' >/dev/null; do
  sleep 60
done

# s backbone, 640 tiles fed at imgsz 1024 (~1.6x magnification on small lights),
# same SAHI mixed recipe + regularization as m+SAHI.
nohup uv run python train.py \
  --model yolo26s.pt \
  --tile --keep-full --downscale 1600 --tile-size 640 --tile-overlap 0.2 \
  --imgsz 1024 --batch 12 --workers 3 \
  --mixup 0.15 --save-period 3 --patience 12 --epochs 50 \
  --name atlas_yolo26s_sahi1024 --device 0 \
  > runs/s1024_train.log 2>&1 &

echo "s@1024 launched PID $! at $(date)" >> runs/queue_s1024.log
