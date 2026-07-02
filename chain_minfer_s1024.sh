#!/bin/bash
# Test the m+SAHI ep13 checkpoint (two inference variants), THEN launch the
# s@1024 magnification run on the freed GPU. Sequential so they never contend.
cd /home/yannic/code/tld-traffic-light-detection
IMG=/home/yannic/code/tld-traffic-light-detection/dataset/test_tld
W=runs/detect/atlas_yolo26m_sahi/weights/best.pt

echo "[chain] $(date) m inference start" >> runs/chain.log
uv run python predict_tiled.py --images "$IMG" --weights "$W" \
  --out predictions_m_sahi.csv --device 0
uv run python predict_tiled.py --images "$IMG" --weights "$W" \
  --out predictions_m_sahi_full640.csv --imgsz-full 640 --device 0
echo "[chain] $(date) m inference done" >> runs/chain.log

# Launch s@1024 (640 tiles fed at imgsz 1024 = ~1.6x magnification on small lights)
nohup uv run python train.py --model yolo26s.pt \
  --tile --keep-full --downscale 1600 --tile-size 640 --tile-overlap 0.2 \
  --imgsz 1024 --batch 12 --workers 3 \
  --mixup 0.15 --save-period 3 --patience 12 --epochs 50 \
  --name atlas_yolo26s_sahi1024 --device 0 \
  > runs/s1024_train.log 2>&1 &
echo "[chain] $(date) s@1024 launched PID $!" >> runs/chain.log
