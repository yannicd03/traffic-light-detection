#!/bin/bash
# Ping when a run reaches a target epoch (or training ends). Reports the
# recall/precision/val_cls/val-F1 trend so the orchestrator can reassess.
#   Usage: epoch_ping.sh '<run-dir-glob>' <target_epoch>
cd /home/yannic/code/tld-traffic-light-detection
GLOB="$1"; TARGET="$2"
R=$(ls -dt $GLOB 2>/dev/null | head -1)
while pgrep -f 'train.py' >/dev/null; do
  R=$(ls -dt $GLOB 2>/dev/null | head -1)
  n=$(awk -F',' 'NR>1{c++} END{print c+0}' "$R/results.csv" 2>/dev/null)
  [ "${n:-0}" -ge "$TARGET" ] && break
  sleep 120
done
echo "=== $R @ ep>=$TARGET (or ended) ==="; date
awk -F',' 'NR==1{for(i=1;i<=NF;i++){if($i~"val/cls")c=i;if($i~"precision")pp=i;if($i~"recall")rr=i;if($i~"mAP50\\(")m=i}}
NR>1{p=$pp;r=$rr;f1=(p+r>0)?2*p*r/(p+r):0; printf "ep %2s: recall=%.3f prec=%.3f val_cls=%.3f valF1=%.4f mAP50=%.4f\n",$1,r,p,$c,f1,$m}' "$R/results.csv" | tail -8
pgrep -f 'train.py' >/dev/null && echo "(still training)" || echo "(training ended)"
