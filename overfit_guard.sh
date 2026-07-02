#!/bin/bash
# Overfit guard for one training run.
#   Usage: overfit_guard.sh '<run-dir-glob>'
# Waits for the run's results.csv, then watches val/cls_loss. Terminates training
# (so the next queued run can start) ONLY if val_cls diverged AND failed to catch
# itself: the min must be >= GRACE epochs in the past AND every one of the last
# GRACE epochs must be >THRESH above that min. A single dip back toward the min
# (or a new lower min) cancels the kill — that's the "give it a few epochs to
# recover" grace. Exits on KILL or when training ends on its own (either way the
# completion notifies the orchestrator). best.pt + save_period checkpoints survive
# a kill, so we never lose the pre-overfit weights.
cd /home/yannic/code/tld-traffic-light-detection
GLOB="$1"; GRACE=4; THRESH=1.02; MINEP=6

RUN=""
while [ -z "$RUN" ] || [ ! -f "$RUN/results.csv" ]; do
  RUN=$(ls -dt $GLOB 2>/dev/null | head -1)
  sleep 30
done
echo "[guard] $(date) monitoring $RUN (grace=$GRACE epochs)"

while pgrep -f 'train.py' >/dev/null; do
  v=$(awk -F',' -v g=$GRACE -v t=$THRESH -v me=$MINEP '
    NR==1{for(i=1;i<=NF;i++) if($i~"val/cls")c=i; next}
    {ep[++n]=$1; cl[n]=$c}
    END{
      if(n<me){print "WAIT"; exit}
      m=cl[1]; mi=1; for(i=1;i<=n;i++) if(cl[i]<m){m=cl[i];mi=i}
      if(n-mi>=g){ ok=1; for(i=n-g+1;i<=n;i++) if(cl[i]<=m*t) ok=0;
        if(ok){printf "KILL|min ep%s(cls=%.3f); last %d epochs all elevated; now=%.3f",ep[mi],m,g,cl[n]; exit} }
      printf "OK|min ep%s(cls=%.3f) now ep%s=%.3f",ep[mi],m,ep[n],cl[n]
    }' "$RUN/results.csv" 2>/dev/null)
  if [ "${v%%|*}" = "KILL" ]; then
    echo "[guard] $(date) TERMINATING $RUN — ${v#*|}"
    pkill -f 'train.py'
    echo "RESULT=KILLED RUN=$RUN"
    exit 0
  fi
  sleep 120
done
echo "[guard] $(date) $RUN ended on its own (patience/epochs)"
echo "RESULT=ENDED RUN=$RUN"
exit 0
