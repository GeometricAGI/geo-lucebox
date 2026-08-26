#!/bin/bash
# Order-balanced (ABBA) timing sequence: GQH, IQ4_XS, IQ4_XS, GQH.
# ABBA puts each arm at position-mean 2.5, so a linear thermal/position drift
# cancels in the arm means instead of being charged to one format.
set -u
PY=/home/deano/bench-venv/bin/python
D=/home/deano/bench-out

cooldown () {
  echo "[cooldown] start $(date -Is)"
  for i in $(seq 1 60); do
    t=$(rocm-smi --showtemp 2>/dev/null | grep "GPU\[0\]" | grep "Sensor edge" | sed 's/.*: //')
    ti=${t%.*}
    echo "[cooldown] ${i} edge=${t}"
    if [ -n "$ti" ] && [ "$ti" -le 36 ]; then echo "[cooldown] reached ${t}C"; return; fi
    sleep 10
  done
  echo "[cooldown] gave up waiting, proceeding"
}

for step in "gqh r1" "iq4xs r1" "iq4xs r2" "gqh r2"; do
  set -- $step
  echo "=============== ARM=$1 READING=$2 $(date -Is) ==============="
  HIP_VISIBLE_DEVICES=0 $PY $D/timing_arm.py "$1" "$2" 2>&1
  echo "[exit] $?"
  pkill -x dflash_server 2>/dev/null
  sleep 5
  cooldown
done
echo "=============== ABBA SEQUENCE DONE $(date -Is) ==============="
