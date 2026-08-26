#!/bin/bash
# HumanEval+ 164 pass@1, both arms, TWICE per arm, order-balanced ABBA.
# Two readings of the SAME arm give that arm its own floor: with the
# temperature=0/top_k=1 fix in place the replies should be byte-identical.
#
# Server lifecycle is owned by quality_humaneval_plus.py, which spawns with
# start_new_session=True and kills its OWN process group in a finally block.
# We deliberately do NOT pkill by name: this box has had co-tenant
# dflash_server processes (a multi-day DeepSeek-V4 server), and a name-matched
# kill would take someone else's job down. Instead we wait for GPU 0 VRAM to
# fall back to baseline, and if it does not, we report and stop rather than
# killing something we cannot prove is ours.
set -u
PY=/home/deano/bench-venv/bin/python
R=/home/deano/lb-upstream
D=/home/deano/bench-out
SMI=$(command -v rocm-smi || echo /opt/rocm-7.2.4/bin/rocm-smi)
DRAFTER=/home/deano/bench-models/Qwen3.8-27B-DFlash2-Q8_0.gguf

edge_c () { $SMI --showtemp 2>/dev/null | grep "GPU\[0\]" | grep "Sensor edge" | sed 's/.*: //'; }
vram_pct () { $SMI --showmemuse 2>/dev/null | grep "GPU\[0\]" | grep "VRAM%" | sed 's/.*: //'; }

# Wait for OUR server to have released the GPU. No name-matched kill.
wait_vram_baseline () {
  for i in $(seq 1 24); do
    v=$(vram_pct); vi=${v%.*}
    if [ -n "$vi" ] && [ "$vi" -le 2 ]; then echo "[vram] back to ${v}% after ${i} check(s)"; return 0; fi
    sleep 5
  done
  echo "[vram] STILL ${v}% after 120s -- something holds GPU 0."
  echo "[vram] dflash_server processes and their owners:"
  ps -eo pid,user,etime,comm | grep dflash_serv || echo "  (none by that name)"
  echo "[vram] NOT killing by name: a co-tenant job could be among them."
  return 1
}

cooldown () {
  for i in $(seq 1 60); do
    t=$(edge_c); ti=${t%.*}
    if [ -n "$ti" ] && [ "$ti" -le 36 ]; then echo "[cooldown] reached ${t}C"; return 0; fi
    if [ -z "$ti" ]; then echo "[cooldown] ABORT: cannot read edge temp via $SMI"; return 1; fi
    sleep 10
  done
  echo "[cooldown] gave up waiting at $(edge_c)C, proceeding"
}

# Fail fast if the temperature probe does not actually parse.
t0=$(edge_c)
if [ -z "${t0%.*}" ]; then echo "FATAL: cannot parse edge temp from $SMI"; exit 1; fi
echo "[preflight] SMI=$SMI edge=${t0}C vram=$(vram_pct)%"

for step in "gqh r1" "iq4xs r1" "iq4xs r2" "gqh r2"; do
  set -- $step
  ARM=$1; RD=$2
  case $ARM in
    gqh)   TGT=/home/deano/bench-models/qwen38-gqh-shaped.gguf ;;
    iq4xs) TGT=/home/deano/bench-models/Qwen3.8-27B-IQ4_XS.gguf ;;
  esac
  echo "=============== QUALITY ARM=$ARM READING=$RD $(date -Is) ==============="
  echo "[pre]  edge=$(edge_c)C vram=$(vram_pct)%"
  rm -f /tmp/hep_results/samples_baseline.jsonl /tmp/hep_results/scores_baseline.json
  HIP_VISIBLE_DEVICES=0 \
  PFLASH_TARGET=$TGT \
  PFLASH_DRAFT=$DRAFTER \
  DFLASH_SERVER_BIN=$R/server/build-wide/dflash_server \
    $PY $R/server/scripts/quality_humaneval_plus.py --configs baseline 2>&1 | tail -25
  echo "[post] edge=$(edge_c)C vram=$(vram_pct)%"
  cp /tmp/hep_results/samples_baseline.jsonl $D/hep_samples_${ARM}_${RD}.jsonl
  cp /tmp/hep_results/scores_baseline.json   $D/hep_scores_${ARM}_${RD}.json
  cp /tmp/hep_results/server_baseline.log    $D/hep_server_${ARM}_${RD}.log
  echo "[saved] hep_samples_${ARM}_${RD}.jsonl $(wc -l < $D/hep_samples_${ARM}_${RD}.jsonl) lines"
  wait_vram_baseline || { echo "ABORT: GPU not released, refusing to continue"; exit 1; }
  cooldown || exit 1
done
echo "=============== QUALITY SEQUENCE DONE $(date -Is) ==============="
