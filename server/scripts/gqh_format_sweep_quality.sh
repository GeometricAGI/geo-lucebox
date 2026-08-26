#!/bin/bash
# HumanEval+ 164 pass@1, both arms, TWICE per arm, order-balanced ABBA.
# Two readings of the SAME arm give that arm its own floor: with the
# temperature=0/top_k=1 fix in place the replies should be byte-identical.
#
# Server lifecycle is owned by quality_humaneval_plus.py, which spawns with
# start_new_session=True and kills its OWN process group in a finally block.
# We deliberately do NOT pkill by name: this box has had co-tenant
# dflash_server processes, and a name-matched kill would take someone
# else's job down. Instead we gate on GPU 0 VRAM *in bytes* against a known
# baseline, and if it does not come back we report and stop rather than
# killing something we cannot prove is ours.
#
# Probe notes (measured on this box, 2026-08-26):
#   * `rocm-smi --showtemp` ALONE intermittently omits GPU[0] entirely, so the
#     old edge-temp probe returned empty and the old cooldown aborted the whole
#     sequence. Passing --showtemp --showmemuse together makes GPU[0] appear.
#     The temp probe is now retried and is REPORT-ONLY: it can never abort.
#     Thermals do not change greedy token output, so no quality gate needs it.
#   * `rocm-smi --showmeminfo vram` is exact and has been 100% reliable; it is
#     the co-tenant gate.
#   * Port 8765 is held by another user on this box, so we use PFLASH_PORT.
set -u
PY=/home/deano/bench-venv/bin/python
R=/home/deano/lb-upstream
D=/home/deano/bench-out
SMI=$(command -v rocm-smi || echo /opt/rocm-7.2.4/bin/rocm-smi)

DRAFTER=/home/deano/bench-models/qwen38-dflash2-q8_0-canonical.gguf
DRAFTER_MD5=a98fb401578886f082315c7031f419a2

PORT=8790
BASELINE_B=59895808                 # GPU 0 idle VRAM, bytes
COTENANT_B=$((BASELINE_B + 536870912))   # baseline + 512 MiB => someone else is resident
RELEASED_B=$((BASELINE_B + 268435456))   # baseline + 256 MiB => our server has let go

# --- probes ---------------------------------------------------------------
# GPU 0 used VRAM in bytes. Reliable.
vram_b () {
  $SMI --showmeminfo vram 2>/dev/null \
    | grep "GPU\[0\]" | grep "Total Used Memory" | sed "s/.*: //" | tr -d " \r"
}
# GPU 0 edge temp. Needs the combined flags; retried; may still be empty.
edge_c () {
  local i v
  for i in 1 2 3 4 5; do
    v=$($SMI --showtemp --showmemuse 2>/dev/null \
        | grep "GPU\[0\]" | grep "Sensor edge" | sed "s/.*: //" | tr -d " \r")
    if [ -n "$v" ]; then echo "$v"; return 0; fi
    sleep 1
  done
  echo ""
}
report () {   # report <label>
  local b t
  b=$(vram_b); t=$(edge_c)
  echo "[$1] gpu0_vram_used_B=${b:-UNREADABLE} baseline_B=$BASELINE_B edge=${t:-UNREADABLE}C"
}

# Hard gate: GPU 0 must be at baseline (nobody else resident) before we start.
require_clean_gpu () {
  local i b
  for i in $(seq 1 24); do
    b=$(vram_b)
    if [ -z "$b" ]; then echo "[gate] cannot read GPU 0 VRAM; retrying"; sleep 5; continue; fi
    if [ "$b" -le "$RELEASED_B" ]; then
      echo "[gate] GPU 0 clean at ${b} B (<= ${RELEASED_B}) after ${i} check(s)"
      return 0
    fi
    if [ "$b" -ge "$COTENANT_B" ]; then
      echo "[gate] GPU 0 holds ${b} B (>= ${COTENANT_B}) -- a CO-TENANT is resident."
    else
      echo "[gate] GPU 0 at ${b} B, above baseline but below co-tenant threshold; waiting"
    fi
    sleep 5
  done
  echo "[gate] GPU 0 STILL at $(vram_b) B after 120s."
  echo "[gate] dflash_server processes and owners:"
  ps -C dflash_server -o pid,user,etime,args 2>/dev/null || echo "  (none by that name)"
  echo "[gate] NOT killing by name: a co-tenant job could be among them."
  return 1
}

# --- preflight ------------------------------------------------------------
echo "[preflight] SMI=$SMI PORT=$PORT"
if [ ! -r "$DRAFTER" ]; then echo "FATAL: drafter unreadable: $DRAFTER"; exit 1; fi
got=$(md5sum "$DRAFTER" | cut -d" " -f1)
if [ "$got" != "$DRAFTER_MD5" ]; then
  echo "FATAL: drafter md5 mismatch: got $got want $DRAFTER_MD5"; exit 1
fi
echo "[preflight] drafter OK $DRAFTER md5=$got"
if [ -n "${GGML_GQH_MMQ:-}" ]; then
  echo "FATAL: GGML_GQH_MMQ is set to '${GGML_GQH_MMQ}'; shipping config needs it UNSET"; exit 1
fi
echo "[preflight] GGML_GQH_MMQ unset (defaults on) -- shipping config"
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FATAL: port $PORT is already in use"; exit 1
fi
echo "[preflight] port $PORT free"
report preflight
require_clean_gpu || { echo "ABORT: GPU 0 not clean at preflight"; exit 1; }

# --- ABBA sequence --------------------------------------------------------
for step in "gqh r1" "iq4xs r1" "iq4xs r2" "gqh r2"; do
  set -- $step
  ARM=$1; RD=$2
  case $ARM in
    gqh)   TGT=/home/deano/bench-models/qwen38-gqh-shaped.gguf ;;
    iq4xs) TGT=/home/deano/bench-models/Qwen3.8-27B-IQ4_XS.gguf ;;
  esac
  echo "=============== QUALITY ARM=$ARM READING=$RD $(date -Is) ==============="
  echo "[target] $TGT"
  require_clean_gpu || { echo "ABORT: GPU 0 not clean before ARM=$ARM RD=$RD"; exit 1; }
  report pre
  rm -f /tmp/hep_results/samples_baseline.jsonl /tmp/hep_results/scores_baseline.json

  HIP_VISIBLE_DEVICES=0 \
  PFLASH_PORT=$PORT \
  PFLASH_TARGET=$TGT \
  PFLASH_DRAFT=$DRAFTER \
  DFLASH_SERVER_BIN=$R/server/build-wide/dflash_server \
    $PY $R/server/scripts/quality_humaneval_plus.py --configs baseline 2>&1 | tail -25
  rc=${PIPESTATUS[0]}

  report post
  if [ ! -s /tmp/hep_results/samples_baseline.jsonl ]; then
    echo "ABORT: ARM=$ARM RD=$RD produced no samples (harness rc=$rc)"
    echo "--- server log tail ---"; tail -30 /tmp/hep_results/server_baseline.log 2>/dev/null
    exit 1
  fi
  cp /tmp/hep_results/samples_baseline.jsonl $D/hep_samples_${ARM}_${RD}.jsonl
  cp /tmp/hep_results/scores_baseline.json   $D/hep_scores_${ARM}_${RD}.json
  cp /tmp/hep_results/server_baseline.log    $D/hep_server_${ARM}_${RD}.log
  echo "[saved] hep_samples_${ARM}_${RD}.jsonl $(wc -l < $D/hep_samples_${ARM}_${RD}.jsonl) lines (harness rc=$rc)"

  require_clean_gpu || { echo "ABORT: GPU not released after ARM=$ARM RD=$RD"; exit 1; }
done
echo "=============== QUALITY SEQUENCE DONE $(date -Is) ==============="
