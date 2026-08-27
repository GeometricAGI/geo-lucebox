#!/bin/bash
# Order-balanced (ABBA) timing sequence over an arm pair.
# ABBA puts each arm at position-mean 2.5, so a linear thermal/position drift
# cancels in the arm means instead of being charged to one format.
#
# Usage:  gqh_format_sweep_timing.sh [ARM_A] [ARM_B]
#   default: gqh iq4xs        (format vs format, shipping defaults)
#   control: gqh gqh_mmqoff   (MMQ on vs off, same target, one variable)
# Arm names must exist in ARMS in gqh_format_sweep_arm.py.
#
# Set GQH_SWEEP_CENSUS=1 to collect the ne11 histogram instead of clean timings
# (separate run: the counters perturb timing slightly and need a graceful exit).
set -u
PY=/home/deano/bench-venv/bin/python
ARM_PY=/home/deano/lb-upstream/server/scripts/gqh_format_sweep_arm.py

A="${1:-gqh}"
B="${2:-iq4xs}"

# This box is SHARED. Never pkill/killall by name: that would take out a
# co-tenant's server too. gqh_format_sweep_arm.py kills its own process group,
# so all this needs to do is verify nothing of OURS leaked, and refuse to reap
# anything owned by anybody else.
reap_our_strays () {
  local pid owner
  # -x matches the exact process NAME (not the argv, so it cannot self-match
  # the way `pgrep -f` does), and we still check the owner for every pid.
  for pid in $(pgrep -x dflash_server 2>/dev/null); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$owner" = "deano" ]; then
      echo "[reap] stray dflash_server pid=$pid owner=$owner -> SIGKILL"
      kill -9 "$pid" 2>/dev/null
    elif [ -n "$owner" ]; then
      echo "[reap] LEAVING pid=$pid owner=$owner (not ours)"
    fi
  done
}

gpu_edge_c () {
  # GPU 0 on this box reports edge temp as N/A (or omits the line) whenever the
  # card has gone to sleep -- which is exactly the cooled-down state we are
  # waiting for. Try junction as a second opinion before believing "no reading".
  local t
  for sensor in "Sensor edge" "Sensor junction"; do
    t=$(rocm-smi --showtemp 2>/dev/null | grep "GPU\[0\]" | grep "$sensor" \
        | sed 's/.*: //' | tr -d ' ')
    case "$t" in
      ''|*N/A*) continue ;;
      *) echo "${t%.*}"; return 0 ;;
    esac
  done
  return 1
}

gpu_vram_used_b () {
  rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" \
    | grep "VRAM Total Used Memory" | sed 's/.*: //' | tr -d ' '
}

# Idle baseline for GPU 0, measured at handover and after the restore.
VRAM_IDLE_B=60030976
VRAM_SLACK_B=67108864

cooldown () {
  echo "[cooldown] start $(date -Is)"
  for i in $(seq 1 60); do
    if ti=$(gpu_edge_c); then
      echo "[cooldown] ${i} temp=${ti}C"
      if [ "$ti" -le 36 ]; then echo "[cooldown] reached ${ti}C"; return; fi
    else
      # No temperature at all. If VRAM is back to the idle baseline then nothing
      # is resident and the card is asleep, i.e. as cool as it is going to get:
      # settle briefly and move on instead of burning the full 10 minutes.
      local used
      used=$(gpu_vram_used_b)
      if [ -n "$used" ] && [ "$used" -le $((VRAM_IDLE_B + VRAM_SLACK_B)) ]; then
        echo "[cooldown] no temp sensor, VRAM ${used}B at idle baseline -> card asleep, settling 30s"
        sleep 30
        return
      fi
      echo "[cooldown] ${i} no temp reading, VRAM=${used:-?}B still above baseline"
    fi
    sleep 10
  done
  echo "[cooldown] gave up waiting, proceeding"
}

echo "############ ABBA: $A $B $B $A  census=${GQH_SWEEP_CENSUS:-0} run=${GQH_SWEEP_RUN:-default} $(date -Is) ############"
for step in "$A r1" "$B r1" "$B r2" "$A r2"; do
  set -- $step
  echo "=============== ARM=$1 READING=$2 $(date -Is) ==============="
  HIP_VISIBLE_DEVICES=0 $PY "$ARM_PY" "$1" "$2" 2>&1
  echo "[exit] $?"
  reap_our_strays
  sleep 5
  cooldown
done
echo "=============== ABBA SEQUENCE DONE $(date -Is) ==============="
