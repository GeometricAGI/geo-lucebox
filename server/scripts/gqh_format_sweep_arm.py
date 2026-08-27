#!/usr/bin/env python3
"""One timing reading for one arm (GQH or IQ4_XS) of the format-vs-format sweep.

Launches a fresh dflash_server with EVERY cache disabled (prefix-cache-slots 0,
prefill-cache-slots 0, prefill-compression off) so a repeat reading can never be
served from a cache, then measures, on the SAME server instance:

  decode   : decode tok/s + accept rate + avg_commit, from the server's own
             [spec-decode] stderr line, cross-checked against usage.timings
  prefill  : prefill tok/s at a short and a long prompt, from
             usage.timings.prefill_ms / prefilled_tokens
  e2e      : server/scripts/bench_he_http.py at its in-tree defaults

rocm-smi edge temperature + VRAM are sampled before and after every phase so a
reading taken under contention or thermal drift can be discarded.

Everything lands in one JSON file; no number is computed anywhere but here.
"""
import json, os, re, subprocess, sys, time, urllib.request

ARM      = sys.argv[1]              # "gqh" | "iq4xs"
READING  = sys.argv[2]             # "r1" | "r2"
# Each sweep gets its own subdirectory. Without this the "gqh" arm of the
# format sweep and the "gqh" arm of the MMQ control write the same filename and
# the second silently overwrites the first.
RUN      = os.environ.get("GQH_SWEEP_RUN", "default")
OUT_DIR  = os.path.join(os.path.expanduser("~/bench-out"), RUN)
ROOT     = os.path.expanduser("~/lb-upstream")
BIN      = f"{ROOT}/server/build-wide/dflash_server"
# The CANONICAL drafter. The earlier value here was Qwen3.8-27B-DFlash2-Q8_0.gguf
# (2,056,414,752 B), which is NOT the canonical artifact -- accept-rate readings
# taken against it are not comparable with anything else. Override only to
# deliberately re-measure the drafter axis.
DRAFTER  = os.environ.get(
    "GQH_SWEEP_DRAFTER",
    os.path.expanduser("~/bench-models/qwen38-dflash2-q8_0-canonical.gguf"))
TARGETS  = {
    "gqh":   os.path.expanduser("~/bench-models/qwen38-gqh-shaped.gguf"),
    "iq4xs": os.path.expanduser("~/bench-models/Qwen3.8-27B-IQ4_XS.gguf"),
}
# An arm is a (target, server-env) pair. Holding the target fixed and moving only
# GGML_GQH_MMQ gives the MMQ on/off control; holding the env fixed and moving the
# target gives the format-vs-format comparison. Both run through this one driver
# so neither can drift from the other in prompts, cache flags or geometry.
# "args" are extra server flags. The b16 arms exist to settle one specific
# question: the accept-rate ordering between GQH and IQ4_XS reversed between an
# older local probe that ran --draft-block-size 16 and this one, which runs the
# shipping default (0 = drafter metadata, block_size 8). Same box, same
# canonical drafter both times, so the drafter cannot be the cause; these arms
# move ONLY the block width and leave prompts, caches and metric alone.
ARMS = {
    "gqh":        {"target": "gqh",   "env": {}, "args": []},
    "iq4xs":      {"target": "iq4xs", "env": {}, "args": []},
    "gqh_mmqoff": {"target": "gqh",   "env": {"GGML_GQH_MMQ": "0"}, "args": []},
    "gqh_mmqon":  {"target": "gqh",   "env": {"GGML_GQH_MMQ": "1"}, "args": []},
    "gqh_b16":    {"target": "gqh",   "env": {}, "args": ["--draft-block-size", "16"]},
    "iq4xs_b16":  {"target": "iq4xs", "env": {}, "args": ["--draft-block-size", "16"]},
    # MMQ width-gate arms. gqh_mmq_max_ne11 defaults to 160, a bound measured
    # BEFORE the v_perm_b32 weight-LUT decode made MMQ faster. Re-measuring the
    # kernel crossover with the gate lifted puts the widest all-shapes-win bound
    # at ~640, with the 512-wide prefill chunk a 7-27% MMQ win. These arms move
    # ONLY the gate, so the real-workload question is asked separately from the
    # kernel curve. 512 and 640 should behave IDENTICALLY on this workload -- the
    # ne11 census shows nothing dispatches in 105..511 -- so a gap between them is
    # a read on the noise floor, not an effect.
    # The OLD compiled default, pinned explicitly. Once the default moves, this
    # is the only way to get the pre-change dispatch back for a same-binary A/B.
    "gqh_ne11_160": {"target": "gqh", "env": {"GGML_GQH_MMQ_MAX_NE11": "160"}, "args": []},
    "gqh_ne11_640": {"target": "gqh", "env": {"GGML_GQH_MMQ_MAX_NE11": "640"}, "args": []},
    "gqh_ne11_512": {"target": "gqh", "env": {"GGML_GQH_MMQ_MAX_NE11": "512"}, "args": []},
}
# Census mode asks the library for the ne11 histogram instead of a clean timing.
# It is a SEPARATE run: the atomic increments are cheap but not free, and the
# table is only flushed by an atexit handler, which needs a graceful shutdown
# rather than the SIGKILL a timing run ends with.
CENSUS = os.environ.get("GQH_SWEEP_CENSUS", "") not in ("", "0")
PORT = 18080
TAG  = f"{ARM}_{READING}"
LOG  = f"{OUT_DIR}/timing_{TAG}.log"
os.makedirs(OUT_DIR, exist_ok=True)

# Caches OFF. These three flags are the whole point: without them a second
# reading of the same arm could be answered out of a cache rather than measured.
CACHE_OFF = ["--prefix-cache-slots", "0",
             "--prefill-cache-slots", "0",
             "--prefill-compression", "off"]

N_WARMUP  = 3     # discarded; first requests pay graph build + weight fault-in
N_DECODE  = 5     # decode readings per arm-reading
N_PREFILL = 5     # prefill readings per prompt size
DECODE_MAX_TOKENS = 256


# GPU 0 idle baseline on this box, measured at handover and re-measured after the
# restore: 60,030,976 B (60.0 MB) with no KFD PIDs. VRAM% alone cannot police a
# shared box -- it is an integer percent of 34 GB, so a whole 342 MB co-tenant
# rounds to 0. The absolute byte figure is what makes contention visible.
VRAM_IDLE_BASELINE_B = 60_030_976
VRAM_CONTENTION_SLACK_B = 64 * 1024 * 1024


def smi():
    """Edge temp (C), VRAM% AND absolute VRAM bytes for GPU 0, plus raw dumps."""
    out = subprocess.run(["rocm-smi", "--showtemp", "--showmemuse"],
                         capture_output=True, text=True).stdout
    temp = vram = None
    # This sensor intermittently reports N/A or omits the line entirely on this
    # card. A reading is not worth aborting a run over, so parse defensively --
    # an uncaught ValueError here would kill the measurement, not just the sample.
    for line in out.splitlines():
        if "GPU[0]" in line and "Sensor edge" in line:
            try:
                temp = float(line.split(":")[-1].strip())
            except ValueError:
                temp = None
        if "GPU[0]" in line and "VRAM%" in line:
            try:
                vram = float(line.split(":")[-1].strip())
            except ValueError:
                vram = None
    mem = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                         capture_output=True, text=True).stdout
    used_b = None
    for line in mem.splitlines():
        if "GPU[0]" in line and "VRAM Total Used Memory" in line:
            try:
                used_b = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    concise = subprocess.run(["rocm-smi"], capture_output=True, text=True).stdout
    return {"edge_c": temp, "vram_pct": vram, "vram_used_b": used_b,
            "t": time.time(), "concise": concise, "meminfo": mem}


def idle_vram_clean(rec_smi):
    """True if GPU 0 VRAM is at our known idle baseline (nobody else resident)."""
    b = rec_smi.get("vram_used_b")
    if b is None:
        return None
    return b <= VRAM_IDLE_BASELINE_B + VRAM_CONTENTION_SLACK_B


def other_gpu_procs():
    """Any process holding GPU 0 besides ours -> contention, discard the reading."""
    out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True).stdout
    return out


def post(payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def chat(content, max_tokens):
    return post({
        "model": "dflash",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    })


# ---- prompts -------------------------------------------------------------
# Short: one ordinary code-completion ask. Long: the same ask preceded by a
# synthetic module so the prompt crosses several 512-token prefill chunks.
SHORT_PROMPT = (
    "Here is a Python helper:\n\n"
    "def merge_intervals(intervals):\n"
    "    intervals = sorted(intervals, key=lambda p: p[0])\n"
    "    out = []\n"
    "    for lo, hi in intervals:\n"
    "        if out and lo <= out[-1][1]:\n"
    "            out[-1][1] = max(out[-1][1], hi)\n"
    "        else:\n"
    "            out.append([lo, hi])\n"
    "    return out\n\n"
    "Explain in one sentence what it does, then state its time complexity."
)

_FILLER_UNIT = (
    "class Record{i}:\n"
    "    \"\"\"Row {i} of the synthetic ledger fixture.\"\"\"\n"
    "    def __init__(self, key, value, weight={i}):\n"
    "        self.key = key\n"
    "        self.value = value\n"
    "        self.weight = weight\n"
    "    def scaled(self):\n"
    "        return self.value * self.weight / ({i} + 1)\n\n"
)
LONG_PROMPT = (
    "Read the following module, then answer the question at the end.\n\n" +
    "".join(_FILLER_UNIT.format(i=i) for i in range(90)) +
    "\nQuestion: how many classes are defined above, and what does "
    "Record7.scaled() return when value=10?"
)

SPEC_RE = re.compile(
    r"\[spec-decode\] tokens=(\d+) time=([\d.]+) s speed=([\d.]+) tok/s "
    r"steps=(\d+) accepted=(\d+)/(\d+) \(([\d.]+)%\) avg_commit=([\d.]+)")


def read_spec_lines(path, seen):
    """Return [spec-decode] summary dicts not yet consumed."""
    fresh = []
    with open(path, errors="replace") as f:
        text = f.read()
    for i, m in enumerate(SPEC_RE.finditer(text)):
        if i < seen:
            continue
        fresh.append({
            "tokens": int(m.group(1)), "time_s": float(m.group(2)),
            "speed_tok_s": float(m.group(3)), "steps": int(m.group(4)),
            "accepted": int(m.group(5)), "draft_positions": int(m.group(6)),
            "accept_pct": float(m.group(7)), "avg_commit": float(m.group(8)),
        })
    return fresh, seen + len(fresh)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    spec   = ARMS[ARM]
    target = TARGETS[spec["target"]]
    rec = {
        "arm": ARM, "reading": READING,
        "target": target, "target_bytes": os.path.getsize(target),
        "drafter": DRAFTER, "drafter_bytes": os.path.getsize(DRAFTER),
        "server_bin": BIN, "server_bin_bytes": os.path.getsize(BIN),
        "cache_flags": CACHE_OFF,
        "git_head": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "run": RUN,
        "arm_env": spec["env"],
        "arm_args": spec.get("args", []),
        "census": CENSUS,
        "env_note": ("shipping defaults except where arm_env overrides: "
                     "GGML_GQH_MMQ unset (=on), GGML_GQH_MMQ_MAX_NE11 unset "
                     "(gate=160), no --specla, no --draft-block-size "
                     "(drafter metadata block_size=8)"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    rec["smi_pre"] = smi()
    rec["pids_pre"] = other_gpu_procs()
    # A co-tenant already resident before we launch invalidates the reading just
    # as surely as one that appears mid-run, so record the verdict up front.
    rec["gpu_clean_pre"] = idle_vram_clean(rec["smi_pre"])
    rec["vram_idle_baseline_b"] = VRAM_IDLE_BASELINE_B

    # Refuse to measure a contended GPU. This box is shared, and a co-tenant
    # resident before launch has two consequences, both bad: the reading is
    # worthless (it competes for bandwidth, and if VRAM is short the load simply
    # OOMs and every figure comes back 0.0/None), and launching anyway piles a
    # second big model onto somebody else's job. Recording an explicit error is
    # what keeps a contended reading out of the report instead of averaging it
    # in -- silent zeros are exactly the failure mode that makes a sweep lie.
    if rec["gpu_clean_pre"] is False and os.environ.get(
            "GQH_SWEEP_ALLOW_CONTENDED", "") in ("", "0"):
        rec["error"] = (
            "GPU 0 not idle before launch: %s B resident vs %s B baseline "
            "(co-tenant present; reading refused, not measured)"
            % (rec["smi_pre"].get("vram_used_b"), VRAM_IDLE_BASELINE_B))
        rec["contended"] = True
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        path = f"{OUT_DIR}/timing_{TAG}.json"
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"WROTE {path}")
        print("REFUSED:", rec["error"])
        return 1

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    for k in ("GGML_GQH_MMQ", "GGML_GQH_MMQ_MAX_NE11", "GGML_GQH_NE11_LOG"):
        env.pop(k, None)          # start from shipping defaults, not inherited state
    env.update(spec["env"])       # then the one variable this arm moves
    if CENSUS:
        env["GGML_GQH_NE11_LOG"] = "1"
    cmd = [BIN, target, "--draft", DRAFTER, "--port", str(PORT), *CACHE_OFF,
           *spec.get("args", [])]
    rec["cmd"] = cmd
    log = open(LOG, "w")
    t_launch = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 600
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",
                                       timeout=2).read()
                ready = True
                break
            except Exception:
                time.sleep(1)
        rec["load_s"] = round(time.time() - t_launch, 1)
        if not ready:
            rec["error"] = "server did not become ready"
            return finish(rec, proc)

        seen = 0
        # --- warmup (discarded) ---
        for _ in range(N_WARMUP):
            chat(SHORT_PROMPT, 64)
        time.sleep(1)
        _, seen = read_spec_lines(LOG, seen)     # drop warmup spec lines
        rec["smi_post_warmup"] = smi()

        # --- decode ---
        dec = []
        for i in range(N_DECODE):
            body, wall = chat(SHORT_PROMPT, DECODE_MAX_TOKENS)
            time.sleep(0.5)
            fresh, seen = read_spec_lines(LOG, seen)
            u = body["usage"]; t = u["timings"]
            dec.append({
                "i": i, "wall_s": round(wall, 3),
                "completion_tokens": u["completion_tokens"],
                "api_decode_tok_s": t["decode_tokens_per_sec"],
                "api_decode_ms": t["decode_ms"],
                "api_accept_rate": u["accept_rate"],
                "spec_decode_ran": u["spec_decode_ran"],
                "cache_hit": t["cache_hit"],
                "cached_prefix_tokens": t["cached_prefix_tokens"],
                "spec": fresh[-1] if fresh else None,
            })
        rec["decode"] = dec
        rec["smi_post_decode"] = smi()

        # --- prefill, two prompt sizes, max_tokens=1 ---
        pf = {}
        for label, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
            rows = []
            for i in range(N_PREFILL):
                body, wall = chat(prompt, 1)
                t = body["usage"]["timings"]
                ptoks = t["prefilled_tokens"]
                rows.append({
                    "i": i,
                    "prompt_tokens": body["usage"]["prompt_tokens"],
                    "prefilled_tokens": ptoks,
                    "prefill_ms": t["prefill_ms"],
                    "prefill_tok_s": round(ptoks / (t["prefill_ms"] / 1000.0), 1)
                                     if t["prefill_ms"] > 0 else None,
                    "cache_hit": t["cache_hit"],
                    "cached_prefix_tokens": t["cached_prefix_tokens"],
                })
                time.sleep(0.3)
            pf[label] = rows
        rec["prefill"] = pf
        _, seen = read_spec_lines(LOG, seen)
        rec["smi_post_prefill"] = smi()

        # --- end-to-end: in-tree bench_he_http.py at its defaults ---
        bench = subprocess.run(
            [os.path.expanduser("~/bench-venv/bin/python"),
             f"{ROOT}/server/scripts/bench_he_http.py",
             f"http://127.0.0.1:{PORT}/v1/chat/completions", "96"],
            capture_output=True, text=True, timeout=2400)
        rec["e2e_stdout"] = bench.stdout
        rec["e2e_stderr"] = bench.stderr[-2000:]
        m = re.search(r"\[bench\] avg tok/s = ([\d.]+)", bench.stdout)
        rec["e2e_avg_tok_s"] = float(m.group(1)) if m else None
        m2 = re.search(r"\[bench\] total tokens=(\d+)\s+total dt=([\d.]+)s", bench.stdout)
        if m2:
            rec["e2e_total_tokens"] = int(m2.group(1))
            rec["e2e_total_dt_s"] = float(m2.group(2))
        fresh, seen = read_spec_lines(LOG, seen)
        rec["e2e_spec_lines"] = fresh
        rec["smi_post_e2e"] = smi()
        rec["pids_post"] = other_gpu_procs()
    finally:
        pass
    return finish(rec, proc)


CENSUS_RE = re.compile(r"^CENSUS\s+(>?\s*\d+)\s+(\d+)\s+(\d+)\s*$", re.M)
CENSUS_HDR = re.compile(
    r"=== GQH ne11 census: (\d+) GQH mul_mat calls, (\d+) past the matvec ===")


def read_census(path):
    """Parse the atexit-flushed ne11 histogram. Returns None if it never flushed,
    which is itself a result worth seeing rather than a silent zero."""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    hdr = CENSUS_HDR.search(text)
    if not hdr:
        if "no GQH mul_mat calls" in text:
            return {"total_calls": 0, "total_past_matvec": 0, "rows": []}
        return None
    rows = []
    for m in CENSUS_RE.finditer(text):
        rows.append({"ne11": m.group(1).replace(" ", ""),
                     "calls": int(m.group(2)),
                     "past_matvec": int(m.group(3))})
    return {"total_calls": int(hdr.group(1)),
            "total_past_matvec": int(hdr.group(2)),
            "rows": rows}


def shutdown(proc, graceful):
    """Stop OUR server only, by pid/pgroup -- never by name, this box is shared.

    Census mode needs the atexit handler to run, so it asks politely first and
    only escalates if the process will not leave."""
    if graceful:
        for sig in (2, 15):        # SIGINT then SIGTERM
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except Exception:
                return
            for _ in range(60):
                if proc.poll() is not None:
                    return
                time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        pass


def finish(rec, proc):
    shutdown(proc, CENSUS)
    proc.wait()
    time.sleep(5)
    if CENSUS:
        rec["ne11_census"] = read_census(LOG)
    rec["smi_after_kill"] = smi()
    rec["gpu_clean_after"] = idle_vram_clean(rec["smi_after_kill"])
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = f"{OUT_DIR}/timing_{TAG}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"WROTE {path}")
    if "error" in rec:
        print("ERROR:", rec["error"])
        return 1
    print(f"  decode tok/s      : {[d['api_decode_tok_s'] for d in rec['decode']]}")
    print(f"  accept (spec)     : {[d['spec']['accept_pct'] if d['spec'] else None for d in rec['decode']]}")
    print(f"  avg_commit        : {[d['spec']['avg_commit'] if d['spec'] else None for d in rec['decode']]}")
    for lbl in ("short", "long"):
        rows = rec["prefill"][lbl]
        print(f"  prefill {lbl:5s} n={rows[0]['prefilled_tokens']}: "
              f"{[r['prefill_tok_s'] for r in rows]}")
    print(f"  e2e avg tok/s     : {rec.get('e2e_avg_tok_s')}")
    cen = rec.get("ne11_census")
    if CENSUS:
        if cen is None:
            print("  ne11 census       : NOT FLUSHED (no clean exit)")
        else:
            print(f"  ne11 census       : {cen['total_calls']} calls, "
                  f"{cen['total_past_matvec']} past matvec")
            for r in cen["rows"]:
                print(f"    ne11={r['ne11']:>6s} calls={r['calls']:<10d} "
                      f"past_matvec={r['past_matvec']}")
    print(f"  edge C pre/post   : {rec['smi_pre']['edge_c']} -> {rec['smi_post_e2e']['edge_c']}")
    print(f"  vram B pre/after  : {rec['smi_pre'].get('vram_used_b')} -> "
          f"{rec['smi_after_kill'].get('vram_used_b')} "
          f"(clean_pre={rec.get('gpu_clean_pre')} clean_after={rec.get('gpu_clean_after')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
