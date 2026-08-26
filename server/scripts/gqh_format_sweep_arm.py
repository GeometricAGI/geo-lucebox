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
OUT_DIR  = os.path.expanduser("~/bench-out")
ROOT     = os.path.expanduser("~/lb-upstream")
BIN      = f"{ROOT}/server/build-wide/dflash_server"
DRAFTER  = os.path.expanduser("~/bench-models/Qwen3.8-27B-DFlash2-Q8_0.gguf")
TARGETS  = {
    "gqh":   os.path.expanduser("~/bench-models/qwen38-gqh-shaped.gguf"),
    "iq4xs": os.path.expanduser("~/bench-models/Qwen3.8-27B-IQ4_XS.gguf"),
}
PORT = 18080
TAG  = f"{ARM}_{READING}"
LOG  = f"{OUT_DIR}/timing_{TAG}.log"

# Caches OFF. These three flags are the whole point: without them a second
# reading of the same arm could be answered out of a cache rather than measured.
CACHE_OFF = ["--prefix-cache-slots", "0",
             "--prefill-cache-slots", "0",
             "--prefill-compression", "off"]

N_WARMUP  = 3     # discarded; first requests pay graph build + weight fault-in
N_DECODE  = 5     # decode readings per arm-reading
N_PREFILL = 5     # prefill readings per prompt size
DECODE_MAX_TOKENS = 256


def smi():
    """Edge temp (C) and VRAM% for GPU 0, plus a raw concise dump for auditing."""
    out = subprocess.run(["rocm-smi", "--showtemp", "--showmemuse"],
                         capture_output=True, text=True).stdout
    temp = vram = None
    for line in out.splitlines():
        if "GPU[0]" in line and "Sensor edge" in line:
            temp = float(line.split(":")[-1].strip())
        if "GPU[0]" in line and "VRAM%" in line:
            vram = float(line.split(":")[-1].strip())
    concise = subprocess.run(["rocm-smi"], capture_output=True, text=True).stdout
    return {"edge_c": temp, "vram_pct": vram, "t": time.time(), "concise": concise}


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
    target = TARGETS[ARM]
    rec = {
        "arm": ARM, "reading": READING,
        "target": target, "target_bytes": os.path.getsize(target),
        "drafter": DRAFTER, "drafter_bytes": os.path.getsize(DRAFTER),
        "server_bin": BIN, "server_bin_bytes": os.path.getsize(BIN),
        "cache_flags": CACHE_OFF,
        "git_head": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "env_note": "shipping defaults: GGML_GQH_MMQ unset (on), "
                    "GGML_GQH_MMQ_MAX_NE11 unset (gate=160)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    rec["smi_pre"] = smi()
    rec["pids_pre"] = other_gpu_procs()

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    for k in ("GGML_GQH_MMQ", "GGML_GQH_MMQ_MAX_NE11"):
        env.pop(k, None)          # shipping defaults, not an override
    cmd = [BIN, target, "--draft", DRAFTER, "--port", str(PORT), *CACHE_OFF]
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


def finish(rec, proc):
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        pass
    proc.wait()
    time.sleep(5)
    rec["smi_after_kill"] = smi()
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
    print(f"  edge C pre/post   : {rec['smi_pre']['edge_c']} -> {rec['smi_post_e2e']['edge_c']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
