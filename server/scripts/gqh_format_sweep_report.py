#!/usr/bin/env python3
"""Collate the format-vs-format sweep into one table, spreads included.

Every figure here is reduced from the per-reading JSON written by timing_arm.py
and the per-item HumanEval+ samples/scores written by quality_humaneval_plus.py.
Nothing is carried in from another box, another drafter, or another day.
"""
import glob, hashlib, json, os, sys

D = os.path.expanduser("~/bench-out")
ARMS = ["gqh", "iq4xs"]
LABEL = {"gqh": "GQH (qwen38-gqh-shaped)", "iq4xs": "IQ4_XS (vendor quant)"}


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"min": min(vals), "max": max(vals), "n": len(vals),
            "mean": sum(vals) / len(vals), "range": max(vals) - min(vals)}


def fmt(s, prec=1):
    if s is None:
        return "n/a"
    if s["range"] == 0:
        return f"{s['min']:.{prec}f} (both readings identical)"
    return f"{s['min']:.{prec}f}–{s['max']:.{prec}f}"


out = {"timing": {}, "quality": {}}

# ---------------- timing ----------------
for arm in ARMS:
    per_reading = {}
    for path in sorted(glob.glob(f"{D}/timing_{arm}_r*.json")):
        rec = json.load(open(path))
        rd = rec["reading"]
        if "error" in rec:
            per_reading[rd] = {"error": rec["error"]}
            continue
        dec = rec["decode"]
        per_reading[rd] = {
            "target_bytes": rec["target_bytes"],
            # within-reading: 5 requests, each its own [spec-decode] line
            "decode_tok_s_within": spread([d["api_decode_tok_s"] for d in dec]),
            "accept_pct_within": spread([d["spec"]["accept_pct"] for d in dec if d["spec"]]),
            "avg_commit_within": spread([d["spec"]["avg_commit"] for d in dec if d["spec"]]),
            "spec_decode_ran": all(d["spec_decode_ran"] for d in dec),
            "any_cache_hit": any(d["cache_hit"] for d in dec)
                             or any(r["cache_hit"] for lbl in rec["prefill"]
                                    for r in rec["prefill"][lbl]),
            "prefill_short_tok_s_within": spread([r["prefill_tok_s"] for r in rec["prefill"]["short"]]),
            "prefill_long_tok_s_within": spread([r["prefill_tok_s"] for r in rec["prefill"]["long"]]),
            "prefill_short_tokens": rec["prefill"]["short"][0]["prefilled_tokens"],
            "prefill_long_tokens": rec["prefill"]["long"][0]["prefilled_tokens"],
            "e2e_avg_tok_s": rec["e2e_avg_tok_s"],
            "e2e_total_tokens": rec.get("e2e_total_tokens"),
            "edge_c_pre": rec["smi_pre"]["edge_c"],
            "edge_c_post": rec["smi_post_e2e"]["edge_c"],
            "load_s": rec["load_s"],
        }
    # across the two readings: this is the order-balanced spread a reader should trust
    good = [v for v in per_reading.values() if "error" not in v]
    out["timing"][arm] = {
        "per_reading": per_reading,
        "across_readings": {
            "decode_tok_s": spread([v["decode_tok_s_within"]["mean"] for v in good]),
            "accept_pct": spread([v["accept_pct_within"]["mean"] for v in good]),
            "avg_commit": spread([v["avg_commit_within"]["mean"] for v in good]),
            "prefill_short_tok_s": spread([v["prefill_short_tok_s_within"]["mean"] for v in good]),
            "prefill_long_tok_s": spread([v["prefill_long_tok_s_within"]["mean"] for v in good]),
            "e2e_avg_tok_s": spread([v["e2e_avg_tok_s"] for v in good]),
        },
    }

# ---------------- quality ----------------
def load_samples(path):
    rows = {}
    for line in open(path):
        if line.strip():
            s = json.loads(line)
            rows[s["task_id"]] = s
    return rows


def load_scores(path):
    d = json.load(open(path))
    return {r["task_id"]: bool(r["ok"]) for r in d["rows"]}, d["passed"], d["total"]


for arm in ARMS:
    entry = {}
    s1p = f"{D}/hep_samples_{arm}_r1.jsonl"
    s2p = f"{D}/hep_samples_{arm}_r2.jsonl"
    c1p = f"{D}/hep_scores_{arm}_r1.json"
    c2p = f"{D}/hep_scores_{arm}_r2.json"
    if not all(os.path.exists(p) for p in (s1p, s2p, c1p, c2p)):
        out["quality"][arm] = {"error": "missing quality artifacts"}
        continue
    s1, s2 = load_samples(s1p), load_samples(s2p)
    v1, p1, t1 = load_scores(c1p)
    v2, p2, t2 = load_scores(c2p)

    # DETERMINISM: are the two same-arm readings byte-identical, reply by reply?
    ids = sorted(set(s1) | set(s2))
    identical = sum(1 for i in ids
                    if i in s1 and i in s2 and s1[i]["raw_reply"] == s2[i]["raw_reply"])
    differing = [i for i in ids
                 if not (i in s1 and i in s2 and s1[i]["raw_reply"] == s2[i]["raw_reply"])]

    # same-arm quality floor, as an ITEM COUNT, not a percentage
    flipped = sorted(i for i in ids if i in v1 and i in v2 and v1[i] != v2[i])
    fail1 = {i for i in v1 if not v1[i]}
    fail2 = {i for i in v2 if not v2[i]}
    entry = {
        "pass_r1": p1, "total_r1": t1, "rate_r1": p1 / t1 if t1 else None,
        "pass_r2": p2, "total_r2": t2, "rate_r2": p2 / t2 if t2 else None,
        "replies_identical": identical, "replies_compared": len(ids),
        "replies_differing_ids": differing[:20],
        "verdicts_flipped_count": len(flipped),
        "verdicts_flipped_ids": flipped,
        "fail_set_r1_n": len(fail1), "fail_set_r2_n": len(fail2),
        "fail_set_intersection_n": len(fail1 & fail2),
        "fail_set_symmetric_diff_n": len(fail1 ^ fail2),
        "fail_set_identical": fail1 == fail2,
        "request_errors_r1": sum(1 for s in s1.values() if s.get("err")),
        "request_errors_r2": sum(1 for s in s2.values() if s.get("err")),
        "samples_sha256_r1": hashlib.sha256(open(s1p, "rb").read()).hexdigest()[:16],
        "samples_sha256_r2": hashlib.sha256(open(s2p, "rb").read()).hexdigest()[:16],
    }
    out["quality"][arm] = entry

# cross-arm: do the two formats fail on the same items?
if all("error" not in out["quality"].get(a, {"error": 1}) for a in ARMS):
    g, _, _ = load_scores(f"{D}/hep_scores_gqh_r1.json")
    q, _, _ = load_scores(f"{D}/hep_scores_iq4xs_r1.json")
    gf = {i for i in g if not g[i]}
    qf = {i for i in q if not q[i]}
    out["cross_arm"] = {
        "gqh_fail_n": len(gf), "iq4xs_fail_n": len(qf),
        "fail_both_n": len(gf & qf),
        "fail_gqh_only": sorted(gf - qf),
        "fail_iq4xs_only": sorted(qf - gf),
    }

json.dump(out, open(f"{D}/summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))
