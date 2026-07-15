# Candidate-restricted LM head (top-k logits) — design, calibration, status

**Branch:** `feat/topk-head-calib`  ·  **Worktree:** `/workspace/lucebox-topk`
**Status (2026-06-18):** premise validated on a smoke sample; instrumentation extended to
sampling; full calibration run pending. No production kernel implemented yet.

---

## 1. Motivation / the idea

The target LM head is the matmul `logits = output.weight @ hidden` over the **full vocab**.
For Qwen3.6-27B the head is `output.weight = [n_embd=5120 × n_vocab=248320]`, quantized **Q6_K
≈ 1.0 GB**, i.e. **~6% of the 16.8 GB** of weights streamed per target forward. The 248k vocab
is unusually large, which makes the head a bigger slice than on a typical 32k-vocab model.

**Idea:** instead of computing logits over all 248k vocab entries, compute them only over a
small **candidate shortlist** `S` of vocab indices, then argmax/sample within `S`. In ggml this
is clean because vocab is the *row* dimension (`ne1`) of `output.weight`: a restricted head is
`ggml_get_rows(output.weight, candidate_ids)` (gathers k quantized rows) followed by a small
matmul. Restricting to k≈256–1024 shrinks the head matmul to <0.5% of its size.

**Where the shortlist comes from:** the DFlash draft already predicts a per-position
distribution (its hidden states projected through the target head, `project_hidden_to_topk`).
Its **top-M** tokens are the natural candidate set `S`. Calibration decides M.

---

## 2. Greedy formulation + the correctness caveat

Greedy spec-decode today is **exact / lossless**: it only commits a draft token when it equals
the target's true full-vocab argmax, and the rejection "bonus" is the true argmax. Output ==
pure target-greedy decoding.

A candidate-restricted head makes argmax run over `S` only. This stays exact **iff the true
argmax ∈ S**. Calibration ("greedy token always in draft top-M") bounds the failure rate, but
"always on a calibration set" ≠ "always at inference" — so this becomes **approximate greedy**,
not lossless. Must validate output quality, not just speed. (You cannot cheaply *verify*
exactness: confirming no out-of-`S` token beats the in-`S` max needs the full matmul.)

### Realistic ceiling
- Head ≈ 6% of weight traffic; restricting it ≈ removes that 6% on the **verify** side.
- But the shortlist itself is produced by `project_hidden_to_topk`, which already runs one full
  target-head matmul over the draft block. So you eliminate the *verify-side* head only.
- **Net realistic decode speedup ≈ ~5%** here (more than a typical model because vocab=248k).
  Modest but real. Worth it only if calibration shows small M suffices.

---

## 3. Extension to sampling (top-k / top-p) — and why it can be *exact*

Generalize the condition: `S` must contain the verifier's **post-filter support**. The sampler
chain order (`server/src/common/sampler.h:6`) is:

```
rep_penalty → freq/pres → top_k → softmax(temp) → top_p → draw
```

The post-filter support is the target's **top-K_s** tokens (then the top-p nucleus within them).
So calibrate M such that **draft-top-M ⊇ target-top-K_s**. Greedy is the K_s=1 case.

**Key insight — normalization:** `top_k` is applied *before* `softmax`. So if `top_k=K_s` is
finite, `softmax` normalizes over just those K_s kept tokens **in both the full and restricted
pipelines** — the global denominator never enters. Therefore, as long as `S ⊇ target-top-K_s`:
- restricted-head `top_k` keeps exactly the true top-K_s (they're the K_s globally-highest and
  all in `S`), `softmax`/`top_p`/draw are identical ⇒ **sampling is EXACT (lossless)**.
- This covers Qwen3.6 defaults (`top_k=20, top_p=0.95`): just need `S ⊇ target top-20`.

**Exceptions:**
- `top_k = 0` (pure top-p): `softmax` normalizes over the kept set; the missing tail inflates
  probabilities and shrinks the nucleus ⇒ **approximate**. Over-cover (`S` carries ≥0.999 mass)
  or correct the threshold with an estimate of the uncovered tail mass.
- pure temperature (no truncation): every token has nonzero prob ⇒ can't shortlist; needs full
  head. (Fine — the optimization only pays off when truncation already discards the tail.)
- penalties run before `top_k`, so strictly the condition is on the *penalized* top-K_s; raw-logit
  draft-top-M is a good superset proxy.

---

## 4. Calibration methodology (the make-or-break test, no kernel needed)

Measure whether the premise holds before building any kernel. Per chain-verify position
(exact alignment: `target_tok[i]` ↔ draft row `i+1`):
- **greedy:** rank of the target argmax within the draft's ranked candidate list. `coverage@M`
  = fraction of positions with rank < M.
- **sampling:** the MAX draft-rank over the target's top-K_s set / top-p nucleus → the smallest
  M with `draft-top-M ⊇ target-top-K_s`. `coverage@M` = fraction fully covered.

**Decision rule:** if `coverage@M` reaches ~99.9% at small M (≈64–256 greedy, somewhat larger
for top_k=20), build the restricted head. If it only saturates at large M, the speedup
evaporates — report and stop.

---

## 5. What's implemented (this branch)

Instrumentation only — gated by env `DFLASH_TOPK_CALIB=1`, in the **chain** verify path of
`server/test/test_dflash.cpp` (run WITHOUT `--ddtree`; chain gives exact position alignment):
1. accumulators declared before the decode loop (`calib_hist`, `samp_hist_k8/k20/p95`);
2. transfer the draft's full per-position logits (`draft_sg.logits → draft_logits_buf`) — the
   fast path otherwise only reads GPU argmax;
3. transfer the target's full logits (`sg.logits → verify_logits_buf`) in the batched path;
4. per-position rank histograms (greedy argmax rank; max draft-rank over target top-8 / top-20 /
   top-p=0.95 nucleus at temp 1.0);
5. print `coverage@M` curves at end of run.

Build: `cmake -B server/build -S server -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_NCCL=OFF` then `cmake --build server/build --target
test_dflash -j`. (NCCL OFF is required — the fork's `ncclCommInitAll` aborts on this box's driver.)

Run: `calib/run_calib.sh` (uses `GPU=${GPU:-1}` — GPU 0 was occupied by another session;
6 tokenized prompts in `calib/*.bin`, generated with the Qwen3.6 tokenizer, thinking off).

### Results so far (smoke: 1 prompt, 210 positions, greedy metric)
```
coverage@k=1   : 74.3%   (draft top-1 == target greedy ≈ per-token accept rate)
coverage@k=64  : 97.6%
coverage@k=256 : 99.5%
coverage@k=512 : 100%     (mean rank 4.9, max rank 354)
```
⇒ Greedy premise holds strongly: target greedy token within draft top-512 (~0.14% of vocab) on
this sample; a safe M≈1024 is likely lossless and still makes the head matmul ~free. Sampling
coverage numbers pending the full multi-prompt run with the extended binary.

---

## 5b. Full calibration results (2026-06-18)

5 prompts (code1/code2/gen1/gen2/math1; math2 dropped on a transient GPU OOM),
**2550 positions**, n_gen=256, chain path, Qwen3.6-27B Q4_K_M target + Q4_K_M 3.6 draft,
position-weighted `coverage@M` (= fraction of positions whose set is fully inside draft top-M):

| coverage@M | greedy (argmax) | top_k=8 | top_k=20 (Qwen def) | top_p=0.95 |
|-----------:|----------------:|--------:|--------------------:|-----------:|
| 128        | 93.7%           | 29.8%   | 3.3%                | 62.5%      |
| 512        | 96.5%           | 57.6%   | 19.1%               | 68.5%      |
| 1024       | 97.5%           | 69.3%   | 35.0%               | 71.8%      |
| 4096       | 99.4%           | 87.0%   | **64.3%**           | 78.4%      |

Per-prompt greedy@M=128 ranged 80%–99% (code/math high, "general" prompts lower);
tails are long (max greedy rank up to ~133k on a few low-confidence positions).

**Verdict.**
- **Greedy: viable.** M≈1024 → ~97.5%, M≈128 → ~94%. A restricted head with M≈1k–2k is
  near-lossless and still <1% of the 248k-wide matmul. Build it.
- **Sampling (top_k / top_p): NOT viable with this draft.** Exact `top_k` needs the draft
  shortlist to contain the target's *entire* top-K_s; even M=4096 covers the full top-20 only
  64% of the time. The draft matches the target's **#1** token well but ranks the rest of the
  nucleus very differently (the target's #15 can sit at draft-rank thousands). Covering the
  nucleus needs an impractically large M, erasing the speedup.
- **Caveat:** this is the weak Q4 3.6 draft (same one that underperformed the benchmark). A
  BF16/stronger draft would likely rank the nucleus more faithfully and could move the sampling
  numbers; re-run the calibration with it before concluding. The structural asymmetry
  (argmax easy, full-nucleus hard) will persist to some degree.

## 5c. Large-M is cheap; MASS-coverage reopens sampling (2026-06-18)

The head scales linearly in M and is only ~6% of step traffic, so large M stays cheap:
M=8192 → head ~0.2% of step (~5.9% speedup), M=32768 → ~0.8% (~5.2%), M=65536 → ~1.6% (~4.4%).
So we can afford M far larger than first assumed.

More importantly, **set-coverage is the wrong metric for sampling quality** — missing a low-prob
nucleus token barely distorts the draw. The right metric is **covered probability MASS**. The
instrumentation now reports, for the top_p=0.95 nucleus, the mean fraction of nucleus *mass*
inside draft-top-M, plus the count of positions with <99% mass covered. Extended M grid to 65536.

Smoke (1 prompt, math1, 225 positions) — mass coverage:
```
mass@M=256   : 98.93%   (32/225 positions <99% covered)
mass@M=4096  : 99.60%   ( 3/225 positions <99%)
mass@M=65536 : 99.9993% ( 0/225 positions <99%)
```
⇒ At M≈4096–8192 (still ~5.6–5.9% speedup) the per-position sampling distortion is <0.5% of
mass; M=65536 is lossless-in-practice. **Sampling is likely viable as an approximate-but-faithful
mode after all** — pending the large-prompt run to confirm the worst-case tail. (Set-coverage of
the full top-20 also reaches ~100% by M=65536 on this prompt.)

### Tooling for scale-up
- `calib/prep_prompts.py --n N` — tokenize N prompts from HumanEval/GSM8K/MATH-500 (Qwen3.6 chat
  template) into `calib/*.bin`. Deps: transformers, datasets, jinja2.
- `calib/run_calib.sh` — `GPU=0 NGEN=256 bash calib/run_calib.sh | tee calib/out/calib.log`
  (chain path, no `--ddtree`; one table per prompt).
- `calib/aggregate.py calib/out/calib.log` — position-weighted aggregate (set + mass coverage).
- CPU note: each position does 2× nth_element+sort over the 248k vocab (host-side), so very large
  prompt counts are CPU-bound regardless of GPU; parallelize across prompts if needed.

### B200 build (Blackwell sm_100, CUDA ≥12.8)
```
git fetch && git checkout feat/topk-head-calib
git submodule update --init --recursive
cmake -B server/build -S server -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=100 -DGGML_CUDA_NCCL=OFF
cmake --build server/build --target test_dflash -j
# place target + draft GGUFs under server/models/ and server/models/draft/
python3 calib/prep_prompts.py --n 300
GPU=0 NGEN=256 bash calib/run_calib.sh | tee calib/out/calib.log
python3 calib/aggregate.py calib/out/calib.log
```

## 5d. Entropy-gated restricted head — calibrating a per-position M (2026-07-14)

§5b/§5c calibrate a single fixed M for *every* position. The follow-on idea: gate on the
draft's own per-position prediction confidence instead of always using one M. Coverage@M is not
uniform across positions — §5b already shows this indirectly ("code/math high, general prompts
lower"); the natural per-position signal for "how uncertain is the draft here" is its **entropy**.

**Direction (confirmed empirically, not just intuited):** low draft entropy (confident) →
high coverage → safe to use the restricted head; high draft entropy (uncertain) → lower
coverage → fall back to the full head. This is the *opposite* of "restrict harder when unsure,"
which would compound the risk exactly where the shortlist is least trustworthy.

**Entropy definition:** temperature-agnostic — computed on raw draft logits as if `T=1`,
independent of whatever `ddtree_temp` is configured for sampling. This keeps the gate a property
of the draft's *intrinsic* confidence rather than an artifact of a sampling knob, and means a
threshold calibrated once doesn't silently drift if `ddtree_temp` changes later.

**What's implemented** (instrumentation only, same `DFLASH_TOPK_CALIB=1` chain path as §5,
`server/test/test_dflash.cpp`):
1. Per-position entropy over draft row `i+1` (the same row supplying the candidate set for
   verify position `i` — the `pp → min(pp+1, q_len-1)` alignment from §5/§7), computed via a raw
   max + `Σ exp(l−max)` + `Σ l·exp(l−max)` pass, no temperature scaling.
2. The existing idealized rank histogram (`calib_hist`) is now also split by a fixed entropy
   bucket (`calib_hist_eb`, edges `{0, 0.5, 1, ..., 12}` nats — bounded by `ln(vocab)≈12.4` for
   this model's 248k vocab) → printed as `[topk-calib-eb]` per-bucket `coverage@M` tables.
3. A **second, non-idealized check**: for a small M grid (`{128,256,512,1024,2048}`), the
   *actual* `extract_topm_cuda` + `restricted_lm_head_q6k` kernels run and their output is
   compared token-for-token against the exact full-head argmax, bucketed the same way → printed
   as `[topk-calib-head]`. This measures the extractor's real approximation error (bin-boundary
   ties aren't rank-ordered — see `topm_extract_cuda.cu`), which the idealized CPU-sorted rank
   histogram in (2) can't see. Needed a small plumbing fix: `build_qwen35_graph` already always
   computes+retains `hidden_states` (`server/src/internal.h:616`), but `graph_builders.cpp`'s
   non-restricted branch wasn't copying it into `StepGraph::hidden_states` — one-line fix lets
   calibration validate the real kernel off the *same* full-head verify build/compute, no second
   graph build needed.
4. `calib/calibrate.py` parses and position-weights both new tables (per-bucket weighting uses
   each bucket's own position count, not the prompt total, since bucket occupancy varies a lot
   prompt to prompt) and prints them alongside the existing report; `--json` picks them up for
   free via the existing `aggregate()` return dict.

**Smoke-tested** first on a synthetic (non-language) token sequence to validate the plumbing
(no NaNs/crashes, bucket counts summed correctly, real-kernel match% tracked idealized
coverage% closely), then run for real (below).

### Real calibration results (2026-07-14)

30 real prompts (10 each HumanEval / GSM8K / MATH-500, Qwen3.6 chat template, thinking off),
**15,360 positions**, `n_gen=256`, chain path, Qwen3.6-27B Q4_K_M target + Q4_K_M 3.6 draft.
`mean_draft_entropy = 1.758` nats overall — this draft/target pair is confident most of the
time; the long tail (entropy > 5) is thin (only 217/15360 positions, 1.4%).

Direction confirmed with real data, not just intuition: real kernel match@M=128 falls
monotonically as entropy rises, from **99.64%** in `[0.00,0.50)` down to **96.66%** in
`[4.00,5.00)` — low draft entropy really does predict high restricted-head fidelity.

**The frontier that answers "pick (T, M)"** — cumulative over all positions with entropy below
T, real kernel `match@M` (i.e. restricted-head output == exact full-head argmax; NOT the
idealized coverage upper bound):

| T (entropy <) | % of positions | match@128 | match@256 | match@512 | match@1024 | match@2048 |
|--------------:|----------------:|----------:|----------:|----------:|-----------:|-----------:|
| 0.5           | 30.7%           | 99.64%    | 99.70%    | 99.75%    | 99.81%     | 99.85%     |
| 1.0           | 40.9%           | 99.36%    | 99.48%    | 99.59%    | 99.68%     | 99.73%     |
| 2.0           | 58.0%           | 99.07%    | 99.34%    | 99.45%    | 99.56%     | 99.66%     |
| 3.0           | 74.7%           | 98.95%    | 99.25%    | 99.36%    | 99.49%     | 99.62%     |
| **4.0**       | **90.4%**       | 98.67%    | 99.06%    | 99.28%    | **99.42%** | 99.55%     |
| 5.0           | 98.6%           | 98.50%    | 98.98%    | 99.23%    | 99.37%     | 99.51%     |
| ∞ (no gate)   | 100%            | 98.47%    | 98.98%    | 99.23%    | 99.38%     | 99.52%     |

**Recommended starting operating point: T≈4.0, M=1024.** Routes 90.4% of positions to the
cheap restricted head at a 99.42% real match rate against exact full-head greedy (the
remaining 9.6%, entropy≥4, fall back to the full head — always exact there). That's a ~0.6%
per-position miss rate concentrated exactly where we chose to still pay for the cheap path —
tightening to T≈2.0 nearly doubles the miss rate's *rarity* (58% coverage, 99.56% match) if a
lower error budget is wanted, or loosening to T≈5.0 barely changes match rate (98.6% coverage
already captures nearly everything — the entropy>5 tail is too rare to matter much either way
in this sample).

**Caveat also worth restating from §5b:** this is one draft/target pair (Q4 3.6 draft) and one
30-prompt sample skewed toward code/math (thinking off) — re-run before committing to numbers
for a different draft, a different prompt mix, or natural-language-heavy workloads, which §5b
showed have measurably worse coverage than code/math.

Raw run: `calib/out/calib.json` (not checked in — `calib/out/` is gitignored; regenerate with
`python3 calib/calibrate.py --prep 30 --gpu 0 --ngen 256 --json calib/out/calib.json`).

## 5e. Natural-language-heavy calibration + combined results (2026-07-14)

The §5d caveat above ("re-run for natural-language-heavy workloads") turned into an actual run.
Added a 4th dataset to `calib/prep_prompts.py` — `tatsu-lab/alpaca` (general instructions,
labeled `nl`), extracted as `instruction + "\n" + input` — specifically because it's *not*
code/math, to isolate how much the §5b/§5d numbers were inflated by that skew. 10 prompts,
reusing the 30 cached code/math logs via `calib/calibrate.py --skip-existing` (only the new `nl`
prompts actually ran) to get a **40-prompt combined** aggregate cheaply.

| | prompts | positions | mean_draft_entropy | greedy coverage@128 |
|---|---:|---:|---:|---:|
| Code/Math only (§5d) | 30 | 15,360 | 1.758 | 98.60% |
| **NL only (Alpaca)** | 10 | 11,445 | **3.979** | **93.11%** |
| Combined | 40 | 26,805 | 2.707 | 96.26% |

Confirms the hypothesis with numbers, not just the "general prompts lower" hand-wave from §5b:
NL prompts run at more than **2× the mean entropy** of code/math and noticeably worse coverage.
The entropy gate is exactly the right lever for this — it's already conditioning on the signal
that separates these two regimes, so a single calibrated `(T, M)` should generalize across a
mixed workload better than a single fixed M would.

**Combined (40-prompt) frontier** — same cumulative-real-match methodology as §5d, now over a
workload that isn't skewed to code/math:

| T (entropy <) | % of positions | match@128 | match@1024 |
|--------------:|----------------:|----------:|-----------:|
| 1.0           | 28.6%           | 99.48%    | 99.74%     |
| 2.0           | 41.6%           | 99.18%    | 99.61%     |
| 3.0           | 55.2%           | 99.03%    | 99.52%     |
| **4.0**       | **69.5%**       | 98.56%    | **99.39%** |
| 5.0           | 82.9%           | 97.66%    | 99.16%     |
| 6.0           | 93.9%           | 96.79%    | 98.94%     |

**Revised recommendation for a mixed/unknown workload: T≈4.0, M=1024** still gives ≈99.4%
match (about the same quality bar as the code/math-only pick), but now only **69.5%** of
positions clear the gate — down from 90.4% when the sample was code/math-skewed. That's the
real cost of not knowing your workload in advance: roughly a third of positions now pay for the
full head instead of a tenth. NL alone is worse still — at the same T=4.0/M=1024 point, NL-only
clears the gate on just 41.4% of positions (vs. code/math's 90.4%), confirming the two workloads
need to be calibrated (or at least sanity-checked) separately rather than assuming one transfers
to the other.

**Still a one-draft, modest-sample-size result** — 40 prompts, one instruction dataset as the
NL proxy (Alpaca skews short/simple; longer conversational or reasoning-heavy NL prompts may
differ). Good enough to size the effect, not to freeze a production threshold.

Raw runs: `calib/out/calib_combined.json` (all 40), `calib/out/calib_nl_only.json`,
`calib/out/calib_codemath_only.json` (none checked in, `calib/out/` gitignored). Regenerate with
`python3 calib/prep_prompts.py --n 40 --out calib --thinking off` then
`python3 calib/calibrate.py --gpu 0 --ngen 256 --skip-existing --json calib/out/calib_combined.json`.

## 5f. Runtime entropy gate + e2e validation (2026-07-14)

Wired the calibrated gate into the DDTree runtime restricted-head path (the tree-mode
`tree_rhead`/chain-mode `rhead` toggles already implemented pre-existing on this branch, gated
by `DFLASH_TOPK_HEAD=M`, previously *ungated* — always on whenever `M>0`). New:
`server/src/common/draft_entropy_cuda.{h,cu}`, a standalone kernel computing per-row
temperature-agnostic entropy directly from `draft_sg.logits` (device-resident already; the
kernel does one more full-vocab-wide device pass but no new full-vocab **host** copy, so it
doesn't reintroduce the D2H the fast decode path is built to avoid — only an `n_positions`-sized
result crosses to host). Gated via new env var `DFLASH_TOPK_HEAD_ENTROPY_T` (unset = old
ungated behavior, fully backward compatible); aggregated per verify step via **MAX** entropy
over that step's draft rows, not mean — a whole-graph decision should be as conservative as its
least-confident row.

**De-risking first (zero new code):** before writing the gate, ran the *existing* ungated
fixed-M=1024 tree head against the exact lossless baseline on 4 real prompts (one per
code/gsm/math/nl label). 3/4 were byte-identical; the NL prompt diverged **74.6%** of tokens,
cascading from a single wrong pick at step 9 — one greedy miss early on sends the entire
autoregressive continuation down a different (locally plausible, but different) path. This
concretely justified building the gate rather than assuming it mattered.

**Gate validation on the same 4 prompts** (`DFLASH_TOPK_HEAD_ENTROPY_T=4.0`): the NL prompt
now routes **0/74 steps** to the restricted head (100% fallback) and is **byte-identical** to
the lossless baseline; the code/math prompts still route 39–97% of steps to the cheap path and
stay byte-identical too. The gate does exactly what it's calibrated to do at the single-prompt
level.

**e2e via `server/scripts/bench_llm.py`** (10 samples each, HumanEval/GSM8K/Math500, baseline
head-off vs. gated `T=4.0/M=1024`, identical prompts both runs):

| Bench | speedup (off → gated) | score (off → gated) |
|---|---|---|
| HumanEval | 2.78x → 2.86x | (unscored) |
| GSM8K | 2.86x → 3.02x | 5/10 → **4/10** |
| Math500 | 3.24x → 3.36x | 1/10 → 1/10 |

Speedup gain is small (+2.8–5.4%), matching §2's ~5% realistic ceiling. **Quality is not fully
preserved at this operating point**: one GSM8K sample (#8) flipped from correct to incorrect
under the gate — a real divergence got through despite T=4.0, consistent with calibration's own
~99.4% (not 100%) per-step match rate there compounding over dozens of gated decisions per
generation. The gate is a large improvement over the *ungated* fixed-M head (which produced
74%-token-divergent output on the NL case) but is still genuinely approximate, not lossless, at
this threshold.

**Verdict: not ready to ship as-is.** The speedup is modest and the quality cost, while small,
is real and measured (not hypothetical). Before adopting: either (a) tighten `T` toward the
§5e frontier's more conservative end (T≈1–2, trading routed-cheap fraction for lower risk), or
(b) build the verify-and-escalate design from the original brainstorm (goal "b": run the
restricted head, detect likely misses — e.g. within-shortlist margin, or disagreement with the
draft's own top-1 — and recompute with the full head only for flagged steps) instead of
accepting a fixed statistical error rate. (a) is a config change with the current code; (b)
needs new plumbing and wasn't built this session.

## 6. Next steps
1. **[done]** Full calibration — see §5b. Greedy validated; sampling needs a better draft.
2. Implement the restricted head for the **greedy** path in `build_qwen35_graph` / the verify graph
   (`server/src/qwen35/graph_builders.cpp`, LM head at ~line 360): plumb candidate ids
   (draft top-M) → `ggml_get_rows(output.weight, ids)` → small matmul → argmax/sample over the
   k logits → map index back to vocab id. Reuse the draft's `project_hidden_to_topk` output as
   the candidate set (it's already computed).
3. Validate quality vs full-head greedy (token-exact match rate) and measure real decode speedup.
4. Sampling is **on hold** pending a better draft (§5b): the Q4 3.6 draft can't cover the
   target's top-k/top-p nucleus at a useful M. Before revisiting, re-run §5b's calibration with
   a BF16/stronger draft. If coverage improves: keep `top_k` finite for exactness (chain order
   makes it lossless when `draft-top-M ⊇ target-top-K_s`), and gate pure-top_p / pure-temperature
   to the full head (or over-cover with a tail-mass correction).
5. **[done, not shipped] Entropy-gated M — runtime gate wired + e2e-tested, see §5f.**
   `DFLASH_TOPK_HEAD_ENTROPY_T` on the existing tree/chain restricted-head toggle, per-verify-step
   (whole-graph) granularity, MAX-entropy aggregate. e2e via `bench_llm.py` at T=4.0/M=1024: +2.8
   to +5.4% speedup, but GSM8K score dropped 5/10→4/10 (one real divergence) — approximate, not
   lossless, as expected, but the speedup/risk trade isn't clearly worth it yet at this
   threshold. Next: either tighten T (§5e's frontier), or build the verify-and-escalate
   alternative (detect likely misses and recompute with the full head, rather than accepting a
   fixed statistical error rate) — see §5f's verdict.

## 7. Key code references
- LM head matmul + argmax: `server/src/qwen35/graph_builders.cpp:360-365` (chain), `:432` (tree),
  `build_lm_head_projection_step` `:446`.
- Candidate source: `Qwen35DFlashTarget::project_hidden_to_topk` `server/src/qwen35/qwen35_dflash_target.cpp:614`;
  `extract_draft_topk` `server/src/common/ddtree.cpp:12`.
- Greedy chain verify + acceptance: `server/test/test_dflash.cpp` (`!seq_verify` path ~3692, accept ~3800);
  generic loop `server/src/common/dflash_spec_decode.cpp:196-207` (layer-split).
- Sampler chain: `server/src/common/sampler.{h,cpp}` (`needs_logit_processing` `sampler.h:38`,
  `sample_logits`).
- Sampled-verify (server, single-GPU): `server/src/qwen35/qwen35_backend.cpp:1741-1779` (gate),
  `:2033-2051` (sampled acceptance walk).
