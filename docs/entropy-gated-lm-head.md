# Entropy-gated restricted LM head

Status: idea / calibration. Off by default. Related:
[`topk-head-optimization.md`](topk-head-optimization.md),
[`topk-head-calibration-results.md`](topk-head-calibration-results.md).

## The problem

The target's greedy verify step computes logits over the **full** vocabulary
(`output.weight` [n_embd × n_vocab] Q6_K @ hidden) just to take an argmax. At
n_vocab ≈ 152k that head reads all ~152k weight rows per verified position and
costs roughly **~5–6% of decode GPU time** (memory-bound; ~37× the row traffic
of a 4096-row restricted head, which alone measures ~0.2%). The full read is
wasteful when we only need the single winning token.

## The idea

The draft model already produces a probability distribution at each position.
Its **top-M** tokens cover the target's true greedy argmax most of the time
(~94% at M=128, ~97.5% at M=1024 — see the calibration results doc). So:

> Instead of the full-vocab head, compute exact target logits **only over the
> draft's top-M candidate tokens**, then argmax within that shortlist.

This is the *candidate-restricted greedy LM head*
(`server/src/common/restricted_lm_head_cuda.cu` +
`topm_extract_cuda.cu`): read M rows of the head instead of ~152k.

It is **lossy**: when the target's true argmax falls outside the draft top-M,
the restricted head returns the wrong token, and DFlash output diverges from
pure greedy-target output.

## Why gate on entropy

Coverage@M is not uniform — it is *worse* exactly where the draft is uncertain.
So don't use one fixed M for every position. Instead gate on the **draft's own
prediction entropy** for that position:

- **Low entropy (confident draft) → restricted head.** Coverage is high here,
  so the cheap path is near-lossless.
- **High entropy (uncertain draft) → full head.** These are the positions the
  restricted head gets wrong, so pay for the full read only here.

The entropy is **temperature-agnostic**: computed on the raw draft logits as if
T=1, independent of the configured sampling temperature.

```
d_max    = max_v  drow[v]
sumexp   = Σ_v    exp(drow[v] - d_max)
weighted = Σ_v    drow[v] * exp(drow[v] - d_max)
entropy  = (d_max + log(sumexp)) - weighted / sumexp     # nats, in [0, ln(vocab)]
```

Alignment note: entropy for verify position `pp` must be read from the draft
row that produced position `pp`, i.e. `pp → min(pp+1, q_len-1)`, not row `pp`.

## Algorithm, step by step

For one DFlash step (draft proposes a set of positions, target verifies them),
with the gate enabled at threshold `T` and shortlist size `M`:

1. **Draft forward.** Run the draft model over the current context. This yields,
   for each draft row, a full-vocab logit vector `drow` (`draft_logits_buf`).

2. **Per-position draft entropy.** For each verify position `pp`, take its
   aligned draft row (`pp → min(pp+1, q_len-1)`) and compute the
   temperature-agnostic entropy from `drow` using the formula above. One scalar
   per position; a single fused pass over the row (`draft_entropy_cuda.cu`).

3. **Extract top-M candidates.** From the same aligned `drow`, extract the M
   highest-logit token ids into a per-position shortlist `cand_ids[pp][0..M)`
   (`extract_topm_cuda` — a coarse-bin threshold select, not a full sort).
   Only needed for positions that will take the restricted path, but it is
   cheap enough to run for all.

4. **Target forward.** Run the target model over the verify positions. The graph
   produces, per position, the hidden vector `h` feeding the LM head
   (`QwenGraphOutputs::hidden_states`). Do **not** yet materialize full-vocab
   logits.

5. **Gate, per position.** Compare the position's draft entropy to `T`:
   - `entropy < T` (confident draft) → **restricted head**: run
     `restricted_lm_head_q6k` over `h` and `cand_ids[pp]` — read only those M
     head rows, dot each with `h`, and argmax within the shortlist. Output token
     = the winning candidate id.
   - `entropy ≥ T` (uncertain draft) → **full head**: the ordinary full-vocab
     `output.weight @ h` followed by argmax over all ~152k tokens.

6. **Verified token.** Either path yields one argmax token per verify position —
   the target's greedy prediction for that position.

7. **Speculative accept/reject.** Feed those verified tokens into the usual
   DFlash accept/reject: walk the drafted sequence, accept while the verified
   token matches the draft, stop (and take the verified token) at the first
   mismatch. Unchanged by this feature — the gate only changes *how* each
   verified token was computed, not what happens with it.

The gate's payoff: for the fraction of positions below `T`, step 5 reads M rows
instead of ~152k, and step 4 already avoided materializing their full logits.
Positions above `T` cost exactly what they cost today.

> This runtime path is off by default. The calibration harness below exercises
> steps 2–6 for *every* position (both paths) to measure how often the
> restricted path would have agreed with the full head — it does not gate.

## Calibration: picking (T, M)

Two things are calibrated jointly — the entropy threshold `T` and the `M` used
below it. The `test_dflash` calibration harness (`DFLASH_TOPK_CALIB=1`) collects
a joint **(entropy-bucket × rank)** histogram plus a **real-kernel fidelity**
check (running the actual `extract_topm_cuda` + `restricted_lm_head_q6k` and
comparing token-for-token against the exact full-head argmax). From the joint
table you read off, for positions below a candidate `T`, the smallest `M` that
reaches the desired coverage (e.g. 99.9%).

Runtime flags (in `test_dflash`): `DFLASH_TOPK_HEAD=M` (master on/off + M),
`DFLASH_TOPK_HEAD_ENTROPY_T=<nats>` (gate threshold).

## Two tiers of guarantee

The gate is a **statistical** mechanism — it makes divergence rare and tunable,
but never provably zero. Keep that separate from an actual losslessness proof:

**Robust (statistical, tunable):** entropy gate, optionally unioning the draft
top-M with a small *static* safety-net set (highest-norm rows and/or
highest-frequency-argmax tokens from calibration). Cheap; lifts coverage; no
full-vocab pass.

**Lossless (provable) — bound-and-branch:** to guarantee DFlash-greedy ≡
full-target-greedy you must touch all of vocab, but cheaply:

1. Restricted head → exact best-in-set logit `L*` over the draft top-M.
2. A cheap coarse full-vocab pass → per-token **upper bound** `UB_v ≥ logit_v`
   (e.g. a Q2_K/int8 coarse head with a per-row quant-error bound, ~40% of the
   full-head traffic; or a low-rank factorization if the head's singular values
   decay fast enough).
3. Prune every `v` with `UB_v < L*`. Survivors `S'` provably contain the true
   argmax.
4. `S' ⊆ top-M` → certified lossless, done. Else exact-eval the tiny remainder
   `S' \ top-M` and re-argmax.

For greedy verify this is a one-sided **acceptance test** ("does anything beat
the drafted token's logit?"), which prunes harder than reconstructing the full
argmax. Target: ~5% → ~1% end-to-end **and** provably lossless, provided the
certification rate (`S' ⊆ top-M`) stays high (>99%).

## Open questions

- Real full-head cost: confirm the ~5–6% via a feature-OFF vs feature-ON A/B in
  `scripts/bench_llm.py` (isolate the head `mul_mat` kernel).
- Coarse-bound choice: measure the head's singular-value decay before betting on
  low-rank; otherwise start with a Q2_K coarse head.
- Certification rate: how often `S' ⊆ top-M` fires (i.e. how often the exact
  fallback is avoided) at a given `M`.