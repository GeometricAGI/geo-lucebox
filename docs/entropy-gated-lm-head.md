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
