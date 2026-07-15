# Candidate-restricted LM head — calibration results

**Date:** 2026-06-18 · **Branch:** `feat/topk-head-calib`
**Companion:** design + methodology in [`topk-head-optimization.md`](topk-head-optimization.md)

Full top-k/top-p shortlist calibration for the candidate-restricted LM head, run on
a single **NVIDIA B200** (Blackwell, sm_100) via `calib/calibrate.py` driving the
`test_dflash` chain-verify path (exact position alignment, `DFLASH_TOPK_CALIB=1`).

## Setup

| | |
|---|---|
| Target | Qwen3.6-27B **Q4_K_M** (`server/models/Qwen3.6-27B-Q4_K_M.gguf`, vocab 248,320) |
| Draft | DFlash 3.6 **Q4_K_M** (`dflash-draft-3.6-q4_k_m.gguf`) |
| Prompts | 198 — 66 each from HumanEval / GSM8K / MATH-500 (Qwen3.6 chat template, thinking off, ≤1024 tok) |
| Decode | greedy (`temp=0`), `n_gen=256`, chain path (no `--ddtree`), `--max-ctx=1024` |
| Positions | **102,045** chain-verify positions |
| GPU | B200, GPU 0, `DFLASH27B_PREFILL_UBATCH=1` (see caveat) |

Sampling metrics (top_k=8 / top_k=20 / top_p=0.95) are computed at fixed internal
temperature 1.0 per position, independent of the greedy decode trajectory.

> **Run quality:** 198/198 prompts succeeded (0 dropped), balanced across all three
> datasets — gsm 37,905 · he 22,935 · math 41,205 positions. This required the
> `DFLASH27B_PREFILL_UBATCH=1` workaround below; without it ~76% of prompts abort and
> the survivors skew heavily away from code (HumanEval).

## Set-coverage — fraction of positions whose entire set ⊆ draft top-M (%)

| M | greedy | top_k=8 | top_k=20 (Qwen def) | top_p=0.95 |
|------:|-------:|--------:|--------------------:|-----------:|
| 1     | 57.30  | 0.00    | 0.00   | 47.02 |
| 2     | 70.24  | 0.00    | 0.00   | 56.23 |
| 4     | 79.88  | 0.00    | 0.00   | 62.19 |
| 8     | 87.50  | 0.06    | 0.00   | 67.00 |
| 16    | 92.68  | 2.68    | 0.00   | 71.08 |
| 32    | 95.94  | 10.53   | 0.02   | 74.72 |
| 64    | 97.64  | 24.51   | 0.66   | 78.10 |
| 128   | 98.56  | 41.73   | 4.11   | 81.10 |
| 256   | 99.02  | 58.15   | 12.77  | 83.59 |
| 512   | 99.27  | 71.41   | 26.01  | 85.97 |
| 1024  | 99.46  | 81.44   | 41.79  | 88.14 |
| 2048  | 99.63  | 88.51   | 57.68  | 90.39 |
| 4096  | 99.78  | 93.46   | 71.72  | 92.68 |
| 8192  | 99.86  | 96.54   | 82.94  | 94.89 |
| 16384 | 99.92  | 98.23   | 90.45  | 96.56 |
| 32768 | 99.95  | 99.11   | 94.93  | 97.73 |
| 65536 | 99.97  | 99.55   | 97.44  | 98.61 |

`mean_rank = 61.6`, `max_rank = 65536` (grid ceiling; a long low-confidence tail,
dominated by code prompts, extends beyond it).

## top_p=0.95 nucleus — MASS coverage

Set-coverage is the wrong quality metric for sampling (missing a low-prob nucleus
token barely distorts the draw). The right one is **covered probability mass**.

| M | mean nucleus mass covered (%) | positions <99% covered (of 102,045) |
|------:|------:|------:|
| 256   | 98.318 | 13,639 |
| 512   | 98.827 | 10,421 |
| 1024  | 99.176 | 7,499  |
| 2048  | 99.460 | 4,924  |
| 4096  | 99.682 | 2,827  |
| 8192  | 99.816 | 1,435  |
| 16384 | 99.895 | 702    |
| 32768 | 99.939 | 340    |
| 65536 | 99.963 | 157    |

## Verdict

- **Greedy: viable.** Target argmax ∈ draft top-M at **99.46% for M=1024** and
  99.78% at M=4096 — near-lossless, and the restricted head is <0.5% of the
  248k-wide matmul. Build the restricted head for the greedy verify path.
- **Sampling, exact set-coverage: not practical.** Exact `top_k` needs the draft
  shortlist to contain the target's *entire* top-K_s; the Q4 draft covers the full
  top-20 only ~72% of the time even at M=4096 (97% at M=65536). The draft ranks the
  target's #1 well but the rest of the nucleus poorly.
- **Sampling, approximate-but-faithful: viable.** By covered mass, M≈8192 captures
  99.8% of the nucleus (only ~1.4% of positions <99% covered) at ~5.9% step speedup;
  M=65536 is lossless-in-practice (99.963%, 0.15% of positions <99%). Gate pure-top_p
  / pure-temperature to the full head, or over-cover with a tail-mass correction.
- A stronger (BF16) draft would likely improve the sampling numbers; the structural
  asymmetry (argmax easy, full nucleus hard) will persist to some degree.

## Caveat — batched-prefill NaN bug (worked around, not fixed)

The batched prefill attention in `test_dflash` emits **all-NaN logits** for certain
`(n_tokens, kv_start)` shapes (e.g. a single 51-token ubatch, or any 2nd+ ubatch with
`kv_start>0`). The greedy `argmax` of an all-NaN vector returns index 0, so the prefill
reports `last_tok=0` and the first draft step aborts with a silent `return 1` right
after `[migrate]`. This affected ~76% of prompts and biased survivors away from code.

Root cause is the batched FA / KQ-mask path — it reproduces on `main` with no calib
env, independent of this work. Token-by-token prefill is numerically correct, so the
calibration runs with `DFLASH27B_PREFILL_UBATCH=1` (the `calib/calibrate.py` default,
`--prefill-ubatch 1`). The underlying kernel bug still needs a real fix; it would
silently corrupt any batched-prefill decode, not just calibration.

## Reproduce

```bash
# toolchain (uv): cmake/ninja + tokenizer deps
uv venv .venv-calib && uv pip install --python .venv-calib/bin/python \
    cmake ninja transformers datasets jinja2 huggingface_hub

# download models (target -> server/models, draft -> server/models/draft)
# build test_dflash (sm_100), prep 200 prompts, run on GPU 0, aggregate -> JSON
PATH="$PWD/.venv-calib/bin:$PATH" .venv-calib/bin/python calib/calibrate.py \
    --all --prep 200 --gpu 0 --ngen 256 --prefill-ubatch 1 \
    --json calib/out/calib200_fixed.json
```

Raw per-prompt tables + the position-weighted aggregate: `calib/out/calib200_fixed.json`.
