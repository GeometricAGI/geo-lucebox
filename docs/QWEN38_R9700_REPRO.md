# Reproducing the Qwen3.8-27B R9700 speculative-decode numbers

Everything below was re-derived from scratch on a clean machine (lucebox6) on
2026-08-24 because none of it was written down. The drafter build in §2 is the
part that matters: get it wrong and acceptance length collapses from ~6 to ~1.7,
which makes every throughput number meaningless.

## Results

One configuration, both arms measured on the same box at the same commit.
**Every number below comes from this setup and nothing else appears here.**

* R9700 (gfx1201), ROCm 7.2.4, `server/build-wide` at this branch's head
* drafter `qwen38-dflash2-q8_0-canonical.gguf`, 2,045,471,776 B, md5
  `a98fb401578886f082315c7031f419a2` -- verified loaded in every server log
* shipping defaults: MMQ on, width gate 160, draft block 8, `GGML_GQH_Q8N` unset
  (flat 127), prefix and prefill caches off -- `cache_hit=true` never observed
* two readings per arm, order-balanced ABBA, GPU at its exact idle VRAM baseline
  before and after every reading

| | GQH-Q3KXL | IQ4_XS | delta |
|---|---:|---:|---:|
| file bytes | 13,440,110,432 | 15,567,824,480 | **-13.67%** |
| HumanEval+ pass@1 | 145/164 | 145/164 | **indistinguishable** |
| decode tok/s | 95.10 - 95.20 | 82.50 - 82.80 | **+15.1%** |
| decode ms/step | 47.78 - 47.89 | 52.33 - 52.50 | **-8.7%** |
| prefill tok/s, 119-token prompt | 701.2 - 711.3 | 862.3 - 871.8 | -18.5% |
| prefill tok/s, 6,850-token prompt | 852.5 - 866.5 | 1064.2 - 1075.7 | **-19.5%** |
| end-to-end tok/s, 10 code prompts | 126.60 - 126.68 | 116.15 - 116.24 | **+9.0%** |
| accept % / avg_commit, prose | 56.9 / 4.56 | 54.2 / 4.33 | GQH higher |

**In one line: 13.7% smaller, indistinguishable quality, decodes ~15% faster,
still prefills slower, and ~9% ahead end-to-end.**

> **PROVENANCE CAVEAT, being fixed.** This table claims both arms at one commit,
> and one cell no longer honours that: the GQH `prefill @6,850` row was patched
> from a **GQH-only** run after the width gate moved, while the IQ4_XS column is
> from the earlier both-arms run. A GQH-only gate change cannot move IQ4_XS, so
> the number is not *wrong* -- but the table's own methodology claim is what makes
> it trustworthy, and patching one cell weakens it. The `end-to-end` row is also
> from an older reading and did not reproduce to its stated band in a later A/B
> (126.79-126.88 against 126.60-126.68), which is within noise but not the same
> measurement. Both arms are being re-measured at the head; until then treat the
> deltas as accurate to about a point and the methodology line as aspirational for
> those two rows.

### Where the speed comes from, and what it still costs

The decode win is real per-step work, not an acceptance artefact: a ~9% narrower
step at equal geometry, with `avg_commit` identical across arms' repeats.

**Short-prompt prefill improved 38% within this branch** and that is what carries
the end-to-end result. Measured same-box, one commit apart: 507.7-515.4 tok/s at
the parent against 701.2-711.3 at the head, non-overlapping ranges. The mechanism
is one `v_perm_b32` replacing four LUT selects in the MMQ weight decode (2,048 of
them in the shipped gfx1201 ISA), and an estimated -25.7% dynamic VALU predicts
the measured -23% prefill on the search's own harness.

**Long-prompt prefill improved 15.7%, by fixing a stale threshold rather than a
kernel.** `gqh_mmq_max_ne11` gated MMQ dispatch at 160, a bound measured BEFORE
`v_perm_b32` made the MMQ kernel ~2.3x faster, so 512-wide prefill chunks were
being routed to the slower dequant->cuBLAS path on the strength of an obsolete
measurement. Re-measured with the gate lifted, MMQ is 7-27% faster than dequant at
width 512. Raising the default to 512 moved prefill @6,850 from 723.3-751.3 to
852.5-866.5 tok/s.

512 rather than the crossover's 640: `qwen35moe_prefill_chunk_limit` is
`min(DFLASH_QWEN35MOE_PREFILL_CHUNK, prompt_len)` with an env default of 512, so
**no prefill can ever present ne11 > 512**. 640 admits nothing extra for any
prompt while spending gqh3's entire remaining margin (0.999 at 640, break-even,
against 0.927 at 512).

That the gate was the cause is measured, not inferred: the MMQ launch counter
reads 0 at gate 511 and 1 at gate 512 for an ne11=512 node, and an `ne11` census
over a shipping serve puts 27,510 of 36,549 gate-visible calls in exactly the 194
and 512 buckets -- 6,850 = 13x512 + 194, with a measured 13.0 ratio. Those buckets
ARE the long prefill, chunked.

So prefill is still the axis where this artifact loses, but by much less: 40.2%
-> 18.5% at the short prompt and 30.6% -> 19.5% at the long one. A
prefill-dominated workload will still not see the end-to-end number above.

### What the quality number does and does not say

pass@1 is **identical**. Determinism was established first: with the harness
pinned to `temperature: 0, top_k: 1`, both readings of each arm returned 164/164
byte-identical replies. Item-by-item the arms are not the same model -- 140 pass
in both, 14 fail in both, 10 discordant 5 each way -- and McNemar exact,
two-sided, gives **p = 1.0000**. Only 54 of 164 replies are byte-identical
between arms, so they emit different tokens on 110 items and still score the same.

Read that as *no measurable quality difference on this benchmark*, not as *the
arms are interchangeable*.

**The instrument has a floor of about one item, and it is not the model.** Two
faults were found in the scorer after these numbers were taken, and both survive
into any pass@1 this harness prints:

* **An unterminated code fence used to fail a correct answer.** The extractor only
  matched a fenced block when the fence was CLOSED, so a reply that ran to its
  token cap mid-block leaked the literal ```` ```python ```` line into the graded
  script and died on `SyntaxError`. 12 of 164 items take that path. It costs about
  one item per reading, equally in both arms -- so comparisons held, but absolute
  totals read low. Fixed; the figures above are the pre-fix numbers and both arms
  gain one item under the fix.
* **The grader is not deterministic.** Re-grading BYTE-IDENTICAL replies flips a
  verdict: `HumanEval/39` (`prime_fib`) alternates pass and fail between grading
  passes, which is an execution-timeout effect, not a sampling one. So the
  "same-arm floor 0 items" above is REPLY determinism. VERDICT determinism is
  weaker, and the floor is about +-1 item.

Consequence for anyone quoting these: **a one-item difference on this benchmark is
not a result.** At the discordance rate observed here, resolving one would need on
the order of 3,700 items -- and that assumes a deterministic scorer, which this is
not. Quote the two arms as indistinguishable, which is what both the McNemar result
and the floor say.

### Two traps for anyone reproducing this

**Use the canonical drafter.** Three other DFlash2-shaped drafters are floating
around and all of them load. `Qwen3.8-27B-DFlash2-Q8_0.gguf` (2,056,414,752) is
the vendor file and shifts acceptance; `qwen38-dflash2-q8_0.gguf` (1,838,540,000)
was built by a converter that silently drops DFlash2's 23 conv and
candidate-selector tensors. Two drivers in `server/scripts/` were found pointing
at the wrong one. Check the md5 above.

**`avg_commit` is not comparable across draft block widths.** It is
`accept% x verify_cap`, and `verify_cap` is the block size, so the same model
reads 4.56 at block 8 and 6.83 at block 16. Comparing figures taken at different
widths produced an apparent accept-rate reversal in our own earlier notes that
was purely this. Quote accept % with its block width, or quote both.

---

## 0. Hardware / toolchain

    GPU     AMD Radeon AI PRO R9700, gfx1201, 31.86 GiB usable
    ROCm    7.2.4 (lucebox6) / 7.2.53211 (lucebox5) - both fine
    build   ninja (lucebox6 has no `make`)

Pick the R9700 explicitly. It is `HIP_VISIBLE_DEVICES=0` on lucebox6 and
`=1` on powerboat (which has a 7900 XTX at index 0). Confirm with
`rocm-smi --showproductname` before trusting any number.

## 1. Build

    cmake -B server/build-bench -S server -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DDFLASH27B_GPU_BACKEND=hip \
      -DDFLASH27B_HIP_ARCHITECTURES="gfx1151;gfx1201" \
      -DDFLASH27B_HIP_SM80_EQUIV=ON \
      -DGGML_HIP_GRAPHS=OFF
    cmake --build server/build-bench --target dflash_server test_dflash -j 16

The documented CUDA path (`cmake -B build -S .`) dies on AMD at
`project(... CUDA)`. Do not use it.

## 2. The drafter - THIS IS THE TRAP

The drafter must end up as **81 tensors, arch `qwen35-dflash-draft`,
`block_size 8`, `target_layer_ids [5,19,33,47,61]`**. Verify before benchmarking.

Source weights: the DFlash2 drafter safetensors, **3,848,817,896 bytes**
(`incoai/Qwen3.8-27B-DFlash2`, or the `z-lab-dflash2` copy on powerboat -
byte-identical). It is the drafter alone, not the 27B target.

    # 1) convert  -> 3,849,350,176 bytes, 81 tensors
    python3 server/scripts/convert_dflash_to_gguf.py \
        z-lab-dflash2/model.safetensors qwen38-dflash2-bf16.gguf

    # 2) quantize -> 2,045,471,776 bytes ("quantized 49 tensors, kept 32 as-is")
    llama-quantize qwen38-dflash2-bf16.gguf qwen38-dflash2-q8_0.gguf q8_0 $(nproc)

Check it:

    python3 - <<'EOF'   # or any gguf KV dump
    # expect: tensors=81  arch=qwen35-dflash-draft
    #         dflash.block_size=8  dflash.target_layer_ids=[5,19,33,47,61]
    #         attention.head_count=32  dflash.mask_token_id=248070
    EOF

### Three wrong drafters that all "work" and all give garbage AL

| artifact | bytes | why it is wrong |
|---|---|---|
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` (incoai, prebuilt) | 2,056,414,752 | arch is the bare string `dflash`, NOT `qwen35-dflash-draft`. Upstream #625/#642 **reject** it outright; only our tree accepts it (a muse-glimmer addition) and then ignores its 23 DFlash2-specific tensors. |
| `quantize_draft_q8.py` output | 1,838,540,000 | that script has no dspark/markov/confidence/aux handling and **silently drops 23 tensors** (58 instead of 81). Loads fine. AL ~1.7. Use `convert_dflash_to_gguf.py` + `llama-quantize` instead. |
| `RadixArk/Qwen3.8-27B-DSpark` converted | 2,718,700,192 | different model (DSpark, not DFlash2): `block_size 7`, `target_layer_ids [4,16,28,40,52]`, `head_count 40`, `mask_token_id 248077`. Wrong capture layers => AL ~1.2-1.8. |

The failure mode is silent: the server loads, generates correct text, and just
runs slow. Always print `[draft] target capture layers from drafter GGUF` and
check it says `5 19 33 47 61`.

## 3. Targets

    IQ4_XS   Qwen3.8-27B-IQ4_XS.gguf     15,567,824,480   bartowski/Qwen3.8-27B-GGUF
    GQH      qwen38-gqh-shaped.gguf      13,440,110,432   our artifact

`GGML_GQH_I8DOT` is **on by default** - the env var only disables it. GQH also
self-caps the tree: `[qwen35-spec] GQH I8/glu-fuse verify is ncols=8; capping
--ddtree-budget N -> 7`, so budgets above ~8 do nothing for GQH but do help
IQ4_XS. Tune each arm separately or the comparison is unfair.

## 4. Running

    export TMPDIR=$HOME/tmp        # /tmp/dflash_bench may belong to another user
    DFLASH_BIN=server/build-bench/test_dflash \
    DFLASH_TARGET=<target.gguf> \
    DFLASH_DRAFT=qwen38-dflash2-q8_0.gguf \
    HIP_VISIBLE_DEVICES=0 \
    python3 server/scripts/bench_he.py --specla --n-gen 128 --ddtree-budget <N>

Tokenize once, then pass `--skip-tokenize` so every arm uses byte-identical
prompt files. Needs `transformers` (venv; the system python is PEP 668 managed).

`bench_he.py` reports mean AL and decode tok/s. The HTTP path
(`bench_he_http.py` against `dflash_server`) measures wall-clock including
prefill and reads the real numbers from the server's own `[spec-decode]` lines -
use it only if `test_dflash` is unavailable, and never mix the two in one table.

## 5. Bugs that block reproduction

* **`test/test_dflash.cpp` `q_len`** - was `DFLASH27B_DRAFT_BLOCK_SIZE`
  (compile-time 16) while `inp_embed` is sized from the GGUF's runtime
  `block_size`. Any drafter not declaring 16 aborts with
  `GGML_ASSERT(offset + size <= ggml_nbytes(tensor))`. Fixed here to
  `dw.block_size > 0 ? dw.block_size : DFLASH27B_DRAFT_BLOCK_SIZE`, matching the
  pattern already used a thousand lines earlier. **The same bug is in #625 and
  #642** (their lines 2280 / 2276) - patch them or they cannot be benchmarked.
* **`--specla` forces DDTree.** For arch `qwen35` it unconditionally sets
  `ddtree_mode = true` (`server_main.cpp`), so the greedy chain path is only
  reachable by omitting `--specla`. Undiscoverable from `--help`.
* **`scripts/quality_humaneval_plus.py`** resolves its dataset as
  `PROJECT_ROOT/dflash/eval/...`, the pre-rename directory name. Unrunnable from
  a clean checkout of this tree; wants a `DFLASH_EVAL_DATASET` override.
* **Adaptive spec policy masks low acceptance.** `DFLASH_QWEN35_SPEC_STEP_RATIO`
  (default 1.9 => accept threshold 0.72) drops to 40 plain AR steps per spec
  probe when acceptance is poor, so a broken drafter reads as `avg_commit 1.05`
  and AR-equal throughput rather than as an error. Set it to `0` when
  diagnosing, never when measuring.

## 6. Sanity values (R9700, lucebox6, canonical drafter)

AR decode, no speculation: IQ4_XS 28.78 tok/s, GQH 25.14 tok/s.
Speculative, default budget: all arms AL 6.00 on IQ4_XS with per-prompt commit
range 4.74-7.53 identical across ours/#625/#642, tok/s within 0.4%.
Same-arm floor across two full repeats: 0.2% or better.

If your AL is below ~5 on HumanEval prompts, the drafter is wrong. Go back to §2.

## 7. Superseded: the ddtree-budget head-to-head

Removed. That table tuned only `--ddtree-budget`, which leaves both arms at the
drafter's trained block width of 8, so it measured the wrong lever and its own
footnote said not to quote it. The draft block width is the lever (section 8),
and the head-to-head that supersedes it is the Results table at the top.

## 8. The real lever: draft block width (and where GQH stops)

Section 7 tuned the ddtree budget and left both arms at the drafter's trained
block width of 8. That is not the lever. Upstream's own post documents it:
DFlash2 ships trained at 8 tokens per block but its acceptance extrapolates, and
`--draft-block-size 16` takes their HumanEval decode from 133.9 to 208.1 tok/s.

Measured here, server path, 10 HumanEval prompts, max_tokens 256, canonical
drafter, decode-only from the engine's own `[spec-decode]` timer:

| arm | block | decode tok/s | end-to-end | avg_commit |
|---|---:|---:|---:|---:|
| ours + IQ4_XS  |  8 | 126.88 | 108.08 |  7.57 |
| ours + IQ4_XS  | 16 | 161.94 | 132.67 | 12.18 |
| **ours + GQH** |  **8** | **154.35** | 88.93 |  7.22 |
| ours + GQH     | 16 |  56.40 |  44.08 | 10.98 |
| #642 + IQ4_XS  |  8 | 126.96 | 108.75 |  7.57 |
| #642 + IQ4_XS  | 16 | 157.18 | 125.90 | 12.18 |

**Honest standing: our best (GQH, block 8, 154.35) is 2-5% BEHIND #642's best
(IQ4_XS, block 16, 157.18-162.82).** GQH wins per step at width 8 by 21.6%
(154.35 vs 126.88) and loses overall because it cannot widen.

Build flags are not the difference. Rebuilding with upstream's published flags
(`-DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON`, gfx1201 only,
`CMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++`) moved nothing:
IQ4_XS block 16 161.94 -> 162.82, GQH block 8 154.35 -> 154.69, GQH block 16
56.40 -> 56.59. All inside noise. Our IQ4_XS block-16 decode of 162.8 against
their published 208.1 on the same file and drafter is therefore box or
environment, not build configuration, and is unresolved. Note our prefill
matches theirs closely (below), so the gap is decode-specific.

### The cliff immediately after ncols 8 -- since fixed

This section records the cliff as originally found. It is no longer a
property of the kernel: giving every dispatched verify width its own
exact-width arm, then giving width 16 its own ROWS, took the block-16 arm
from 56.4 to 195.8 tok/s (3.47x). Read the table below as the diagnosis
that motivated that work, not as current behaviour.

The spec cap is `1 + n_nodes`, so a cap of 7 means the kernel runs at **ncols 8**.
Sweeping GQH by width (`gqh_cap_spec_ddtree_budget`, `gqh_headers.cpp:65`, caps
at `native_block - 1`, and `native_block` follows `--draft-block-size`):

| block | cap | ncols | decode tok/s | avg_commit |
|---:|---:|---:|---:|---:|
|  8 |  7 |  **8** | **154.35** |  7.22 |
|  9 |  8 |  9 |  69.03 |  8.16 |
| 11 | 10 | 11 |  55.22 |  9.37 |
| 13 | 12 | 13 |  63.37 | 10.32 |
| 14 | 13 | 14 |  60.65 | 10.53 |
| 16 | 15 | 16 |  56.40 | 10.98 |

Acceptance rises exactly as it should (7.22 -> 10.98, tracking IQ4_XS's
7.57 -> 12.18), so the drafter extrapolates on our artifact too. Throughput
still falls 2.2x the moment you leave ncols 8.

**The cause is NOT register pressure**, and getting this wrong sends the fix in
the wrong direction. `gqh.cu` around line 620 does document a VGPR wall - at
ROWS==3 the grid reaches 99-100 VGPRs by ncols 7 and occupancy drops from 16
waves/SIMD to 12 - but occupancy is not what limits this kernel. The geo-evo
`gqh_mtp_multicol` run measured it directly (iteration 6): an LDS-burn probe
drove this same arm from 16 waves/SIMD down through 10 / 7 / 6 / 5 / 3 and the
scored shape **did not move**; it collapses only at 1 wave/SIMD. So the VGPR
table describes occupancy, not throughput.

What actually changes across the cliff is work per instruction. The ncols == 8
arm is instruction-issue bound (its own comment says so). `GQH_MULTICOL_SPEC_MAX`
is 12, but 9..12 use **ROWS==1** exact-width, which discards the row reuse that
makes ncols 8 efficient, and 13..16 fall to the generic instantiation. Hence cap
12 (63.37) is no better than cap 15 (56.40): both pay more instructions per
useful FMA than the ROWS>1 arm at 8.

For the same reason, LDS staging is aimed at the wrong constraint. It was tried
and rejected on measurement anyway - staging N=8 activations in 8 KB LDS got
GQH3 under the VGPR cliff but ran 94-101 tok/s against register-xshared's 104
("do not revive without a smaller tile") - but the lesson is that it bought
occupancy, which this kernel does not want.

So widening GQH is not a config change and not a missing instantiation. It needs
a wide-ncols path that keeps row reuse without spilling.

## 9. Prefill and the MMQ path

GQH registers only the ggml converter hooks (`dequantize_gqh3_to_fp16_cuda` ->
`gqh_convert` -> `gqh3_decode_cuda`). There is **no MMQ kernel for the GQH types**
- `mmq.cu`/`mmq.cuh` contain no GQH3/GQH4/GQH2_H at all. The matvec dispatch
declines anything wider than `GQH_MAX_COLS` (16) with the comment "wider batches
keep the dequant->GEMM path", so every prefill dequantises the weight tensors to
fp16 and then runs a cuBLAS GEMM.

Prefill throughput, same build, `--max-ctx 32768`, prefill from the server timer:

| prompt tokens | IQ4_XS tok/s | GQH tok/s | ratio |
|---:|---:|---:|---:|
|    218 |  311.4 | 167.7 | 0.54 |
|  1,018 | 1018.0 | 783.1 | 0.77 |
|  4,018 | 1085.9 | 787.8 | 0.73 |
| 12,018 |  946.3 | 719.6 | 0.76 |

Two distinct costs, both from the missing MMQ path:

* **A fixed per-prefill adder.** GQH prefill is *identical* at 218 and 1,018
  tokens (1.3 s both), i.e. independent of prompt length - you are paying for the
  weights, not the prompt. On the HumanEval prompts it shows as a steady 0.3 s
  against IQ4_XS's 0.1 s across all ten requests. 13.44 GB of weights expand to
  ~27 GB of fp16, ~40 GB of traffic, plus one kernel launch per tensor.
* **~25% lower prefill throughput at length**, once that adder amortises.

This is why GQH's end-to-end (88.93) sits so far under its decode (154.35), a
0.58 ratio where IQ4_XS manages 0.81.

For scale: our IQ4_XS prefill of 1018 tok/s at 1K tokens is in line with
upstream's published 945 tok/s at 1.4K, so prefill is not where this box differs
from theirs.

### One fix for both problems -- half right, and the half that failed

The prediction here was that an MMQ path would remove the dequant from prefill
**and** give the wide verify widths a batched quantised kernel. **The second half
was right; the first was wrong, and it was measured wrong twice.**

MMQ re-decodes the weight tile once per output column tile -- about 16 times at a
2K prompt and 68 at 8.7K -- where dequant unpacks once and streams fp16. So MMQ
*loses* at prefill widths, and loses harder as the prompt grows: +4.2% at 2K and
+6.2% at 8.7K with one rung converted, roughly doubling to +8.3..13.8% once both
rungs were. The regression scaling with both prompt length and MMQ's share of the
artifact is what confirmed the mechanism rather than a coding error.

What MMQ does win is the narrow chunks. An `ne11` census over a real verify plus
a 6,850-token prefill puts 9,039 of 85,839 GQH `mul_mat` calls (24.7% of those
that get past the matvec) inside the gate, and there MMQ is **1.68x** on a
119-token prefill. Decode never reaches it at all -- the matvec owns widths 1..16
and returns first -- so **MMQ can be credited for prefill and never for decode.**

Hence the dispatch is width-gated (`gqh_mmq_max_ne11()`, default 160) rather than
all-or-nothing. The threshold turns out not to be load-bearing: the workload's
widths are bimodal with a hole, clustering at 8, 39-119 and 512, so nothing lands
between 105 and 511 and any threshold in that window dispatches identically.

Caveat on all end-to-end numbers here: everything ran `--prefix-cache-slots 0` to
keep arms comparable. Workloads with shared prefixes amortise the fixed prefill
cost, so the production penalty is smaller than these rows -- but cold-prefill
latency is real and it is what a first request feels.

### Measured: MMQ loses prefill at length and wins the code workload

The "one fix" paragraph above was a prediction. Both GQH rungs the artifact
contains now have an MMQ path (`GGML_GQH_MMQ=1`, off by default) and the
prediction is measurably wrong where it was most confident and right where it
was most cautious.

A decision rule was written down before measuring, and is reported against
verbatim below so the result cannot be rationalised after the fact.

#### Prefill: MMQ loses, and loses more the longer the prompt

Target only, no drafter, `--max-ctx 32768 --prefix-cache-slots 0`, prefill from
the server timer, three warm requests per cell, arms interleaved because this rig
drifts upward (§10). Stage 1 had GQH3 only on MMQ (139 of 396 header-bearing
tensors); Stage 2 has GQH3 and GQH4, i.e. all 396.

| prompt tokens | dequant | MMQ, GQH3 only | MMQ, both rungs |
|---:|---:|---:|---:|
|   353 |  0.5 s |  0.5 s |  0.5 s |
| 2,033 |  2.4 s |  2.5 s (+4.2%) |  2.6-2.7 s (+8.3 to +12.5%) |
| 8,703 | 10.9 s | 11.7 s (+6.2%) | 12.2-12.5 s (+11.9 to +13.8%) |

Each arm reproduced to 0.1 s within a run. 353 tokens is below the timer's
resolution warm; its COLD first request is the other way round and repeatably so,
1.3 s on MMQ against 1.8 s on dequant, which is the fixed adder this section
opened with - though a first request also carries warm-up, so treat it as
suggestive.

**Verdict against the pre-registered rule: third branch. MMQ still loses, and is
still worse at 8.7K than at 2K.** The re-decode mechanism is confirmed, now by
two independent predictions rather than one: the regression grows with prompt
length, AND it roughly doubled when MMQ's share of the artifact went from 35% to
100%. MMQ re-loads and re-decodes the weight tile once per output column tile -
at `mmq_x` 128 that is ~16 re-decodes at 2K tokens and ~68 at 8.7K - where the
dequant path unpacks once and lets cuBLAS stream fp16. GQH cannot buy its way out
with a bigger tile: it already uses the 128x128 shape and `mmq_x` stops at 128.
IQ4_XS survives the same re-streaming because its unpack is a nibble and a
`__byte_perm` LUT; GQH's is bit-sliced planes or uint4 codes plus a curved-grid
select per code.

#### But the code workload gains 20%, and that flips the conclusion

`final_table.sh`, two readings per arm. IQ4_XS cannot be affected by
`GGML_GQH_MMQ` and is the machine control.

| arm | metric | MMQ off | MMQ on | delta |
|---|---|---:|---:|---:|
| GQH b16 | he_tok_s (HE-10, decode only) | 194.02 / 195.98 | 191.25 / 191.25 | -1.4 to -2.4% |
| GQH b16 | code_e2e_tok_s | 115.90 / 116.85 | 140.14 / 140.64 | **+20.6%** |
| GQH b16 | code_decode_tok_s | 138.06 / 139.33 | 152.74 / 153.42 | **+10.5%** |
| GQH b8 | code_e2e_tok_s | 112.26 / 112.42 | 131.27 / 130.53 | **+16.5%** |
| IQ4_XS b16 | he_tok_s | 159.97 / 160.36 | 159.29 / 160.09 | -0.3% |
| IQ4_XS b16 | code_e2e_tok_s | 119.26 / 119.63 | 119.27 / 118.93 | -0.3% |
| IQ4_XS b8 | code_e2e_tok_s | 101.79 / 101.97 | 101.83 / 101.71 | -0.1% |

The control is flat to 0.3% on every metric and both repeats, so the GQH code-axis
movements are real. `avg_commit` is identical to four decimals in both arms, so
this is not an acceptance-length effect.

Consequence for the comparison this document exists to settle: on the code
workload GQH went from LOSING to IQ4_XS (116.85 against 119.63) to beating it by
17.9% (140.64 against 119.27).

`code_decode_tok_s` excludes prefill and still gains 10.5%, so not all of this is
the fixed adder - part of it is inside the decode loop. The plausible mechanism is
the one §9 predicted for the verify widths: a verify batch wider than
`GQH_MAX_COLS` used to fall through to dequant->GEMM, which the fused-matvec
comment already calls pathological, and now takes MMQ instead. That is a
hypothesis; it has not been instrumented.

#### Decode moved, slightly, and it is not yet explained

`he_tok_s` on GQH b16 fell about 2% (191.25 against 194.02-195.98, where the off
arm's own run-to-run spread is 1.0% and the control moved 0.3%). That is at or
just above the ~1.3% scorer floor, so it is a small real effect rather than noise.
Nothing in the MMQ path should touch a width the matvec still owns. The suspect is
that `ggml_cuda_should_use_mmq` returning true for GQH also feeds the
graph-capture eligibility predicate and `supports_op`, so enabling it can change
capture or fusion decisions on subgraphs whose arithmetic never changes.
Unverified.

#### Where this leaves MMQ

Not killed, and not on by default either. The shape that fits the measurements is
a WIDTH-GATED dispatch: MMQ for the wide-verify band just past `GQH_MAX_COLS`,
where it replaces a dequant->GEMM fallback that is pathological, and
dequant->cuBLAS kept for true prefill widths, where re-decode per column tile
loses to unpacking once. There is precedent for exactly this in the same file -
the mix qtypes already decline wide batches via `mix_mmq_max_ne11`. Finding that
crossover is the next measurement, and until it exists `GGML_GQH_MMQ` stays
opt-in.

Note on hardware coverage: everything above is gfx1201, where the WMMA branch
executes and the dp4a arm is dead code. The dp4a tile indices in the GQH loaders
are unverified and nothing runnable on this box tests them.

## 10. Where the search runs

The widening work is driven by the geo-evo kernel loop, target `gqh_wide_he`
(geo-evo branch `geo-loop/gqh-wide-verify`), reusing the `gqh_n8_he` adapter
with `GQH_HE_BLOCK_SIZE=16` so the narrow target is unchanged when it is unset.

**The block-16 arm is now at 195.84 tok/s** (R9700, canonical drafter, 10
HumanEval prompts, decode-only from the engine's own `[spec-decode]` timer),
which is the 3.47x recorded in section 8. Any goal below that is already met,
so a search aimed at it scores every candidate as passing on its first
iteration and ranks on nothing.

    GQH_HE_GOAL_TPS=205.0

205 rather than 196: the scorer's floor is ~1.3% and nothing under ~2-3% is a
result (below), so a goal has to sit clear of the noise around the number it is
trying to beat. 205 is about 4.7% above 195.84. It is a threshold chosen from
the current best and the noise floor, not a measurement -- recalibrate it after
any move that shifts the best, and recalibrate it **on the box that will run
the search**, per the closing note in this section.

    DO NOT USE: GQH_HE_GOAL_TPS=163.0, GQH_HE_AL_MIN=10.5

Both are stale and both are stale in the same way -- they were set when the
block-16 arm ran at 56 tok/s:

* `163.0` was a bar the wide arm could not reach. It is now beaten by 20%.
* `GQH_HE_AL_MIN=10.5` was the gate that made the search honest, because at
  the time the throughput ranking alone preferred abandoning wide verify:

      block-8  (narrow): he_tok_s = 154.19  avg_commit =  7.22  -> REJECTED
      block-16 (wide)  : he_tok_s =  55.97  avg_commit = 10.98  -> accepted

  The narrow arm was 2.75x faster and was rejected anyway. That is no longer
  the trade: the wide arm now wins on throughput as well (195.84 against
  154.19, +27%), so the AL floor no longer buys the search anything it would
  not choose on its own, and leaving it in only narrows the space. Keep an AL
  floor if you want a guard against a candidate that silently reverts to
  narrow verify -- but set it from the arm you are actually measuring and say
  which, rather than inheriting 10.5 because it is written here.

Two measurement disciplines carried over from the earlier `gqh_mtp_multicol`
run on this same kernel, both learned the hard way there:

* The scorer's floor is ~1.3%: one code state re-read five times gave
  1.4705 / 1.4728 / 1.4745 / 1.4765 / 1.4895, all ISA-identical. Nothing under
  ~2-3% is a result.
* The bare rig drifts ~1.3% upward over about two minutes, monotonically, so
  **interleave** build+measure (base, cand, base, ...) rather than batching, and
  keep some benched shapes ISA-identical so those shapes measure the machine.

Consequence for scheduling: the loop needs the R9700 to itself. Both candidate
boxes are single-R9700, and a co-tenant benchmark makes every number arguable.
Absolute tok/s is NOT portable between them - our IQ4_XS block-16 decode is
162.8 here against upstream's published 208.1 on the same file and drafter, a
gap that survived rebuilding with their exact flags - so a goal threshold must
be calibrated on the box that will run the search. The 195.84 the bar above is
derived from is measured on lucebox6, which is also where every number in
sections 8 and 9 was taken. Ratios port between boxes; thresholds do not.
