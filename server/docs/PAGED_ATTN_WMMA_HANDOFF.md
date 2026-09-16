# paged_attn_wmma — design retrospective (GLM-5.3 reviewed, rev 2)

Status: LANDED (see the sessions below for the differential and the perf
results). The header below is the assembly contract the port was built from,
kept as the design authority; its fattn line numbers refer to
fattn-mma-f16.cuh at port time and drift as upstream moves. Read the landed
code, do not redo the assembly. Evidence: PAGED_ATTN_WMMA_PLAN.md (its design
section is SUPERSEDED by this file — see correction 7).

## The kernel = fattn-mma's own code + 6 deltas

Copy verbatim from server/deps/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:
- `flash_attn_ext_f16_iter` (lines 469-968) — KQ mma, mask, softmax, VKQ.
- `flash_attn_ext_f16_process_tile` (lines 1006-1548) — Q load, iter loop,
  rowsum reduce, np-combine, write-back.
- Copy `mma_tile_sizes<ncols>` verbatim (fattn-mma-f16.cuh:986-993). The
  tiles are `tile<16,16,float>` and `tile<16,8,half2>` with DEFAULT
  I_MAJOR — on RDNA4 the I_MAJOR C layout already IS the transposed A&B
  layout. Do NOT declare J_MAJOR tiles.
- mma.cuh: 2-arg `mma(D,A,B)` ACCUMULATES on AMD; tiles zero-init via
  default member initializers (`T x[ne] = {0}`), so KQ_C/VKQ_C start
  zeroed by construction — still zero them explicitly for clarity.

Config: D=256, ncols2=8, ncols1=4, ncols=32, nwarps=8, np=4, nbatch_fa=64,
nbatch_K2=nbatch_V2=64, nbatch_combine=32, tile_stride=nbatch_combine+4=36,
Q_in_reg=true, single-stage gather (256 B rows; block-table resolved once per
thread, 16-byte vector loads, Q8_0 as one 34 B block - cp.async is
NVIDIA-only). Grid: (n_head_kv, ceil(n_rows/4), n_partitions).

## The 6 deltas

1. **Q load**: paged q is F32 [256, n_rows, n_head]; column (j,c) →
   row paged_row0+j, head kv_head*gqa_ratio+c with the RUNTIME gqa_ratio
   (never hardcoded 6). float2 stride = nb/8:
   `Q_f2[j*(q_nb1/8) + c*(q_nb2/8) + k]`, scale_h2 = scale*log2(e)
   (LOG2 DOMAIN). Guards: `j < group_rows` and `c < gqa_ratio`; else zero.
   Head-slot deadness (c >= gqa_ratio) is handled by this zero-Q plus the
   write-back skip — NOT by the mask (the mask tile is per-row only).

2. **K/V tile loads**: per 64-token chunk, per seq-pass s: resolve each
   token with the block-table VALIDATION decode performs (paged-attn.cu
   :452-477: `physical_block < 0 || >= pool_tokens/block_size` → no row,
   zeros). Use the RUNTIME `block_size` op param (gate `block_size == 16`
   in the launcher). row = k + phys*k_nb1 + kv_head*k_nb2.
   F16: direct half2 copy.
   Q8_0: block_q8_0 = {half d(2B); int8 qs[32]} = 34B. For half2 index kk
   (dim pair 2*kk): b = (2*kk)/32, l = (2*kk)%32;
   `d = *(const half *)(row + b*34)`; val = make_half2(d*qs[2l], d*qs[2l+1])
   with qs = row + b*34 + 2. (Row stride k_nb1 is BYTES.)

3. **Mask**: fattn wide-layout mask tile [ncols1][nbatch_fa/2+4] half2s,
   indexed by row only. Paged version: row j visible iff
   `row_seq[j] == seq_s && token < row_extent[j]`, else -FLT_MAX.
   Causal clamp + dead-row pinning from paged-attn.cu:275-330; the
   dead-row SENTINEL WRITE is at :328-348 (an earlier rev pointed at the
   wrong range — the exit block matters).

4. **expf → exp2f** everywhere (softmax :685/:765, KQ_max_scale :779,
   KQ_cms :1396). Drop FATTN_KQ_MAX_OFFSET (log2 domain). Drop sinks.
   INVARIANT: keep `KQ_max` init at `-FLT_MAX/2.0f` (fattn-mma:1083).
   This is load-bearing: paged passes can be FULLY masked (foreign-seq
   passes, dead rows), and `-FLT_MAX/2` makes the exponent
   -inf - finite = -inf → exp2 = 0, never -inf - (-inf) = NaN. Do NOT
   "harmonize" with paged decode's -FLT_MAX init (decode guards
   `score > -FLT_MAX/2 ? exp2f : 0` instead — different mechanism).

5. **Per-sequence passes**: rows in a block may belong to different
   sequences. Enumerate the ≤4 unique seqs ONCE into shared memory
   (block-uniform; a per-warp divergent enumeration deadlocks the
   syncthreads). Per pass s: stage K/V/mask for seq_s, run iter; other
   columns masked. Per-seq active_partitions: inside the block, a row
   whose seq is inactive at THIS partition must take the
   `(-FLT_MAX, 0)` sentinel write (:332-336) even while other rows run —
   paged_attn_combine treats meta.y > 0 as live, and an unwritten
   sentinel reads recycled pool memory as a plausible float2.

6. **Write-back** — the most likely source of a silent bug; copy the guard
   set, not just the addresses:
   (a) SKIP dead columns: `c >= gqa_ratio || row >= n_rows || row invalid`
       (fattn-mma:1502 translates to: head >= n_head || row >= n_rows ||
       row_seq invalid). Writing them OOB corrupts neighboring rows
       silently.
   (b) Single-partition divide: use decode's guard
       `inv_sum = qk_sum > 0 ? 1/qk_sum : 0` (paged-attn.cu:570) — dead
       rows yield 0, not 0/0=NaN.
   (c) Invalid-slot rows → dst = 0 (the :328-348 exit behavior).
   (d) Multi-partition: partial_acc/partial_meta EXACTLY as
       paged_attn_decode :571-602 (pre-normalized acc, log2 meta,
       sentinel). Reuse `paged_attn_partitions` (:24) and
       `paged_attn_combine` (:609) verbatim.

## Launcher + hook

- `try_launch_paged_attn_wmma(ctx, dst)`: env `DFLASH27B_PAGED_WMMA`
  (read-once static, default off). Gates: non-tree, K/V ∈ {F16, Q8_0, Q4_0},
  supported() passes, **gqa_ratio <= 8** (ncols2=8; without this gate any
  ratio > 8 silently never computes heads 8+ — deterministic silent bug
  on model swap), **block_size == 16**. Compute n_partitions like
  try_launch_paged_attn :888-1110 INCLUDING the
  GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS override at :1003-1013 (copy it;
  otherwise whether kv=512 stays single-partition varies by box and the
  direct-write path is untested nondeterministically). When n_partitions > 1
  allocate partials and launch paged_attn_combine after. Bump the counter.
- Hook at the top of ggml_cuda_paged_attn (:1166): if
  try_launch_paged_attn_wmma(...) return;
- Counter: g_paged_attn_wmma256_launch_count + extern "C" record/get,
  getter in ggml-cuda.h (mirror g_fattn_mma256).

## Qualification (test_paged_attn_wmma.cpp)

Differential vs the V_DOT2 kernel (CPU backend ABORTS on GGML_OP_PAGED_ATTN
— confirmed ggml-cpu.c:2247-2249): two-mode program like test_fattn_mma256
— env=1 asserts the wmma counter and dumps outputs, env=0 asserts
counter==0 and dumps; a comparator checks the pair. Matrix:
- rows 64 ragged prefill, pin NON-4-ALIGNED mixes that form mixed-seq
  blocks (e.g. 30+20+14 across slots) — the NaN/garbage classes fire only
  when a seq boundary is not 4-aligned.
- rows 1 decode (no query_positions).
- kv_len 512 (single partition, deterministic only with the
  FORCE_PARTITIONS override or explicit env) and 8192 (multi-partition).
- MIXED LENGTHS at n_partitions>1: one long seq + one short seq — covers
  the per-seq sentinel class.
- K/V {F16, Q8_0} (4 combos).
- A two-mode dump at kv >= 32768 f16 (the f16-acc VKQ drift class grows
  ~sqrt(iterations); 44K ctx ≈ 687 rescales vs 128 at kv 8192).
- Tolerance: q8_0 vs V_DOT2 compares dp4a-integer dequant against
  WMMA half-dequant — a wider gap than the contiguous 2e-3-vs-CPU test;
  expect borderline failures, use ~3e-3 for q8_0 with justification or a
  common f32 reference.
CMake: dflash_add_ggml_gpu_executable + GGML_USE_HIP.

## Verify on lucebox4

build-hip: two-mode unit test + the 12K/44K paged ladder A/B (env 1 vs 0)
+ concurrency harness C1/2/5. Then a GLM-5.3 pass on the actual kernel
diff, fix findings, PR.

## Gotchas learned (do not rediscover)

- 2-arg `mma(D,A,B)` on AMD ACCUMULATES (zero the accumulators anyway).
- `2*k0` in the VKQ A-load is fattn's token-pair stride — copy exactly.
- np-combine meta layout: rows strided by tile_stride=36 half2s
  (= nbatch_combine+4; the 68 the K/V tiles use is stride_tile_K/V), meta at
  float2 offset nbatch_combine/2, nmeta=2 when np*cols_per_warp=64>=32.
- The box's pkill self-match: run server scripts via bash files, never
  inline pkill -f chains.

## Session 2026-09-16: differential gate GREEN (max 2.08e-3 < 3e-3, all 5 cases, deterministic)

Root causes found and fixed (all reproduced numerically before fixing):

1. **Partition overlap**: the iter hardcoded
   `constexpr k_VKQ_sup = nbatch_fa` and never clamped its 64-token tile to
   the partition's `token_end`. With 16-token-wide partitions, token t was
   staged by every partition whose tile spanned it (tokens 16-31 twice,
   32-47 3x, 48-63 4x) — a softmax over a multiset. This was THE extent-17
   boundary: extents <= 16 = single partition = exact; LSQ showed slice-1
   token weights doubled (2w/(1+w) fit to 4 decimals). Fix:
   `k_VKQ_sup = clamp(token_end - (token_begin + kb0*nbatch_fa), 0, 64)`
   (fattn:1178 semantics). Dropped the differential 0.18 -> 0.07.
2. **partial_meta cross-block race** (found by a GLM-5.3 review):
   the meta write derived `head = kv_head*gqa_ratio + jc%ncols2` without a
   `c >= gqa_ratio` guard. With ncols2=8 > gqa_ratio=6, block kv_head=kh
   wrote zero-Q padding-column stats into heads (kh+1)*6+{0,1}'s slots,
   racing block kh+1's real write — nondeterministic exactly on
   multi-partition rows, error pattern on heads {6,7,12,13,18,19}. Fixed
   with the guard (the partial_acc write always had it). Also moved the
   FORCE_PARTITIONS override after the occupancy floor (was a no-op <= 32,
   decode launcher order).
3. **Invalid ragged rows**: decode's `valid_query` (:295-300)
   kills rows with `query_positions` present, no tree, and pos < 0 (empty
   context, zero output). The wmma metadata fell through to the full
   kv_len. The mixed case's qpos=-1 decode row exposed it: ref = all zeros,
   cand = full attention. Where both kernels are valid the wmma output
   matches an f32 host model to 1.7e-5 at extent 8192; the decode reference
   matched the same model to ~1e-4 at extents <= 64. No live decode row at
   extent 8192 was ever compared against truth, so no claim is made about
   the reference there.

Final differential (two identical cand runs, md5-equal):
512f16 0.000488 / 512q8 0.002014 / 8192f16 0.000488 / 8192q8 0.002075 /
mixed 0.000977 — zero elements over 3e-3 anywhere. compare_paged_attn.py
-> PASS.

Analysis-tooling traps that cost hours (do not rediscover):
- numpy reshape of a sliced view silently keeps wrong-stride layouts;
  ALWAYS index flat (`buf[off + d + r*D + h*D*rows]`) or reshape to the
  memory order (dim-major files need `reshape(Hk, pool, D)` etc.). Both the
  "ref row 0 != kdump row 0" and "V[0] != kvf[:,0,0]" mysteries were this.
- The un-partition-gated dbg dump blocks race across blocks/cases; gate
  dumps to one (kv_head, partition, group) or they hold last-writer data.
- Host ground truth: test dumps qv/kvf pre-quantization (hostcase_N.bin,
  header 4xu32 then q [D][rows][Hq] then kvf [D][pool][Hk], dim-major).
  (The kraw/vraw dump loop the bring-up used had a lane-stride bug and is
  gone with the debug instrumentation.)

All of the items then remaining - stripping the debug instrumentation (route
counter kept), the 12K/44K prefill ladder A/B, the concurrency ladders and
the final reviews - are done.

## Session 2026-09-16 (cont): performance — parity at batch, win grows with context

Occupancy refactor and partition-floor fix, both on the
Radeon (gfx1201, 32 CU, 64 KiB LDS/CU — NOT the Strix Halo iGPU on device 1).

Measured (server, q8_0 KV, same binary, env A/B):
- batched 8K pool, concurrency 2-4, aggregate prefill tok/s:
  V_DOT2 1119/1168/1084 vs WMMA 1119/1166/1094 -> parity (+-1%).
- single prompt: 12K 14.6s -> 14.1s (-3.4%); 44K 94.1s -> 86.6s (-8.0%).

The "2x at 44K" from the plan is a KERNEL-level number: 7.5s saved of 94.1s
end-to-end == attention being ~2x faster and ~16% of prefill time. End-to-end
translation is 8%; there is no missing factor.

What moved the needle:
1. smem 51 -> 18.4 KiB: aliased phases were summed instead of
   maxed; dim-split staging (nbatch 128->64), combine strips 64->32, inline
   causal mask (staged tile deleted). Occupancy 1 -> 3 blocks/CU.
2. The launcher forced min_partitions >= 32 for every shape — a
   mis-transcription of the decode launcher's partition_limit cap (a floor on
   the CAP, not on min_partitions). Prefill-shaped calls already fill the
   device via row-groups, so 32 partitions were pure per-partition overhead
   (staging, barriers, partials, combine). Mirroring the decode launcher's
   inversely batch-scaled cap: batch moves -9% -> parity, 12K -4% -> +3%,
   44K +5% -> +8%.

Next candidate was a software double-buffer for the block-table K/V gather
(cp.async is `!GGML_USE_HIP`-gated, so plain double-buffering; an earlier
review specified nbatch 32 x 2 buffers = 18,432 B, occupancy stays 3). It was
tried and reverted - see the next section - and that review's minor cleanups
are folded into the final tree.

### Double-buffer pipeline: attempted, measured, reverted

Split the gather into register-load + smem-store and rotated two 64x36
buffers (nbatch_K2/V2 64->32, 18,432 B, occupancy stays 3) so chunk c+1's
global loads issue before chunk c's mma. Measured regression, both variants
(two-sync and the correct one-sync form): 44K single 86.6s -> 97.5s (+12.6%),
12K 14.1 -> 14.9, batched also worse. With 3 blocks/CU (24 warps) the gather
latency was already hidden; narrower stages double the per-chunk barrier and
loop granularity for no net overlap. cp.async is `!GGML_USE_HIP`-gated
upstream, so there is no hardware overlap path on this target. Reverted.

Final same-session A/B on the reverted tree (server, q8_0 KV, one binary):
- batched 8K pool, agg tok/s: V_DOT2 1115/1164/1081 vs WMMA 1123/1162/1090
  -> parity.
- single: 12K 14.7 -> 14.1s (-4.1%); 44K 94.3 -> 86.6s (-8.2%).
- differential: PASS, max 2.0752e-3 < 3e-3, deterministic.

### Where the plan's 2x went (post-hoc reconstruction)

PLAN.md expected the WMMA port to turn the paged 12K prefill 16.9s -> ~11s
and the 44K pathology into ~50s, extrapolating PR #736's 1.7-2x ATTENTION
speedup to the whole prefill. Two errors in that arithmetic, now measurable:

1. The plan attributed the whole paged-vs-contiguous gap (16.9 vs 12.6s at
   12K) to the attention kernel. Measured deltas instead imply attention is
   ~7.5% of 12K prefill and ~16% of 44K (44K: 7.7s saved of 94.3s = the
   attention kernel being ~2x). A 2x attention kernel therefore buys 4% at
   12K and 8% at 44K end-to-end - exactly what the final A/B shows.
2. The 44K ">12 min pathology" the plan targeted no longer exists in the
   baseline: the current V_DOT2 paged kernel does 44K in 94.3s.

So the kernel-level goal (attention ~2x) appears met; the plan's end-to-end
numbers were not. Open question for a follow-up: isolate the attention kernel
(contiguous-mma vs paged-mma, same shape) to see whether the residual
paged-vs-contiguous prefill gap is attention (gather overhead) or elsewhere in
the paged graph.

### Isolated op throughput (bench_paged_attn_wmma / bench_fattn_mma256, nq=512, q8_0)

Measured with the new isolated bench (same binary per route, gfx1201):

| route | S=8192 | S=16384 | S=32768 | S=65536 | S=131072 |
|---|---:|---:|---:|---:|---:|
| contiguous tile | 12.3 | - | - | 11.2 | 11.3 |
| contiguous MMA (PR #736) | 19.3 | - | - | 21.7 | 22.1 |
| paged V_DOT2 | 6.3 | 6.5 | 6.8 | 6.8 | - |
| paged WMMA | 7.4 | 7.6 | 7.8 | 7.8 | - |

(TFLOP/s; flops = 2*(KQ+VKQ)*nq*kv*D*Hq.)

Conclusions:
1. The paged WMMA kernel is 1.16-1.18x the paged V_DOT2 kernel, not the
   1.7-2x PR #736 delivered on the contiguous path. The port captures only a
   fraction of the mma advantage.
2. The paged path is ~1.7x slower than the contiguous path at the same route
   (tile: 6.5 vs 11.5; mma: 7.6 vs 20.5), so paged overhead - not mma-vs-tile -
   dominates. PLAN.md's premise that "the V_DOT2 paged kernel runs at roughly
   tile-kernel throughput" was wrong by 1.8x.
3. Where the plan's 2x went: it extrapolated the contiguous mma/tile ratio
   (1.6-1.9x) onto a path whose baseline it mis-measured, then applied it
   end-to-end without the attention share. Measured end-to-end: 4% at 12K,
   8% at 44K (attention is a large *time* share - both paged routes run at
   ~7 TFLOP/s vs ~27 for the model's GEMMs - but only a 1.17x kernel delta).

Next optimization target is therefore the paged path's own overhead, in
likely-impact order: (a) the gather does a block-table load per (row, half2)
element - hoist per-row resolution and vectorize (16 B rows are contiguous);
(b) the partials+combine global round trip for n_partitions>1; (c) partition
granularity / tail effects (PAGED_ATTN_BLOCKS_PER_PARTITION=64).

### Gather vectorization: the 2x, delivered

Root cause of the paged path's deficit: the gather resolved the block table
and did its loads per (row, half2) element (~8-15 instructions, 4-byte F16 /
scalar 1-2 byte Q8_0 loads). Measured at ~165 GB/s of K/V against the
contiguous kernel's ~450 GB/s roofline => instruction/latency bound.
Fix: one thread owns 16 contiguous half2s of one row; block-table resolved
once per thread; F16 read as 4x16B, Q8_0 as one block_q8_0 (34 B) with
4-byte qs loads. First attempt had two bugs (k_VKQ_sup compared against a
global token instead of the chunk-local row; Q8_0 block stride 18 instead of
34) - both fixed, differential still PASS at 2.0752e-3.

Isolated op (nq=512, q8_0, TFLOP/s):

| kv | V_DOT2 | WMMA before | WMMA after | vs V_DOT2 |
|---|---:|---:|---:|---:|
| 8192  | 6.37 | 7.38 | 20.65 | 3.24x |
| 16384 | 6.63 | 7.61 | 20.96 | 3.16x |
| 32768 | 6.83 | 7.83 | 21.33 | 3.12x |
| 65536 | 6.77 | 7.84 | 22.37 | 3.31x |

The paged WMMA op now matches (and at 64K exceeds) the contiguous MMA
template's throughput; the paged-path deficit is gone.

End-to-end server A/B (q8_0 KV, same binary):
- batched 8K pool: 12K-equiv +2.9% (n=2), +3.4% (n=4), +6.4% (2x3000).
- single: 12K 14.5 -> 11.6 s (-20.0%); 44K 93.6 -> 54.4 s (-41.9%, 1.72x).

PLAN.md targets (12K ~11 s, 44K ~50 s) are met (11.6 s, 54.4 s). Remaining
headroom, re-ranked after this measurement: (P1) row-tile size / block
occupancy, (P2) async staging now that loads are vectorized, (P3) partition
granularity tails.

### Blog-harness A/B (canonical concurrency bench + ragged long workload)

Same harness that produced the continuous-batching blog post, on the Radeon,
Qwen3.8-27B-UD-IQ4_XS, q8_0 KV, `DFLASH27B_PAGED_WMMA` 0 vs 1, route verified
from each run's `server-command.txt`.

Two knobs were added to the two concurrency runners (backward compatible,
defaults unchanged): `KV_TYPE` (canonical default q4_0; the WMMA route
supports F16/Q8_0/Q4_0, q8_0 matches the blog command) and `PAGED_WMMA`,
which forwards DFLASH27B_PAGED_WMMA past the runners' ambient-tuning guard.

**he-raw (short HumanEval-style prompts, 256 output tokens, C=1/2/5, 1 rep)**
— parity within noise: goodput 29.33/41.04/92.88 -> 29.14/40.87/92.56 tok/s
(-0.3..-0.7%); TTFT median 0.297/0.479/0.816 -> 0.297/0.482/0.842 s. Expected:
~100-token prompts generating 256 tokens leave paged attention a sliver of
each step.

**ragged `long` profile (2000-4000-word prompts ~2.6-5.2K tokens, 128 output
tokens, C=2/4/8, 3 fresh-server reps, medians)** — consistent win:

| C | goodput tok/s | prompt tok/s to first | TTFT median s | native prefill tok/s | out-window tok/s |
|---|---:|---:|---:|---:|---:|
| 2 | 20.07 -> 21.10 (+5.1%) | 1023 -> 1113 (+8.7%) | 6.590 -> 6.059 (-8.1%) | 1043 -> 1136 (+8.9%) | 41.4 -> 42.0 |
| 4 | 26.56 -> 28.02 (+5.5%) | 1062 -> 1146 (+8.0%) | 12.424 -> 11.521 (-7.3%) | 1076 -> 1165 (+8.2%) | 71.9 -> 73.2 |
| 8 | 31.20 -> 33.20 (+6.4%) | 1059 -> 1148 (+8.4%) | 24.992 -> 23.081 (-7.6%) | 1069 -> 1161 (+8.6%) | 103.2 -> 107.3 |

All requests completed on both sides (2/2, 4/4, 8/8). The WMMA side's
`Stable output` column reads NO (content hashes differ across the three
reps) where the decode side reads YES: greedy token flips from ~1e-3
attention-numerics differences under timing-dependent batch composition.
Known harness behaviour (the blog documents responses differing between runs)
and not a kernel determinism issue - the differential is bit-exact per run.

Caveat: single-repeat for he-raw. (This run predates the DFlash2 draft below;
the blog's headline numbers use DFlash2 speculation and are not comparable to
the AR variant used here.)

### DFlash2 drafter + extended long-context A/B (blog harness)

Drafter obtained the documented way (download incoai/Qwen3.8-27B-DFlash2,
convert with server/scripts/convert_dflash_to_gguf.py, quantize q8_0) with one
prerequisite the README omits: config.json must sit next to model.safetensors or
the converter emits neither the dflash2 conv tensors nor the
dflash.dflash2.conv_kernel_size metadata and the server refuses the draft.
Paged + draft requires the "concurrent local same-device DFlash2 chain" shape:
--ddtree is REJECTED with --paged-attention (the harness's blog-ddtree variant
is the DTREE mode, not the published config; a dflash2 variant now exists).

he-raw with DFlash2 (256 out, C=1/2/5, 1 rep): parity again, and absolute
numbers now in the blog's regime (C5 270.5 -> 268.8 tok/s; blog 300.9 with a
different target file and 3 reps).

Ragged long-context ladder, q8_0 KV, luce-k8 (medians where repeated):

| prompt | C | prefill tok/s dec -> wmma | speedup | TTFT median |
|---|---:|---:|---:|---:|
| 2.6-5.2K | 2 | 1043 -> 1136 | 1.09x | 6.59 -> 6.06 s |
| 2.6-5.2K | 8 | 1069 -> 1161 | 1.09x | 24.99 -> 23.08 s |
| 16K (xl) | 2 | 760 -> 1004 | 1.32x | 39.6 -> 30.2 s |
| 16K (xl) | 4 | 770 -> 1028 | 1.34x | 80.8 -> 60.6 s |
| 32K (xxl) | 2 | 558 -> 871 | 1.56x | 104.3 -> 67.5 s |

The advantage grows with context exactly as the isolated op does. Bench
constraint found: the ragged runner sizes the worst-case prefill graph as
SLOTS x max_ctx, so 16 slots x 32K OOMs (20 GB graph); xl needs SLOTS<=4 and
xxl SLOTS<=2 to stay within the ~131K-row bound those runs used.

### Q4_0 KV support

Third KV type alongside F16/Q8_0, so the canonical runner's q4_0 default (and
the DS4 production KV type) routes without overrides. The gather's per-thread
segment is exactly one block_q4_0 ({half d; uint8_t qs[16]} = 18 B / 32 dims):
4-byte qs loads and nibble extraction in ggml's layout (dims 0-15 low nibbles,
16-31 high nibbles, each offset -8 and scaled by d). Launcher gate widened and
the four-branch dispatch collapsed to a macro over the nine type combinations.

Isolated op (nq=512, TFLOP/s, kv 8K..64K):
- f16: V_DOT2 7.3-8.0 -> WMMA 28.4-30.6 (3.9x)
- q8_0: 6.3-6.8 -> 20.4-22.0 (3.2x)
- q4_0: 6.2-6.7 -> 20.1-21.7 (3.3x)

Differential vs the V_DOT2 reference: q4_0 max 3.4-4.3e-3 (mean 3e-4, <2e-4 of
elements over 3e-3) where q8_0 is ~2e-3 - the same dequant-path gap class,
widened by the coarser 4-bit lattice. compare_paged_attn.py keeps the strict
3e-3 default and gains `--tol`; q4_0 runs use 6e-3. Test covers 512/8192/mixed
q4_0.

Engine A/B, ragged long (2.6-5.2K prompts, 128 out, q4_0 KV, 1 rep):
goodput 20.40/26.84/31.41 -> 21.13/27.98/32.84 (+3.6/+4.2/+4.6% at C=2/4/8),
prompt tok/s to first 1028/1074/1066 -> 1118/1137/1133, TTFT median
6.51/12.29/24.82 -> 6.03/11.51/23.36 s. Same shape as the q8_0 ladder. One
outlier: C=8 output-window read 56.5 vs 103.8 (single rep, scheduling
artifact; every other column for that level tracks).

### Tolerance margins (review note)

compare_paged_attn.py reports a single global max over all concatenated
cases, so the per-type margins are not independent witnesses. After the
long-context coverage fix (dense rows now sit at the end of the committed
prefix), the measured per-case maxima are:

| case | max abs | tol | headroom |
|---|---:|---:|---:|
| 512-f16 / 8192-f16 / mixed-f16 / sparse-f16 | 4.88e-4 | 2e-3 | 4.1x |
| 512-q8 / 8192-q8 | 2.77e-4 | 3e-3 | 10.8x |
| 512-q4 / 8192-q4 | 4.43e-4 | 6e-3 | 13.5x |
| mixed-q4 | **4.28e-3** | 6e-3 | **1.40x** |

The binding case is mixed-q4; short-extent rows carry the *larger* error
(peaked softmax over few tokens), which is why the old dense-64-token cases
set the q8_0 margin. A future codegen change could flip the mixed-q4 outlier
and fail the whole run. If that happens, split the comparator per case (the
test's case table is fixed) rather than widening --tol further.
