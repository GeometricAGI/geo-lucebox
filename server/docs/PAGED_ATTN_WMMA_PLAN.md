# Paged attention WMMA plan (head-256, RDNA4)

Follow-up to PR #736 (tensor-core fattn). Target: the continuous-batching
path (`--paged-attention`), which routes full-attention layers through
`ggml_paged_attn_ext` (paged-attn.cu) and never reaches the fattn dispatch
PR #736 changed.

## Evidence (gfx1201, Qwen3.8-27B UD-IQ4_XS, q8_0 KV)

| Shape | Contiguous prefill | Paged prefill |
|---|---:|---:|
| 12.2K prompt | 12.6 s (tile) / 12.9 s (MMA) | **16.9 s** |
| 44K prompt | 71 s (tile) | **> 12 min** (killed) |

The paged path is 1.3x slower at 12K and >10x slower at 44K. Root causes,
from reading paged-attn.cu:

1. **No tensor cores.** Scores use `multi_vec_dot_kq_*` (V_DOT2_F32_F16,
   2 MAC/lane/cycle); the V accumulation is `dequantize_v` + scalar FMA.
2. **Lane-per-token layout** (each lane owns one token's score) makes a
   cross-lane WMMA conversion impossible inside the existing loop.
3. **Partition re-reads:** context splits over blocks (64 per partition),
   each partition re-dequantizes its K/V rows; the packed prefill chunk
   (512 rows) multiplies the partition grid.

## Design: paged_attn_wmma<D=256, type_K, type_V>

Mirror fattn-mma's proven fragment machinery (mma.cuh tiles, WMMA ops,
online-softmax reduction) with the block-table gather replacing the
contiguous K/V strides:

- Block = (sequence, partition). 4 warps, ncols=32 token columns (2
  fragments of 16), ncols2=1 — the paged path has no GQA batching, each
  warp owns a KV head group via `kv_head` like today.
- K tile staging: per 16-token fragment, resolve the block table
  (`block_table[logical_block * bt_nb0 + physical_seq * bt_nb1]`) into
  shared memory, one stage per pipeline step — same partition/token-end
  logic as `paged_attn_decode` (reuse `paged_attn_partitions`,
  `paged_attn_tree_visible` verbatim).
- Q fragments loaded once per block from the quantized-Q global buffer
  (`paged_attn_quantize_q`) for q8_0/Q4_0, or raw f16 for f16 KV.
- V accumulation: WMMA VKQ like fattn-mma, weights read back from the
  KQ shared tile.
- Partial merge for multi-partition contexts unchanged (write_partials +
  the existing merge kernel).
- Scope stage 1: non-tree path, head-256, F16 + Q8_0 KV, min_partitions
  path only (prefill+decode rows); tree and Q4_0 stay on the V_DOT2
  kernel. Dispatch: env-gated `DFLASH27B_PAGED_WMMA=1` until qualified,
  same rollout as PR #736.

## Qualification

- Differential test vs the V_DOT2 paged kernel (same graphs, both routes
  via the launch counter) and vs the CPU reference for small shapes.
- The canonical concurrency harness (he-raw, C=1/2/5) plus the long-prompt
  paged TTFT ladder (12K/44K) A/B.

## Expected

The V_DOT2 paged kernel runs at roughly tile-kernel throughput; the WMMA
conversion should recover the same 1.7-2x attention speedup PR #736
delivered for the contiguous path, applied to the batched prefill slope
(16.9s -> ~11s at 12K, and the >10x 44K pathology back to ~50s).
