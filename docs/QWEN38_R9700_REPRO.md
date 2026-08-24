# Reproducing the Qwen3.8-27B R9700 speculative-decode numbers

Everything below was re-derived from scratch on a clean machine (lucebox6) on
2026-08-24 because none of it was written down. The drafter build in §2 is the
part that matters: get it wrong and acceptance length collapses from ~6 to ~1.7,
which makes every throughput number meaningless.

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

## 7. Tuned head-to-head vs upstream #642 (R9700, lucebox6, 2026-08-24)

Each arm tuned independently on `--ddtree-budget`; HumanEval 10 prompts,
`--specla --n-gen 128`, canonical drafter from §2, greedy.

| budget | #642 + IQ4_XS | ours + IQ4_XS | ours + GQH |
|---:|---:|---:|---:|
| 3  | 68.21 (AL 3.46) | - | - |
| 4  | 76.37 (AL 3.95) | 76.39 (AL 3.95) | 65.14 (AL 3.80) |
| 5  | 83.80 (AL 4.40) | - | - |
| 6  | 89.76 (AL 4.79) | 89.92 (AL 4.79) | 65.06 (AL 4.62) |
| 7  | 94.13 (AL 5.17) | - | 100.47 (AL 5.10) |
| 8  | **96.79 (AL 5.83)** | **96.33 (AL 5.83)** | 101.69 (AL 5.10) |
| 16 | 78.22 (AL 5.76) | - | 101.77 (AL 5.10) |
| 22 | 76.34 (AL 5.84) | 76.27 (AL 5.84) | **101.82 (AL 5.10)** |
| 32 | 73.38 (AL 5.99) | 73.41 (AL 5.99) | - |
| 48 | 65.98 (AL 6.03) | - | 101.76 (AL 5.10) |

**Best vs best: ours+GQH 101.82 tok/s at 13,440,110,432 B, versus #642+IQ4_XS
96.79 tok/s at 15,567,824,480 B - +5.2% throughput for 13.7% fewer bytes.**

Two things this table is for.

**The gain is the format, not the stack.** Our arm on IQ4_XS peaks at 96.33 vs
#642's 96.79 - a 0.5% difference, i.e. the two stacks are equivalent and all of
the win comes from the GQH artifact. Matched-budget rows agree even more closely
(76.27 vs 76.34 at 22; 89.92 vs 89.76 at 6; 76.39 vs 76.37 at 4).

**Never compare these two at a single budget.** The arms have opposite tuning
curves. IQ4_XS peaks sharply at 8 and decays hard (96.79 -> 65.98 by 48); GQH is
flat from 7 upward because it self-caps at `ncols=8`, but falls off a cliff
*below* 7 (100.47 at 7, 65.14 at 4) where the fused kernel is not filled. So:

* at the default budget 22, GQH looks **+33%** ahead - flattering, wrong
* at budget 4, GQH looks **15% behind** - unflattering, also wrong
* tuned per arm, the honest figure is **+5.2%**

An independent check: the campaign status doc records +7% on the same pairing
from a separate run, so +5.2% is in family.

### Reproducing exactly this table

    for b in 3 4 5 6 7 8 16 22 32 48; do
      DFLASH_BIN=<arm>/server/build-bench/test_dflash \
      DFLASH_TARGET=<target.gguf> \
      DFLASH_DRAFT=qwen38-dflash2-q8_0.gguf \
      HIP_VISIBLE_DEVICES=0 TMPDIR=$HOME/tmp \
      python3 server/scripts/bench_he.py --specla --n-gen 128 \
        --ddtree-budget $b --skip-tokenize
    done

Patch `test_dflash.cpp` in the upstream arm first (§5) or every run aborts.
Same-arm floor measured over two full repeats: 0.2% or better (56.06/56.17 and
101.78/101.77), so the 5.2% gap is well outside it.
