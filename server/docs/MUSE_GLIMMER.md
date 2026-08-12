# Muse-Glimmer serving (muse-glimmer)

Serves `meta-models/Muse-Glimmer-30B` — a dense 29.8B agentic VLM — as
`general.architecture = "muse-glimmer"`. **LM tower only**; image input needs
the vendor `mmproj` and is not implemented here.

## Platforms

Both GPU backends are first-class and built from one source path — no
CUDA-only primitives are used, and the ATEM/segmenter/tool-parser code is plain
C++ that compiles and runs identically under either.

| Backend | Verified on | Notes |
|---|---|---|
| **CUDA** | **H200 (sm90)** | Full quality gate + parity + throughput. No sm90-specific code; other archs should build but are unverified. |
| **HIP/ROCm** | **gfx1201** (R9700 AI), **gfx1151** (Strix Halo) | Both targets in one binary, ROCm 7.2.2. Spec-decode correctness verified on both (token-identical to greedy); `--draft` is a throughput *loss* here. |

## Artifacts

| Artifact | Bytes | bpw | Qtypes | Needs |
|---|---|---|---|---|
| `muse-v4.gguf` | 16,755,869,152 | 4.807 | stock K-quants (q2_k…q8_0) | nothing beyond this backend |
| `muse-lowbpw-r1.gguf` | 11,499,877,216 | 3.303 | K-quants + 71 tensors at qtype 105/106 | dmix2 sidecar registration (kernels already upstream from the DS4 line) |

Quality (122-item agentic golden suite, scored through this server's
`/v1/chat/completions` so the run exercises its own ATEM rendering, response
segmentation and tool parsing):

| Artifact | via dflash_server | llama.cpp reference band |
|---|---|---|
| `muse-v4` | **105/122** | 103–106 |
| `muse-lowbpw-r1` | **101/122** | 101–103 |

Both land inside their reference bands, so this serving path costs no quality.
`muse-lowbpw-r1` scores AGENT 9/12 against `muse-v4`'s 8.

**Quality is unchanged by GPU vendor and by speculative decode.** The same
122-item suite, `muse-lowbpw-r1`, served through `dflash_server` at pinned
geometry, scored **101/122 on gfx1201 with `--draft` on** against **101/122 on
H200** — identical totals, with the per-axis counts moving by at most one item,
inside the ±2 the suite is noisy to:

| axis | H200 (CUDA, AR) | gfx1201 (HIP, `--draft`) |
|---|---|---|
| MATH | 30/40 | 31/40 |
| CODE | 37/40 | 38/40 |
| TOOL | 25/30 | 24/30 |
| AGENT | 9/12 | 8/12 |
| **total** | **101/122** | **101/122** |

That is the whole point of an output-verified drafter: it is not a quality
trade. Mean acceptance across the suite's 148 requests was **0.314** — higher
than the 0.15–0.24 of open-ended chat, because reasoning and code
continuations are more predictable.

## Feature support

muse-glimmer serves dense AR prefill + decode, `--fa-window`, and DFlash
speculative decode against the vendor drafter (`--draft`, monolithic only).
Everything else is `Never` in `model_capabilities.h` because
`MuseBackendConfig` carries no field for it, and the table is cross-checked
against that struct at compile time so the two cannot drift silently — adding
a config field forces the row to be updated.

| Feature | Flag | muse-glimmer | For contrast |
|---|---|---|---|
| Speculative decode | `--draft` | **Yes** (monolithic). 1.05–1.45× on CUDA; a **slowdown on AMD** — see below | qwen35 both, gemma4 monolithic |
| Draft over IPC | `--draft-ipc-bin` | **No** | qwen35 |
| Draft tree / budget / temp | `--ddtree*` | **No** | qwen35 both |
| Verify width | `--verify-width` | **No** | laguna |
| FA window | `--fa-window` | **Yes** (monolithic) | qwen35, gemma4 |
| Draft SWA | `--draft-swa` | **No** | qwen35 |
| Paged attention | `--paged-attention` | **No** | qwen35 monolithic |
| Layer split / multi-GPU | `--target-devices` | **No** | qwen35, gemma4, deepseek4, laguna |
| PFlash prefill compression | `--prefill-compression` | **No** | qwen35, qwen3 |
| Expert offload | `--spark`, `--adaptive-experts` | **N/A** (dense) | qwen35moe, laguna |
| Park / unpark | — | **No** — returns failure | |
| Snapshots | — | **No** — returns failure | |
| Compress | — | **No** — returns failure | |
| Vision / image input | — | **No** (LM tower only) | |

The unimplemented daemon verbs return failure rather than claiming success:
parking frees weights and KV to hand the GPU to another process, and nothing
here reloads them, so a success reply would strand the daemon in a state it
cannot leave.

### `--fa-window`

Caps how far back the **full-attention** layers look during decode. The SWA
layers are untouched — they already carry the model's own window. Default 0
(unlimited), which is bit-identical to the unwindowed path (verified: 202048 f32
logits compare byte-for-byte with `MUSE_FA_WINDOW=0` against a build predating
the flag). Out-of-range values are rejected at init rather than clamped.

**Treat a non-zero value as unproven on this model.** It has only 13
full-attention layers and they are what carry global context, so windowing them
can drop the system prompt and tool definitions out of view — which is exactly
how tool calling breaks. Nothing here has been scored on the golden suite at any
non-zero window, and the backend prints a warning at startup saying so. It is
also not a free speedup: at a 128-token prompt a 32-token window measured
13.819 ms/tok against 13.837 unwindowed — i.e. nothing beyond run-to-run noise,
because the attention span is trivial at that length. Any benefit is a
long-context effect and is unmeasured.

A/B it without standing a server up:

```bash
MUSE_FA_WINDOW=512 MUSE_N_PROMPT=2048 MUSE_GGUF=… ./bench_muse_decode
```

### Speculative decode — verified-correct on CUDA and HIP; a speedup only on CUDA

`--draft ~/models/muse-glimmer-gguf/dflash-kquant.gguf` loads the vendor
drafter (arch `dflash`, 5 blocks, `n_embd` 6656, block size 16, mask token
201818, capture layers `[2,14,26,38,50]` — all read from the drafter's GGUF,
none hardcoded) and runs greedy-chain DFlash speculation:
`MuseDFlashTarget` (`muse_dflash_target.{h,cpp}`) implements
`common/dflash_target.h`; the loop lives in `muse_backend.cpp`
(`do_spec_decode`). Greedy requests only — a sampling request falls back to AR
rather than silently sampling from the draft. The bonus token is folded into
the next round's verify batch (llama.cpp convention), so each round costs one
target forward, not two.

The rollback is row-scoped, and the reasoning matters because it is what makes
it cheap:

- **Full-attention layers need no saved data at all.** Row index *is* the
  absolute position, so a speculative write never lands on a row an earlier
  position occupies; rolling back is a cursor move.
- **SWA layers save only the rows the span will clobber** — at most
  `depth` rows per layer, as one or two contiguous spans, moved by a single
  ggml graph. ~640 KB at depth 16 against ~110 MB for copying the whole cache
  the way gemma4's adapter does. `rollback_to(base, commit_n)` restores only
  the rejected suffix, so accepted KV survives.

**Measured finding: the data restore is currently redundant.** A speculative
span at `[P+1, P+D]` clobbers slot `(P+i) % S`, and the probe at `P` can only
read that slot if `i >= S - window`, i.e. `i` exceeds the ring headroom. But
`muse_step` already refuses any forward longer than the headroom. So every
speculative span that *can* run touches only rows already outside every future
query's window. The mechanism is kept anyway: it is the correct general
implementation, and it stops being redundant the moment ring sizing, chunk
policy or speculation depth changes. `test_muse_kv_snapshot` therefore proves
the save/restore on cache **bytes** (scribble-and-restore, plus a one-shot
guard) rather than through logits, where a negative control cannot be made to
bite — the first version of that test failed its own control, which is how the
property above was found.

**Correctness is the tested property.** `test_muse_spec_decode` asserts the
speculative output is token-identical to greedy AR: **64/64 identical on all
three GPUs — H200, gfx1201 and gfx1151**. Getting there on gfx1201 took a
better prompt: with word-counting, the target's own margins fell inside that
GPU's (widest) drift band by position 19 and the coverage guard failed the run
rather than let a 19-token demonstration pass as a 64-token one. Numeric
counting keeps the margins outside the band and the check covers the full
horizon. There is one calibrated exception no implementation can remove: ggml
uses a matrix-vector
kernel at one token and a GEMM at many, so one batched verify shifts the logits
against one-at-a-time decode. Where the target's top-2 margin is inside that
drift band, which token wins is kernel-scheduling luck; the test requires
identity up to the first in-band margin and fails any divergence at a wide
margin. Its mechanism control double-norms the drafter's hidden states:
acceptance must drop (proves the projection is on the path) while output stays
identical (proves verification governs).

**The drift is platform-dependent, so the threshold is measured, not fixed.**
Same 16-token batch, same artifact:

| | max Δlogit | rms | K/V max Δ |
|---|---|---|---|
| H200 (CUDA) | 0.42 | 0.082 | 0.41 |
| gfx1151 (HIP) | 0.56 | 0.091 | 0.36 |
| gfx1201 (HIP) | **0.77** | 0.143 | 0.73 |

The AMD kernels drift ~1.8× further than CUDA — the same order as llama.cpp's
own CUDA-vs-CPU rms 0.107 that the graph-parity gate is calibrated against.
`test_muse_verify_capture` therefore uses the drift it measures *in that run*
as its own tie threshold (self-calibrating, and tighter on H200 than the
constant it replaced), and `test_muse_spec_decode`, which drives the backend
and cannot measure drift itself, carries 1.0 to clear all three platforms. A
0.5 constant calibrated on H200 alone sat *below* gfx1201's real drift and
would have reported a legitimate tie-break there as an implementation bug.

**The one convention that mattered: the block mask is BIDIRECTIONAL.** The
generic draft graph masked the noise block causally (the Qwen3-style drafters
here were converted against that), but the DFlash paper (arXiv:2602.06036,
Fig. 2/4: "tokens attend bidirectionally within the same block") and the
reference implementation this artifact was built for (llama.cpp
`models/dflash.cpp`: "cache-aware, non-causal attention") both run the block
bidirectionally. Running the muse drafter causally starved later slots and
cost nothing visible except acceptance: avg_commit 2.91 → **6.40** on a
rigid-continuation prompt the moment the mask matched training. The loader now
keys this off the drafter arch (`DraftWeights::block_bidirectional`). Two more
conventions verified against the reference: NEOX RoPE (measured
indistinguishable from NORMAL, and NEOX is what the reference uses) and RAW
noise embeddings (`ggml_get_rows` of the target table, no norm — measured
within noise of the RMS-normed variant, reference behaviour kept).

**Performance is a CUDA-only win. Do not enable `--draft` on the AMD parts.**
Same code, same drafter, same 5-prompt chat set at 256 tokens; acceptance is
statistically indistinguishable across all three GPUs, so the difference is
entirely the cost of the batched verify:

| GPU | artifact | AR | with `--draft` | ratio | accept |
|---|---|---|---|---|---|
| H200 (CUDA) | muse-v4 | ~70 tok/s | 73–101 | **1.05–1.45×** | 0.15–0.20 |
| gfx1201 (R9700 AI) | muse-lowbpw-r1 | 27.6–27.8 | 13.6–21.1 | **0.49–0.76×** | 0.15–0.24 |
| gfx1151 (Strix Halo) | muse-lowbpw-r1 | 16.7 | 4.7–7.6 | **0.28–0.46×** | 0.14–0.23 |

The 16-token verify forward costs ~1.9 AR steps on H200, ~4.4 on gfx1201 and
~6.6 on gfx1151. Speculation can only pay when a batch of D costs less than the
D AR steps it replaces, and on these AMD kernels it does not — the same
batched-GEMM weakness the dense mix matvec work exposed from the other
direction. The backend prints the measured ratio for its platform at startup.

**It also depends on the artifact, and one pairing is a trap.** Same drafter,
same H200, same prompts; only the target changes:

| target | GB | accept | AR tok/s | draft tok/s | ratio |
|---|---|---|---|---|---|
| bf16 | 55.73 | 0.225 | 55.0 | 121.7 | **2.21×** |
| official kquant-17gb | 16.76 | 0.222 | 73.4 | 114.7 | 1.56× |
| `muse-v4` | 16.76 | 0.182 | 70.7 | 91.1 | 1.29× |
| `muse-lowbpw-r1` | 11.50 | 0.181 | 52.6 | 43.5 | **0.83×** |

Acceptance barely moves across these (0.18–0.23); the ratio moves 2.7×. What
predicts the payoff is AR-step cost ÷ batched-verify cost — bf16 wins *because*
its AR step is expensive enough to amortise the block.

**Do not combine `--draft` with `muse-lowbpw-r1`:** it is a slowdown even on
CUDA. The qtype-105/106 kernels were tuned for the batch-1 matvec path (the
2.06× win recorded below); nothing serves batch 16, so the verify is
disproportionately expensive. Speculative decode currently wants a K-quant
artifact on CUDA.

**Acceptance rate is not a quality signal.** A deliberately broken 9.2 GB
`q2k-pure` control scored the *highest* mean acceptance of any artifact tested
(0.246, above bf16) while emitting 256 tokens of empty visible output — a
degenerate model is trivially draftable. Full write-up in the geo-quant
research log; do not use acceptance to rank artifacts.

**Unsloth GGUFs do not load here** (`GGML_ASSERT(buf != NULL && "tensor buffer
not set")` for both UD-Q4_K_XL and UD-IQ2_XS). The muse loader binds the tensor
set our own and the first-party builds carry; use llama.cpp for those.

On CUDA it is still short of the vendor's 3.1× llama.cpp figure, for two
measured reasons:

1. **A 16-token verify forward costs ~1.9 AR steps on these kernels** (26.7 ms
   vs 14.3 ms after moving verify argmax onto the GPU; it was 33 ms before).
   For a memory-bound dense model it should approach parity — the vendor's
   numbers imply near-parity batch cost. Profiling territory (ncu on the
   batch-16 forward), not read-the-source territory.
2. **The context features are re-projected through the drafter's wk/wv every
   round** (~4.8 ms at 2k ctx). The reference injects them into a persistent
   draft KV cache once per accepted token instead.

Ablations that measured neutral and were left at the reference convention:
drafter RoPE NORMAL vs NEOX, capture-layer shift ±1 (worse both ways),
RMS-normed noise embeddings, target out_norm in the projection (worse, as
expected — the drafter already normalizes).

Diagnostic knobs, all env-gated and off by default: `MUSE_SPEC_DEBUG` (dump
draft/target token arrays per round), `MUSE_DRAFT_NORM_EMBED`,
`MUSE_CAPTURE_SHIFT`, `DFLASH_DRAFT_ROPE`.

## Architecture — a gemma4-shaped model

The KV schema is the gemma4 schema under a different prefix, and every
distinctive gemma4 feature is present: q/k RMS norms, `post_attention_norm` /
`post_ffw_norm`, `attention.sliding_window` + `sliding_window_pattern` (SWA ring
buffer), `final_logit_softcapping` (20.0), `logit_scale` (0.1961), untied
`output.weight`.

Shape: 52 layers, `n_embd` 6656, `n_ff` 19968, GQA 32q/2kv @ head_dim 128,
vocab 202048, window 2048 with 39/52 layers on SWA from a period-8 pattern.

Deltas from gemma4:

1. **`blk.N.attn_gate.weight`** — a per-layer gate on the attention path with no
   gemma4 analogue. The one genuinely new graph node.
2. **Dense only** — no experts; gemma4's MoE, per-layer-embedding and KV-sharing
   machinery is all dead weight here.
3. **ATEM chat format** (below).

### Six details not inferable from tensor names

Ported from llama.cpp's `src/models/muse-glimmer.cpp` — the implementation the
artifacts were gated against — and the parity harness earned its keep on the
last one:

1. input embeddings get an **unweighted** RMS norm before layer 0;
2. post-attention / post-FFN norms use eps **1e-8**, not the model's 1e-5
   `f_norm_rms_eps`;
3. RoPE runs on the **SWA layers only** — full-attention layers are NoPE;
4. the attention gate projects the **pre**-attention normed hidden state, is
   sigmoid'd, and multiplies the attention output **before** `wo`;
5. SDPA scale is the standard `1/sqrt(head_dim)` — gemma4 next door uses 1.0
   because its Q/K norms absorb it; muse does not;
6. rope type is **NORMAL, not NeoX**. Copying gemma4's NeoX measured rms 1.08 /
   max 4.63 against the reference *while leaving argmax intact on a short
   prompt* — i.e. it looks fine until it is benchmarked.

Parity tolerance is calibrated, not guessed: llama.cpp's own CUDA-vs-CPU logits
on this artifact differ by rms 0.107, so the gate is rms ≤ 0.25 plus exact
argmax and top-8 agreement. Measured **rms 0.120** (CUDA ref) / 0.131 (CPU ref),
argmax and top-8 identical, on a full 52-layer forward over the real 16.76 GB
artifact.

### KV cache and SWA ring

Per-layer cache: full-attention layers get `max_ctx` rows, SWA layers a
`swa_size` ring indexed by absolute position % `swa_size`. Appends use
`set_rows` with row indices as graph inputs, so node properties stay stable for
CUDA-graph replay.

The ring is sized **window + chunk headroom**, not window: with ring == window a
prefill chunk evicts rows it still needs to read (measured: chunked vs stepwise
diverged rms 2.11). Occupancy is tracked on the ring, visibility on the window.
The ring mask derives each slot's occupant (largest `p < total` with
`p % S == slot`) rather than assuming a contiguous span, because mid-chunk the
ring holds a rotated view and a start..end span would either mask live rows or
expose stale ones.

## ATEM chat format

`<|start|>role[ to=recipient]<|message|>…<|eom|>`/`<|eot|>`, with
`<atem:function_calls>` blocks and a `to=self` reasoning channel.

- The renderer is native C++, verified **byte-exact against prompts rendered by
  the model's own chat template** across five cases including a tool round-trip
  (`test/data/muse_atem`).
- `AtemSegmenter` splits responses by recipient into reasoning / content /
  tool-call channels, incrementally and token-driven so the streaming and
  non-streaming paths share it.
- `tool_parser.cpp` checks the ATEM dialect **first**, so an argument value
  containing another dialect's opener cannot be half-consumed by a later
  pattern.

Two things that silently break serving if got wrong:

- **Control tokens must be emitted.** Suppressing them measured **TOOL 0/30**.
- **`<|eom|>` ends a segment, not the turn.** Only `<|eot|>`/eos end a turn.
  Treating `<|eom|>` as a stop token truncates every reply at the end of its
  reasoning and returns an empty answer with `finish_reason=stop`.

## qtype 105/106 sidecar (dmix2)

`muse_dmix2.cpp` parses the name-keyed `geoquant.dmix2.sidecar` KV and registers
each resident mix tensor with the `ggml_cuda_rocmfp{2,3}_mix_register_host`
registry the DS4 line already carries (dense ⇒ `n_experts = 1`). Verified on the
real 11.5 GB artifact: 71 entries parsed, 71 registered — on CUDA and on Strix
Halo.

Cover is checked both ways so neither direction fails quietly:

- every **resident** mix tensor must have an entry — a missing one would decode
  against fixed levels, which is plausible output from the wrong numbers;
- every entry must name a mix tensor **of the file** rather than of this load, so
  a layer-split target legitimately covering a subset is not mistaken for drift.

**Host-resident mix tensors are refused by name.** Decode for these qtypes is
GPU-only, and on unified memory (Strix Halo) a host pointer may read something
valid-but-wrong instead of faulting — so the refusal cannot be left to a
segfault. Full offload of every mix layer is mandatory.

Other fail-closed behaviour: the **lm_head is untied**, and falling back to
`token_embd` would still emit fluent text, so the loader refuses; the
**SWA pattern array is 8 long for 52 layers**, and read as per-layer, 44 layers
would silently get the wrong span.

## Decode throughput

`bench_muse_decode` — prefill once, time N single-token steps, no sampling.
Reproduces the golden suite's throughput figure in ~30 s instead of ~80 min.

```bash
MUSE_GGUF=/path/muse-lowbpw-r1.gguf MUSE_N_DECODE=64 ./bench_muse_decode
```

Batch 1, current kernels:

| | H200 (CUDA) | gfx1201 | gfx1151 |
|---|---|---|---|
| `muse-v4` | 72.4 tok/s | — | — |
| `muse-lowbpw-r1` | 53.6 tok/s | 28.4 tok/s | 16.5 tok/s |

### The dense mix matvec had never been tuned for a dense model

`muse-lowbpw-r1` decoded at 26.0 tok/s. ncu on `mix_matvec_rocmfp3_kernel`
measured **1.65 % DRAM throughput, 97 % L1 throughput, 98.4 % L1 hit rate**, and
56 % of warp cycles stalled on **LG throttle** — not bandwidth bound, bound on
the *number* of narrow global load instructions. SASS confirmed it: **320 32-bit
`LDG.E.CONSTANT` for activations against 35 `LDG.E.U16` for weights**, i.e. 90 %
of load issue was the activation vector, entirely unvectorized.

Two bit-exact changes: stage each block's 32 contiguous activations with
`float4` (8 loads, not 32), and give each warp two output rows so one activation
stage feeds two accumulators.

The second is not new — the MoE and 3-D-slice kernels in the same files have
done it since the DS4 line, for this exact reason. **The dense 2-D kernel never
got it, because the model that family was built for is MoE and never took that
path.** Muse-Glimmer is the first dense consumer of qtype 105/106.

Result: **2.06× decode**, kernel 151 → 40 µs, global load instructions 1.12 M →
318 k, warp cycles per issued instruction 59.5 → 10.7, no register spilling.
`muse-v4` untouched (13.81 vs 13.77 ms/tok). Bit-exactness is verified, not
asserted: 202048 f32 logits compare byte-for-byte identical before and after via
`MUSE_DUMP_LOGITS`.

On AMD the same change is **neutral** — gfx1201 −1.9 %, gfx1151 −0.4 %. The
load-issue ceiling that dominates on NVIDIA is not what binds on wave32 RDNA,
and the AMD kernel was already hand-tuned (load-from-floor wide staging, an
`amdgpu_waves_per_eu(12,12)` pin against a measured occupancy cliff). It does
not regress either, so one path serves both and no per-backend fork is
warranted.

**Corollary: the AMD side is now the laggard.** Before this change the AMD parts
beat CUDA on their own kernels (35.8 vs 38.5 ms); after it CUDA is nearly 2×
faster than the R9700 AI (18.7 vs 35.2 ms). The ROCmFPX line has untapped
headroom on AMD, bound by something this change does not touch — and finding it
wants the same method that worked here: profile first (rocprof / omniperf), do
not reason from the source.

Known remaining headroom on the kernel:

- Partly **launch-bound on small tensors** — ncu reports 0.6 full waves for
  `attn_gate` (out=4096), since doubling rows-per-warp halved the grid. A
  rows-per-warp choice made at launch from `out` would suit both ends; it is
  bit-exact either way (verified: 2-row output == 1-row output).
- **fp3 weight staging is still 7 × `LDG.U16` per 14-byte block.** fp2 already
  does load-from-floor wide staging. The fp3 analogue needs a 24 B window (14 B
  block at a 2-aligned offset), which overruns the tensor end by up to 10 B, so
  it needs the same registration-time shape guard fp2 carries — not a bare port.

## Tests

| Test | Covers | Needs artifact |
|---|---|---|
| `test_muse_atem_chat_template` | ATEM grammar units | no |
| `test_muse_atem_reference_prompts` | byte-exact vs the model's own template | no |
| `test_atem_stream` | response segmentation + ATEM tool dialect | no |
| `test_muse_loader` | hparams, SWA pattern, tensor map, refusals | yes (`MUSE_GGUF`) |
| `test_muse_graph_parity` | full 52-layer forward vs llama.cpp logits | yes + `MUSE_REF_LOGITS` |
| `test_muse_generate` | prefill-N == prefill-(N-1) + 1 step (**bit-identical**) | yes |
| `test_muse_kv_snapshot` | row-scoped save/restore on cache bytes + ring wrap | yes |
| `test_muse_verify_capture` | verify == sequential (argmax, KV bytes, logit drift) + capture coverage | yes |
| `test_muse_spec_decode` | spec output token-identical to greedy AR + acceptance | yes + `MUSE_DRAFT_GGUF` |
| `bench_muse_decode` | decode throughput | yes |

Artifact-dependent tests skip with 77 when `MUSE_GGUF` is unset.
`test_muse_generate` loads on CPU for its reference comparison, so it cannot run
against the 105/106 artifact by design (decode for those qtypes is GPU-only).

## Known gaps

- Everything in the feature table above.
- **Speculative decode is 1.05–1.45× on H200 against the vendor's 3.1×**
  (verified-correct; see the section above). The two measured gaps: batch-16
  verify at ~1.9 AR-step cost (profile with ncu), and per-round feature
  re-projection instead of a persistent draft KV cache.
- ~~No end-to-end forward has exceeded the 2048-token window.~~ **Closed:**
  `test_muse_kv_snapshot` builds a cache with `ring < max_ctx` and prefills
  `ring + 32` tokens (2144 against a 2112 ring), so the ring genuinely wraps and
  the chunked prefill is exercised across the wrap.
- The quality gate has been run on CUDA only; the HIP builds are verified for
  load, decode and mix registration but not scored.
- One 1.1 s first-timed-step outlier on Strix Halo did not reproduce with a
  longer warmup; medians were stable to ±0.2 %. Not root-caused.
