# Muse-Glimmer serving plan (geo-lucebox)

Target: serve the geo-quant Muse-Glimmer-30B GGUF artifacts on lucebox.

## The artifacts

| Artifact | Bytes | bpw | Qtypes | Kernel work needed |
|---|---|---|---|---|
| `muse-v4.gguf` | 16,755,869,152 | 4.807 | **stock K-quants only** (q2_k…q8_0) | **none** |
| `muse-lowbpw-r1.gguf` | 11,499,877,216 | 3.303 | K-quants + 71 tensors at qtype 105/106 | dmix2 sidecar registration (kernels already upstream) |

Both are LM-only (52 layers); image input needs the vendor mmproj. Gate
evidence and reproduction: `geo-quant` repo, `docs/runbooks/museglimmer_gguf_rewrite.md`.

## Architecture — it is a gemma4-shaped model

`general.architecture = "muse-glimmer"`. The KV schema is the gemma4 schema
under a different prefix, and every distinctive gemma4 feature lucebox already
implements is present:

| Feature | gemma4 backend | muse-glimmer artifact |
|---|---|---|
| q/k RMS norms (`attn_q_norm`, `attn_k_norm`) | yes | yes (parameter-free upstream → written as F32 constants) |
| `post_attention_norm`, `post_ffw_norm` | yes | yes |
| `attention.sliding_window` + `sliding_window_pattern` (SWA ring buffer) | yes | yes |
| `final_logit_softcapping` | yes | yes (20.0) |
| `logit_scale` | yes | yes |
| untied `output.weight` | yes | yes |

Deltas to implement:

1. **`blk.N.attn_gate.weight` (4096×6656)** — a per-layer gate on the
   attention path that gemma4 has no analogue for. This is the one genuinely
   new graph node.
2. **Dense only** — no experts; the MoE branches of the gemma4 graph are dead
   weight for this family.
3. **GQA 32q/2kv** with `key_length`/`value_length` from KV.
4. **Chat format** — ATEM (`<atem:function_calls>`, `to=self` reasoning
   channel, `<|eom|>`/`<|eot|>`). Control tokens MUST be emitted; suppressing
   them measured TOOL 0/30 on the golden suite.

## AMD / HIP

The target hardware includes Strix Halo (gfx1151, unified memory) and the
R9700 AI (gfx120x), so HIP is a first-class build for this family, not a
port afterthought:

- S1 adds no device code — the ATEM renderer and its tests are plain C++,
  compiled and run identically under both backends.
- S2/S3 must stay inside the ops the HIP backend already carries for gemma4
  (that family builds and runs on HIP today), so the muse graph should reuse
  its node vocabulary rather than reaching for CUDA-only primitives.
- S4 inherits the ROCmFPX mix kernels, which are HIP-native by origin (the
  DS4/Strix line) and carry `.hip.cu` paths plus wave64-safe reductions
  upstream. The 128-weight row-alignment and full-offload refusals are the
  same on both backends.
- Strix Halo caveat carried from the ds4 line: unified memory means "GPU
  resident" and "host resident" are not the memory-space distinction they
  are on discrete parts — the mix registry's host-resident refusal must be
  re-verified there rather than assumed from the CUDA result.

## Staged plan

- **S1 — recognition (no kernels, no graph). DONE.** `ChatFormat::ATEM`
  (native renderer, `muse-glimmer` → ATEM dispatch), `model_card` family
  row, and `ChatMessage::name` so tool turns can be addressed by name.
  Verified byte-exact against prompts rendered by the model's OWN chat
  template (`test/data/muse_atem`, five cases incl. a tool round trip):
  `test_muse_atem_reference_prompts` + a grammar unit test. Both are
  CPU-only and backend-agnostic, so they run identically under the CUDA and
  HIP builds. The capability-table row lands with S2 — it is cross-checked
  at compile time against a backend config struct that does not exist yet.
- **S2 — loader. DONE.** `src/muse/{muse_internal.h,muse_loader.cpp}`:
  hparams, interleaved-SWA pattern, tokenizer ids, and the per-layer tensor
  map including `attn_gate`. Dense-only — none of gemma4's MoE / per-layer
  embedding / KV-sharing machinery. Verified against BOTH shipped artifacts
  (`test_muse_loader`, `MUSE_GGUF=…`, skips without one): 52 layers, 6656 /
  19968, 32q-2kv @128, vocab 202048, window 2048, softcap 20.0,
  logit_scale 0.1961, and 39/52 SWA layers from the period-8 pattern. Two
  traps it closes: the pattern array is 8 long for 52 layers (read as
  per-layer, 44 layers silently get the wrong span), and the lm_head is
  UNTIED — falling back to `token_embd` would still emit fluent text, so
  the loader refuses instead. `<|eom|>` has no KV key and is resolved from
  the token list (200007); without it the ATEM sampler cannot end a
  reasoning turn or a chained tool call.
  Note for S4: the 105/106 artifact already loads here — lucebox's ggml
  knows those qtypes — so S4 is sidecar registration, not type plumbing.
- **S3 — graph builders. DONE (block level).** `src/muse/muse_graph.cpp`
  ports the semantics from llama.cpp's `src/models/muse-glimmer.cpp` — the
  implementation the artifacts were gated against — as
  `build_muse_{inp_norm,attn_block,layer,head}`. **Parity verified on the
  real 16.76 GB artifact**: full 52-layer forward, argmax and top-8 identical
  to llama.cpp, rms 0.120 (CUDA ref) / 0.131 (CPU ref).

  Six details are not inferable from tensor names, and the harness earned
  its keep on the last one:
  1. input embeddings get an UNWEIGHTED RMS norm before layer 0;
  2. post-attention / post-FFN norms use eps **1e-8**, not the model's
     1e-5 `f_norm_rms_eps`;
  3. RoPE runs on the **SWA layers only** — full-attention layers are NoPE;
  4. the attention gate projects the PRE-attention normed hidden state, is
     sigmoid'd, and multiplies the attention output BEFORE `wo`;
  5. SDPA scale is the standard `1/sqrt(head_dim)` — gemma4 next door uses
     1.0 because its Q/K norms absorb it; muse does not;
  6. rope type is **NORMAL**, not NeoX. Copying gemma4's NeoX measured
     rms 1.08 / max 4.63 against the reference while leaving argmax intact
     on a short prompt — i.e. it looks fine until it is benchmarked.

  Tolerance is calibrated, not guessed: llama.cpp's own CUDA-vs-CPU logits
  on this artifact differ by rms 0.107, so the gate is rms <= 0.25 plus
  exact argmax and top-8 agreement.
- **S3b — KV cache + step driver. DONE.** `src/muse/muse_step.cpp`:
  per-layer cache (full layers `max_ctx` rows, SWA layers a `swa_size` ring
  indexed by absolute position % swa_size), per-step full and ring masks,
  `set_rows` append with row indices as graph inputs (stable node properties
  for CUDA-graph replay), and `muse_step()` covering prefill and decode.
  `test_muse_generate` checks that one prefill of N tokens and
  (prefill N-1 + one decode step) produce the SAME logits — same backend,
  so any divergence is a cache/mask/position bug, not reduction order.
  Measured: **bit-identical (max|d| = 0.0)**, and the incremental path also
  matches the llama.cpp reference (rms 0.131).

  The ring mask derives each slot's occupant (largest p < total with
  p % S == slot) rather than assuming a contiguous span, because mid-chunk
  the ring holds a rotated view and a start..end span would either mask live
  rows or expose stale ones. **Honest limit:** the end-to-end runs so far all
  had `swa_size == max_ctx`, so the ring has not actually wrapped in a full
  forward — wrapping needs a prompt longer than the 2048-token window. The
  wrap arithmetic is unit-tested model-free (`muse_swa_slot_visible`,
  including the mid-chunk future-slot and eviction cases); a >2048-token
  forward is still owed.

Remaining for a servable path: the ModelBackend implementation (parking,
snapshots, drafter hooks), daemon wiring, backend factory + capability-table
row, and batching/CUDA-graph replay tuning.
- **S4 — qtype 105/106 sidecar. DONE.** `src/muse/muse_dmix2.cpp` parses the
  name-keyed `geoquant.dmix2.sidecar` KV and registers each resident mix
  tensor with the `ggml_cuda_rocmfp{2,3}_mix_register_host` registry the ds4
  line already carries (dense ⇒ `n_experts = 1`). Verified on the real
  11.5 GB artifact: 71 entries parsed, resident tensors registered.

  Cover rules, chosen so neither direction can go wrong quietly:
  every RESIDENT mix tensor must have an entry (a missing one would decode
  against fixed levels — plausible output from the wrong numbers), and every
  entry must name a mix tensor **of the file** rather than of this load, so a
  layer-split target legitimately covering a subset is not mistaken for
  drift. Host-resident mix tensors are refused by name: decode is GPU-only,
  and on unified memory (Strix Halo) a host pointer may read something valid
  but wrong instead of faulting.
- **S5 — gate parity. DONE (CUDA).** The 122-item agentic golden suite against
  lucebox-served artifacts reproduces the llama.cpp bands: **muse-v4 105/122**
  (band 103–106) and **muse-lowbpw-r1 101/122** (band 101–103). Notably r1
  scores AGENT 9/12 against v4's 8 — the 3.30 bpw artifact holds multi-step
  agentic behaviour. AMD is still owed.

## Decode throughput: the dense mix matvec was never tuned for a dense model

`bench_muse_decode` (prefill once, time N single-token steps, no sampling)
reproduces the golden suite's throughput in ~30 s instead of ~80 min, which is
what made the following tractable.

Measured on an H200, muse-lowbpw-r1:

| | decode | prefill |
|---|---|---|
| before | 25.95 tok/s (38.53 ms/tok) | 499 tok/s |
| after | **53.56 tok/s (18.67 ms/tok)** | 502 tok/s |

**2.06× decode, and the output is bit-identical** — 202048 f32 logits compare
byte-for-byte equal between the two builds (`MUSE_DUMP_LOGITS`). Quality is
therefore unchanged by construction, not merely within a noise band. muse-v4
is untouched (13.81 vs 13.77 ms/tok).

The diagnosis is worth recording because it inverts the intuition for a
quantized kernel. ncu on `mix_matvec_rocmfp3_kernel` measured **1.65% DRAM
throughput, 97% L1 throughput, 98.4% L1 hit rate**, and 56% of all warp cycles
stalled on **LG throttle**. It was not bandwidth bound at all — it was bound on
the *number* of narrow global load instructions. SASS confirmed it exactly:
**320 32-bit `LDG.E.CONSTANT` for activations against 35 `LDG.E.U16` for
weights**, i.e. 90% of load issue was the activation vector, entirely
unvectorized.

Two bit-exact changes:

1. **float4 activation staging.** The 32 activations a block consumes are
   contiguous (128 B), so they cost 8 loads instead of 32. Same values, same
   ascending-j fold.
2. **Two output rows per warp.** Both rows consume the SAME activations, so one
   stage feeds two accumulators — halving activation load issue per unit work.

The second is not a new idea: the MoE and 3-D-slice kernels in the same files
have done it since the DS4 line, for this exact reason. **The dense 2-D kernel
never got it, because the model this family was built for is MoE and never took
that path.** Muse-Glimmer is the first dense consumer of qtype 105/106.

Result on the kernel itself: 151 µs → 40 µs, global load instructions 1.12 M →
318 k, warp cycles per issued instruction 59.5 → 10.7, no register spilling
(`local_ld` = 0). L1 dropped from 97% to 54% and nothing is saturated now.

Remaining headroom, for whoever picks this up:

- The kernel is now partly **launch-bound on small tensors** — ncu reports
  0.6 full waves for `attn_gate` (out=4096), and doubling rows-per-warp halved
  the grid. A rows-per-warp choice made at launch from `out` would suit both
  ends; it is bit-exact either way (verified: 2-row output == 1-row output).
- **fp3 weight staging is still 7× `LDG.U16` per 14-byte block.** fp2 already
  does load-from-floor wide staging (2 × u64 + funnel shift). The fp3 analogue
  needs a 24 B window (14 B block at a 2-aligned offset), which overruns the
  tensor end by up to 10 B — so it needs the same registration-time shape guard
  fp2 carries, not a bare port. Deliberately not attempted here.
- **These changes are untested on AMD.** The shapes they replace were tuned on
  wave64/CDNA-era parts and the fp2 kernel still carries an
  `amdgpu_waves_per_eu(12,12)` pin chosen against a measured occupancy cliff.
  Wider loads and register blocking *should* help there too, but that is a
  hypothesis until measured on gfx1151/gfx1201 — and a divergent result is the
  case for genuinely forking the launch config per backend.

S1–S3 make `muse-v4` (the byte-parity artifact that beats the vendor GGUF)
servable with no kernel work at all. S4 adds the 3.30 bpw artifact.
