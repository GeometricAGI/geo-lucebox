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

## Staged plan

- **S1 — recognition (no kernels, no graph).** `gguf_inspect` + `model_card`
  + `chat_template` + capability-table row for `muse-glimmer`; a test that
  parses the real artifact's KV (52 / 6656 / 32q-2kv / softcap 20.0) and
  asserts the family resolves. Cheap, testable, and it is what tells us the
  artifact is legible to the server at all.
- **S2 — loader.** Tensor map for the muse topology including `attn_gate`,
  reusing the gemma4 loader's norm/SWA/softcap plumbing.
- **S3 — graph.** Dense 52-layer forward with the attn-gate node, softcap,
  logit scale. Numerical parity target: logits vs `llama.cpp` on the
  `muse-rocmfpx-cuda` branch for a fixed prompt.
- **S4 — qtype 105/106.** Register the dmix2 sidecar (KV-embedded, name-keyed)
  through the existing `ggml_cuda_rocmfp{2,3}_mix_register_host` registry that
  the ds4 line already carries; full-offload refusal identical to ds4.
- **S5 — gate parity.** Run the 122-item agentic golden suite against
  lucebox-served muse-v4 and compare with the llama.cpp bands
  (muse-v4 103–106, muse-lowbpw-r1 101–103, official 100–101, bf16 103).

S1–S3 make `muse-v4` (the byte-parity artifact that beats the vendor GGUF)
servable with no kernel work at all. S4 adds the 3.30 bpw artifact.
