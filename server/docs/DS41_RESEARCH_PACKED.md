# DS4.1 mixed research checkpoint bridge

The `deepseek41` loader can attach a geo-quant research checkpoint to a GGUF
containing the dense weights, model metadata, vocabulary and engram hash layout.
Routed experts are authenticated and loaded once into shared per-layer GPU allocations, individually packed, onto the
chosen backend. The production attention, routing and decode graphs dispatch
through a packed-expert operation whose descriptors hold stable weight pointers.
Changing routing IDs does not reload or re-quantize weights. Shared allocations avoid the large per-allocation GPU granularity overhead of tens of thousands of small expert buffers. The research runner's
per-token matrix cache is not involved.

Supported expert payloads: GQH ternary (16/4, 32/4 and 32/3 scale layouts), GQH2_H,
GQH3, GQH4, INT3 group 64/128, NVFP4 and native MXFP4. Existing AWQ compensation
is applied once, before BF16 rounding, using the authenticated artifact's scales.
Native FP8 experts are rejected until a scaled binding is implemented. Each matrix
retains its selected format; heterogeneous layers need no uniform-format padding.

This first integration requires a complete, resident, single-device load. It does
not implement cross-device expert placement, streaming experts, or a shipping GGUF
encoding for these descriptors. Descriptors contain runtime device pointers and
must never be serialized or migrated between backends. Their owning weights live
until every referencing graph has been released.

## Execution and storage contract

Dense weights are decoded with geo-quant's reference codec into the envelope.
Small normalization/routing tensors are widened exactly to F32. Expert weights
are decoded to BF16 values within the routed product, with F32 accumulation.
This preserves decoded **weights**, but is a new production execution contract:
it does not claim the native research runner's FP8 activation rounding or identical
whole-model output. Evaluate the engine against the frozen golden suite before
using its results to make quantization quality claims.

Native engram weight and scale ranges are resolved from the pinned native index,
validated against safetensors metadata, and read with bounded `pread` calls. No
extra interleaved copy of the roughly 189 GiB tables is needed. This path uses the
OS page cache. The artifact must remain immutable while a server uses it; the
native index identity pins the mapping, not the content of every native shard.

The envelope is an additional dense-weight file. It does not reduce artifact
storage or establish that a Lucebox deployment fits. Measure runtime residency,
workspace and KV memory separately on the target hardware.

## Build and export

Build with the normal `LUCE_GPU_BACKEND=cuda` or HIP configuration. PCRE2 (`pcre2-8`
headers and library) enables the exact three-pass DS4.1 Unicode pre-tokenizer.
A build lacking PCRE2 explicitly rejects this tokenizer instead of silently using
Qwen splitting. OpenSSL is optional and accelerates one-time payload SHA256 checks.

Use a geo-quant checkout that provides the DS4.1 research codecs and an authenticated
binding plan for the artifact:

```sh
python server/scripts/export_ds41_research_gguf.py \
  --artifact /data/research-artifact \
  --plan /data/model-plan.json \
  --engram-layout /data/engram/layout.bin \
  --engram-layout-sha256 <layout-sha256-from-manifest> \
  --geoquant /path/to/geo-quant \
  --output /data/production/model.gguf
```

The exporter refuses to overwrite an existing envelope, validates source identities,
and writes through a partial file. The artifact path is relative to the envelope.
Move both together or export another envelope when staging elsewhere. Temporary
dense tensor spooling requires additional disk space during export.

```sh
CUDA_VISIBLE_DEVICES=0 server/build/luce_server /data/production/model.gguf \
  --target-device cuda:0 --host 127.0.0.1 --port 8094 \
  --max-ctx 16384 --chunk 32 --max-concurrency 1
```

## Validation

`test_packed_experts` compares mixed-format CPU/CUDA products with the existing
independent row decoders, including non-identity AWQ, changed and duplicate routes,
multiple tokens, broadcast/per-route inputs, and repeated execution of one graph.
`test_packed_arena` checks allocation footprints, disjoint weights/scales and borrowed lifetimes.
`test_ds4_engram` checks native split-table reads against interleaved rows, threaded
reads, moves, overflow and overlap rejection. The existing conditioning tests cover
registry ownership and lifetime. `test_ds41_hc_f32` checks native F32 and legacy
F16 hyper-connection projections against an independent CPU reference, including
single-token and batched execution. Native F32 controllers remain F32 in both
the CUDA projection and CPU fallback.

```sh
ctest --test-dir server/build -R 'test_packed_(experts|arena)_|test_gqh_compensation_|DeepSeek4EngramFixture' --output-on-failure
python server/tests/test_ds41_tokenizer_parity.py \
  --harness server/build/test_tokenizer_harness \
  --gguf /data/production/model.gguf \
  --tokenizer /data/research-artifact/tokenizer.json \
  --requests /data/frozen/requests-gpu-*.json
```

The DS4 sparse D=512 attention implementation is available to CUDA and HIP.
`ds41_attention_cuda` compares masked/indexed F16 and F32 cache attention,
including sink normalization, with an independent scalar oracle. It covers
255/256/257 and 2048 cache rows, single-token and batched queries, and changed
inputs on graph replay. `ds4_maskless_prefill_*` additionally checks explicit
masks against analytic visibility, inverse RoPE, compressed-cache ratios 1/2/4,
and long-context selected-row schedules on either backend.

```sh
ctest --test-dir server/build -R 'ds41_attention_cuda|ds4_maskless_prefill_' --output-on-failure
```

CUDA/H200 is the initial execution target. A successful CUDA gate does not qualify
HIP, Lucebox placement, concurrent paged execution, or whole-model golden quality.

## CUDA paged concurrency (from the DS4.1 CUDA research branch)

Resident single-GPU CUDA serving also accepts `--paged-attention
--max-concurrency 2`. It retains native F32 hyper-connection controller
projections in gathered graphs. CUDA tensor/expert parallel placement and
legacy `deepseek4` CUDA paged serving remain unsupported. Allow GPU memory
for two sequences and the gathered graph scratch space in addition to the
resident weights.

The optional CUDA/HIP `test_ds4_paged_prefill_model` accepts a JSON file
containing exactly two requests with `token_ids` as its second positional
argument after the GGUF. With `LUCE_DS4_SPEC=0`, it checks 140-token
concurrent-versus-serial prefixes and chronological prefill reconstruction
at tokens 53 and 129. This checks consistency within paged serving;
separate golden evaluation is still needed against the non-paged engine.

On an H200 with the tight128 mixed research artifact (2026-09-26), the
two-slot CUDA pilot passed the frozen coding, GPQA, and AIME questions.
It produced 3,379 tokens in 311.2 seconds, or 10.9 aggregate output
tokens/s including prefill. The non-paged single-slot baseline produced
2,168 tokens in approximately 306.7 seconds on the same three prompts.
Longer generated reasoning erased the throughput improvement on this
small pilot. These runs are separate evaluation cohorts; two-slot CUDA
is not yet established as a faster replacement for the full quality run.

