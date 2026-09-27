# Resident CUDA decode profiling (2026-09-27)

The tight128 mixed research artifact was measured on one H200 while seven
other H200s continued golden evaluation. Expert weights remain resident;
engram tables use the existing host reader. No quantization or sampling
changes are involved in these optimizations.

## Findings

An initial 12-second Nsight capture recorded 3,266 `cudaMalloc` and 3,287
`cudaFree` calls, with about 1.97 seconds spent in those API calls. This is
API time, not an additive decomposition of wall time: synchronization can
include outstanding GPU work. Attention graph keys change with compressed
history size and previously created new device workspaces for these shapes.

A subsequent node-level CUDA-graph capture showed the generic packed-expert
kernel accounting for 83.4% of GPU kernel time in the sampled short-context
decode window. This percentage excludes CPU time and is not a claim that
83.4% of end-to-end latency is in that kernel.

## Changes and validation

* Native controller normalization keeps its ordered FP32 reduction, division,
  square root and reciprocal on the GPU. It removes an intermediate host
  readback. FP16/F32 controller outputs match the diagnostic legacy path
  bit-for-bit; two frozen 128-token generations also match exactly.
* `LUCE_DS41_REUSE_ATTN_WORKSPACE=1` reuses a layer's attention allocator and
  reserves capacity in 4 MiB increments. Old native graph captures are
  invalidated before reuse. The existing cache budget still applies. In the
  matched 64-token capture, allocation/free calls fell from 2,432/2,436 to
  zero. Both frozen 128-token generations match exactly.
* `LUCE_DS41_TYPED_EXPERTS=1` dispatches by format outside the weight loop,
  allowing constant-format decoding without changing the dot-product
  accumulation or reduction order. CPU-oracle and exact generic-versus-typed
  GPU tests cover all 12 supported formats, conditioning scales, duplicate
  and changing routes, and broadcast/per-route inputs. Two frozen 128-token
  generations match exactly.

The controller optimization defaults on for CUDA; set
`LUCE_DS4_HC_DEVICE_RMS=0` to reproduce the legacy path. Workspace reuse and
typed expert kernels are opt-in while extended-context canary validation
runs. Do not toggle the typed-kernel setting inside a process with captured
CUDA graphs: start a new server with the desired setting.

## Fixed-output measurements

| Comparison | Reference decode | Optimized decode | Scope |
| --- | ---: | ---: | --- |
| Controller normalization | 31.682 s | 31.352 s | Combined two 128-token prompts; about 1% latency reduction |
| Attention workspace reuse | 15.721 s | 11.807 s | Untraced coding prompt; 8.14 to 10.84 tokens/s |
| Typed experts, workspace reuse enabled | 22.229 s | 18.934 s | Combined two 128-token prompts; 11.52 to 13.52 tokens/s |

For the typed comparison, CUDA graphs were disabled in both arms so a
captured launch could not retain the previous kernel setting. The roughly
13.4–13.6 tokens/s result is a short-context fixed-output measurement, not
an established long-context or Lucebox throughput figure. Prefill also
improved in that comparison. Request order was reversed for the second
prompt to reduce cache-warmth bias.

## Reproduction

Build `test_ds41_hc_f32` and `test_packed_experts` in the CUDA configuration.
The controller test runs without model arguments; the optional full-model
comparisons are:

```sh
LUCE_DS4_SPEC=0 test_ds41_hc_f32 model.gguf frozen-token-requests.json
LUCE_DS4_SPEC=0 test_ds41_hc_f32 model.gguf frozen-token-requests.json workspace
LUCE_DS4_SPEC=0 test_ds41_hc_f32 model.gguf frozen-token-requests.json experts
```

The JSON contains a `requests` array with frozen `token_ids`. These tests
require each request to produce the full 128 tokens. For matched profiling,
set `LUCE_DS41_PROFILE_COMPARE=1` and run under Nsight with
`--capture-range=cudaProfilerApi --capture-range-end=repeat:2
--cuda-graph-trace=node`. The test captures tokens 16–80 of each arm on the
first prompt; the second prompt remains untraced.
