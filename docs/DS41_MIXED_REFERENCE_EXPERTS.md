# DS4.1 mixed reference experts

The research-artifact loader accepts raw GGUF Q2_K and IQ2_XXS expert matrices
alongside the existing conditioned research formats. Imported payloads retain their
standard block layout; they have no research header or GQH palette registration.
Scale validation and exact byte extent checks run before device upload. The shared
CPU/CUDA/HIP decoder reconstructs BF16 weights, matching the research execution
contract. Q2_K uses separate multiply/subtract before BF16 rounding to avoid an FMA
rounding discrepancy.

On 2026-09-27, the mixed routed-matmul and packed-arena tests passed on CPU, NVIDIA
H200, Lucebox 6 gfx1201 (32 GB Radeon) and gfx1151 (Radeon 8060S). They cover changing
routes, broadcast and per-route inputs, graph reuse, typed/generic parity, raw-wire
round-trip and invalid ranges. These are kernel/loader qualifications, not whole-model
quality or throughput results. ROCm used an isolated build with both architectures
and four compile jobs; the existing Lucebox checkout was left untouched.

For H200 diagnostic replicas, `LUCE_DS41_PACKED_PEER_DEVICE=CUDA1` permits whole expert
layers to spill from the primary device to a peer device. Explicit peer-access
activation is required; failed enablement stops loading. Packed and native fallback
weights share the selected placement, while descriptors stay on the compute GPU.
Each GPU retains 8 GiB headroom in this mode. This is remote weight access, not tensor
parallel computation. A two-H200 kernel test passed; full-model qualification remains
pending. No ROCm peer-placement support is claimed.

The planner and runtime byte auditor account for both imported wire types. Golden
acceptance remains separate from kernel parity and from native-streamed diagnostics;
reference experts with native dense weights differ from a full Q2 GGUF with Q8 dense
weights. Always record that dense policy when comparing outputs and sizes.


`bench_packed_experts` supplies actual DS4.1 projection shapes with separately
allocated 384-expert pools and changing top-six routes. It tiles known fixture
payloads and checks sampled outputs against CPU decoding; timings are projection
microbenchmarks, not model throughput. The opt-in environment setting
`LUCE_DS41_PACKED_WARP_REDUCE=1` preserves the existing addition tree while finishing
it with warp shuffles. Tests compare its outputs bit-for-bit with the original
reduction on CUDA and HIP, including CUDA peer weights. Measured gains were small
and hardware-dependent, with one slight gfx1151 regression; default remains off.

A separate `--dense-reference reference.gguf` exporter mode copies dense payloads
without requantization. It checks logical shapes, preserves wire types, recognizes
the published Engram aliases, and records every payload digest in the export report.
The 2026-09-27 Q2 control envelope contains 884 tensors (9.370 GiB of payload);
all were read back and matched. It excludes the unused visual routing biases and
is currently qualified for text diagnostics only. The packed expert arithmetic and
native Engram storage path still differ from ordinary reference GGUF loading.
`test_ds41_reference_dense_export.py` covers packed/F32 round-trip, aliases, missing
tensors and shape rejection. Whole-model quality remains a separate test.

### Lucebox peer access qualification

Do not enable AMD packed peer placement based on `hipDeviceCanAccessPeer` alone.
On both Lucebox 5 and 6, R9700-to-Strix data access failed: a standalone kernel and
`hipMemcpyPeer` returned 16,384/16,384 incorrect words. The opposite direction and
local reads passed. Host-staged copies passed both ways. The packed loader's CUDA-only
peer restriction remains intentional. Disabling SDMA in isolated child processes did
not fix the failure; no system settings were changed.

Actual-shape projection tests on box 6 measured roughly 0.57–1.03 ms on the R9700
with local weights, 4.80–6.53 ms when it read Strix weights, and 1.24–1.99 ms on Strix
with local weights. These are synthetic projection tests, not model TPS. Prefer
local weight computation and explicitly tested activation transport when developing
heterogeneous execution.

[Host-staging probe](results/lucebox-peer-20260927/host_staging_bench.cpp) and
[raw results](results/lucebox-peer-20260927/host-staging-results.jsonl) measure both
transfer legs with synchronization: 20 KiB took median 23–26 us, 120 KiB 60–65 us,
640 KiB 257–283 us, and a 9.49 MiB expert triplet 3.38–3.54 ms. Every case passed
bitwise validation. Pinned memory gave little benefit in this synchronous probe;
these results do not establish overlapped runtime performance.

### Lucebox HIP architecture preflight (2026-09-28)

Check the resolved `CMAKE_HIP_ARCHITECTURES` in the build cache, not just the
configure command. Lucebox 6 needs both `gfx1151` (Strix) and `gfx1201` (R9700).
The previous resolver overwrote an explicit `-DCMAKE_HIP_ARCHITECTURES` with the
Strix default unless `LUCE_HIP_ARCHITECTURES` or `AMDGPU_TARGETS` was supplied.
The resolver now captures the caller's CMake setting before language discovery
and honors it after those two project-specific overrides. The actual resolved
cache must include every device used for inference.

Build and run `test_backend_compute` on every visible device before loading a
model. It executes RMS normalization plus scaling and compares with a CPU result;
`test_backend_peer_copy` alone exercises transport and can pass even when GPU
kernel targets are missing. The initial full reference loaded but crashed in HIP
kernel launch on the first prompt. A smaller-residency GDB reproduction reached
the same failure, including with GGML fusion disabled. The corrected dual-target
build and subsequent model canary are separate qualification gates; no throughput
or quality result is implied by a successful build or copy test.


### Consistent physical UMA memory queries

The exported `ggml_backend_cuda_get_device_memory` formerly returned raw runtime
figures, whereas the backend-device API applied Linux UMA `MemAvailable`. Cache
sizing and headroom logging used the former; on Lucebox 6 it reported 1,280 GiB
total and about 121 GiB free even with a large resident model. Both entry points
now share the UMA query and report physical `MemTotal` plus `MemAvailable`; a
failed physical query reports zero free space rather than silently allocating
against the misleading runtime total. Zero available memory is valid and must
not fall back to the larger runtime figure. Dedicated-device queries are unchanged.

`test_backend_memory` passed on H200 and R9700. With `uma` on Strix during a live
model run, both APIs reported 134,311,915,520 total bytes and 30,927,663,104 free,
exactly matching `/proc/meminfo`. The separate build did not change the serving
binary for the ongoing quality comparison. Logs are under
`docs/results/lucebox-peer-20260927/memory-query-*`.
This is shared physical RAM, not additional GPU capacity. Generic queries remain
conservative about reclaimable TTM pages; the existing engine page-pool estimator
is a separate admission mechanism and must not be double-counted.


### Remaining mixed-format Lucebox integration

The successful streamed ordinary-Q2 hardware run does not qualify streamed
research artifacts. `deepseek4_packed_experts.cpp` owns immutable resident
per-expert descriptors; `ds4_packed_or_regular_product` consumes those through
`ggml_packed_experts_mul_mat_id`. By contrast, `make_ds4_moe_layer_desc` passes
ordinary uniform-type tensors to the existing hybrid path.
`LayerExpertRegions` assumes one fixed byte stride per projection for all experts
in a layer. A mixed ternary/Q2/native artifact cannot be substituted into those
regions without changing the storage and compute contracts.

The next integration should reuse the existing hybrid routing/ownership split
while providing explicit per-expert, per-projection byte regions and qualified
packed descriptors. Compute on the backend that owns the weights; transport
activations and weighted outputs through the qualified host-staged path. Cold
experts need bounded scratch residency and synchronization before descriptor
reuse. Retain SwiGLU clamping, routing weights, global/local ID remapping and
shared-expert semantics from `eval_ds4_hybrid`. Test mixed formats, repeated
routes, ragged prefill, evictions and both ownership directions against the
resident packed reference before a full-model run. Removing the CUDA peer guard
is not this integration.

Native FP8 dense storage is a separate possible saving. The original
`ggml_dsv41_fp8_matmul` took only BF16 input/output. It now also accepts F32
input and produces F32 output, avoiding an implicit BF16 activation/output round
trip. `test_ds41_fp8_io` passed on CPU, two H200s, R9700 and Strix: the tiny case
detects accidental input rounding; the other cases cover scale-block boundaries,
a real projection width and a 32-token prefill. The original BF16 path remains
covered. Results and source/binary hashes are in
`results/dsv41-fp8-io-20260928/qualification.json`.

This is arithmetic qualification only. It does not establish production matmul
reduction parity, full-model quality, throughput or a deployable memory saving.
The operation still rejects grouped/strided inputs; production tensor binding,
grouped output projections and measured runtime memory remain integration work.
Preserving source weight values alone is not an activation or quality guarantee.


The physical-memory-query build subsequently loaded the complete Q2 reference
with the same requested 20/92 GiB ownership, passed the exact-answer canary and
completed its 128-token probe, then released the GPUs. Its serving log reported
the physical Strix pool and available headroom through the corrected query.
This is a smoke test, not another golden score. The resolved server and shared
library hashes are recorded in
`results/lucebox-peer-20260927/memory-query-full-model.json`; the earlier full
quality screen retains its separate frozen binary identity.
