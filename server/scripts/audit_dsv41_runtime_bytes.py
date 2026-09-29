"""Audit retained research tiers using current packed runtime payload costs.

This does not allocate a model or prove fit: native tensors retain source byte
costs, and alignment, native conversion, caches and runtime overhead are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path


def replacement_bytes(entry):
    rows, cols = entry['shape']
    if any(type(n) is not int or n <= 0 for n in (rows, cols)):
        raise ValueError('Invalid matrix shape')
    method = entry['method']
    if method in ('gguf_q2_k', 'gguf_iq2_xxs', 'ggml_q2_k', 'ggml_iq2_xxs'):
        if cols % 256:raise ValueError('Invalid GGUF block width')
        return rows * (cols // 256) * (84 if method.endswith('q2_k') else 66)
    if method in ('int3_g64', 'int3_g64_gptq', 'int3_g128', 'int3_g128_gptq'):
        group = 64 if method.startswith('int3_g64') else 128
        if cols % group:
            raise ValueError('Invalid INT3 width')
        return rows * (cols // group) * (4 + group * 3 // 8)
    if method in ('nvfp4_mse', 'nvfp4_gptq'):
        if cols % 64:
            raise ValueError('Invalid NVFP4 width')
        return rows * (cols // 64) * 36
    palette = method.endswith('_palette')
    plain = method.removesuffix('_palette')
    for base, stride in [('gqh_t_g32_r4', 57), ('gqh_t_g32_r3', 56),
                         ('gqh_t', 61), ('gqh2_h', 73), ('gqh3', 105), ('gqh4', 137)]:
        for fit in ('', '_gptq', '_ratio_gptq', '_signround_gptq', '_awq_gptq'):
            if plain != base + fit:
                continue
            if cols % 256 or (not base.startswith('gqh_t') and (palette or not fit)):
                raise ValueError('Unsupported packed shape or method')
            return rows * (cols // 256) * stride + (cols * 4 if fit == '_awq_gptq' else 0)
    raise ValueError('No qualified runtime representation for ' + method)


def audit(path):
    blob = path.read_bytes()
    report = json.loads(blob)
    if report['status'] != 'complete' or report['native_replay'] != 'bit_exact':
        raise ValueError('Incomplete or incompatible report: ' + str(path))
    tiers = {}
    for name, tier in report['tiers'].items():
        file_bytes = tier['bytes']
        replacements = tier['choices']
        native_bytes = file_bytes - sum(e['bytes'] for e in replacements.values())
        if native_bytes < 0:
            raise ValueError('Negative native residual')
        packed = sum(replacement_bytes(e) for e in replacements.values())
        tiers[name] = dict(serialized_bytes=file_bytes, native_source_bytes=native_bytes,
                           replacement_device_payload_bytes=packed,
                           runtime_payload_lower_bound_bytes=native_bytes + packed,
                           damage=tier['damage'])
    if not tiers:
        raise ValueError('Empty candidate surface')
    file_best = min(tiers, key=lambda k: tiers[k]['serialized_bytes'])
    runtime_best = min(tiers, key=lambda k: tiers[k]['runtime_payload_lower_bound_bytes'])
    return dict(layer=report['layer'], report=str(path), report_sha256=hashlib.sha256(blob).hexdigest(),
                minimum_serialized_tier=file_best, minimum_runtime_tier=runtime_best, tiers=tiers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--override-report', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    reports = {}
    for path in sorted((a.study / 'sensitivity').glob('layer-*/report.json')):
        result = audit(path)
        if result['layer'] in reports:
            raise ValueError('Duplicate layer')
        reports[result['layer']] = result
    for path in a.override_report:
        result = audit(path)
        reports[result['layer']] = result
    marker_path = a.study / 'routed-campaign.json'
    marker_blob = marker_path.read_bytes()
    expected = json.loads(marker_blob)['layers']
    if any(layer not in range(expected) for layer in reports):
        raise ValueError('Layer outside campaign')
    result = dict(scope='Retained-tier payload audit only; not a full model placement or fit proof',
                  exclusions=['Unmeasured layers', 'Fixed model tensors outside layer ownership',
                              'Native runtime conversion overhead', 'Backend alignment and allocation overhead',
                              'Caches, workspaces and host metadata',
                              'Candidates previously pruned using serialized cost; runtime optimum unproven'],
                  marker_sha256=hashlib.sha256(marker_blob).hexdigest(),
                  measured_layers=sorted(reports), missing_layers=sorted(set(range(expected)) - reports.keys()),
                  sum_measured_layer_minimum_serialized_bytes=sum(r['tiers'][r['minimum_serialized_tier']]['serialized_bytes'] for r in reports.values()),
                  sum_measured_layer_minimum_runtime_lower_bound_bytes=sum(r['tiers'][r['minimum_runtime_tier']]['runtime_payload_lower_bound_bytes'] for r in reports.values()),
                  layers=[reports[k] for k in sorted(reports)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
