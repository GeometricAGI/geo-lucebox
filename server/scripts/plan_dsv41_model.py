"""Compile a text-backbone tensor/ownership plan from DS4.1 checkpoint headers.

Reads bounded metadata only, never tensor payloads. This is a binding plan, not
payload authentication, an allocator decision, or proof of model execution/fit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from audit_dsv41_runtime_bytes import replacement_bytes

LIMIT = 64 << 20
WIDTH = {'BF16': 2, 'F32': 4, 'I8': 1, 'I32': 4, 'I64': 8,
         'F8_E4M3': 1, 'F8_E8M0': 1}
CONTRACT = 'native-fp8-activations-decoded-bf16-weight-gemm-v1'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def bounded(path):
    require(path.stat().st_size <= LIMIT, f'Metadata exceeds bound: {path}')
    raw = path.read_bytes()
    require(len(raw) <= LIMIT, 'Metadata grew beyond bound')
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def confined(root, name):
    p = Path(name)
    require(not p.is_absolute(), 'Absolute checkpoint path')
    p = (root / p).resolve()
    require(p.is_relative_to(root), 'Checkpoint path escapes root')
    return p


def compile_plan(snapshot, artifact=None):
    root = snapshot.resolve()
    config_raw = bounded(root / 'config.json')
    index_raw = bounded(root / 'model.safetensors.index.json')
    config = json.loads(config_raw)
    require(config['model_type'] == 'deepseek_v41', 'Unsupported model family')
    c = config['text_config']
    n, dim, inter, hc = (c[k] for k in ('num_hidden_layers', 'hidden_size',
                                        'moe_intermediate_size', 'hc_mult'))
    experts, heads, hd = (c[k] for k in ('n_routed_experts', 'num_attention_heads', 'head_dim'))
    require(all(type(x) is int and x > 0 for x in (n, dim, inter, hc, experts, heads, hd)), 'Invalid dimensions')
    require(hd == 512 and c['qk_rope_head_dim'] == 64 and c['index_head_dim'] == 128
            and hc <= 8 and c['n_shared_experts'] == 1 and 0 < c['num_experts_per_tok'] <= 16,
            'Geometry outside qualified runtime')
    ratios = c['compress_ratios'][:n]
    require(len(ratios) == n and all(type(r) is int and r in (0, 1, 2) for r in ratios), 'Invalid ratios')
    kv, ix = c['kv_source_layer_ids'], c['index_source_layer_ids']
    for owners in (kv, ix):
        require(owners == sorted(set(owners)) and all(type(i) is int and 0 <= i < n and ratios[i] for i in owners), 'Invalid source owners')
    require(set(kv) <= set(ix), 'KV owner must also own index query')
    candidate = c['candidate_source_layer_id']
    require(candidate in ix and ratios[candidate] == 1, 'Unsupported candidate owner')
    mapping = json.loads(index_raw)['weight_map']
    headers, shard_info = {}, {}
    for shard in sorted(set(mapping.values())):
        path = confined(root, shard)
        size = path.stat().st_size
        with path.open('rb') as f:
            prefix = f.read(8)
            require(len(prefix) == 8, 'Truncated shard')
            length, = struct.unpack('<Q', prefix)
            require(length <= LIMIT and length <= size - 8, 'Invalid shard header size')
            raw = f.read(length)
            require(len(raw) == length, 'Short shard header')
        headers[shard] = json.loads(raw)
        shard_info[shard] = dict(file_bytes=size, header_bytes=length,
                                 header_sha256=digest(raw))
    used, tensors, matrices = set(), {}, {}

    def tensor(name, dtype, shape, role, compute=None):
        require(name in mapping, f'Missing required tensor: {name}')
        shard = mapping[name]
        t = headers[shard][name]
        require(t['dtype'] == dtype and t['shape'] == shape, f'Shape/dtype mismatch: {name}')
        lo, hi = t['data_offsets']
        require(all(type(x) is int for x in (lo, hi)) and 0 <= lo <= hi
                and hi - lo == math.prod(shape) * WIDTH[dtype]
                and 8 + shard_info[shard]['header_bytes'] + hi <= shard_info[shard]['file_bytes'],
                f'Invalid tensor extent: {name}')
        used.add(name)
        tensors[name] = dict(dtype=dtype, shape=shape, shard=shard,
                             offset=8 + shard_info[shard]['header_bytes'] + lo,
                             source_bytes=hi-lo, role=role,
                             compute_dtype=compute or dtype)
        return name

    def dense(name, shape, dtype='BF16', role='layer_dense', compute=None):
        return tensor(name, dtype, shape, role, compute)

    def matrix(name, rows, cols, fp4=False, fp8_activation=True, role='layer_projection'):
        weight = tensor(name+'.weight', 'I8' if fp4 else 'F8_E4M3',
                        [rows, cols//2 if fp4 else cols], role)
        scale = tensor(name+'.scale', 'F8_E8M0',
                       [rows, cols//32] if fp4 else [(rows+31)//32, (cols+31)//32], role)
        w, s = tensors[weight], tensors[scale]
        require(w['shard'] != s['shard'] or w['offset']+w['source_bytes'] <= s['offset']
                or s['offset']+s['source_bytes'] <= w['offset'], 'Overlapping weight/scale')
        size = w['source_bytes']+s['source_bytes']
        matrices[name] = dict(rows=rows, columns=cols, source='native',
                              native_format='fp4' if fp4 else 'fp8',
                              research_fp8=fp8_activation, role=role,
                              device_payload_bytes=size,
                              host_payload_bytes=size*(2 if fp4 else 1),
                              runtime_tensors=([dict(type='MXFP4', shape=[cols, rows])] if fp4 else
                                [dict(type='I8', shape=[cols, rows]), dict(type='I8', shape=[(cols+31)//32, (rows+31)//32])]))
        return name

    globals_ = dict(embedding=dense('embed.weight', [c['vocab_size'], dim], role='global'),
                    norm=dense('norm.weight', [dim], role='global'),
                    head=dense('head.weight', [c['vocab_size'], dim], role='global'))
    layers = []
    replaceable_dense = {}
    for i in range(n):
        prefix = f'layers.{i}'
        layer = dict(id=i, ratio=ratios[i], kv_owner=None, topk_owner=None,
                     owns_kv=i in kv, owns_index_query=i in ix,
                     applies_candidates=i in ix and i > candidate and bool(ratios[i]), dense={}, projections={})
        if ratios[i]:
            for key, owners in [('kv_owner', kv), ('topk_owner', ix)]:
                prior = [j for j in owners if j <= i]
                require(prior and ratios[prior[-1]] == ratios[i], 'Missing/incompatible source owner')
                layer[key] = prior[-1]
            require(layer['topk_owner'] >= layer['kv_owner'], 'Index source predates key source')
            if layer['applies_candidates']:
                require(layer['kv_owner'] <= candidate, 'Candidate source has a different key owner')
        def ld(key, suffix, shape, dtype='BF16', compute=None):
            layer['dense'][key] = dense(prefix+'.'+suffix, shape, dtype, compute=compute)
        def lm(key, suffix, rows, cols, activation=True):
            layer['projections'][key] = matrix(prefix+'.'+suffix, rows, cols, fp8_activation=activation)
        for part in ('attn', 'ffn'):
            ld(part+'_norm', part+'_norm.weight', [dim])
            ld('hc_'+part+'_fn', 'hc_'+part+'_fn', [(2+hc)*hc, dim*hc], 'F32')
            ld('hc_'+part+'_base', 'hc_'+part+'_base', [(2+hc)*hc], 'F32')
            ld('hc_'+part+'_scale', 'hc_'+part+'_scale', [3], 'F32')
        ld('sink', 'attn.attn_sink', [heads], 'F32')
        ld('q_norm', 'attn.q_norm.weight', [c['q_lora_rank']])
        ld('kv_norm', 'attn.kv_norm.weight', [hd])
        ld('router', 'ffn.gate.weight', [experts, dim])
        ld('route_bias', 'ffn.gate.bias', [experts], 'F32')
        for key, rows, cols in [('wq_a', c['q_lora_rank'], dim), ('wq_b', heads*hd, c['q_lora_rank']),
                              ('wkv', hd, dim), ('wo_a', c['o_groups']*c['o_lora_rank'], heads*hd//c['o_groups']),
                              ('wo_b', dim, c['o_groups']*c['o_lora_rank'])]:
            lm(key, 'attn.'+key, rows, cols, key != 'wo_a')
        if i in kv:
            ld('compressor_kv', 'attn.compressor.wkv.weight', [hd, dim], compute='F32' if ratios[i] == 2 else 'BF16')
            if ratios[i] == 2:
                ld('compressor_gate', 'attn.compressor.wgate.weight', [hd, dim], compute='F32')
            ld('compressor_norm', 'attn.compressor.norm.weight', [hd])
            ld('index_key', 'attn.indexer.wk.weight', [c['index_head_dim'], hd])
            ld('index_norm', 'attn.indexer.k_norm.weight', [c['index_head_dim']])
        if i in ix:
            lm('index_query', 'attn.indexer.wq_b', c['index_n_heads']*c['index_head_dim'], c['q_lora_rank'])
            ld('index_weights', 'attn.indexer.weights_proj.weight', [c['index_n_heads'], dim])
        def expert(base, native_fp4, role):
            return [matrix(base+'.'+key, rows, cols, native_fp4, True, role)
                    for key, rows, cols in [('w1', inter, dim), ('w3', inter, dim), ('w2', dim, inter)]]
        layer['experts'] = [expert(f'{prefix}.ffn.experts.{j}', True, 'routed_streamable') for j in range(experts)]
        layer['shared'] = expert(prefix+'.ffn.shared_experts', False, 'shared')
        if i in c['engram_layer_ids']:
            ei = c['engram_layer_ids'].index(i)
            erows, edim = c['engram_num_embeddings'][ei], c['engram_head_dim']
            table = prefix+'.engram.embed'
            layer['engram'] = dict(hash_layer=ei,
                weight=tensor(table+'.weight', 'F8_E4M3', [erows, edim], 'engram_nvme'),
                scale=tensor(table+'.scale', 'F8_E8M0', [erows, edim//32], 'engram_nvme'))
            ld('engram_q', 'engram.q_weight', [hc, dim])
            ld('engram_k', 'engram.k_weight', [hc, dim])
            lm('engram_wkv', 'engram.wkv', dim*(hc+1), (c['engram_max_ngram_size']-1)*c['engram_n_heads']*edim)
        for key in ('index_key', 'index_weights', 'compressor_kv'):
            if key in layer['dense'] and (key != 'compressor_kv' or ratios[i] == 1):
                replaceable_dense[layer['dense'][key].removesuffix('.weight')] = (layer, key)
        layers.append(layer)
    # Text-only target: explicitly enumerate unused image routing, vision and
    # optional MTP tensors, rather than silently dropping unknown backbone data.
    excluded = {}
    for name in mapping.keys()-used:
        reason = ('vision' if name.startswith(('vision.', 'aligner.', 'image_')) else
                  'draft' if name.startswith('mtp.') else
                  'image_router_bias' if name.endswith('.ffn.gate.bias_vl') else None)
        require(reason is not None, 'Unclassified checkpoint tensor: '+name)
        excluded[name] = reason
    artifact_identity = None
    artifact_native_index_identity = None
    artifact_engram_shards = {}
    artifact_engram_tensors = {}
    if artifact is not None:
        ar = artifact.resolve()
        raw = bounded(ar/'geoquant_artifact.json');a = json.loads(raw)
        require(a['format'] == 'geoquant-v41-research-v1' and a['status'] == 'complete'
                and a['contract'] == CONTRACT and a['source_index_sha256'] == digest(index_raw), 'Artifact identity/contract mismatch')
        artifact_index_raw = bounded(ar/'model.safetensors.index.json')
        remaining = json.loads(artifact_index_raw)['weight_map']
        for name, entry in a['replacements'].items():
            if name in replaceable_dense:
                layer, key = replaceable_dense[name]
                tensor_name = layer['dense'][key]
                t = tensors[tensor_name]
                require(t['dtype'] == 'BF16' and len(t['shape']) == 2, 'Invalid research dense source')
                matrices[name] = dict(rows=t['shape'][0], columns=t['shape'][1],
                                      native_format='bf16', research_fp8=False, role='layer_projection')
                del layer['dense'][key]
                layer['projections'][key] = name
                t['runtime_replaced'] = True
            require(name in matrices, 'Replacement outside qualified text matrices: '+name)
            m = matrices[name]
            require(entry['shape'] == [m['rows'], m['columns']], 'Replacement shape mismatch')
            require(name+'.weight' not in remaining and name+'.scale' not in remaining, 'Duplicate native replacement')
            path = confined(ar, entry['file'])
            require(type(entry['bytes']) is int and path.stat().st_size == entry['bytes'], 'Replacement file size mismatch')
            require(len(entry['sha256']) == 64 and all(x in '0123456789abcdef' for x in entry['sha256']), 'Invalid replacement digest')
            cost = replacement_bytes(entry)
            method = entry['method']
            base = next((b for b in ('ggml_q2_k', 'ggml_iq2_xxs', 'gguf_q2_k', 'gguf_iq2_xxs', 'gqh_t_g32_r4', 'gqh_t_g32_r3', 'gqh_t', 'gqh2_h', 'gqh3', 'gqh4', 'int3_g64', 'int3_g128', 'nvfp4')
                         if method == b or method.startswith(b+'_')), None)
            require(base is not None, 'Unknown runtime format')
            types = {'ggml_q2_k':'Q2_K', 'ggml_iq2_xxs':'IQ2_XXS', 'gguf_q2_k':'Q2_K', 'gguf_iq2_xxs':'IQ2_XXS', 'gqh_t':'GQH_T', 'gqh_t_g32_r4':'GQH_T_G32_R4', 'gqh_t_g32_r3':'GQH_T_G32_R3',
                     'gqh2_h':'GQH2_H', 'gqh3':'GQH3', 'gqh4':'GQH4', 'int3_g64':'DSV41_INT3_G64',
                     'int3_g128':'DSV41_INT3_G128', 'nvfp4':'NVFP4'}
            runtime = [dict(type=types[base], shape=[m['columns'], m['rows']])]
            if '_awq_gptq' in method:
                runtime.append(dict(type='F32', shape=[m['columns']]))
            # Conservative scratch: file + unpack/repack wire + two AWQ scale
            # copies. Runtime loader still validates actual budget and hashes.
            m.update(source='research', runtime_tensors=runtime, method=entry['method'], file=entry['file'], sha256=entry['sha256'],
                     file_bytes=entry['bytes'], device_payload_bytes=cost,
                     host_payload_bytes=entry['bytes']+cost+2*m['columns']*4+5)
        removed = {name+suffix for name in a['replacements'] for suffix in ('.weight', '.scale')}
        text_omitted = {name for name, reason in excluded.items() if reason in ('vision', 'draft')}
        full_remaining = mapping.keys() - removed
        text_remaining = full_remaining - text_omitted
        require(set(remaining) in (full_remaining, text_remaining),
                'Artifact native fallback index is incomplete or changed')
        native_groups = {}
        for name, native_shard in remaining.items():
            require(native_shard in a['native_files'], 'Artifact index references an unlisted native file')
            original_shard = mapping[name]
            native_groups.setdefault(original_shard, set()).add(native_shard)
        require(all(len(group) == 1 for group in native_groups.values())
                and len({next(iter(group)) for group in native_groups.values()}) == len(native_groups),
                'Artifact native files mix original source shards')
        source_engram_shards = {t['shard'] for t in tensors.values() if t['role'] == 'engram_nvme'}
        shared = a.get('hardlinked_original_engram_shards', {})
        require(isinstance(shared, dict) and (not shared or
                (set(shared.values()) == source_engram_shards and len(shared) == len(source_engram_shards))),
                'Artifact engram hardlink mapping is incomplete')
        for source_shard in source_engram_shards:
            native_shard = next(iter(native_groups[source_shard]))
            native_path = confined(ar, native_shard)
            require(native_path.stat().st_size == a['native_files'][native_shard]['bytes'],
                    'Artifact engram native shard size mismatch')
            if shared:
                require(shared.get(native_shard) == source_shard and native_path.samefile(confined(root, source_shard)),
                        'Artifact engram shard is not the pinned original file')
            require(all(remaining[name] == native_shard for name, t in tensors.items()
                        if t['role'] == 'engram_nvme' and t['shard'] == source_shard),
                    'Artifact engram index differs from source shard grouping')
            artifact_engram_shards[source_shard] = native_shard
            with native_path.open('rb') as stream:
                prefix = stream.read(8)
                require(len(prefix) == 8, 'Truncated artifact engram header')
                header_bytes, = struct.unpack('<Q', prefix)
                require(header_bytes <= LIMIT and header_bytes <= native_path.stat().st_size-8,
                        'Artifact engram header exceeds bound')
                header = json.loads(stream.read(header_bytes))
            for name, t in tensors.items():
                if t['role'] != 'engram_nvme' or t['shard'] != source_shard:
                    continue
                item = header[name]
                lo, hi = item['data_offsets']
                require(item['dtype'] == t['dtype'] and item['shape'] == t['shape']
                        and type(lo) is int and type(hi) is int and 0 <= lo <= hi
                        and hi-lo == t['source_bytes']
                        and 8+header_bytes+hi <= native_path.stat().st_size,
                        'Artifact engram tensor metadata mismatch')
                artifact_engram_tensors[name] = dict(shard=native_shard, offset=8+header_bytes+lo)
        artifact_native_index_identity = digest(artifact_index_raw)
        artifact_identity = digest(raw)
    dense_bytes = sum(t['source_bytes'] for t in tensors.values()
                      if t['role'] in ('global', 'layer_dense') and not t.get('runtime_replaced', False))
    promoted = sum(math.prod(t['shape'])*2 for t in tensors.values()
                   if t['dtype'] == 'BF16' and t['compute_dtype'] == 'F32')
    engram_bytes = sum(t['source_bytes'] for t in tensors.values() if t['role'] == 'engram_nvme')
    summary = dict(layers=n, experts=n*experts, matrices=len(matrices), dense_source_bytes=dense_bytes,
                   additional_dense_F32_promotion_bytes=promoted, engram_nvme_bytes=engram_bytes,
                   all_matrix_device_payload_bytes=sum(m['device_payload_bytes'] for m in matrices.values()),
                   replaced_matrices=sum(m['source'] == 'research' for m in matrices.values()),
                   maximum_matrix_host_payload_bytes=max(m['host_payload_bytes'] for m in matrices.values()))
    return dict(format='dsv41-text-binding-plan-v1', scope='text backbone, engram enabled; vision and draft explicitly excluded',
                source_index_sha256=digest(index_raw), config_sha256=digest(config_raw), artifact_manifest_sha256=artifact_identity,
                artifact_native_index_sha256=artifact_native_index_identity,
                artifact_engram_shards=artifact_engram_shards,
                artifact_engram_tensors=artifact_engram_tensors,
                config=c, globals=globals_, layers=layers, matrices=matrices, tensors=tensors, shards=shard_info,
                excluded=excluded, summary=summary,
                remaining=['Authenticate immutable shard contents; header hashes do not authenticate tensors',
                           'Backend alignment, actual allocations, host metadata, direct-I/O scratch and kernel pools',
                           'Placement and residency selection; counts above are not a fit proof',
                           'Native engram hash layout/tokenizer mapping sidecar',
                           'Full model execution and frozen golden validation'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--artifact', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();plan = compile_plan(a.snapshot, a.artifact)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan, separators=(',', ':'))+'\n')
    print(json.dumps(plan['summary'], indent=2))

if __name__ == '__main__':
    main()
