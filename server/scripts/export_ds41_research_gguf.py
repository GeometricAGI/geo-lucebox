#!/usr/bin/env python3
"""Export the dense/tokenizer envelope for resident mixed research experts.

Expert payloads and native engram tables remain in the immutable artifact;
GGUF records its manifest/index identities. Dense matrices are decoded using
geo-quant's reference codec to BF16, without a second weight quantization.
This is a production-engine execution contract, not native FP8 activation parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy_reference_tensor(writer, tensors, target, shape):
    reference = tensors.get(target)
    if reference is None:
        # The loader accepts both native and published GGUF aliases.
        aliases = {'engram_q.weight': 'engram_q_norm.weight',
                   'engram_k.weight': 'engram_k_norm.weight',
                   'engram_wkv.weight': 'engram_kv.weight'}
        for suffix, alias in aliases.items():
            if target.endswith('.' + suffix):
                reference = tensors.get(target[:-len(suffix)] + alias)
                break
    if reference is None:
        raise ValueError(f'Missing reference dense tensor {target}')
    reference_shape = tuple(int(v) for v in reference.shape[::-1])
    if reference_shape != tuple(shape):
        raise ValueError(f'Reference dense shape mismatch for {target}: {reference_shape} != {tuple(shape)}')
    # Reader data carries the packed byte shape. Writer converts uint8 storage
    # dimensions back to logical dimensions; passing logical raw_shape is wrong.
    writer.add_tensor(target, reference.data, raw_dtype=reference.tensor_type)
    return dict(tensor=target, shape=list(shape), reference_tensor=reference.name,
                dtype=reference.tensor_type.name, bytes=reference.n_bytes,
                reference_payload_sha256=sha(memoryview(reference.data).cast('B')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--engram-layout', type=Path, required=True)
    ap.add_argument('--engram-layout-sha256', required=True)
    ap.add_argument('--geoquant', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dense-reference', type=Path,
                    help='Diagnostic: copy exact non-expert tensor payloads from this GGUF; report each source digest')
    args = ap.parse_args()
    # Avoid scripts/profile.py shadowing the Python standard library profile module.
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(args.geoquant.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deps/llama.cpp/gguf-py'))
    import numpy as np
    import torch
    import gguf
    from geoquant.models.deepseek_v41.research_codec import decode
    from geoquant.formats.native_dequant import maybe_dequant_native_weight
    torch.set_num_threads(8)
    root = args.artifact.resolve()
    plan = json.loads(args.plan.read_bytes())
    manifest_bytes = (root / 'geoquant_artifact.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    index_bytes = (root / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)['weight_map']
    config_bytes = (root / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    for actual, expected, label in (
        (sha(manifest_bytes), plan['artifact_manifest_sha256'], 'manifest'),
        (sha(index_bytes), plan['artifact_native_index_sha256'], 'native index'),
        (sha(config_bytes), plan['config_sha256'], 'config'),
        (manifest['source_index_sha256'], plan['source_index_sha256'], 'source index'),
    ):
        if actual != expected:
            raise ValueError(f'{label} identity mismatch')
    if args.output.exists():
        raise FileExistsError(args.output)
    dense_reader = gguf.GGUFReader(str(args.dense_reference)) if args.dense_reference else None
    dense_tensors = {t.name:t for t in dense_reader.tensors} if dense_reader else {}
    dense_identity = (dict(path=str(args.dense_reference.resolve()),
                          bytes=args.dense_reference.stat().st_size,
                          mtime_ns=args.dense_reference.stat().st_mtime_ns)
                      if dense_reader else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.partial')
    if tmp.exists():
        raise FileExistsError(tmp)
    w = gguf.GGUFWriter(str(tmp), 'deepseek41', use_temp_file=True)
    w.add_name('DS4.1 geo-quant mixed research weights / production execution')
    c = plan['config']
    def array(key, values, subtype):
        w.add_key_value(key, list(values), gguf.GGUFValueType.ARRAY, subtype)
    for key, val in c.items():
        if key in ('rope_theta', 'compress_rope_theta'): continue
        if isinstance(val, bool):
            w.add_bool('deepseek41.' + key, val)
        elif isinstance(val, int):
            w.add_uint32('deepseek41.' + key, val)
        elif isinstance(val, float):
            w.add_float32('deepseek41.' + key, val)
    # These aliases are read as floats, even when the HF config uses integers.
    for key in ('rope_theta', 'compress_rope_theta'):
        w.add_float32('deepseek41.' + key, float(c[key]))
    for key, val in c['rope_scaling'].items():
        name = 'deepseek41.rope_scaling.' + key
        if key == 'original_max_position_embeddings': w.add_uint64(name, val)
        elif isinstance(val, (int, float)): w.add_float32(name, float(val))
    for key in ('compress_ratios', 'kv_source_layer_ids', 'index_source_layer_ids'):
        array('deepseek41.' + key, c[key][:c['num_hidden_layers']], gguf.GGUFValueType.UINT32)
    w.add_uint32('deepseek41.block_count', c['num_hidden_layers'])
    w.add_uint32('deepseek41.embedding_length', c['hidden_size'])
    w.add_string('geoquant.research.artifact', os.path.relpath(root, args.output.parent.resolve()))
    w.add_string('geoquant.research.manifest_sha256', sha(manifest_bytes))
    w.add_string('geoquant.research.source_index_sha256', plan['source_index_sha256'])
    w.add_string('geoquant.research.native_index_sha256', sha(index_bytes))
    w.add_string('geoquant.research.execution', 'luce-ds41-resident-packed-bf16-weights-v1')
    layout = args.engram_layout.read_bytes()
    if sha(layout) != args.engram_layout_sha256 or layout[:8] != b'GQEH1\0\0\0':
        raise ValueError('Engram layout identity/magic mismatch')
    nl, ng, nh, dim, vocab, compressed, pad = struct.unpack_from('<7I', layout, 8)
    if (nl, ng, nh, dim, vocab, compressed) != (len(c['engram_layer_ids']), c['engram_max_ngram_size'],
            c['engram_n_heads'], c['engram_head_dim'], c['vocab_size'], c['engram_compressed_vocab_size']):
        raise ValueError('Engram geometry mismatch')
    pos = 36
    token_map = list(struct.unpack_from(f'<{vocab}i', layout, pos)); pos += vocab * 4
    ids, rows, multipliers, primes, offsets = [], [], [], [], []
    for _ in range(nl):
        il, nr = struct.unpack_from('<IQ', layout, pos); pos += 12
        mul = list(struct.unpack_from(f'<{ng}Q', layout, pos)); pos += 8 * ng
        pr = list(struct.unpack_from(f'<{(ng-1)*nh}Q', layout, pos)); pos += 8 * (ng-1)*nh
        ids.append(il); rows.append(nr); multipliers.extend(mul); primes.extend(pr)
        off=0
        for n in pr: offsets.append(off); off += n
        if off != nr: raise ValueError('Engram row count mismatch')
    if pos != len(layout) or ids != c['engram_layer_ids'] or rows != c['engram_num_embeddings']:
        raise ValueError('Engram layer layout mismatch')
    ep = 'deepseek41.engram.'
    for key,val in [('head_count',nh),('key_length',dim),('max_ngram_size',ng)]: w.add_uint32(ep+key,val)
    w.add_int32(ep+'pad_id',pad)
    for key,val in [('layer_ids',ids),('token_map',token_map)]: array(ep+key,val,gguf.GGUFValueType.INT32)
    for key,val in [('rows',rows),('multipliers',multipliers),('primes',primes),('offsets',offsets)]: array(ep+key,val,gguf.GGUFValueType.UINT64)
    w.add_string('geoquant.research.engram_layout_sha256',sha(layout))
    tok = json.loads((root/'tokenizer.json').read_bytes())
    tokens = [f'[unused{i}]' for i in range(c['vocab_size'])]
    types = [5]*len(tokens)
    for text,i in tok['model']['vocab'].items(): tokens[i]=text; types[i]=1
    for t in tok['added_tokens']:
        if t['id'] < len(tokens): tokens[t['id']]=t['content']; types[t['id']]=3 if t['special'] else 4
    w.add_tokenizer_model('gpt2'); w.add_tokenizer_pre('deepseek-v4')
    w.add_token_list(tokens); w.add_token_types(types)
    w.add_token_merges([' '.join(m) if isinstance(m,list) else m for m in tok['model']['merges']])
    for key in ('bos_token_id','eos_token_id','pad_token_id'): w.add_uint32('tokenizer.ggml.'+key,config[key])
    w.add_bool('tokenizer.ggml.add_bos_token',False)
    w.add_string('geoquant.research.tokenizer_sha256',sha((root/'tokenizer.json').read_bytes()))
    headers={}
    dtype={'BF16':torch.bfloat16,'F32':torch.float32,'I32':torch.int32,'I64':torch.int64,
           'I8':torch.int8,'F8_E4M3':torch.float8_e4m3fn,'F8_E8M0':torch.float8_e8m0fnu}
    def native(name):
        if name not in index: return None
        path=(root/index[name]).resolve()
        if not path.is_relative_to(root): raise ValueError('Native path escapes artifact')
        if path not in headers:
            with path.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                if n>64<<20: raise ValueError('Native header too large')
                headers[path]=(8+n,json.loads(f.read(n)))
        base,header=headers[path];m=header[name];lo,hi=m['data_offsets']
        if not 0<=lo<=hi<=path.stat().st_size-base: raise ValueError('Native extent invalid')
        with path.open('rb') as f: f.seek(base+lo);data=bytearray(f.read(hi-lo))
        if len(data)!=hi-lo: raise ValueError('Short native read')
        return torch.frombuffer(data,dtype=dtype[m['dtype']]).reshape(m['shape'])
    bindings=[]
    globals={'embedding':'token_embd.weight','norm':'output_norm.weight','head':'output.weight'}
    for role,name in plan['globals'].items(): bindings.append((name,globals[role],False))
    dense={'attn_norm':'attn_norm.weight','ffn_norm':'ffn_norm.weight','sink':'attn_sinks.weight',
        'q_norm':'attn_q_a_norm.weight','kv_norm':'attn_kv_a_norm.weight','router':'ffn_gate_inp.weight',
        'route_bias':'exp_probs_b.bias','compressor_kv':'attn_compressor_kv.weight',
        'compressor_gate':'attn_compressor_gate.weight','compressor_norm':'attn_compressor_norm.weight',
        'index_key':'indexer.attn_k.weight','index_norm':'indexer.k_norm.weight','index_weights':'indexer.proj.weight',
        'engram_q':'engram_q.weight','engram_k':'engram_k.weight'}
    for h in ('attn','ffn'):
        for suffix in ('fn','base','scale'): dense[f'hc_{h}_{suffix}']=f'hc_{h}_{suffix}.weight'
    proj={'wq_a':'attn_q_a.weight','wq_b':'attn_q_b.weight','wkv':'attn_kv.weight','wo_a':'attn_output_a.weight',
          'wo_b':'attn_output_b.weight','engram_wkv':'engram_wkv.weight','index_query':'indexer.attn_q_b.weight'}
    proj.update(dense)
    for layer in plan['layers']:
        prefix=f"blk.{layer['id']}."
        for role,name in layer['dense'].items(): bindings.append((name,prefix+dense[role],False))
        for role,name in layer['projections'].items(): bindings.append((name,prefix+proj[role],True))
        for name,suffix in zip(layer['shared'],('ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight')):
            bindings.append((name,prefix+suffix,True))
    report=[]
    for n,(source,target,matrix) in enumerate(bindings):
        replacement=manifest['replacements'].get(source) if matrix else None
        if replacement:
            path=(root/replacement['file']).resolve()
            if not path.is_relative_to(root): raise ValueError('Research path escapes artifact')
            blob=path.read_bytes()
            if len(blob)!=replacement['bytes'] or sha(blob)!=replacement['sha256']: raise ValueError('Research payload identity mismatch')
            t=decode(blob,replacement['method'],replacement['shape'])
        else:
            source_name=source+'.weight' if matrix else source
            t=native(source_name)
            if t is None: raise ValueError(f'Missing dense source {source_name}')
            t=maybe_dequant_native_weight(source_name,t,native)
        # GGML elementwise norm/gate/engram operators consume F32 tensors.
        if not matrix and target not in ('token_embd.weight','output.weight'):
            t=t.float()
        t=t.contiguous()
        if dense_reader is not None:
            report.append(dict(source=source, **copy_reference_tensor(w, dense_tensors, target, t.shape)))
            continue
        if t.dtype==torch.bfloat16:
            data=t.view(torch.int16).numpy().view(np.uint16); qtype=gguf.GGMLQuantizationType.BF16
        else:
            data=t.numpy();qtype=None
        w.add_tensor(target,data,raw_dtype=qtype)
        report.append({'source':source,'tensor':target,'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()})
        if n%25==0: print(f'{n+1}/{len(bindings)} {target}',flush=True)
    if dense_identity is not None and (args.dense_reference.stat().st_size != dense_identity['bytes']
            or args.dense_reference.stat().st_mtime_ns != dense_identity['mtime_ns']):
        raise ValueError('Reference GGUF changed during dense export')
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file(progress=True);w.close()
    os.replace(tmp,args.output)
    args.output.with_suffix('.export.json').write_text(json.dumps({'artifact_manifest_sha256':sha(manifest_bytes),
        'execution':'production engine; BF16-decoded weights; native FP8 activation parity not claimed',
        'dense_reference':dense_identity,
        'gguf_bytes':args.output.stat().st_size,'tensors':report},indent=2)+'\n')
    print(f'Exported {args.output}: {args.output.stat().st_size/(1<<30):.3f} GiB',flush=True)

if __name__ == '__main__': main()
