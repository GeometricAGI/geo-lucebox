"""Exact packed dense tensor transport, including GGUF/native Engram aliases."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deps/llama.cpp/gguf-py'))
import gguf
import numpy as np
spec = importlib.util.spec_from_file_location('exporter', ROOT / 'scripts/export_ds41_research_gguf.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class DenseReferenceExport(unittest.TestCase):
    def test_packed_and_float_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d)/'source.gguf', Path(d)/'dest.gguf'
            w = gguf.GGUFWriter(str(source), 'deepseek41')
            packed = gguf.quants.quantize(np.arange(128, dtype=np.float32).reshape(2, 64), gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor('output.weight', packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor('blk.1.engram_q_norm.weight', np.arange(16, dtype=np.float32).reshape(2, 8))
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            r = gguf.GGUFReader(str(source)); tensors = {t.name: t for t in r.tensors}
            w = gguf.GGUFWriter(str(dest), 'deepseek41')
            reports = [exporter.copy_reference_tensor(w, tensors, 'output.weight', (2, 64)),
                       exporter.copy_reference_tensor(w, tensors, 'blk.1.engram_q.weight', (2, 8))]
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            output = {t.name: t for t in gguf.GGUFReader(str(dest)).tensors}
            for report in reports:
                actual = output[report['tensor']]
                original = tensors[report['reference_tensor']]
                self.assertEqual(actual.tensor_type, original.tensor_type)
                np.testing.assert_array_equal(actual.shape, original.shape)
                np.testing.assert_array_equal(actual.data, original.data)
                self.assertEqual(exporter.sha(memoryview(actual.data).cast('B')), report['reference_payload_sha256'])
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                exporter.copy_reference_tensor(None, tensors, 'output.weight', (64, 2))
            with self.assertRaisesRegex(ValueError, 'Missing reference'):
                exporter.copy_reference_tensor(None, tensors, 'missing.weight', (2, 8))


if __name__ == '__main__':
    unittest.main()
