"""Reference reconstruction must be byte-identical and fail closed on corruption."""
import hashlib,importlib.util,io,json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('stager',Path(__file__).resolve().parents[1]/'scripts/stage_ds41_reference.py')
stager=importlib.util.module_from_spec(spec);spec.loader.exec_module(stager)
sha=lambda b:hashlib.sha256(b).hexdigest()

class Response(io.BytesIO):
    status=206
    def __init__(self,data,header):super().__init__(data);self.headers={'Content-Range':header}
    def __enter__(self):return self
    def __exit__(self,*a):self.close()

class ReferenceStaging(unittest.TestCase):
 def run_case(self,corrupt=False,bad_range=False):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);native=root/'native';native.mkdir();out=root/'out';out.mkdir()
   index=b'{"weight_map":{}}';(native/'model.safetensors.index.json').write_bytes(index)
   rows=5;w=np.arange(rows*256,dtype=np.uint8).reshape(rows,256);s=np.arange(rows*8,dtype=np.uint8).reshape(rows,8)
   header={'fixture':True};hb=json.dumps(header).encode();base=8+len(hb)
   raw=struct.pack('<Q',len(hb))+hb+w.tobytes()+s.tobytes();(native/'native.safetensors').write_bytes(raw)
   interleaved=np.concatenate((w,s),axis=1).tobytes();reference=b'header-payload'+interleaved+b'footer'
   offset=len(b'header-payload');chunks=[dict(kind='download',offset=0,bytes=offset,sha256=sha(reference[:offset])),dict(kind='native',offset=offset,bytes=len(interleaved),sha256=sha(interleaved),table=0,row=0,rows=rows),dict(kind='download',offset=offset+len(interleaved),bytes=6,sha256=sha(b'footer'))]
   if corrupt:chunks[1]['sha256']='0'*64
   plan=dict(source_sha256=sha(reference),source_bytes=len(reference),url='https://fixture/model',native_index_sha256=sha(index),min_free_gib=0,
    tables=[dict(file='native.safetensors',rows=rows,source_bytes=len(raw),header_sha256=sha(json.dumps(header,sort_keys=True).encode()),weight_offset=base,scale_offset=base+w.nbytes)],chunks=chunks)
   (out/'plan.json').write_text(json.dumps(plan))
   def request(req,timeout):
    start,end=map(int,req.headers['Range'].removeprefix('bytes=').split('-'))
    return Response(reference[start:end+1],f'bytes {start}-{end}/{len(reference)+(1 if bad_range else 0)}')
   with patch.object(sys,'argv',['stage','--root',str(out),'--native',str(native)]),patch.object(stager.urllib.request,'urlopen',request):
    if corrupt or bad_range:
     with self.assertRaises((ValueError,RuntimeError)):stager.main()
     self.assertFalse((out/'DeepSeek-V4.1-Flash-Q2.gguf').exists())
     self.assertEqual(json.loads((out/'state.json').read_text())['status'],'failed')
    else:
     stager.main();self.assertEqual((out/'DeepSeek-V4.1-Flash-Q2.gguf').read_bytes(),reference)
     self.assertEqual(json.loads((out/'state.json').read_text())['status'],'complete')
 def test_reconstructs_exact_file(self):self.run_case()
 def test_changed_native_content_rejected(self):self.run_case(corrupt=True)
 def test_wrong_http_range_rejected(self):self.run_case(bad_range=True)


class InvalidExtentPlan(unittest.TestCase):
 def test_gaps_overlap_and_extra_tail_fail(self):
  for offsets,size in [([0,3],5),([0,1],4),([0,2],5)]:
   with self.assertRaises(ValueError):
    stager.validate_extents(dict(chunks=[dict(offset=o,bytes=2,kind='download') for o in offsets],source_bytes=size))

if __name__=='__main__':unittest.main()
