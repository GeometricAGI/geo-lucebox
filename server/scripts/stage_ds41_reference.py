"""Stage the exact reference GGUF without retransmitting existing Engram data."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,os,shutil,struct,threading,time,urllib.request
import numpy as np

def validate_extents(plan):
 offset=0
 for chunk in plan['chunks']:
  if (type(chunk['offset']) is not int or type(chunk['bytes']) is not int
      or chunk['offset']!=offset or chunk['bytes']<=0
      or chunk['kind'] not in ('download','native')):
   raise ValueError('Staging chunks must cover the destination exactly once in order')
  if chunk['kind']=='native':
   table=plan['tables'][chunk['table']]
   if (type(chunk['row']) is not int or type(chunk['rows']) is not int
       or chunk['row']<0 or chunk['rows']<=0
       or chunk['row']+chunk['rows']>table['rows']
       or chunk['bytes']!=chunk['rows']*264):
    raise ValueError('Invalid native Engram row extent')
  offset+=chunk['bytes']
 if offset!=plan['source_bytes'] or offset<=0:
  raise ValueError('Staging extents do not cover the reference file')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);a=ap.parse_args()
 root=a.root;raw=(root/'plan.json').read_bytes();plan=json.loads(raw);identity=hashlib.sha256(raw).hexdigest()
 validate_extents(plan)
 index=a.native/'model.safetensors.index.json'
 if hashlib.sha256(index.read_bytes()).hexdigest()!=plan['native_index_sha256']:raise ValueError('Native index mismatch')
 for t in plan['tables']:
  p=a.native/t['file']
  if not p.resolve().is_relative_to(a.native.resolve()):raise ValueError('Native shard path escapes source directory')
  if p.stat().st_size!=t['source_bytes']:raise ValueError('Native shard extent mismatch')
  with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  if hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()!=t['header_sha256']:raise ValueError('Native shard metadata mismatch')
 output=root/'DeepSeek-V4.1-Flash-Q2.gguf';partial=output.with_suffix('.gguf.partial')
 if output.exists():raise FileExistsError(output)
 statepath=root/'state.json';done={};lock=threading.Lock();stop=threading.Event();started=time.monotonic()
 if statepath.exists():
  prior=json.loads(statepath.read_text())
  if prior['plan_sha256']!=identity:raise ValueError('Resume plan changed')
  done=prior['completed']
 remaining=sum(c['bytes'] for i,c in enumerate(plan['chunks']) if str(i) not in done)
 if shutil.disk_usage(root).free<remaining+plan['min_free_gib']*2**30:raise RuntimeError('Insufficient space for remaining payload plus reserve')
 fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o600)
 os.ftruncate(fd,plan['source_bytes'])
 def state(status,error=None):
  with lock:
   obj=dict(status=status,pid=os.getpid(),plan_sha256=identity,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
     elapsed_seconds=time.monotonic()-started,completed=dict(done),completed_bytes=sum(plan['chunks'][int(i)]['bytes'] for i in done),
     downloaded_bytes=sum(plan['chunks'][int(i)]['bytes'] for i in done if plan['chunks'][int(i)]['kind']=='download'),error=error)
   tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(statepath)
 def check_space():
  if stop.is_set():raise RuntimeError('Another staging worker failed')
  if shutil.disk_usage(root).free<(plan['min_free_gib']+1)*2**30:raise RuntimeError('Disk reserve reached; preserving partial transfer')
  if time.monotonic()-started>30*3600:raise RuntimeError('Thirty-hour staging ceiling reached')
 def save(i,data):
  chunk=plan['chunks'][i];view=memoryview(data).cast('B')
  if len(view)!=chunk['bytes'] or hashlib.sha256(view).hexdigest()!=chunk['sha256']:raise ValueError(f'Chunk {i} payload mismatch')
  check_space();n=0
  while n<len(view):
   written=os.pwrite(fd,view[n:],chunk['offset']+n)
   if written<=0:raise OSError('Short destination write')
   n+=written
  os.fdatasync(fd)
  with lock:done[str(i)]=chunk['sha256']
  state('staging')
 def native_chunks():
  sources=[os.open(a.native/t['file'],os.O_RDONLY) for t in plan['tables']]
  try:
   for i,c in enumerate(plan['chunks']):
    if c['kind']!='native' or str(i) in done:continue
    check_space();t=plan['tables'][c['table']];source=sources[c['table']];row,rows=c['row'],c['rows']
    w=os.pread(source,rows*256,t['weight_offset']+row*256);s=os.pread(source,rows*8,t['scale_offset']+row*8)
    if len(w)!=rows*256 or len(s)!=rows*8:raise OSError('Short native read')
    out=np.empty((rows,264),dtype=np.uint8);out[:,:256]=np.frombuffer(w,dtype=np.uint8).reshape(rows,256);out[:,256:]=np.frombuffer(s,dtype=np.uint8).reshape(rows,8)
    save(i,out)
  finally:
   for source in sources:os.close(source)
 def download(i):
  c=plan['chunks'][i];begin=c['offset'];end=begin+c['bytes']-1
  for attempt in range(6):
   check_space()
   try:
    req=urllib.request.Request(plan['url']+f'?download=true&geoquant_range={begin}-{end}',headers={'Range':f'bytes={begin}-{end}'})
    with urllib.request.urlopen(req,timeout=180) as response:
     if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{plan["source_bytes"]}':raise ValueError('Server did not honor exact byte range')
     data=response.read(c['bytes']+1)
    save(i,data);return
   except (OSError,TimeoutError) as e:
    if attempt==5:raise
    state('retrying',f'chunk {i}: {type(e).__name__}; attempt {attempt+1}')
    time.sleep(min(60,5*2**attempt))
 try:
  # Recheck marked bytes before trusting a resumed partial file.
  for key,expected in done.items():
   c=plan['chunks'][int(key)];h=hashlib.sha256();offset=c['offset'];left=c['bytes']
   while left:
    b=os.pread(fd,min(left,8<<20),offset)
    if not b:raise OSError('Short resumed chunk')
    h.update(b);left-=len(b);offset+=len(b)
   if expected!=c['sha256'] or h.hexdigest()!=expected:raise ValueError('Resumed chunk failed checksum')
  state('staging')
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   futures=[pool.submit(native_chunks)]
   # Two bounded network workers; each visits disjoint numbered chunks.
   def lane(n):
    for j,i in enumerate(i for i,c in enumerate(plan['chunks']) if c['kind']=='download'):
     if j%2==n and str(i) not in done:download(i)
   futures += [pool.submit(lane,n) for n in range(2)]
   for f in concurrent.futures.as_completed(futures):
    try:f.result()
    except BaseException:stop.set();raise
  if len(done)!=len(plan['chunks']):raise RuntimeError('Incomplete staging')
  state('verifying');h=hashlib.sha256();offset=0
  while offset<plan['source_bytes']:
   b=os.pread(fd,min(16<<20,plan['source_bytes']-offset),offset)
   if not b:raise OSError('Short final file')
   h.update(b);offset+=len(b)
  if h.hexdigest()!=plan['source_sha256']:raise ValueError('Final reference checksum mismatch')
  os.fsync(fd);os.close(fd);fd=None;partial.replace(output);state('complete')
 except BaseException as e:state('failed',repr(e));raise
 finally:
  if fd is not None:os.close(fd)
if __name__=='__main__':main()
