"""Read only atomic completed roots; never alter the live solver checkpoint."""
import hashlib,io,json,os,time
from pathlib import Path
import numpy as np
root=Path('/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115')
source=root/'results/n512/checkpoint.npz'
out=root/'captured_roots';out.mkdir(exist_ok=True)
wanted={6000,6500,9000};last_mtime=None
started=time.monotonic()
while wanted and time.monotonic()-started<3600:
    stat=source.stat()
    if stat.st_mtime_ns!=last_mtime:
        raw=source.read_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            re=int(meta['completed_reynolds'])
            if re in wanted:
                destination=out/f're{re}-n512.npz'
                if destination.exists() and destination.read_bytes()!=raw:
                    raise ValueError('Different captured root already exists')
                destination.write_bytes(raw)
                assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256(raw).digest()
                print(json.dumps(dict(captured_reynolds=re,sha256=hashlib.sha256(raw).hexdigest())),flush=True)
                wanted.remove(re)
        last_mtime=stat.st_mtime_ns
        if re>=10000:break
    time.sleep(3)
print(json.dumps(dict(missing=sorted(wanted))),flush=True)
