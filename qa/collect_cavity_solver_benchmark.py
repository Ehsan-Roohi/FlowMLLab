"""Collect compact frozen solver evidence; leave full recovery fields on Unity."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, datetime


def collect():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    args=p.parse_args()
    remote=r'''
import pathlib,json,hashlib,subprocess,datetime
import numpy as np
r=pathlib.Path(ROOT)
data={'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(r),'trials':[],'metadata':{}}
for name in ['protocol.json','staging_manifest.json','selection.json','submission_receipt.json','production_submission.json','strong_submission.json','hardware_audit.json','paired_fields_audit.json']:
 p=r/name
 if p.exists():data['metadata'][name]=json.loads(p.read_text())
for phase in ['pilot','pilot_strong','production']:
 for p in sorted((r/phase).glob('n*/*/result.json')):
  d=json.loads(p.read_text());entry={'path':str(p.relative_to(r)),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'result':d}
  fp=p.parent/'final_fields.npz'
  if fp.exists() and d.get('success') and d.get('repeat')==0:
   assert hashlib.sha256(fp.read_bytes()).hexdigest()==d['final_fields_sha256']
   with np.load(fp,allow_pickle=False) as z:
    m=d['cells']//2
    entry['profiles']={'x':z['x'].tolist(),'y':z['y'].tolist(),'u_center':z['u'][:,m].tolist(),'v_center':z['v'][m,:].tolist(),'omega_horizontal_center':z['omega'][m,:].tolist()}
  data['trials'].append(entry)
 for p in sorted((r/phase).glob('n*/status.json')):data['metadata'][str(p.relative_to(r))]=json.loads(p.read_text())
print(json.dumps(data,allow_nan=False))
'''
    code='ROOT='+repr(args.root)+'\n'+remote
    ssh=[r'C:\Windows\System32\OpenSSH\ssh.exe','-i',str(args.identity),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-m','hmac-sha2-512-etm@openssh.com','-c','aes256-ctr','roohie_umass_edu@login.unityhpc.org',str(Path(args.root).parent).replace('\\','/')+'/venv/bin/python -']
    # Windows Path does not represent POSIX parent consistently: use string split.
    ssh[-1]=args.root.rsplit('/',1)[0]+'/venv/bin/python -'
    run=subprocess.run(ssh,input=code,text=True,encoding='utf-8',capture_output=True,timeout=90)
    if run.returncode:raise RuntimeError(run.stderr)
    evidence=json.loads(run.stdout)
    for entry in evidence['trials']:
        d=entry['result']
        if d.get('success'):
            assert d['solution_gate'] and d['final_residual_linf']<1e-8
            assert d['reference_discrepancies']['velocity_relative_l2']<1e-8
            assert d['reference_discrepancies']['omega']['relative_l2']<1e-7
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'evidence.json';temp=target.with_suffix('.pending.json')
    temp.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8');temp.replace(target)
    print(json.dumps({'trials':len(evidence['trials']),'passed':sum(e['result']['success'] for e in evidence['trials']),'evidence':str(target)}))


if __name__=='__main__':collect()
