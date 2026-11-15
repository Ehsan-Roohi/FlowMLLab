"""Archive a closed, verified cavity solver experiment without removing originals."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,tarfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve()
    assert r.name=='solver-benchmark-v1' and r.parent.name=='roohie_umass_edu-flowml-ghia-highre-20261115'
    q=subprocess.run(['squeue','-u','roohie_umass_edu','-h','-o','%i|%Z'],text=True,capture_output=True,check=True,timeout=20).stdout
    assert not any(str(r) in line for line in q.splitlines()),'A benchmark allocation remains live'
    assert not list(r.glob('**/writer.lock')),'A writer lock remains'
    for phase,n in [('pilot',128),('pilot_strong',128),('production',256),('production',512)]:
        status=json.loads((r/phase/f'n{n}'/'status.json').read_text())
        assert status['state']=='complete' and len(status['completed'])==status['planned']
    checked=[]
    for phase in ['pilot','pilot_strong','production']:
        for f in sorted((r/phase).glob('n*/*/result.json')):
            d=json.loads(f.read_text());item={'path':str(f.relative_to(r)),'success':d['success']}
            if d['success']:
                assert d['solution_gate'] and d['final_residual_linf']<1e-8
                assert d['reference_discrepancies']['velocity_relative_l2']<1e-8
                assert d['reference_discrepancies']['omega']['relative_l2']<1e-7
                with np.load(f.parent/'recovery.npz',allow_pickle=False) as z:
                    state=z['state'];meta=json.loads(str(z['metadata']))
                    assert np.isfinite(state).all()
                    assert hashlib.sha256(state.tobytes()).hexdigest()==d['final_state_sha256']
                    assert meta['seed_sha256']==d['seed_sha256']
                    assert meta['history'][-1]['reynolds']==d['target_re']
                if 'final_fields_sha256' in d:assert sha(f.parent/'final_fields.npz')==d['final_fields_sha256']
                item['recovery_verified']=True
            checked.append(item)
    files=[]
    for f in sorted(r.rglob('*')):
        if f.is_symlink():raise ValueError('Unexpected symlink: '+str(f))
        if not f.is_file():continue
        rel=f.relative_to(r)
        if rel.parts[0] in ['transfer','__pycache__'] or '__pycache__' in rel.parts or f.name in ['archive_manifest.json','archive_receipt.json']:continue
        assert not f.name.endswith(('.pending.npz','.pending.json','.staging')),str(f)
        st=f.stat();files.append({'path':str(rel),'bytes':st.st_size,'sha256':sha(f)})
    estimate=sum(f['bytes'] for f in files)
    allocated=int(subprocess.run(['du','-s','-B1',str(r)],capture_output=True,text=True,check=True,timeout=30).stdout.split()[0])
    assert allocated+estimate+10485760<600000000,'Archive would exceed declared new-data budget'
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'files':files,'verified_trials':checked,'original_data_preserved':True,'declared_budget_bytes':600000000,'allocated_before_archive_bytes':allocated}
    mp=r/'archive_manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    transfer=r/'transfer';transfer.mkdir(exist_ok=True);target=transfer/'solver_benchmark_verified.tar.gz';assert not target.exists()
    tmp=target.with_suffix('.partial');assert not tmp.exists()
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for item in files:tar.add(r/item['path'],arcname=item['path'],recursive=False)
        tar.add(mp,arcname='archive_manifest.json',recursive=False)
    for item in files:
        f=r/item['path'];assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],'Source changed during archival'
    with tarfile.open(tmp,'r:gz') as tar:
        members={m.name:m for m in tar.getmembers()};assert len(members)==len(files)+1
        for item in files:
            member=members[item['path']];assert member.isfile() and member.size==item['bytes']
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==item['sha256'],'Archive member mismatch'
    os.replace(tmp,target)
    receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_path':str(target),'archive_bytes':target.stat().st_size,'archive_sha256':sha(target),'manifest_sha256':sha(mp),'files':len(files),'trials':len(checked),'successful_trials':sum(v['success'] for v in checked),'original_data_preserved':True,'durable_project_archive':'Not claimed: read-only project-location lookup stalled; verified Lexar archive is the intended external copy.'}
    (r/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
