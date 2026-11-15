"""Convert verified immutable captures of completed continuation roots."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
from flowmllab.cavity_steady import SteadyCavity
from flowmllab.cavity_highre import audit_field
from run_cavity_highre import atomic_npz,atomic_json,digest
EXPECTED={6000:'0b60b322b5f0dfdc1f40a08d53929f5d29694b81cab520071b33633b7814f910',
          6500:'b994567fa899fe75020d8365ad7e49609975214f09a623b6eb12cf2ac462c31f',
          9000:'dffb579c3ba6e19ba45ef9248e4c7ae866149217b6bebe8fab7a9e7060918371'}


def main():
    p=argparse.ArgumentParser();p.add_argument('captures',type=Path);a=p.parse_args()
    solver=SteadyCavity(512);out=ROOT/'results/cavity_highre_ghia/n512'
    for re,expected in EXPECTED.items():
        path=a.captures/f're{re}-n512.npz'
        if digest(path)!=expected:raise ValueError('Captured checkpoint hash mismatch')
        z=np.load(path,allow_pickle=False);meta=json.loads(str(z['metadata']))
        if meta['completed_reynolds']!=re:raise ValueError('Capture does not contain a completed target root')
        fields=solver.fields(z['state']);audit=audit_field(fields,re)
        if max(audit.values())>=1e-6:raise ValueError('Captured root fails independent equation audit')
        final=out/f're{re}.npz';atomic_npz(final,**fields)
        j,i=np.unravel_index(np.argmin(fields['psi']),fields['psi'].shape)
        config=dict(meta['config'],reynolds=re,captured_completed_continuation_root=True,
                    captured_checkpoint_sha256=expected)
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        atomic_json(final.with_suffix('.json'),dict(config=config,signature=signature,parent_continuation_signature=meta['signature'],
            reynolds=re,cells=512,converged=True,**audit,field_sha256=digest(final),
            captured_checkpoint_sha256=expected,continuation_history=meta['history'],
            cumulative_compute_seconds=meta['compute_seconds'],slurm_job_id='1767993_2',
            vortex_x=float(fields['x'][i]),vortex_y=float(fields['y'][j]),psi_min=float(fields['psi'][j,i]),
            ghia_used_during_solution=False,branch='2-D steady Navier-Stokes root captured after convergence',
            conversion_source_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()))
        print(re,audit,flush=True)


if __name__=='__main__':main()
