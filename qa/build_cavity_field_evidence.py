"""Build plots only after the fixed neural test is scored."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowmllab import cavity_field_evidence as evidence
from flowmllab.cavity_highre import file_hash

def main():
    out=ROOT/'results/cavity_field_deeponet'
    cases,info=evidence.load(ROOT)
    rows=evidence.diagnostics(cases)
    with (out/'physical_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14})
    for re,fields in cases.items():
        for name,plot in [('fields',evidence.plot_fields),('profiles',evidence.plot_profiles)]:
            fig=plot(fields,re)
            fig.savefig(out/f're{re}_{name}.png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.08)
            plt.close(fig)
    receipt=dict(frozen_selection_sha256=file_hash(out/'frozen_selection.json'),
                 physical_audit='wall/divergence preservation is separate from nonlinear PDE accuracy',
                 plot_scope='native 513 by 513 fields; shared speed and linear error scales within each Reynolds case',
                 files_sha256={p.name:file_hash(p) for p in sorted(out.glob('*.png'))})
    (out/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
