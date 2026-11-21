"""Replot the retained DSMC comparison after a text-only percent-sign repair.
No solver or neural fit runs; retain the exact original Run All identity.
"""
from pathlib import Path
import os,sys,json,hashlib,base64,copy
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/course_refresh_early/mpl')
sys.path.insert(0,str(ROOT))
from common.article_validation import build_dsmc_wall_pressure_validation

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
 stem='AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student'
 nbpath=ROOT/'notebooks/week03'/f'{stem}.ipynb'
 out=ROOT/'results/course_refresh_early'/stem
 receiptpath=ROOT/'qa/course_refresh_early_dsmc_presentation.json'
 old_receipt=json.loads(receiptpath.read_text()) if receiptpath.exists() else {}
 baseline=json.loads((ROOT/'student_runs/article_validation/fig10a_mohammadzadeh_validation_metrics.json').read_text())
 sources=[ROOT/'results/dsmc_validation/mohammadzadeh_fig3_dsmc_points.csv']
 sources+=[ROOT/'results/dsmc_validation/runs'/n/'wall_pressure.csv' for n in ('ref_n40_seed07','ref_n40_seed19','ref_n40_seed31','ref_n60_seed07')]
 sources+=[ROOT/'results/dsmc_validation/runs/ref_n60_seed07/metadata.json']
 source_hash={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
 original_nb=nbpath.read_bytes();nb=json.loads(original_nb)
 code_sha=hashlib.sha256('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code').encode()).hexdigest()
 old_png=base64.b64decode(nb['cells'][3]['outputs'][0]['data']['image/png'])
 result=build_dsmc_wall_pressure_validation(ROOT,ROOT/'student_runs/article_validation')
 assert result['metrics']==baseline
 assert source_hash=={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
 data=result['figure']['png'].read_bytes()
 updated=copy.deepcopy(nb);updated['cells'][3]['outputs'][0]['data']['image/png']=base64.b64encode(data).decode()
 protected=copy.deepcopy(updated);protected['cells'][3]['outputs'][0]['data']['image/png']=nb['cells'][3]['outputs'][0]['data']['image/png']
 assert protected==nb
 nbpath.write_text(json.dumps(updated,indent=1,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
 (out/'cell_03_00.png').write_bytes(data)
 (out/'retained_wall_pressure_validation.pdf').write_bytes(result['figure']['pdf'].read_bytes())
 execution=json.loads((out/'execution.json').read_text())
 assert execution['code_sha256']==code_sha
 execution.setdefault('execution_notebook_sha256',execution['notebook_sha256'])
 execution['notebook_sha256']=sha(nbpath)
 for f in execution['inline_figures']:
  if f['path'].endswith('cell_03_00.png'):f['sha256']=sha(out/'cell_03_00.png')
 execution['post_execution_presentation_revision']={'reason':'Two literal percent signs corrected in retained wall-pressure comparison annotation; no data, code, metrics, other outputs or timing changed.','receipt':'qa/course_refresh_early_dsmc_presentation.json','no_simulation_or_training':True}
 write(out/'execution.json',execution)
 record={'original_executed_notebook_sha256':execution['execution_notebook_sha256'],'original_presentation_notebook_sha256':old_receipt.get('original_presentation_notebook_sha256',hashlib.sha256(original_nb).hexdigest()),'original_inline_png_sha256':old_receipt.get('original_inline_png_sha256',hashlib.sha256(old_png).hexdigest()),'final_notebook_sha256':sha(nbpath),'code_sha256':code_sha,'all_other_notebook_values_unchanged':True,'metrics_before_and_after':baseline,'metrics_equal':True,'source_hashes_unchanged':source_hash,'helper_before_sha256':'73f97ae93cc60ff107322b1b78e3e22b7ec2296cdd8d8ccd338e5ec55910b9e4','helper_after_sha256':sha(ROOT/'common/article_validation.py'),'script_sha256':sha(Path(__file__)),'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in (out/'cell_03_00.png',out/'retained_wall_pressure_validation.pdf')},'claim':'Presentation-only replot from retained numerical arrays. The 624.05 s original Run All timing is unchanged; no solver or neural fit executed.'}
 write(receiptpath,record)
 print('Retained validation replot verified; all metrics and six input fingerprints unchanged.')
if __name__=='__main__':main()
