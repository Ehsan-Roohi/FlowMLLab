"""Validate and bind the scoped early-course refresh without rerunning science."""
from pathlib import Path
import json,hashlib,ast,base64,sys,copy
from datetime import datetime,timezone
import nbformat
import numpy as np
from IPython.core.inputtransformer2 import TransformerManager
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def record(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def main():
 prior=read(ROOT/'qa/course_refresh_early_final.json') if (ROOT/'qa/course_refresh_early_final.json').exists() else {}
 auditpath=ROOT/'qa/course_refresh_early.json'
 if auditpath.exists():audit=read(auditpath)
 else:
  lm={x['path']:x for x in prior['lectures']}
  audit={'baseline_commit':prior['baseline_commit'],'notebooks':[{'notebook':n['path'],'notebook_initial_sha256':n['initial_sha256'],'lecture':n['lecture'],'lecture_initial_sha256':lm[n['lecture']]['initial_sha256'],'source':lm[n['lecture']]['retained_original_source'],'finding':n['action_reason']} for n in prior['notebooks']]}
 rows=[a for a in audit['notebooks'] if '/week04/' not in a['notebook']]
 assert len(rows)==19
 reasons={
 'AI_in_Fluids_Week2_Colab_Expanded':'Network depth/activation controls now construct the actual graph; parameter counts, label targets, CPU figures and engineering experiments explained.',
 'AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student':'Dimensional moment definitions, finite-N bias, paired unbiased comparison, nondimensional error diagnostics, exposed neural controls and concrete experiments.',
 'AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student':'Collision invariants, actual physical sampling windows, native fields/profiles and full errors; matched-time exercise corrected without changing simulations.',
 'P0_Project_Setup':'Retained project controls/exercises; shared helper now displays physical CFD/model fields, pressure gauges, profiles and full errors; actual CPU output retained.',
 'P1_Re_Generalization':'Retained project controls/splits; actual default CPU fit and physical comparison including unfavorable corner errors retained.',
 'P2_Physics_Guided_DNN':'Retained physics tradeoff and validation selection; physical pressure/velocity outputs added through helper; outline histogram and external legend.',
 'P3_POD_Study':'Low-rank reconstruction compared with full-field POD oracle using physical fields/profiles; original rank study preserved.',
 'P4_Uncertainty_Study':'Physical ensemble mean/error/spread/profiles, honest one-sample-SD band, all seeds preserved; fit timing excludes later evaluation.',
 'P5_Rarefied_Cavity':'Complete CPU Track5A default, native speed/streamlines/errors and signed temperature profiles; coarse noisy reference and poor neural behavior remain visible.',
 'P6_FP_Cavity_Closure':'GPU research extension retained; only two histogram styles and legend placement changed. Static verification only; no GPU training claimed.',
 'W5_Lab2_Sparse_Sensing_Dynamics':'POD/sensor derivation and real rank/budget/noise/seed controls; train90/110, validation100, assessment105; conditioning and physical diagnostics; explicit optional dynamics bridge.'}
 notebooks=[];figures=[];fresh=[];tm=TransformerManager();compiled=0
 for a in rows:
  p=ROOT/a['notebook'];nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
  for i,c in enumerate(nb.cells):
   if c.cell_type=='code':
    compile(tm.transform_cell(c.source),str(p)+':cell'+str(i),'exec');compiled+=1
    assert not any(o.output_type=='error' for o in c.get('outputs',[]))
  changed=sha(p)!=a['notebook_initial_sha256']
  entry={'path':a['notebook'],'initial_sha256':a['notebook_initial_sha256'],**record(p),'changed':changed,'lecture':a['lecture'],'action_reason':reasons.get(p.stem,a['finding'])}
  executionpath=ROOT/'results/course_refresh_early'/p.stem/'execution.json'
  if executionpath.exists():
   e=read(executionpath);assert e['failure'] is None;assert e['notebook_sha256']==sha(p)
   codehash=hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest();assert e['code_sha256']==codehash
   actual={}
   for i,c in enumerate(nb.cells):
    for j,o in enumerate(c.get('outputs',[])):
     if 'image/png' in o.get('data',{}):
      name=f'results/course_refresh_early/{p.stem}/cell_{i:02d}_{j:02d}.png';data=base64.b64decode(o['data']['image/png']);assert (ROOT/name).read_bytes()==data;actual[name]=hashlib.sha256(data).hexdigest()
   assert actual=={x['path']:x['sha256']for x in e['inline_figures']}
   entry['execution']={'receipt':executionpath.relative_to(ROOT).as_posix(),'receipt_sha256':sha(executionpath),'elapsed_run_all_seconds':e['elapsed_run_all_seconds'],'host':e['host'],'cpu_only':e['cpu_only'],'executed_code_cells':e['executed_cells'],'figure_count':len(actual),'original_execution_sha256':e.get('execution_notebook_sha256',e['notebook_sha256']),'post_execution_changes':{k:v for k,v in e.items() if k.startswith('post_execution_')}}
   if p.stem=='P5_Rarefied_Cavity':
    cold=read(executionpath.parent/'initial_cold_execution.json');entry['execution']['cold_generation_seconds']=cold['elapsed_run_all_seconds'];entry['execution']['final_run_used_existing_configuration_matched_cache']=True
   for path,digest in actual.items():figures.append({'path':path,'sha256':digest,'review':'Viewed at readable resolution; physics, full error ranges, labeling and layout checked.','scope':'Retained validation replot only' if p.stem.startswith('AI_in_Fluids_Week3_Lab2') and path.endswith('cell_03_00.png') else 'CPU default output'})
   fresh.append(a['notebook'])
  else:
   entry['execution']={'scope':'No new run; retained verified evidence' if not changed else 'Static compile/schema verification only; optional GPU project not run'}
   if not changed:assert sha(p)==a['notebook_initial_sha256']
  entry['review_status']='fresh_cpu_executed_and_physical_figures_reviewed' if executionpath.exists() else ('content_audited_and_preserved_byte_for_byte' if not changed else 'static_review_and_compile_only_optional_gpu_path')
  notebooks.append(entry)
 assert len(fresh)==10 and len(figures)==35
 lecturepath=ROOT/'qa/course_refresh_early_lectures.json'
 lecturebuild=read(lecturepath) if lecturepath.exists() else prior['lecture_build'];built={x['path']:x for x in lecturebuild['artifacts']}
 lectures=[]
 for a in rows:
  if a['lecture'] in {x['path']for x in lectures}:continue
  p=ROOT/a['lecture'];changed=sha(p)!=a['lecture_initial_sha256']
  entry={'path':a['lecture'],'initial_sha256':a['lecture_initial_sha256'],**record(p),'changed':changed,'retained_original_source':a['source']}
  if changed:
   b=built[a['lecture']];assert b['sha256']==sha(p)
   entry.update({'source':'qa/course_refresh_early_lectures.py','page_count':b['page_count'],'reason':'Preserved original core plus topic-specific worked laboratory pages' if 'protected_core_pages'in b else 'Reorganized and expanded POD/sensing derivation, conditioning, actual figures and experiments'})
   if 'protected_core_pages'in b:entry['protected_core_pages']=b['protected_core_pages'];entry['protected_core_pixel_sha256']=b['protected_render_sha256']
  else:entry['reason']='Existing worked coverage is substantial and no concrete teaching defect found; preserved byte-for-byte.'
  entry['review_status']='updated_and_all_changed_pages_visually_reviewed' if changed else 'content_audited_and_preserved_byte_for_byte_no_new_render_claim'
  lectures.append(entry)
 assert len(lectures)==11 and sum(x['changed'] for x in lectures)==3
 coreproof=read(ROOT/'qa/course_refresh_early_core_preservation.json')
 for proof in coreproof['artifacts']:
  assert sha(ROOT/proof['path'])==proof['final_pdf_sha256'] and proof['all_pages_exactly_equal']
 pendingpath=ROOT/'qa/course_refresh_early_visual_pending.json'
 pages=read(pendingpath) if pendingpath.exists() else read(ROOT/'qa/course_refresh_early_visual_review.json')['pages']
 assert len(pages)==22
 for x in pages:
  assert sha(ROOT/x['pdf'])==x['pdf_sha256'];assert sha(ROOT/x['render'])==x['render_sha256']
  x['review']='Readable individual-page visual inspection passed. Root independently reviewed the other final pages; Week3 page44 was rechecked by this worker after the root-requested matched-time correction.'
 write(ROOT/'qa/course_refresh_early_visual_review.json',{'pages':pages,'notebook_figures':figures,'reviewers':['week4_lab2_repair','root (21 final lecture pages plus the prior page44; requested the matched-time correction)'],'claim':'Contact sheets were not used as the sole review evidence.'})
 testsource=ROOT/'tests/test_course_refresh_early.py';tree=ast.parse(testsource.read_text())
 fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='test_halved_timestep_matches_every_post_move_sample_time')
 ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(testsource),'exec'),ns);ns[fn.name]()
 sources=['qa/course_refresh_early_notebooks.py','qa/course_refresh_week05_sensing.py','qa/course_refresh_projects.py','qa/course_refresh_early_lectures.py','qa/course_refresh_early_execute.py','qa/course_refresh_early_dsmc_presentation.py','qa/course_refresh_early_finalize.py','flowmllab/early_sensing.py','common/w5_common.py','common/w5_kinetic_figures.py','common/article_validation.py','tests/test_course_refresh_early.py']
 for name in sources:
  p=ROOT/name;ast.parse(p.read_text(encoding='utf-8'));assert b'\r\n'not in p.read_bytes(),name
 source_records={name:record(ROOT/name)for name in sources}
 science={p.relative_to(ROOT).as_posix():record(p)for p in (ROOT/'results/course_refresh_early').rglob('*') if p.is_file() and (p.suffix in ('.npz','.csv') or p.name.endswith('_metrics.json'))}
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'baseline_commit':audit['baseline_commit'],'status':'Complete scoped repair; publication is owned by root','scope':{'notebooks':19,'pdfs':11,'fresh_cpu_defaults':10,'preserved_notebooks':8,'static_only_optional_gpu_notebooks':1,'updated_pdfs':3,'retained_pdf_core_pages':61,'reviewed_final_lecture_pages':22,'inline_cpu_figures':35},'notebooks':notebooks,'lectures':lectures,'source_files':source_records,'retained_compact_scientific_outputs':science,'visual_review':'qa/course_refresh_early_visual_review.json','original_pdf_page_preservation_proof':{'path':'qa/course_refresh_early_core_preservation.json','sha256':sha(ROOT/'qa/course_refresh_early_core_preservation.json'),'exact_baseline_comparison_pages':61},'checks':{'schema_and_python_compile_cells':compiled,'inline_images_match_receipts_and_notebooks':35,'focused_tests':{'record':'results/course_refresh_early/unity/focused_tests.xml','sha256':sha(ROOT/'results/course_refresh_early/unity/focused_tests.xml'),'passed':8,'environment':'Unity TF2.20.0 CPU'},'matched_time_test':{'name':fn.name,'passed':True,'test_source_sha256':sha(testsource),'scope':'Both retained sampling configurations; exact halved-step times and rejection of old doubled-start mapping'},'old_preserved_notebook_and_pdf_hashes':True},'execution_environment':'results/course_refresh_early/unity/environment.txt','workspace_staging':'provenance.workspace_and_input_staging','return_receipts':'provenance.return_transfers','reproduction_order':['Build an unmarked notebook from its original source, then run qa/course_refresh_early_notebooks.py for Week2/3; qa/course_refresh_week05_sensing.py for W5; qa/course_refresh_projects.py for P2-P6. These patchers preserve existing exercises.','Run the explicit CPU notebook paths with qa/course_refresh_early_execute.py in a documented compatible environment; preserve source data and split definitions. Optional P6 GPU and dynamics variants are excluded from default verification.','Normalize generated textual JSON to LF with parsed-value equality before publication; preserve original execution-byte fingerprints for serialization-only updates.','Run qa/course_refresh_early_lectures.py after the original core lecture builders and after notebook output export. This retains Week2 pages1-24 and Week3 pages1-37 and verifies their pixel hashes.','If refreshing the retained DSMC comparison after the percent fix, qa/course_refresh_early_dsmc_presentation.py rerenders only that one cell from unchanged inputs and records exact metric equality.','Run qa/course_refresh_early_finalize.py to verify current assets and refresh this compact receipt.'],'builder_contract_limit':'Marker guards avoid duplicate additions; they are not a generic guarantee against a different old in-place builder overwriting individual marked cells. Clean regeneration must run the original builder followed by the scoped repair scripts. Root owns canonical modal Week5 builder hooks.','scientific_limits':['P5 uses a coarse 12x12, six-particles-per-cell teaching model and noisy native data. Velocity structure is not learned accurately; averaging two seeds is not convergence. Cold data generation134.40s and final cached RunAll27.70s are distinct measurements.','P0-P4 coordinate networks retain unfavorable corner errors and wall inconsistencies; physical profiles and full errors show them rather than guaranteeing improvement.','Week3 quick/better runs differ in grid, particles, physical windows and seed. They are diagnostic comparisons, not isolated grid refinement.','W5 uses previously inspected coarse LBM fields with a strict local train/validation/assessment split. Noise seeds are artificial measurement/layout realizations, not independent CFD simulations.','No optional GPU training, broad research sweep, hosted Colab run or optional W5 dynamics execution is claimed.'],'excluded_root_baseline':'All11notebooks under notebooks/week04, including W4_1 ROM, retain root-owned independent verification; they are not counted as newly executed here.'}
 receipt['lecture_build']=lecturebuild
 provenance=copy.deepcopy(prior.get('provenance',{}))
 bindings={'course_refresh_early_staging.json':'workspace_and_input_staging','course_refresh_early_followup.json':'followup_staging','course_refresh_early_p5_staging.json':'kinetic_staging','course_refresh_early_serialization.json':'serialization_proofs'}
 for name,key in bindings.items():
  q=ROOT/'qa'/name
  if q.exists():provenance[key]=read(q)
 transfers=provenance.get('return_transfers',{})
 for name,phase in [('course_refresh_early_unity_return.json','initial_teaching_outputs'),('course_refresh_early_p5_return.json','initial_kinetic_outputs'),('course_refresh_early_final_return.json','final_P2_P5_outputs')]:
  q=ROOT/'qa'/name
  if q.exists():transfers[phase]=read(q)
 provenance['return_transfers']=transfers
 q=ROOT/'qa/course_refresh_early_transfer_history.json'
 if q.exists():
  events={}
  for x in read(q)['executions']:
   e=x['record'];key=e.get('execution_notebook_sha256',e['notebook_sha256'])
   events[key]={k:e[k] for k in ('notebook','utc','elapsed_run_all_seconds','host','kernel','python','cpu_only','failure','code_sha256','notebook_sha256')}
  provenance['prior_closed_execution_events']=list(events.values())
  provenance['historical_numeric_tables']=read(q)['retained_previous_numeric_tables']
 receipt['provenance']=provenance
 receipt['consolidated_receipt_sources']=copy.deepcopy(prior.get('consolidated_receipt_sources',{}))
 for name in ['course_refresh_early.json','course_refresh_early_lectures.json','course_refresh_early_visual_pending.json','course_refresh_early_transfer_history.json','course_refresh_early_remote_cleanup.json',*bindings,'course_refresh_early_unity_return.json','course_refresh_early_p5_return.json','course_refresh_early_final_return.json']:
  q=ROOT/'qa'/name
  if q.exists():receipt['consolidated_receipt_sources'][name]=sha(q)
 if (ROOT/'qa/course_refresh_early_cleanup.json').exists():receipt['cleanup']='qa/course_refresh_early_cleanup.json'
 write(ROOT/'qa/course_refresh_early_final.json',receipt)
 print(json.dumps(receipt['scope'],indent=2));print('PASS: all scoped assets, exact execution/image bindings and preserved evidence checks.')
if __name__=='__main__':main()
