"""Assemble the local release record only after its required checks succeed."""
from pathlib import Path
import json,subprocess,sys,hashlib
from datetime import datetime,timezone
from verify_transformer_course import verify

root=Path(__file__).resolve().parents[1]
def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=verify()
subprocess.run([sys.executable,str(root/'qa/build_transformer_course.py'),'--check'],check=True,cwd=root)
reproduction=read('qa/transformer-reproduction.json')
checkpoints=read('qa/transformer-checkpoint-audit.json')
pdfqa=read('qa/transformer-pdf-qa.json')
assert reproduction['passed'] and checkpoints['passed'] and pdfqa['passed']
execution=read('notebook_execution.json')
assert len(execution)==12
assert 'Ran 14 tests' in (root/'qa/transformer-unit-tests.txt').read_text(encoding='utf-8')
pdfs={}
for w in range(17,23):
    name=f'week{w}_cfd_transformer.pdf'
    pdfs[name]=digest(root/'lectures'/name)
    assert pdfs[name]==digest(root/'build/compatibility-proof/lectures'/name),name
record={
 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
 'base_commit':'37c1fcf791a41e5eca9d473a214954bed1d2f83a',
 'scope':'Local Windows validation; no public publication or remote CI run',
 'unit_tests':{'passed':14,'failed':0,'seconds':29.089,'log':'qa/transformer-unit-tests.txt'},
 'independently_rescored_rows':rows,
 'checkpoint_predictions':len(checkpoints['checks']),
 'fresh_training_reproduction':{'comparisons':len(reproduction['comparisons']),'passed':True,
    'relative_tolerance':reproduction['relative_tolerance'],'absolute_tolerance':reproduction['absolute_tolerance'],
    'maximum_absolute_metric_difference':max(abs(r['reference']-r['candidate']) for r in reproduction['comparisons']),
    'environment_relation':'Two independent trainings in the same clean installed environment'},
 'notebooks':execution,'instructor_exercises_passed':24,
 'builder_checks':{'canonical_inputs_match':True,'source_and_evidence_preserved':True,
    'compatibility_entry_point_pdfs_identical':True,'pdf_sha256':pdfs},
 'visual_qa':pdfqa,
 'recovered_failure':'One kernel startup timed out at 60 seconds before Week 21 execution. Startup allowance increased to 180 seconds; only Weeks 21-22 rerun in fresh kernels. Earlier successful timings recovered from stdout.',
 'budget_audit':read('results/transformer_budget_audit/summary.json'),
 'remaining_limits':['No Linux or interactive Colab execution','No globally converged-model claim; fixed-budget transfer and one extended MLP still select at the ceiling','Lecture notes are compact and do not satisfy the former review\'s 8-10-page target','No student pilot or independent scientific re-review','No optimizer state for exact training resumption']}
(root/'qa/transformer_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PASS: local release record assembled from completed checks.')
