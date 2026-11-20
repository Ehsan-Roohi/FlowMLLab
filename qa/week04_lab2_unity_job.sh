#!/bin/bash
#SBATCH --job-name=w4lab2-cpuqa
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail

work="${W4_QA_ROOT:?Set the isolated, allocated Workspace QA directory}"
test -d "$work/repo"
test -x "$work/venv/bin/python"
cd "$work/repo"
mkdir -p "$work/tmp" tmp/week04_lab2
export TMPDIR="$work/tmp"
export CUDA_VISIBLE_DEVICES=-1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$work/repo"
export MPLCONFIGDIR="$work/repo/tmp/week04_lab2/mpl"
mkdir -p "$MPLCONFIGDIR"
py="$work/venv/bin/python"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$work/start_utc.txt"
"$py" qa/week04_lab2_build.py
"$py" qa/week04_lab2_gradient_check.py
"$py" -m pytest tests/test_cavity_lab2.py -q -p no:cacheprovider > qa/week04_lab2_tests.txt
cat qa/week04_lab2_tests.txt
"$py" qa/week04_lab2_execute.py
"$py" -m pip freeze > qa/week04_lab2_environment.txt
"$py" - <<'PY'
from pathlib import Path
import os,json,hashlib,tarfile,platform
from datetime import datetime,timezone
root=Path.cwd()
work=Path(os.environ['W4_QA_ROOT'])
receipt={
    'job_id':os.environ.get('SLURM_JOB_ID'),
    'partition':os.environ.get('SLURM_JOB_PARTITION'),
    'allocated_cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
    'hostname':platform.node(), 'python':platform.python_version(),
    'start_utc':(work/'start_utc.txt').read_text().strip(),
    'end_utc':datetime.now(timezone.utc).isoformat(),
    'scratch_working_directory':str(root),
    'workspace_expiry':'2026-12-15 14:04:59',
    'durable_destination':r'D:\ChatGPT Backup\Projects\FlowMLLab\Teaching_Current\repo',
    'shared_environment_mutated':False,
    'input_archive_sha256':hashlib.sha256((work/'unity_inputs.tar.gz').read_bytes()).hexdigest(),
    'job_script_sha256':hashlib.sha256((root/'qa/week04_lab2_unity_job.sh').read_bytes()).hexdigest(),
    'capacity_note':'Shared filesystem availability is not personal quota; personal quota was unavailable.'
}
(root/'qa/week04_lab2_unity_run.json').write_text(json.dumps(receipt,indent=2))
files=list((root/'results/week04_lab2').rglob('*'))
files=[p for p in files if p.is_file()]
files += [root/p for p in (
    'notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb',
    'qa/week04_lab2_execution.json','qa/week04_lab2_gradient_check.json',
    'qa/week04_lab2_unity_run.json','qa/week04_lab2_tests.txt',
    'qa/week04_lab2_environment.txt')]
manifest={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest_path=root/'qa/week04_lab2_return_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
files.append(manifest_path)
archive_path=work/'verified_outputs.tar.gz'
with tarfile.open(archive_path,'w:gz') as archive:
    for p in sorted(files):
        archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
checksum=hashlib.sha256(archive_path.read_bytes()).hexdigest()
(work/'verified_outputs.sha256').write_text(checksum+'  verified_outputs.tar.gz\n')
(work/'SUCCESS').write_text(datetime.now(timezone.utc).isoformat()+'\n')
print(json.dumps({'archive':str(archive_path),'bytes':archive_path.stat().st_size,
                  'sha256':checksum,'verified_files':len(manifest)},indent=2))
PY
