"""Create a portable Weeks 17-22 archive with data, evidence and editable sources."""
from pathlib import Path
import argparse,hashlib,json,zipfile

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
paths=set()
for pattern in ('flowmllab/**/*.py','course/lessons/*.json','data/modal_labs/*',
                'results/transformer_course_v3/*','results/transformer_budget_audit/*','results/transformer_sensor_ablation/*',
                'lectures/source/week*_transformer_course.md','lectures/week*_cfd_transformer.pdf',
                'notebooks/week1[789]/*','notebooks/week2[012]/*','instructor/week*/*.ipynb',
                'qa/*transformer*','tests/test_transformer*.py','docs/TRANSFORMER*.md',
                'docs/assets/transformer-contours/*','weeks/week1[789]/*','weeks/week2[012]/*',
                'RELEASE_NOTES_v*.md'):
    paths.update(p for p in root.glob(pattern) if p.is_file())
for name in ('pyproject.toml','LICENSE','README.md','COURSE_MAP.md','START_HERE.md','CITATION.cff','.zenodo.json',
             'docs/RESULTS_GUIDE.md','notebooks/week07_3/README.md','notebooks/week07_4/README.md',
             'data/modal_labs/README.md','results/modal_labs/metrics.json','notebook_execution.json'):
    p=root/name
    if not p.is_file():raise FileNotFoundError(p)
    paths.add(p)
# A sparse checkout must not silently produce an incomplete teaching package.
for week in range(17,23):
    for name in (f'course/lessons/week{week}.json',
                 f'notebooks/week{week}/W{week}_CFD_Transformer.ipynb',
                 f'instructor/week{week}/W{week}_Solutions.ipynb',
                 f'lectures/week{week}_cfd_transformer.pdf',
                 f'weeks/week{week}/README.md',
                 f'docs/assets/transformer-contours/week{week}.png'):
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        paths.add(p)
readme='''# Portable FlowMLLab Weeks 17-22

Start with docs/TRANSFORMER_COURSE.md. This archive contains the redesigned six
weeks, their inputs, executed student and instructor notebooks, editable lecture
sources, six PDFs, retained model checkpoints and numerical validation records.
It is a course subset; links to earlier weeks refer to the full public repository.

Use an isolated Python 3.12 environment. Install with:
    python -m pip install -e ".[transformer,test]" -c qa/transformer-environment.lock.txt
Then run:
    python qa/verify_transformer_course.py
    python qa/build_transformer_course.py --check

The recorded lock and local validation were produced on Windows. GitHub Actions
checks Ubuntu; consult the linked workflow result for the published revision.
Interactive Colab and student workload acceptance remain open. Instructor solutions
are in instructor/. Students should start
with notebooks/week17/W17_CFD_Transformer.ipynb and complete the coding tasks.
Retained models need not be trained to read the course. Fresh training writes to
a NEW empty directory with qa/run_transformer_course.py --output NEW_EMPTY_DIR.
See docs/TRANSFORMER_SECOND_REVIEW_RESPONSE.md for current correction status and remaining classroom acceptance checks. The earlier review responses and validation JSON are historical.
'''
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(paths):z.write(p,'FlowMLLab/'+p.relative_to(root).as_posix())
    z.writestr('FlowMLLab/PORTABLE_README.md',readme)
    z.writestr('FlowMLLab/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(args.output) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():assert hashlib.sha256(z.read('FlowMLLab/'+name)).hexdigest()==digest,name
checksum=hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_suffix('.zip.sha256').write_text(checksum+'  '+args.output.name+'\n',encoding='utf-8')
print(f'PASS: {len(manifest)} source files, archive CRC and every SHA-256 checked; {args.output.stat().st_size} bytes')
