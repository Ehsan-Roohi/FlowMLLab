"""Build reviewed Weeks 1.2–6 teaching artifacts without numerical regeneration.

Run from any directory with the repo's dependencies and XeLaTeX installed.
Only explicit --notebooks / --pdf actions write artifacts. Retained results,
scientific plots and datasets are read-only inputs to this renderer.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'qa')]


def build_pdfs(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    from qa.lecture_typography import font_paths
    font_paths()  # reject silent font substitution before writing any PDF
    sources = {
        'week02_lecture': 'week02_supervised_learning_rarefaction',
        'week04_lecture': 'week04_cavity_surrogates_deeponet',
        'weeks05_06_project_guide': 'week05_06_project_guide',
        'week02_1_probabilistic_uq_notes': 'week02_1_probabilistic_uq',
    }
    for source, target in sources.items():
        for _ in range(2):
            proc = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                   '-output-directory=' + str(build_dir.resolve()),
                                   'lectures/source/' + source + '.tex'],
                                  cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (build_dir / (source + '.console.txt')).write_text(proc.stdout)
            if proc.returncode:
                raise RuntimeError(f'{source} failed; inspect {build_dir}:\n{proc.stdout[-1500:]}')
        shutil.copy2(build_dir / (source + '.pdf'), ROOT / 'lectures' / (target + '.pdf'))
    import build_week01_2 as pressure
    manifest, _ = pressure.load_results(pressure.OUT)
    pressure.lecture(manifest)  # do not call its mixed numerical/figure main()
    import build_week04_2_stokes_correction as stokes
    retained = stokes.RESULTS
    stokes.build_pdf({'errors': retained / 'week04_2_error_matrix.png',
                      'lids': retained / 'week04_2_lid_families.png',
                      'history': retained / 'week04_2_validation_history.png'})
    import build_week04_2_grid51 as grid
    import pandas as pd
    grid.pdf(pd.read_csv(grid.R / 'grid_comparison.csv'), pd.read_csv(grid.R / 'grid_shift.csv'))
    import build_modal_materials as modal
    modal.make_pdf('week05')
    # Week7 shares only the typography renderer; no experiment is rerun.
    modal.make_pdf('week07')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if not (args.notebooks or args.pdf):
        parser.error('Choose --notebooks and/or --pdf explicitly')
    if args.notebooks:
        from qa.notebook_review_updates import PATCH_FILE, review_notebook_file
        for relative in json.loads(PATCH_FILE.read_text()):
            review_notebook_file(ROOT / relative)
    if args.pdf:
        if args.build_dir:
            build_pdfs(args.build_dir)
        else:
            with tempfile.TemporaryDirectory(prefix='flowmllab-lecture-build-') as folder:
                build_pdfs(folder)


if __name__ == '__main__':
    main()
