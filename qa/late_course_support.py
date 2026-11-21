"""Repository-aware setup cells for Weeks 15--22 and the optional bridge."""
from textwrap import dedent

TRANSFORMER_PATHS = [
    "flowmllab", "course/lessons", "data/modal_labs",
    "results/transformer_course_v3", "results/transformer_sensor_ablation",
    "notebooks/week07_3", "notebooks/week07_4", "docs",
]

def bootstrap_source(marker, paths, root_name="ROOT"):
    """Return a self-contained cell prefix; no import from a missing checkout."""
    modules = {
        'flowmllab/transformer_course.py': ['flowmllab/late_teaching_plots.py'],
        'results/week15_postaudit': ['flowmllab/geometry_operator_lesson.py'],
        'results/step_operator_audit/wake_predictions.tgz': [
            'qa/step_architecture_v5.py', 'results/step_operator_audit/source/dataset.npz',
            *[f'results/week15_ordinary_deeponet/ordinary-deeponet-seed{seed}/case_metrics.csv'
              for seed in (17,29,43)]],
        'results/family_transolver/comparison.csv': ['flowmllab/late_teaching_plots.py'],
        'results/step_transolver/comparison.csv': ['flowmllab/late_teaching_plots.py'],
        'results/week16_lowboom': ['qa/week16/benchmark.py', 'qa/week16/cfd.py'],
        'flowmllab/geometry_time_bridge.py': ['flowmllab/geometry_time_bridge.py'],
    }
    required = list(dict.fromkeys([marker, *paths, *modules.get(marker, [])]))
    return dedent(f'''\
        from pathlib import Path
        import sys, os, subprocess, importlib.util
        {root_name} = next((p for p in [Path.cwd(), *Path.cwd().parents]
                           if (p / {marker!r}).exists()), None)
        try:
            _in_colab = importlib.util.find_spec("google.colab") is not None
        except (ModuleNotFoundError, ValueError):
            _in_colab = False
        if _in_colab:
            _course_target = {root_name} if {root_name} is not None else Path("/content/FlowMLLab")
            if not _course_target.exists():
                subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                                "--sparse", "https://github.com/Ehsan-Roohi/FlowMLLab.git",
                                str(_course_target)], check=True)
            _course_complete = all((_course_target / p).exists() for p in {required!r})
            if not _course_complete and (_course_target / ".git").exists():
                _sparse = subprocess.run(["git", "-C", str(_course_target), "config", "--bool",
                                          "core.sparseCheckout"], capture_output=True, text=True, check=False)
                if _sparse.stdout.strip() != "true":
                    raise RuntimeError("This full checkout is missing required lesson files; use a complete course checkout or a new Colab runtime.")
                # Add this lesson's dependencies; keep any previous sparse selection.
                subprocess.run(["git", "-C", str(_course_target), "sparse-checkout",
                                "add", *{paths!r}], check=True)
            elif not _course_complete:
                raise RuntimeError("The extracted course folder is incomplete for this lesson. Use the complete portable bundle or a new Colab runtime.")
            {root_name} = _course_target
        if {root_name} is None or not all(({root_name} / p).exists() for p in {required!r}):
            raise RuntimeError("Open this notebook in a FlowMLLab checkout, or use its Open in Colab link and run this setup cell first.")
        {root_name} = {root_name}.resolve()
        if str({root_name}) not in sys.path:
            sys.path.insert(0, str({root_name}))
        print("Course checkout:", {root_name})
        ''')
