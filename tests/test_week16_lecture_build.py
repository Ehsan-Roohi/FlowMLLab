"""Isolated-build routing and retained waveform/POD regression checks."""
from pathlib import Path
from contextlib import nullcontext
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("week16_lecture_builder", ROOT / "qa/week16/build_week16_lecture.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class IsolatedWeek16BuildTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="week16-build-test-", dir=ROOT / "tmp")
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.reference = self.repo / "results/week16_lowboom/reference"
        self.assets = self.repo / "lectures/source/week16_tex_assets"
        self.tex = self.repo / "lectures/source/week16_supersonic_shape_optimization.tex"
        self.pdf = self.repo / "lectures/week16_supersonic_shape_optimization.pdf"
        self.reference.mkdir(parents=True)
        self.assets.mkdir(parents=True)
        part = self.repo / "lectures/source/week16_parts/part1.tex"
        part.parent.mkdir(parents=True)
        part.write_text(r"\includegraphics{retained.png}", encoding="utf-8")
        self.tex.write_text(r"\input{lectures/source/week16_parts/part1.tex}", encoding="utf-8")
        (self.assets / "retained.png").write_bytes(b"reviewed-figure")
        self.pdf.write_bytes(b"reviewed-publication-pdf")
        for name in ("clean_dataset_v801.npz", "cone_refinement_v801.json", "clean_model_audit_v801.json"):
            (self.reference / name).write_bytes(b"preserved-evidence")
        self.patch = patch.multiple(builder, ROOT=self.repo, R=self.reference, ASSETS=self.assets,
                                    TEX=self.tex, PDF=self.pdf)
        self.patch.start()
        self.out = self.base / "isolated-build"

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def compile_stub(self, output_dir, source_tree):
        self.assertEqual(source_tree, output_dir / "source")
        result = output_dir / self.pdf.name
        result.write_bytes(b"new-isolated-pdf")
        return result

    def test_default_compiles_reviewed_copies_without_regenerating(self):
        with patch.object(builder, "build_figures") as figures, patch.object(builder, "compile_pdf", self.compile_stub):
            receipt = builder.main(["--output-dir", str(self.out)])
        figures.assert_not_called()
        self.assertEqual(receipt["mode"], "compile-only with reviewed figures")
        self.assertTrue(receipt["reviewed_inputs_unchanged"])
        self.assertEqual(self.pdf.read_bytes(), b"reviewed-publication-pdf")
        self.assertEqual((self.assets / "retained.png").read_bytes(), b"reviewed-figure")
        self.assertEqual(receipt["reviewed_inputs"]["lectures/source/week16_tex_assets/retained.png"],
                         receipt["staged_inputs"]["lectures/source/week16_tex_assets/retained.png"])

    def test_explicit_regeneration_is_routed_only_to_isolated_stage(self):
        def regenerate(destination):
            self.assertTrue(destination.is_relative_to(self.out))
            (destination / "retained.png").write_bytes(b"isolated-regeneration")
        with patch.object(builder, "build_figures", regenerate), patch.object(builder, "compile_pdf", self.compile_stub):
            receipt = builder.main(["--output-dir", str(self.out), "--rebuild-figures"])
        self.assertEqual((self.assets / "retained.png").read_bytes(), b"reviewed-figure")
        self.assertEqual(self.pdf.read_bytes(), b"reviewed-publication-pdf")
        self.assertNotEqual(receipt["reviewed_inputs"]["lectures/source/week16_tex_assets/retained.png"],
                            receipt["staged_inputs"]["lectures/source/week16_tex_assets/retained.png"])

    def test_publication_paths_and_checkout_ancestors_are_rejected(self):
        for output in (self.repo, self.repo.parent, self.assets, self.repo / "results", self.repo / "lectures/new"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                builder.isolated_directory(output)
        with self.assertRaises(ValueError):
            builder.contained_output(self.out, "../escape")

    def test_compiler_writes_only_isolated_auxiliaries_and_pdf(self):
        stage, _ = builder.stage_sources(self.out)
        calls = []
        def run(command, cwd, check):
            calls.append((command, cwd, check))
            destination = self.out / "tex" / self.pdf.name
            writer = PdfWriter()
            for _ in range(24):
                writer.add_blank_page(width=595, height=842)
            with destination.open("wb") as stream:
                writer.write(stream)
        with patch.object(builder, "configure_matplotlib"), patch.object(builder.shutil, "which", return_value="/usr/bin/xelatex"), patch.object(builder.subprocess, "run", run):
            result = builder.compile_pdf(self.out, stage)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(cwd==stage and check for _,cwd,check in calls))
        self.assertTrue(all(f"-output-directory={self.out / 'tex'}" in cmd for cmd,_,_ in calls))
        self.assertEqual(result, self.out / self.pdf.name)
        self.assertEqual(self.pdf.read_bytes(), b"reviewed-publication-pdf")
        self.assertFalse((self.repo / "lectures/week16_supersonic_shape_optimization.aux").exists())

    def test_reused_tex_output_cannot_alias_the_publication(self):
        stage, _ = builder.stage_sources(self.out)
        (self.out / "tex").mkdir()
        alias = self.out / "tex" / self.pdf.name
        try:
            os.link(self.pdf, alias)
            linked_stat = nullcontext()
        except OSError:
            # exFAT cannot create hardlinks. Exercise the same admission branch
            # with link-count metadata; Linux CI tests a native hardlink above.
            alias.write_bytes(self.pdf.read_bytes())
            original_stat = Path.stat
            def stat(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                if path == alias:
                    fields = list(result)
                    fields[3] = 2
                    return os.stat_result(fields)
                return result
            linked_stat = patch.object(Path, "stat", stat)
        with linked_stat:
            with patch.object(builder.subprocess, "run") as run, self.assertRaises(ValueError):
                builder.compile_pdf(self.out, stage)
            run.assert_not_called()
            self.assertEqual(self.pdf.read_bytes(), b"reviewed-publication-pdf")
            with self.assertRaises(ValueError):
                builder.contained_output(self.out, Path("tex") / self.pdf.name)

    def test_waveform_and_training_only_pod_curves_are_actually_drawn(self):
        x = np.linspace(0, 1, 21)
        y = np.array([np.sin((i+1)*x) + .03*i*x*x for i in range(10)])
        splits = np.array(["train"]*4 + ["validation"]*2 + ["test"]*2 + ["extrapolation"]*2)
        parameters = np.arange(20).reshape(10,2)/20
        np.savez(self.reference / "clean_dataset_v801.npz", x=x, waveforms=y, splits=splits, parameters=parameters)
        cone = {"levels":[{"mesh":{"cells":n},"analytical_comparison":{"relative_error":e}} for n,e in [(100,.2),(400,.1)]]}
        audit = {"same_mesh":{s:{"wave_relative_l2":.1,"peak_mean_relative_error":.2,"drag_mean_relative_error":.3} for s in set(splits)},
                 "finer_mesh":{"per_case_wave_relative_l2":[.2,.15,.1]}}
        (self.reference / "cone_refinement_v801.json").write_text(json.dumps(cone), encoding="utf-8")
        (self.reference / "clean_model_audit_v801.json").write_text(json.dumps(audit), encoding="utf-8")
        captured = {}
        def capture(fig, path, **kwargs):
            captured[Path(path).name] = [(np.asarray(line.get_xdata()).copy(), np.asarray(line.get_ydata()).copy())
                                         for ax in fig.axes for line in ax.lines]
        with patch.object(builder, "configure_matplotlib"), patch.object(Figure, "savefig", capture):
            builder.build_figures(self.out / "figures")
        self.assertEqual(len(captured), 10)
        selected = [2,5,7,9]
        self.assertEqual(len(captured["waveforms_by_split.png"]), 4)
        for index, (xx, yy) in zip(selected, captured["waveforms_by_split.png"]):
            np.testing.assert_array_equal(xx, x)
            np.testing.assert_array_equal(yy, y[index])
        training = y[:4]
        _, sigma, vt = np.linalg.svd(training-training.mean(axis=0), full_matrices=False)
        np.testing.assert_allclose(captured["pod_energy.png"][0][1], np.cumsum(sigma**2)/np.sum(sigma**2), rtol=1e-14)
        self.assertEqual(len(captured["pod_modes.png"]), 3)
        for k, (xx, yy) in enumerate(captured["pod_modes.png"]):
            np.testing.assert_array_equal(xx, x)
            np.testing.assert_allclose(yy, vt[k], rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
