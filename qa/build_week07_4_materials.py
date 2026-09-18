#!/usr/bin/env python3
"""Build the Week 7.4 notebook and lecture PDF from retained evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qa'))
from add_colab_entrypoints import bootstrap
NOTEBOOK = ROOT / "notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb"
PDF = ROOT / "lectures/week07_4_diverse_wake_pretraining.pdf"
RESULTS = ROOT / "results/week07_4_diverse_pretraining"


def md(text):
    cell_id = hashlib.sha256(("markdown\0" + text).encode("utf-8")).hexdigest()[:12]
    return {"cell_type": "markdown", "metadata": {}, "source": text, "id": f"m-{cell_id}"}


def code(text):
    cell_id = hashlib.sha256(("code\0" + text).encode("utf-8")).hexdigest()[:12]
    return {"cell_type": "code", "metadata": {}, "source": text, "execution_count": None,
            "outputs": [], "id": f"c-{cell_id}"}


def build_notebook() -> None:
    cells = [
        md("""# Week 7.4 - Diverse-wake pretraining and label-efficient lift decoding
<!-- MIE690A article-aligned validation v4 -->

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb)

This independent extension asks the MAPA question on fluid data: can unlabelled
pretraining across many trajectories reduce the labels needed on a new one?
The default path reads frozen evidence; set `RUN_FULL=True` to regenerate it."""),
        code(bootstrap('notebooks/week07_4') + """from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
from IPython.display import Image, display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'flowmllab').is_dir())
RESULTS = ROOT/'results/week07_4_diverse_pretraining'
RUN_FULL = False
if RUN_FULL:
    subprocess.run([sys.executable, str(ROOT/'qa/run_week07_4_protocol.py')], check=True)
metrics = json.loads((RESULTS/'metrics.json').read_text())
print('development', metrics['protocol']['development'])
print('validation', metrics['protocol']['validation'])
print('untouched test', metrics['protocol']['test'])"""),
        md("""## 1. Information contract

Complete Reynolds trajectories stay on one side of the split.  The MAE uses
only development velocity fields; Re105 selects stopping and ridge strength.
Ridge selection fits on all 2761 development lift labels and scores 251
validation labels. These shared source-label costs are additional to k target
labels. The study measures target-label efficiency, not total-label efficiency.
For each test trajectory labels come from `[0,150)`, a 30-frame gap follows,
and `[180,251)` is scored.  The target is instantaneous lift, not field
completion, so POD is a serious representation baseline rather than a nearly
closed-form solver of the task."""),
        code("""protocol = metrics['protocol']
display(pd.Series({k: protocol[k] for k in ('label_pool','test_window','mask_ratio','pretrain_steps','field','downstream_label')}))
from flowmllab import diverse_wake_pretraining as dw
data_root = ROOT/'data/week07_4_wakes'
manifest = dw.load_manifest(data_root)
for case in manifest['cases']:
    assert dw.sha256(data_root/case['file']) == case['sha256'] == metrics['data_hashes'][case['file']]
print('All 16 dataset hashes match the retained run.')"""),
        md(r"""## Definitions before implementation

A snapshot is the transverse velocity $v/U$ on a $32\times78$ crop.
Each $4\times6$ patch contains 24 values; there are 104 patches. Pretraining
minimizes mean squared error only on hidden patches. At inference all patches
are visible. Averaging encoder tokens gives $z\in\mathbb{R}^{64}$.

The ridge probe fits $\hat C_L=z^Tw+b$ with an L2 penalty on $w$.
NRMSE is $100\sqrt{\langle(\hat C_L-C_L)^2\rangle}/\sigma(C_L)$,
where the denominator uses only the scored reference window. It measures
decoding of instantaneous force given the current field, not autonomous
forecasting. Lift is generated alongside fields by the solver, so the label
budget is an educational simulation of scarce observations, not a measured
reduction in LBM production cost."""),
        code("""from flowmllab.masked_pretraining import PatchLayout, RidgeReadout
layout = PatchLayout()
frames, lift, times = dw.load_case(data_root, 105)
patches = layout.patchify(frames[:2])
print('snapshot shape:', frames.shape, 'patch shape:', patches.shape)
assert np.allclose(layout.unpatchify(patches), frames[:2])
print('Patch round-trip is exact; lattice time range:', times[[0,-1]])"""),
        md("""## 2. Masked pretraining and representation transfer

Seventy-five percent of 4x6 patches are hidden.  The encoder sees visible tokens
only; the decoder reconstructs masked velocity.  At downstream time the visible
tokens are mean pooled to a fixed-dimensional 64-vector (not mathematically
mask-invariant). A ridge readout learns
lift from k frames.  The identical random encoder and POD-32 coefficients receive
the same frames and the same validation-only ridge selection."""),
        code("""display(Image(filename=str(RESULTS/'pretraining_loss.png')))
print(metrics['model'])"""),
        md("""## 3. Principal result: label efficiency on unseen trajectories

The curve aggregates labelled-frame draws within each trajectory, then reports
the mean and range across four independent test Reynolds cases.  This avoids
treating masks from one trajectory as independent flows."""),
        code("""display(Image(filename=str(RESULTS/'label_efficiency.png')))
rows=[]
for k in protocol['ks']:
    rows.append({'k':k, **{m:metrics['curves'][m][str(k)]['mean'] for m in metrics['curves']}})
display(pd.DataFrame(rows).round(2))
records = pd.DataFrame(metrics['records'])
per_case = records.groupby(['reynolds','k','method'])['nrmse'].mean().unstack('method')
display(per_case.loc[(slice(None), [32,64,128]), :].round(2))"""),
        md("""## 4. One untouched trajectory

Re115 below was not used for architecture, stopping, POD rank or ridge strength.
All methods use the same 32 early labels; the displayed window is separated by
the frozen gap."""),
        code("display(Image(filename=str(RESULTS/'re115_lift_decoding.png')))"),
        md("""## 5. Reading the evidence honestly

The pretrained encoder beats the random encoder at every k.  Its 12.31% NRMSE
at k=32 is already below the random encoder's 16.02% at k=128: better mean
accuracy with one-quarter as many target labels. POD remains better through k=32;
pretraining passes it only at k=64 and 128. These are quick educational LBM
trajectories, not grid-independent DNS or a force-validation benchmark.
Only one encoder initialization was trained; the three seeds resample labels,
not pretraining. POD rank is fixed at 32 and is not optimized over ranks. Initial
condition and Reynolds number vary together, with one run per Reynolds number;
this does not isolate initial-condition generalization. The inspected test set
is now retained evidence and must not guide further tuning. Average superiority
is not a significance test or a guarantee for each trajectory.

### Exercises
1. Replace instantaneous lift with a five-snapshot-ahead target.
2. Add fixed 2D coordinate and relative-position encodings.
3. Repeat with a geometry or inlet-profile shift held out by family.
4. Compare representation CKA across Reynolds number and perturbation sign."""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def build_pdf() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title74", parent=styles["Title"], fontSize=24, leading=29,
                              textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="H174", parent=styles["Heading1"], textColor=colors.HexColor("#17324D"), spaceAfter=10))
    styles.add(ParagraphStyle(name="Body74", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=9))
    story = [Spacer(1, .65*inch), Paragraph("Week 7.4", styles["Title74"]),
             Paragraph("Diverse-wake pretraining and label-efficient force decoding", styles["Title74"]),
             Paragraph("A trajectory-split, MAPA-inspired experiment on 16 educational LBM wakes", styles["Body74"]),
             Spacer(1, .25*inch), Image(str(RESULTS / "label_efficiency.png"), width=6.7*inch, height=4.25*inch), PageBreak()]
    sections = [
        ("Why a new week?", "Week 7.3 asks a network to complete a periodic low-rank field, a task gappy POD almost solves analytically. MAPA transfers a representation learned from heterogeneous unlabelled signals to an external label. Week 7.4 follows that design: the MAE sees velocity only; the downstream target is cylinder lift."),
        ("Frozen split", "Development: Re 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 130. Validation: Re 105. Untouched test: Re 95, 115, 125, 135. Target labels come from frames [0,150); frames [180,251) are scored. No random frame split crosses trajectory roles."),
        ("Model", "A 90k-parameter masked autoencoder hides 75% of 4 by 6 patches and trains for 6000 AdamW steps. The encoder sees visible tokens only. Downstream, all visible tokens are mean pooled to a 64-component representation; ridge strength is selected on Re105."),
        ("Baselines", "The identical random encoder tests the value of pretraining. POD-32 coefficients provide a strong classical representation. Every representation receives exactly the same k labels and ridge protocol."),
        ("Definitions and reading the metric", "A 32 by 78 snapshot is divided into 104 patches, each with 24 velocity values. The hidden-patch MSE supplies the pretraining signal. The downstream ridge readout minimizes squared lift error plus an L2 penalty on its weights. NRMSE (%) is 100 times RMSE divided by the scored lift standard deviation. The current field is available when decoding current lift; this is not an autonomous forecast. Lift is produced alongside the solver fields, so label scarcity is simulated rather than an observed saving in CFD cost."),
    ]
    for title, body in sections:
        story += [Paragraph(title, styles["H174"]), Paragraph(body, styles["Body74"])]
    story += [PageBreak(), Paragraph("Retained numerical evidence", styles["H174"])]
    methods = ["pretrained_encoder", "random_encoder", "pod32"]
    table = [["k", "Pretrained", "Random", "POD-32"]]
    for k in metrics["protocol"]["ks"]:
        table.append([str(k)] + [f"{metrics['curves'][m][str(k)]['mean']:.2f}" for m in methods])
    tab = Table(table, colWidths=[.7*inch, 1.25*inch, 1.25*inch, 1.25*inch])
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#AAB7C4")),("ALIGN",(0,0),(-1,-1),"CENTER"),
                             ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF3F7")]),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [Paragraph("Lift NRMSE (%), mean across four unseen trajectories", styles["Body74"]), tab, Spacer(1,.2*inch),
              Paragraph("At k=32, pretrained 12.31% is below random-at-k=128 16.02%: better mean accuracy with one-quarter as many target labels. POD leads through k=32; pretrained passes it in the mean at k=64 and 128.", styles["Body74"]),
              Image(str(RESULTS / "re115_lift_decoding.png"), width=6.7*inch, height=3.65*inch), PageBreak(),
              Paragraph("Scope and reproducibility", styles["H174"]),
              Paragraph("Ridge selection uses 2761 development and 251 validation lift labels in addition to the reported target-label budget. Only one encoder initialization is retained; the three seeds are label draws. POD rank 32 is fixed, not rank-optimized. One run per Reynolds number confounds Reynolds and initial-condition variation. The test cases are now inspected retained evidence; future tuning needs fresh tests. Mean differences are not statistical significance claims.", styles["Body74"]),
              Paragraph("The retained success is representation transfer for an external force label, not a claim that transformers universally beat POD. The solver uses D=6 quick D2Q9-TRT trajectories and periodic transverse boundaries; the data are not grid-independent DNS. Re140 was rejected after the solver became non-physical. The histories do not support a validated Strouhal claim.", styles["Body74"]),
              Paragraph("Reproduce: generate data with qa/generate_week07_4_wakes.py; train and score with qa/run_week07_4_protocol.py; inspect hashes and every record in results/week07_4_diverse_pretraining/metrics.json.", styles["Body74"]),
              Paragraph("References", styles["H174"]),
              Paragraph("Tang, Spalding and Cogan (2026), Pretraining for Sample-Efficient Neural Interfaces (MAPA), arXiv:2609.13507. He et al. (2022), Masked Autoencoders Are Scalable Vision Learners. Everson and Sirovich (1995), Karhunen-Loeve procedure for gappy data.", styles["Body74"])]
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=letter, rightMargin=.65*inch, leftMargin=.65*inch,
                            topMargin=.55*inch, bottomMargin=.55*inch,
                            title="Week 7.4 - Diverse-wake pretraining")
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#526878'))
        canvas.drawString(47, 24, 'FlowMLLab / Ehsan Roohi / Week 7.4')
        canvas.drawRightString(565, 24, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--notebook-only", action="store_true"); parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    if not args.pdf_only: build_notebook(); print(NOTEBOOK.relative_to(ROOT))
    if not args.notebook_only: build_pdf(); print(PDF.relative_to(ROOT))
