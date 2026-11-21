"""Maintain the late-course laboratory maps in their editable lecture sources."""
from pathlib import Path
import re
from week15_operator_section import LECTURE

ROOT=Path(__file__).resolve().parents[1]

W16=r"""\topic{20}{From the lecture to the executable CPU laboratory}
\textbf{Aim.} Complete the physics-to-surrogate chain using the retained accepted CFD cases, then explain why a newly optimized shape still needs a new governing-equation calculation. Ordinary Run All performs small CPU calculations and fits; it does not launch SU2.

\begin{center}\small
\begin{tabularx}{\linewidth}{p{.16\linewidth}X X}
\toprule
Notebook stage & Algorithm and information & Inspect and interpret\\\midrule
Sections 1--4 & Pressure normalization, fixed-volume geometry and Taylor--Maccoll ODE & Units, constant volume and the analytic cone verification\\
Sections 5--9 & Geometry partition, training-only POD, ridge/GP/MLP fits & Fresh validation errors and all three methods' pressure profiles\\
Section 10 & Saved neural checkpoint, fixed normalization and stored splits & Current inference against the retained verification record\\
Sections 11--13 & Surrogate grid search and the independently solved retained design & A new proposal versus an already verified design; they are not interchangeable\\\bottomrule
\end{tabularx}
\end{center}

\textbf{What students can change.}
\begin{itemize}
\item \texttt{POD\_RANK=12}: number of pressure modes.
\item \texttt{MLP\_WIDTH=32} and \texttt{MLP\_DEPTH=2}: network capacity.
\item \texttt{MLP\_MAX\_ITER=3000}: optimization budget.
\item \texttt{MODEL\_SEED=16}: reproducible initialization.
\end{itemize}
The tanh MLP uses L-BFGS; its iteration limit is a budget, not evidence of convergence. The retained neural checkpoint is a separate fixed artifact and is not replaced by these controls.

\textbf{Algorithm.}
First preserve the 24/6/8/6 training, validation, interpolation-test and extrapolation groups. Fit the pressure mean and POD basis on the 24 training cases only, transform the geometry inputs using training statistics, and fit ridge, Gaussian-process and MLP maps to coefficients/drag. Compare candidate models on validation using the declared criterion. The new all-method plot uses the first validation case fixed in the split, so viewing it does not select an easy test case. Only then freeze the choice and inspect the existing test-waveform comparisons.

\textbf{Read a pressure comparison.}
The upper panel compares $\Cp=(p-p_\infty)/(\frac12\rho_\infty U_\infty^2)$ at the same observation line and physical axial locations. The lower panel shows signed predicted-minus-CFD $\Cp$; a phase shift of a compression peak produces opposite-sign lobes. Do not reduce the conclusion to peak error alone: shock location, expansion strength, waveform norm and pressure drag can improve differently. Every method sees the same truth and coordinate array.

\textbf{A controlled experiment.}
Compare POD ranks 6 and 12 at fixed split/seed, then widths 16 and 32 at fixed rank/depth. Record fit time, warnings, selected model, validation waveform/peak/drag errors and both positive and negative changes in the signed profiles. Choose from validation before evaluating test cases. Increasing depth and increasing the iteration ceiling simultaneously does not isolate capacity.

\textbf{The optimization boundary.}
Changing a surrogate changes the proposed optimum. The retained CFD files validate their recorded shape and conditions only: they do not validate a different proposed shape merely because the notebook can plot both. Submit the proposal's geometry coordinates and constraint values; mark it unverified until a separate accepted CFD calculation is available.
"""

BRIDGE="""
## Read the two diagnostic experiments

**Aim:** explain the difference between a one-step map and a model that consumes its own predictions. The notebook's first new figure plots the same manufactured scalar state with observed, teacher-forced and autonomous histories. Each autonomous update drops the oldest state and appends the prediction. The teacher-forced control instead appends truth, so it answers a different question even when both curves start at the same value. Poisoning future truth must leave the autonomous prediction unchanged while its scored error changes.

The second figure compares a manufactured periodic reference with a controlled amplitude/phase perturbation. Its lower panel retains the signed difference. A small phase shift can create a large pointwise error without changing amplitude. Fit the same declared time window for both signals, report amplitude ratio and circular phase error separately, and inspect the residual before interpreting a mixed-frequency signal with a single-harmonic fit. A constant prediction has undefined phase.

These figures are diagnostics of the implemented contracts. The original four exercises remain: group ownership, boundary-strip weighting, autonomous feedback, and phase with missingness. Keep the geometry partition fixed when making windows; a longer history is not permission to move one geometry into both training and testing. Submit the requested adverse-case checks and the bounded claim ledger, not only the visually closest curve.
"""

PHYSICAL={
17: [
 ('week17_sensors_01.png','Re=110, frame 100: native transverse velocity v/U with the sixteen observation locations used in the attention-mechanism demonstration. Coordinates are in cylinder diameters. The rectangular wake ROI boundary is not a wall.'),
],
18: [
 ('week18_causal_response_01.png','The same randomly initialized POD decoder before and after a future-input perturbation. Earlier causal outputs must remain unchanged; unmasked outputs can change. This tests information access, not trained forecast accuracy.'),
],
19: [
 ('week19_reconstruction_02.png','Re=110, frame 100: native v/U reference and QR/D-optimal gappy reconstruction with signed error. The source POD basis was fitted to Re=90 training frames; the comparison uses the same declared observation-noise realization.'),
 ('week19_reconstruction_04.png','The same native-row profiles compare the source-POD representation control, QR/D-optimal reconstruction and lattice reconstruction. Use signed differences to separate missing representation from poorly conditioned sensing.'),
],
20: [
 ('week20_classroom_fit_01.png','Re=100 validation frame 120 after the fresh 120-update classroom fit. It is a validation diagnostic of this short fit, separate from the retained 1,200-update protocol.'),
 ('week20_retained_sensing_01.png','Re=105, frame 120, seed 17, fixed half-sensor condition: retained Sensor-set prediction and matched native CFD. Missing-sensor policies differ between systems and must be stated when comparing them.'),
 ('week20_retained_sensing_05.png','Same-case signed native-row profiles for all four estimators. Retain each loss as well as each gain; a useful set interface does not guarantee a lower reconstruction error than a classical estimator.'),
],
21: [
 ('week21_rollout_01.png','Re=110, frame 240, seed 17: native vorticity D/U and the retained autonomous Transformer prediction. Only frames 156–159 initialized the trajectory; it was not reset at the validation/test boundary.'),
 ('week21_rollout_06.png','Same native-row comparison for Transformer, history MLP, DMD and persistence. The POD oracle uses the exact target field and indicates representation error; it is not a deployable forecast.'),
 ('week21_rollout_07.png','Signed vorticity at the fixed native probe nearest x/D=4, y/D=0.5 over the complete autonomous trajectory. Phase drift and amplitude error can differ even when a single snapshot looks plausible.'),
],
22: [
 ('week22_transfer_02.png','Target Re=110, 84 target labels, seed 17, frame 240: a retained transfer-arm field comparison on the native CFD grid. Keep source exposure, target labels and computational work distinct.'),
 ('week22_transfer_06.png','Same-case signed native-row profiles for all five retained transfer controls. The target-POD control changes representation information; matched additional steps change computational work. Neither can be collapsed into a single pretrained-versus-scratch score.'),
],
}

def physical_sections():
    begin='<!-- physical-laboratory:start -->';end='<!-- physical-laboratory:end -->'
    for week,figures in PHYSICAL.items():
        p=ROOT/f'lectures/source/week{week}_transformer_course.md'
        s=p.read_text(encoding='utf8')
        if begin in s:s=s[:s.index(begin)].rstrip()+'\n\n'+s[s.index(end)+len(end):].lstrip()
        block=begin+'\n\n## Read the physical diagnostic\n\n'
        for filename,caption in figures:
            block+=f'![{caption}](../../results/late_course_teaching/{filename})\n\n'
        block+=end+'\n\n'
        anchor='## Printable laboratory worksheet'
        if anchor not in s:raise ValueError(f'Missing worksheet anchor: {p}')
        s=s.replace(anchor,block+anchor,1)
        s=s.replace('The POD masked/unmasked curves below come from',
                    'The POD masked/unmasked curves in the physical diagnostic come from')
        p.write_text(s,encoding='utf8',newline='\n')

def main():
    p=ROOT/'lectures/source/week15_geometry_generalization.md'
    s=p.read_text(encoding='utf8')
    s=s.replace("**A long-form, evidence-first lecture for Ehsan Roohi's FlowMLLab**",
                '**Neural operators: mechanisms, a CPU training laboratory and retained CFD comparisons**')
    s=s.replace('**Release date:** 19 September 2026','**Teaching edition:** 20 November 2026; retained research protocols keep their original identities.')
    s=s.replace('The accompanying notebook is the executable companion. The lecture explains what each calculation means; the notebook rechecks the split, hashes, stored arrays, metrics, learning-rate decisions, Reynolds-number trends, and figure provenance. Neither document runs new CFD.',
                'The main notebook begins with mechanism demonstrations and a complete small CPU operator-training experiment. Its exact manufactured reference is separate from the retained CFD research results. Accessible comparison tables and figures follow; archive-dependent split, metric and checkpoint audits are optional and clearly labeled. Neither route runs new CFD.')
    begin='<!-- local-operator-lab:start -->';end='<!-- local-operator-lab:end -->'
    if begin in s:s=s[:s.index(begin)].rstrip()+'\n\n'+s[s.index(end)+len(end):].lstrip()
    example="""
### Read the executed manufactured example

The fixed default has 748 trainable branch parameters. Validation selected update 589 in the recorded CPU run; L-BFGS reached its 100-iteration ceiling, so this is a budgeted result, not a convergence certificate. At a=0.32 and 0.52, relative discrete test errors were about 0.110% and 0.124%, versus 1.039% and 0.461% for linear interpolation. These exact-reference teaching cases do not establish accuracy on the retained CFD geometries. Your changed configuration may improve or lose to either baseline.

![Training and validation histories use different error definitions and scales; the dotted line marks validation-only checkpoint selection.](../../results/late_course_teaching/week15_history_01.png)

![Manufactured test geometry a=0.32. The exact speed field and common-level predictions are followed by full-range vector errors; both methods use the same mapped nodes and reference.](../../results/late_course_teaching/week15_test_01.png)

![At a=0.32, the signed velocity-profile differences reveal the error even when the overlaid profiles almost coincide. The physical column x/L=0.375 was fixed before evaluation.](../../results/late_course_teaching/week15_test_02.png)

![The second withheld manufactured geometry, a=0.52, uses the same frozen model and declared comparison. Color ranges are shared between the two errors within this case; each case retains its own full range.](../../results/late_course_teaching/week15_test_03.png)

![The second test geometry's signed profiles. Both velocity components and their signed errors should be inspected before changing their training weights.](../../results/late_course_teaching/week15_test_04.png)
"""
    section=begin+'\n\n'+LECTURE+'\n\n'+example+'\n\n'+end+'\n\n'
    s=s.replace('## 1. The scientific question',section+'## 1. The scientific question',1)
    # Keep the overview caption inside the text block; larger detail panels follow.
    s=re.sub(r'(!\[(?:Core|Extended) comparison at Re=\d+\]\([^\n]+?\))(?:\{[^\n]*\})?',
             r'\1{height=7.2in}',s)
    s=s.replace('![Train-validation-test velocity ladder](../../results/week15_postaudit/Train_Validation_Test_Ladder.png)\n\n','')
    s=s.replace('- all 18 matched runs: $\\rho=0.5645$, $p=0.0147$;\n- 12 author-code runs: $\\rho=0.8462$, $p=0.000521$.',
                '| Retained 19,200-update comparison | Runs | Spearman $\\rho$ | $p$ |\n'
                '|---|---:|---:|---:|\n'
                '| All matched models | 18 | 0.5645 | 0.0147 |\n'
                '| Author-code models | 12 | 0.8462 | 0.000521 |')
    route='## Self-contained classroom route'
    if route in s:
        s=s[:s.index(route)]+route+'\n\n'+(
            'Begin with the manufactured two-step geometry/SDF example, explicit branch–trunk contraction, Fourier mode filter, and query-set-dependence counterexample. These small mechanism demonstrations explain the operators. For the worked contraction, [1,0.2,0.5] dotted with [1,0.8,0.3] gives 1.31. Explain why this pointwise output is invariant to adding unrelated queries, and why a query-mean-normalized trunk need not be.\n\n'
            'Then run the complete manufactured-kinematic CPU operator experiment: fit the training-only POD trunk and trainable branch MLP, select the checkpoint using validation geometries only, and compare the two withheld geometries against their exact velocity fields. Change one depth, optimizer, or loss-weight control at a time, inspecting both field errors and signed profiles. This establishes performance on the declared analytic teaching problem, not on the retained CFD geometries.\n\n'
            'Finish with the accessible retained CFD tables and native figures, keeping their original protocols separate from the fresh teaching fit. The optional external LR archive supports reconstruction and audit of all 342 retained prediction fields. Without that archive, the notebook explains the unavailable audit route once; it does not count unexecuted archive checks as passed. Allow 60–90 minutes for the initial guided reading and default run, and additional time for the four controlled experiments.\n')
    p.write_text(s,encoding='utf8',newline='\n')
    p=ROOT/'lectures/source/week16_parts/part6_cpu_lab.tex'
    p.write_text(W16+r"""
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{results/late_course_teaching/week16_validation_models_01.png}
\caption{First predeclared validation shape at Mach 1.8: all three fresh CPU surrogate fits share the same CFD pressure signature. Signed discrepancies expose displaced waves and amplitude errors. This plot is used for development, not as a final-test result.}
\end{figure}
""",encoding='utf8',newline='\n')
    p=ROOT/'lectures/source/week16_supersonic_shape_optimization.tex'
    s=p.read_text(encoding='utf8')
    include=r'\input{lectures/source/week16_parts/part6_cpu_lab.tex}'
    if include not in s:s=s.replace(r'\input{lectures/source/week16_parts/part5.tex}',include+'\n'+r'\input{lectures/source/week16_parts/part5.tex}')
    p.write_text(s,encoding='utf8',newline='\n')
    p=ROOT/'lectures/source/optional_geometry_time_bridge.md';s=p.read_text(encoding='utf8')
    s=s.replace('The supplied LinkedIn URL could not be read during the preceding audit. No content, author claim or result from that post is used here. A social post is not substituted for the paper or its numerical evidence.','Primary papers and versioned numerical artifacts provide the evidence for this module; social-media descriptions are not numerical validation.')
    marker='## Read the two diagnostic experiments'
    if marker in s:s=s[:s.index(marker)].rstrip()+'\n\n'+s[s.index('## References and evidence limits',s.index(marker)):]
    anchor='## References and evidence limits'
    before,after=s.split(anchor,1)
    figures="""
![Observed, teacher-forced and autonomous histories for the same manufactured scalar update. Feedback changes the problem even when initialization and one-step map are identical.](../../results/late_course_teaching/weekbridge_feedback_phase_01.png)

![A controlled amplitude/phase perturbation and its signed difference. The example isolates two distinct waveform errors without claiming CFD or paper reproduction.](../../results/late_course_teaching/weekbridge_feedback_phase_02.png)
"""
    s=before.rstrip()+'\n\n'+BRIDGE.strip()+'\n\n'+figures.strip()+'\n\n'+anchor+after
    p.write_text(s,encoding='utf8',newline='\n')
    physical_sections()

if __name__=='__main__':main()
