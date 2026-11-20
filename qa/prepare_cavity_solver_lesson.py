"""Append the frozen solver experience to Week 4 lecture source, idempotently.

Uses the verified compact evidence already stored in results/cavity_solver_benchmark.
No simulation or model fitting is performed. Earlier lecture page objects are retained.
"""
from pathlib import Path
import csv, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cavity_solver_benchmark'
UNIT='matched-cavity-solver-benchmark'

def block(kind,text): return {'type':kind,'text':text}
def math(latex): return {'type':'math','latex':latex}
def table(rows,widths): return {'type':'table','rows':rows,'widths':widths}
def figure(name,caption,width=465,max_height=360):
    return {'type':'figure','path':f'results/cavity_solver_benchmark/{name}.png','caption':caption,'width':width,'max_height':max_height}
def page(title,*blocks): return {'title':title,'benchmark_unit':UNIT,'blocks':list(blocks)}

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==expected,name
    with (OUT/'timings.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    keyed={(int(r['cells']),r['scenario'],r['method']):r for r in rows}
    paired=json.loads((OUT/'paired_fields_audit.json').read_text())['pairs']
    vmax=max(p['paired_velocity_relative_l2'] for p in paired)
    wmax=max(p['paired_omega_relative_l2'] for p in paired)
    time_rows=[['Nodes per side','Target Re','Direct (s)','GMRES/MG (s)','MG/direct']]
    setup_rows=[['Nodes / Re','Setup (s)','Setup / solve','Needed setup cut']]
    for n in [256,512]:
        for case,re in [('warm7500',7500),('warm10000',10000)]:
            a=keyed[n,case,'direct'];b=keyed[n,case,'gmres_mg']
            td=float(a['wall_seconds_median']);tm=float(b['wall_seconds_median']);ts=float(b['preconditioner_setup_seconds_median'])
            time_rows.append([str(n+1),str(re),f'{td:.3f}',f'{tm:.3f}',f'{tm/td:.3f}'])
            setup_rows.append([f'{n+1} / {re}',f'{ts:.2f}',f'{100*ts/tm:.1f}%',f'{100*(tm-td)/ts:.1f}%'])
    td=float(keyed[256,'continuation','direct']['wall_seconds_median'])
    tm=float(keyed[256,'continuation','gmres_mg']['wall_seconds_median'])
    pages=[
      page('Measured experience: choosing a cavity solver',
        block('body','<b>Question:</b> would replacing the sparse-direct linear solve with Newton-Krylov and multigrid make our high-Re cavity calculations faster? A matched experiment answered this for our current implementation: <b>the candidate took 1.95-2.17 times as long</b> in the four repeated warm-case comparisons. We retain the existing production solver for these workloads.'),
        block('heading','What changed, and what stayed fixed?'),
        block('body','Both methods solve the same discrete steady streamfunction-vorticity equations. The grid, no-slip walls and moving lid, initial field, exact Jacobian, damped Newton iteration, line search and final acceptance tests are fixed. Only the method used to obtain the Newton correction changes.'),
        math(r'F(q_k)=0\;\mathrm{at\ convergence},\quad J_k\delta q_k=-F(q_k),\quad q_{k+1}=q_k+\alpha_k\delta q_k'),
        block('small','Here q collects the interior streamfunction and vorticity unknowns; F is the discrete equation residual; J is its Jacobian; k is the Newton iteration; delta q is the correction; and alpha is the accepted line-search step. The direct method factorizes the sparse Jacobian. GMRES is the Generalized Minimal Residual iterative method.'),
        math(r'\widetilde J_k=R_k J_k C,\quad M_k^{-1}\widetilde J_k z_k=-M_k^{-1}R_k F(q_k),\quad\delta q_k=Cz_k'),
        block('small','R is an invertible diagonal row scaling; C is an invertible diagonal variable scaling; z is the scaled Newton correction. The preconditioner acts on the scaled Jacobian J-tilde. These transformations preserve the original correction equation.'),
        block('body','The operator M<super>-1</super> applies one coupled geometric Galerkin multigrid V-cycle. It uses bilinear prolongation, scaled-transpose restriction, ILUT smoothing (incomplete LU with threshold dropping), and a direct solve on the coarsest grid. Invertible row and variable scaling help the linear algebra; checks against the original unscaled equations are still required.'),
        block('heading','Why record a result that did not improve runtime?'),
        block('body','A plausible method is not automatically a practical improvement. Preserving its failed pilots, setup cost and accepted solutions makes the result reusable: future work can target the measured bottleneck instead of repeating the same unsupported assumption. This is a CFD solver experiment, not neural-network training.')
      ),
      page('Matched timings: include the cost of setup',
        block('body','The table gives <b>median solve wall time over three repetitions</b> per method, grid and warm case. Each trial uses one CPU core and one BLAS/OpenMP thread on the same Intel Xeon Gold 5220 node (2.20 GHz). Python 3.12.3, NumPy 2.2.6 and SciPy 1.15.3 were recorded. Paired methods run sequentially in fresh processes, in AB, BA, AB order; the two grid studies use separate cores of the same allocation.'),
        table(time_rows,[84,73,91,119,98]),
        block('body','Re=7500 starts from the same retained Re=6500 solution; Re=10000 starts from the same retained Re=9000 solution. A grid with 257 or 513 nodes has 256 or 512 cells per side. <b>These are warm continuation steps, not cold starts.</b> Minima, maxima and per-trial values are retained with the notebook evidence.'),
        block('heading','A separate, longer continuation test'),
        block('body',f'On 257-by-257 nodes, the same 20-root path from Re=1000 to 10000 took <b>{td:.3f} s</b> with the direct method and <b>{tm:.3f} s</b> with GMRES/MG: {tm/td:.3f} times as long. Both start from the same retained 128-cell Re=1000 seed, interpolated to the target grid. There is only <b>one repetition per method</b> for this comparison.'),
        block('heading','What the timer includes'),
        block('body','Solve time includes residual evaluation, Jacobian assembly, preconditioner construction, linear solves and line search. It excludes queueing, imports, seed generation, initial file I/O, checkpoint writes and post-run verification. A failed pilot is not a fast successful solve. Memory records are whole-process peak RSS, not just factor storage; the candidate also used slightly more memory in these runs.'),
        block('small','The main evaluation contains 26 successful trials, forming 13 matched pairs. The same source and packed initial-state hashes were checked for every pair. Neither this timing table nor historical Ghia timings establish a universal or hardware-independent algorithm ranking.')
      ),
      page('Diagnosis: few iterations, expensive preparation',
        figure('nonlinear_convergence','513-by-513 nodes; representative repeat 0. The original equation residual is the maximum of the Poisson and steady-vorticity defects. The dotted line is the final 10<super>-8</super> residual threshold. Both methods take four Newton steps; the candidate takes eight GMRES iterations in total.',465,195),
        math(r'T_{\rm solve}=T_{\rm residual/J}+T_{\rm setup}+T_{\rm Krylov}+T_{\rm search/other}'),
        block('small','T denotes elapsed time in each non-overlapping phase. Setup constructs the multigrid hierarchy and its incomplete factors; Krylov includes preconditioner applications. In the recorded profiling, linear_total_seconds already contains setup: do not add setup to that total again.'),
        table(setup_rows,[112,91,120,142]),
        block('body','The strong ILUT hierarchy is rebuilt at every Newton step. Its construction accounts for <b>89.3-95.2%</b> of warm solve time, computed as median setup time divided by median solve time. It makes each linear solve converge in few iterations, but the setup cost overwhelms that benefit.'),
        block('small','The final column is a conditional cost-model calculation, not an achieved acceleration: it is the setup reduction needed to tie the direct median if all other costs remain fixed. Reusing a hierarchy can change iteration count and robustness; it needs a fresh correctness and timing test. No reuse improvement was implemented or measured here.')
      ),
      page('Check the flow before declaring a faster solve',
        figure('physical_profiles','513-by-513 nodes; representative repeat 0. Solid black: direct Newton. Dashed red: Newton with GMRES/MG. Left: vertical-centreline u; middle: horizontal-centreline v; right: signed horizontal-centreline vorticity. U is lid speed and L is cavity width. The almost coincident curves are expected, not a missing method.',465,295),
        math(r'D_u=\left[\frac{\sum_i((u_i^{MG}-u_i^{D})^2+(v_i^{MG}-v_i^{D})^2)}{\sum_i((u_i^D)^2+(v_i^D)^2)}\right]^{1/2}'),
        block('small','Superscripts MG and D denote the candidate and direct solvers; i ranges over every common native grid node. Vorticity uses the analogous relative L2 norm. The complete-field audit is separate from these centreline plots.'),
        block('body',f'Across five representative full-field pairs, the largest velocity discrepancy was <b>{vmax/1e-14:.3g} x 10<super>-14</super></b> and the largest vorticity discrepancy was <b>{wmax/1e-14:.3g} x 10<super>-14</super></b>. These quantify agreement between two numerical solvers. They are <b>not</b> errors against Ghia observations, a continuum exact solution, or experimental flow.'),
        block('body','A tiny nonlinear residual identifies a discrete steady root. It does not remove mesh error, prove the high-Re branch is temporally stable, or establish three-dimensional flow accuracy. Keep the existing mesh-sensitivity, Ghia-validation and temporal checks separate.')
      ),
      page('Keep the unsuccessful attempts in the record',
        block('body','Candidate settings were explored only on the smaller 128-cell pilot grid. Each setting was tried at both Re=7500 and 10000. The bounded pilot selected the fastest setting that passed both cases; the selection was frozen before any larger-grid candidate evaluation.'),
        table([['ILUT fill / drop','Outcome across two pilot cases'],['4 / 10<super>-3</super>','2 singular incomplete-factor failures'],['8 / 10<super>-4</super>','1 singular factor; 1 original-equation gate failure'],['12 / 10<super>-5</super>','2 original-equation gate failures'],['20 / 10<super>-6</super>','2 successful trials; slower than the stronger setting'],['40 / 10<super>-8</super>','2 successful trials; selected for the main evaluation']],[110,355]),
        block('small','Fill factor controls permitted incomplete-factor storage; drop tolerance removes sufficiently small entries during factorization. A singular incomplete factor does not prove that the original PDE system has no solution. A small residual in a scaled or preconditioned system must not replace a check of the original equations.'),
        block('heading','Acceptance was not weakened to make the candidate pass'),
        block('body','Final checks require original nonlinear max residual below 10<super>-8</super>, combined velocity relative L2 difference below 10<super>-8</super>, and vorticity relative L2 difference below 10<super>-7</super> against the retained same-grid CFD solution. These are agreement checks. The unchanged outer linear-defect gate is 10<super>-5</super>; the candidate additionally enforces an original unscaled linear-defect threshold of 10<super>-7</super>. The inner tolerances are therefore not identical.'),
        block('heading','Reproducibility without repeating the expensive work'),
        block('body','The compact records, plots and hashes are in <link href="https://github.com/Ehsan-Roohi/FlowMLLab/tree/main/results/cavity_solver_benchmark"><u>results/cavity_solver_benchmark</u></link>. Lab 7 audits and reanalyses these records on CPU. The full archive retains inputs, all 40 trial records, frozen code, failed pilots, recovery states and representative fields. Full-grid runs are deliberate reproduction work, not a prerequisite for the exercises.'),
        block('small','This candidate is not Ghia et al.\'s coupled strongly implicit multigrid algorithm. The result evaluates one implementation on two grids. It does not imply that multigrid or Newton-Krylov methods are generally slower than sparse direct solves.')
      ),
      page('Engineering exercises: learn from the evidence',
        block('body','Use the executed appendix in <b>Week 4 Lab 7, Section 8</b>. Lab 8, Section 9 links the result to the cost of generating CFD labels. Run All reads frozen data; it does not run the PDE solver, submit a cluster job or train a model.'),
        block('heading','A. Reproduce the performance decision'),
        block('body','From the raw production trials, reproduce four warm-case rows: direct and candidate median seconds, min/max ranges, and the ratio candidate/direct. Audit all 13 pairs for matched seeds, source, grid, tolerance and thread count. List the six failed pilot trials separately. Explain why neither a failed short run nor the one-repeat continuation pair should be pooled into the three-repeat medians.'),
        block('heading','B. Distinguish numerical agreement from validation'),
        block('body','Inspect both physical profiles and residual histories, then report the two maximum full-field discrepancies from the five-pair audit. State which data would be needed to claim physical accuracy, mesh independence and temporal stability. The nearly 10<super>-14</super> agreement between solvers is not a new neural-network or Ghia validation result.'),
        block('heading','C. Set an improvement target before another run'),
        math(r'T_{\rm hyp}(r)=T_{\rm MG}-T_{\rm setup}+rT_{\rm setup},\quad r_* = \frac{T_D-(T_{\rm MG}-T_{\rm setup})}{T_{\rm setup}}'),
        block('body','Here <i>r</i> is the retained fraction of setup cost (0 to 1); <i>T</i><sub>D</sub> and <i>T</i><sub>MG</sub> are measured direct and candidate medians; <i>T</i><sub>setup</sub> is median setup time. Compute 100(1-<i>r</i><sub>*</sub>) for each warm case: about <b>53.6-57.4% setup reduction</b> is needed just to tie the baseline under this model. Propose when a reused hierarchy would be rebuilt and which original-equation checks must still pass. Label this a proposal, not a measured speedup.'),
        block('heading','D. Keep the neural cost comparison honest'),
        block('body','For a surrogate, write total cost as CFD label generation plus training plus per-query inference. State which of these costs this benchmark actually measured. Do not substitute a warm continuation timer for an arbitrary cold CFD solve, or infer improved prediction accuracy from faster label generation.'),
        block('small','Submit one numerical timing/setup table, a separate failure list, annotated physical-profile observations, and a short proposed-reuse protocol. Include the evidence manifest hash and software version. A correct decision to keep the baseline is a useful engineering outcome.')
      )
    ]
    source=ROOT/'lectures/source/cavity_expansion.json'
    content=json.loads(source.read_text(encoding='utf-8'))
    name='week04_cavity_surrogates_deeponet'
    old=content[name]
    original=[p for p in old if p.get('benchmark_unit')!=UNIT]
    assert len(original)==17,('Unexpected prior extension pages',len(original))
    content[name]=original+pages
    source.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme='''# Matched cavity solver experience

The existing sparse-direct Newton solver was faster than the tested GMRES/multigrid candidate at Re=7500 and 10000 on 257-by-257 and 513-by-513 nodes. The candidate took **1.95-2.17 times as long** in three-repeat warm comparisons. Setup consumed **89.3-95.2%** of candidate solve time. We retain the baseline for these tested workloads.

This is a measured implementation result, not a universal verdict about multigrid and not a reimplementation of Ghia's CSI-MG.

## Learn from the experiment

- [Lab 7: CFD audit and executable solver appendix, Section 8](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb)
- [Lab 8: what this means for neural training data, Section 9](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb)
- [Week 4 lecture, solver-benchmark evidence appendix](../../lectures/week04_cavity_surrogates_deeponet.pdf)
- [Full measured report](REPORT.md), [summary timings](timings.csv), [raw trial records](evidence.json), [whole-field agreement audit](paired_fields_audit.json)

The notebook reanalyses retained measurements on CPU. No solver or training run is needed. The six unsuccessful candidate pilots remain in the record; failed runs are not counted as fast successful solves.

![Matched velocity and signed vorticity profiles](physical_profiles.png)

Both solvers converge to effectively the same discrete solution. Full-field differences near 1e-14 describe solver agreement, not physical or Ghia error.

![Original nonlinear residual histories](nonlinear_convergence.png)

## Timing and correctness boundaries

Warm Re7500 uses the retained Re6500 field; warm Re10000 uses Re9000. The 257-node 20-root continuation test has only one repetition per method. Solve timing includes Jacobian/preconditioner setup but excludes seed creation, imports, queue, file I/O and post-run verification. The recorded linear total already includes setup. Identical final acceptance tests do not imply identical inner tolerances: the candidate additionally checks the original linear defect at 1e-7.

Preconditioner reuse is an untested proposal. A cost model predicts that 53.6-57.4% less setup cost would be needed to tie the baseline if other costs stayed fixed; no such acceleration is claimed.

## Evidence and reproduction

`manifest.json` records SHA256 for the byte-verified compact evidence. The separately retained 146,274,361-byte full archive has SHA256 `fbfd423b06883c36268849d08647ebcc68d797ae49aeb82ae55090dd1c27313e`. It preserves 40 trial records, frozen numerical code, exact inputs, complete successful recoveries and representative full fields. It is not downloaded by Run All.

Experimental source: [linear solver](../../flowmllab/cavity_krylov.py), [single-trial benchmark](../../qa/benchmark_cavity_solvers.py), [bounded pilot](../../qa/run_cavity_solver_strong_pilot.py), [paired matrix](../../qa/run_cavity_solver_matrix.py), [summary audit](../../qa/summarize_cavity_solver_benchmark.py). The original production solver is unchanged. Full-grid reruns require deliberate resource planning and fresh allocated scratch; do not run a saved site-specific batch path blindly.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    week=ROOT/'weeks/week04/README.md'
    text=week.read_text(encoding='utf-8')
    marker='## Measured solver experience'
    if marker not in text:
        text+='\n'+marker+'\n\nLabs 7 and 8 now end with a measured CPU solver comparison: why fewer Krylov\niterations did not reduce cavity solve time. The appendix includes successful\nand failed trials, signed physical profiles, setup-cost analysis and exercises\nusing saved records. [Evidence and lesson links](../../results/cavity_solver_benchmark/README.md).\n'
        week.write_text(text,encoding='utf-8')
    print(json.dumps({'new_lecture_pages':len(pages),'prior_extension_pages_unchanged':len(original),'result_directory':str(OUT)}))

if __name__=='__main__':main()
