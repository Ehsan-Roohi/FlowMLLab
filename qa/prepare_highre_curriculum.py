"""Integrate verified high-Re CFD and neural evidence without replacing old labs."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]

def write(path,text):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(text.strip()+'\n',encoding='utf-8',newline='\n')
def readcsv(path):
    with (ROOT/path).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def block(text,kind='body'):return dict(type=kind,text=text)
def figure(path,caption,height=360):return dict(type='figure',path=path,caption=caption,max_height=height)

def main():
    out='results/cavity_field_deeponet'
    scores=readcsv(out+'/test_scores.csv')
    selected=json.loads((ROOT/out/'frozen_selection.json').read_text())
    verification=json.loads((ROOT/'results/cavity_highre_ghia/verification.json').read_text())
    assert len(verification['cases'])==45 and verification['all_fields_pass']
    methods=['selected_three_seed_mean','linear_in_Re','linear_in_log_Re']
    table=[['Re','Neural mean','Linear in Re','Linear in log Re']]
    for re in (3200,7500):
        vals=[next(float(s['relative_l2']) for s in scores if int(s['Re'])==re and s['method']==m) for m in methods]
        table.append([str(re)]+[f'{100*x:.4f}%' for x in vals])
    markdown='\n'.join(['| '+' | '.join(row)+' |' for row in [table[0],['---']*4]+table[1:]])
    write('results/cavity_highre_ghia/README.md',"""# Newly solved cavity CFD across the Ghia Reynolds range

These are actual steady Navier-Stokes fields, not curves fitted to benchmark
samples. Forty-five fields cover 15 Reynolds numbers on 128, 256 and 512 cells
per side. The seven Ghia cases are included; additional cases support the
separate full-field neural experiment. Original low-Re runs are preserved.

[Run Lab 7](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) |
[Equations and execution protocol](../../docs/CAVITY_HIGH_RE_PROTOCOL.md) |
[Independent field/hash audit](verification.json) |
[Neural predictions](../cavity_field_deeponet/README.md)

<img src="highre_cfd_ghia.png" width="900" alt="Our three CFD grids against actual Ghia samples at all four higher Reynolds numbers">

<img src="highre_streamlines.png" width="900" alt="Black streamlines of newly solved high-Re cavity flows">

The red crosses retain predeclared suspicious printed entries. Both raw and
screened scores are in [ghia_scores.csv](ghia_scores.csv); none is replaced.
Re=3200's finest screened u/v errors are 3.039% / 4.773%; Re=10000's are
2.058% / 2.524%. Refinement need not monotonically approach a sparse historical
benchmark with its own numerical error.

## Numerical meaning and limits

[Mesh sensitivity](mesh_sensitivity.csv) includes both centreline and native
full-field differences. The 256-to-512 vector discrepancies at Re=3200,5000,7500,
10000 are 2.03%, 3.14%, 4.72%, 6.32%. These fields solve their discrete equations
but are **not demonstrated mesh-independent physical truth**. Strong corner
differences remain. A neural model learns the stated 512-cell discretization.

[Matched-time checks](time_sensitivity.csv) perturb the same 256-cell root and
compare dt=0.002 with 0.001 through tU/L=10. All paired time-step checks pass.
The Re=10000 perturbation grows by 8.7% over this short interval, while the other
three decay. This single perturbation and short horizon do not prove or
disprove asymptotic stability.

<img src="temporal_audit.png" width="800" alt="Matched-time perturbation trajectories and centre velocity changes">

## Reproduction and data

Each n128, n256 and n512 directory contains native x,y,u,v,psi,omega arrays and
JSON configuration, source hashes, convergence history and final SHA256.
Independent checks re-evaluate Poisson, vorticity transport, wall constraints,
streamfunction/velocity consistency and continuity. No pressure is reconstructed.

CPU runs used Unity Workspace flowml-ghia-highre-20261115. Its actual scratch
path, expiry, source identities and job records are in the protocol. Final
scientific fields were byte-verified on Lexar D:. The raw fields and provenance
are retained; transient live checkpoint traffic is not mirrored here.
""")
    write(out+'/README.md',f"""# Full-field cavity POD-DeepONet: fixed whole-Re tests

A training-only POD basis and a tanh branch predict both velocity components on
the native 513-by-513 grid. This is a supervised scalar-parameter surrogate,
not a data-free PINN. No Ghia sample enters fitting. Pressure is not predicted.

[Run Lab 8](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) |
[Frozen design and validation amendment](../../docs/CAVITY_FIELD_NEURAL_PROTOCOL.md) |
[CFD discretization audit](../cavity_highre_ghia/README.md) |
[Primary-source literature check](../../docs/CAVITY_ALL_RE_NEURAL_LITERATURE.md)

Training Re: 100,250,500,750,1000,2500,4000,5000,6000,9000,10000.
Validation Re: 400,6500. Test Re: 3200,7500.
Selection: **{selected['selected_configuration']}**, mean validation vector
error across both Re and seeds 7,23,61. The delivered field averages those seeds.
Both interpolation baselines receive exactly the same training fields.

{markdown}

These are full-field vector relative L2 errors against the retained 512-cell
CFD solution, not speed-only errors, Ghia profile errors or exact physical error.

<img src="re3200_fields.png" width="900" alt="Held-out Re3200 CFD, selected neural prediction, interpolation and shared-scale errors">

<img src="re3200_profiles.png" width="800" alt="Re3200 signed u and v comparisons with CFD, Adam, selected model and interpolation">

<img src="re7500_fields.png" width="900" alt="Held-out Re7500 physical velocity fields and shared-scale errors">

<img src="re7500_profiles.png" width="800" alt="Re7500 signed velocity profile errors and Ghia samples">

## What improved, and what remains limited

The [validation scores](validation_scores.csv) retain every architecture/seed,
including deterioration after depth or per-mode weighting. All five candidates
have the same maximum function/gradient-evaluation budget. The capacity-matched
five-layer control distinguishes depth from a large parameter-count increase.

The initial nine-case training round had a 3.251% POD projection floor at
validation Re=400. Its [unscored retained trial](selection_round_1/ROUND_DECISION.json)
motivated two new independent CFD training cases at 500 and 750, before testing.
This is a validation-driven amendment, not a prospectively fixed training set.

The [projection table](projection_lower_bound.csv) uses test truth solely for
an oracle representation bound after prediction freezing. It is not a predictor.
[Physical diagnostics](physical_diagnostics.csv) separately re-evaluate wall,
divergence and nonlinear vorticity residuals. Linear basis constraints do not
enforce nonlinear momentum balance.

The 256-to-512 CFD vector discrepancy is 2.03% at Re3200 and 4.72% at Re7500.
Those sensitivities can exceed surrogate error. This experiment demonstrates
accuracy against a specified discretization, not mesh independence, physical
stability, a pressure solution or a data-free neural PDE solver.

## Reproduce without changing the experiment

The notebook verifies hashes and displays the retained result on CPU.
qa/run_cavity_field_operator.py has separate train and score stages. Training
refuses to run after test scores exist. The selected configuration, training
data reads, every model hash and frozen prediction hash are recorded in
[frozen_selection.json](frozen_selection.json); [scoring_receipt.json](scoring_receipt.json)
records target access after freezing. Source, weights, Adam moments and the
training RNG initialization state are retained. SciPy's internal L-BFGS history
is not serialized; these are completed fits, not resumable mid-L-BFGS checkpoints.
""")
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8')
    old='Reference tables cover all seven Ghia Reynolds numbers; retained CFD fields for\n3200, 5000, 7500 and 10000 are still missing.'
    new="""**New higher-Re CFD.** The four cases below were independently solved on
128, 256 and 512 cells per side; circles remain the original Ghia samples.

<img src="results/cavity_highre_ghia/highre_cfd_ghia.png" width="760" alt="New three-grid CFD against Ghia at Re3200,5000,7500 and10000">

[Run the new CFD lab](notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) ·
[Native fields and mesh/time sensitivity](results/cavity_highre_ghia/README.md).
These are converged discrete roots; remaining high-Re mesh sensitivity is reported."""
    assert old in s;s=s.replace(old,new)
    new=f"""### Full-field neural follow-on to Week 4

The new POD-DeepONet learns our matched CFD velocity fields across Re=100–10000.
Re=3200 and 7500 are withheld in full. Architecture and interpolation convention
are selected on Re=400 and 6500; neither test field is used for fitting or selection.

<img src="results/cavity_field_deeponet/re3200_fields.png" width="760" alt="Held-out full-field CFD, POD-DeepONet, interpolation and linear-scale vector errors">

{markdown}

Errors compare with the 512-cell CFD discretization. Its remaining mesh
sensitivity is larger than some surrogate errors.
[Run the new field-neural lab](notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) ·
[Both tests, signed profiles, seed variation and limitations](results/cavity_field_deeponet/README.md).

"""
    s=s.replace('### Week 4.1 — Classical reduced-order models',new+'### Week 4.1 — Classical reduced-order models')
    write('README.md',s)
    p=ROOT/'weeks/week04/README.md';s=p.read_text(encoding='utf-8')
    s=s.replace('**Module guides:**',"""| Lab 7: new high-Re CFD and mesh/time checks | [Open notebook](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) |
| Lab 8: full-field POD-DeepONet and interpolation | [Open notebook](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) |

**Module guides:**""")
    s=s.replace('High-Re profile fitting does not create missing CFD fields.','The historical profile fit remains separate. New Labs 7 and 8 now add the actual high-Re CFD and full-field neural experiment.')
    s+='\n[New CFD evidence](../../results/cavity_highre_ghia/README.md) · [Full-field neural tests](../../results/cavity_field_deeponet/README.md).\n'
    write('weeks/week04/README.md',s.replace('\n\n| Lab 7:','\n| Lab 7:'))
    p=ROOT/'flowmllab/cavity_course.py';s=p.read_text(encoding='utf-8')
    needle="        rows.append({'Re':re,"
    assert needle in s
    s=s.replace(needle,"        new_field=root/f'results/cavity_highre_ghia/n512/re{re}.npz'\n        if new_field.exists():path=new_field\n"+needle)
    write('flowmllab/cavity_course.py',s)
    p=ROOT/'qa/check_cavity_notebooks.py';s=p.read_text(encoding='utf-8')
    s=s.replace("'notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb',","'notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb',\n 'notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb',\n 'notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb',")
    s=s.replace('Three new cavity notebooks','Five cavity notebooks').replace('the three new cavity notebooks','the five cavity notebooks')
    write('qa/check_cavity_notebooks.py',s)
    # Preserve every base lecture page and the prior extension content.
    p=ROOT/'lectures/source/cavity_expansion.json';doc=json.loads(p.read_text(encoding='utf-8'))
    for pages in doc.values():
        for page in pages:
            for b in page['blocks']:
                if b.get('text','').startswith('A selectable Reynolds number does not certify'):
                    b['text']='A selectable Reynolds number does not certify a solver. New Lab 7 now supplies actual CFD at all four higher Ghia Reynolds numbers on 128, 256 and 512 cells per side. Matched low-Re and additional training fields complete 45 solutions. All pass independent discrete-equation checks; material high-Re mesh sensitivity remains.'
                if 'Missing high-Re fields must remain labelled missing' in b.get('text',''):
                    b['text']=b['text'].replace('Missing high-Re fields must remain labelled missing in any neural-training manifest.','Use the new field/hash manifest in Lab 7; distinguish a converged discrete solution from a mesh-independent physical reference.')
    pages=doc['week04_cavity_surrogates_deeponet']
    pages.extend([
      dict(title='New CFD: solving the four higher Ghia cases',blocks=[
       block('Lab 7 adds Re=3200,5000,7500,10000 on three uniform grids. Central streamfunction-vorticity differences and Thom wall vorticity match the earlier Week-4 spatial formulation. Damped Newton with Reynolds continuation solves the steady equations; Ghia values are never used in the solve. Newton iteration is not physical time.'),
       dict(type='math',latex=r'\nabla_h^2\psi+\omega=0,\qquad Re^{-1}\nabla_h^2\omega-uD_x\omega-vD_y\omega=0.'),
       figure('results/cavity_highre_ghia/highre_cfd_ghia.png','Our 128/256/512-cell profiles against the original Ghia samples. Crosses retain predeclared suspicious printed entries; raw and screened errors are both reported.',320),
       block('Each native field is checked independently for array identity, Poisson and vorticity residuals, wall values, velocity/streamfunction consistency and divergence. Residuals below 1e-6 establish discrete roots; they do not establish grid independence.'),
       block('Files: results/cavity_highre_ghia/verification.json, ghia_scores.csv. Exercises: Lab 7, Sections 3 and 7.','small')]),
      dict(title='Mesh and time sensitivity are separate questions',blocks=[
       figure('results/cavity_highre_ghia/highre_streamlines.png','Finest-grid black streamlines. Automatic seeding can miss small vortices; inspect signed streamfunction contours in the corner exercise.',180),
       dict(type='table',widths=[70,130,130,130],rows=[['Re','256-to-512 vector difference','Short-time perturbation ratio','dt-pair RMS/U'],
        ['3200','2.03%','0.813','1.07e-12'],['5000','3.14%','0.880','2.23e-12'],['7500','4.72%','0.940','8.73e-12'],['10000','6.32%','1.087','2.57e-11']]),
       block('The vector mesh difference uses common native samples and is not an exact-error bound. Both temporal runs use the same 256-cell initial perturbation, dt=0.002/0.001, and tU/L=0..10. Tiny paired differences test temporal discretization, not total CFD accuracy.'),
       block('The Re=10000 perturbation grows by 8.7% over this finite horizon. Neither this observation nor decay of one perturbation proves long-time stability. A steady 2-D root also need not be a stable physical 3-D flow. Pressure is not reconstructed.'),
       block('Engineering task: compare signed profiles and corner extrema across all grids; state which numerical criterion fails as well as which passes.')]),
      dict(title='A full-field POD-DeepONet with whole-Re tests',blocks=[
       block('Lab 8 learns both velocity components from our matched native CFD fields. It is supervised parameter-to-field learning, not a data-free PINN. Stack q=[u;v], compute the mean and orthonormal POD modes only from training fields, and learn the coefficients with a tanh branch.'),
       dict(type='math',latex=r'\widehat{\mathbf{q}}(Re)=\overline{\mathbf{q}}+\Phi\,\mathbf{a}_\theta(\log_{10}Re-3).'),
       dict(type='table',widths=[80,380],rows=[['Role','Complete Reynolds cases'],['Training','100,250,500,750,1000,2500,4000,5000,6000,9000,10000'],['Validation','400,6500'],['Test','3200,7500']]),
       block('Two and five layers, Adam and Adam-to-L-BFGS, global/per-mode coefficient scales, and a capacity-matched depth control use three seeds and the same maximum evaluation budget. Select by mean validation vector error; retain all seed outcomes.'),
       dict(type='math',latex=r'J=\frac{1}{n_{\rm tr}r}\sum_{j,k}\left[f_{\theta,k}(z_j)-a_{jk}/s_k\right]^2.'),
       block('s_k is either one common training RMS or each mode RMS. Per-mode weights emphasize weak modes but can harm optimization/generalization. More layers are not automatically better. Interpolation receives precisely the same training fields.')]),
      dict(title='Representational error and held-out prediction',blocks=[
       block('The original nine-case POD had a 3.251% projection floor at validation Re=400. Adding independent training solves at 500 and 750 addressed missing spatial content; test cases were unchanged and unscored. This validation-led amendment is retained, not presented as a fixed design chosen in advance.'),
       dict(type='math',latex=r'\mathbf{q}_{\rm proj}=\overline{\mathbf{q}}+\Phi\Phi^T(\mathbf{q}_{\rm true}-\overline{\mathbf{q}}).'),
       block('Projection uses target truth and is only an oracle lower bound. It is not the neural prediction. The model, basis and test coefficients were hash-frozen before scoring. The reported neural field is the selected three-seed average.'),
       dict(type='table',widths=[60,130,130,140],rows=table),
       block('These percentages measure the full velocity-vector discrepancy against the 512-cell CFD solution. They are not speed-only errors or proof of exact physical accuracy. The 256-to-512 sensitivities at the two tests are 2.03% and 4.72%.'),
       figure('results/cavity_field_deeponet/re3200_profiles.png','Signed profiles distinguish the Adam baseline, selected neural result and interpolation, alongside CFD and actual Ghia samples.',275)]),
      dict(title='Look at the flow and where the errors occur',blocks=[
       figure('results/cavity_field_deeponet/re7500_fields.png','Held-out Re=7500: CFD, neural and interpolation speed use one colour scale; the vector-error maps share a linear scale. Native samples are retained.',390),
       block('A fixed linear CFD basis preserves the common wall values and discrete continuity. Its nonlinear vorticity-transport residual is generally nonzero. Compute this residual separately: small supervised error and small divergence do not imply satisfaction of nonlinear momentum.'),
       block('Inspect both centreline components, vortex structure and spatial errors. A small global norm can hide local failure; use regional diagnostics for weak recirculation. No pressure or long-time-stability claim follows from these velocity comparisons.')]),
      dict(title='Engineering work: five measured comparisons',blocks=[
       block('A. From the Lab 8 validation table, report per-Re mean and seed range for all five configurations. Compare optimizer at fixed depth, depth at fixed width, the capacity-matched control, and coefficient scaling. Identify both improvements and deterioration.'),
       block('B. Compare the retained first-round and final validation errors. Explain the role of the projection floor and the two extra training fields; separate this change in information from optimizer effects.'),
       block('C. For each held-out Re, report vector/component L2, maximum signed centreline error and its location. Compare the selected ensemble with both interpolation conventions. Supply physical profiles, not bar charts.'),
       block('D. Compare neural error with the native mesh sensitivity and independently computed nonlinear residual. Explain why a mesh difference is not a certified error bound and exact divergence does not establish momentum balance.'),
       block('E. Enable the optional CPU cell and change one architecture or optimizer setting. Keep fitting and selection on training/validation only. Published test scores are disclosed; further test-based tuning is exploratory and needs fresh withheld cases for a blind claim.'),
       block('Deliverables: one CSV of measured results, two signed-profile figures, and an interpretation separating representation, optimization and CFD discretization. Record source/model/data hashes. Earlier laboratories remain available and unchanged.'),
       block('Methods in context: FFV-PINN solves separate data-free square-cavity cases with finite-volume residuals and correction loss; Roohi-McDevitt assesses regional accuracy in a regularized D/W=5 cavity. Neither should be conflated with this supervised parameter surrogate. Primary-source links: docs/CAVITY_ALL_RE_NEURAL_LITERATURE.md.','small')])])
    write('lectures/source/cavity_expansion.json',json.dumps(doc,indent=2))
    for name in ('pyproject.toml','flowmllab/__init__.py','CITATION.cff'):
        p=ROOT/name;s=p.read_text(encoding='utf-8').replace('1.12.1','1.13.0')
        if name=='CITATION.cff':s=s.replace('doi: "10.5281/zenodo.24247466"','doi: "10.5281/zenodo.22074169"')
        write(name,s)
    p=ROOT/'.zenodo.json';z=json.loads(p.read_text())
    z['version']='1.13.0'
    z['description']='FlowMLLab v1.13.0 adds 45 actual cavity CFD fields on three grids, all seven Ghia cases, high-Re mesh/time sensitivity, and a full-field POD-DeepONet with whole-Re tests, matched interpolation baselines and retained validation-driven training-data augmentation. Two new executed student notebooks and six Week-4 lecture pages explain representation, optimization and physical/numerical limits. Earlier notebooks, frozen profile models and original lecture pages are preserved. High-Re mesh sensitivity remains material; neural error is relative to the stated CFD discretization, not exact physical truth.'
    for item in z.get('related_identifiers',[]):
        if isinstance(item.get('identifier'),str):item['identifier']=item['identifier'].replace('v1.12.1','v1.13.0')
    write('.zenodo.json',json.dumps(z,indent=2))
    print('Prepared current curriculum, result pages, six lecture pages and release metadata.')

if __name__=='__main__':main()
