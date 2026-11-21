"""Editable teaching section for the local Week15 operator experiment."""
import nbformat as nbf

INTRO=r"""## Local CPU laboratory: learn a geometry-to-field operator

**Aim.** Train a small operator model, vary its capacity and optimization, and judge its spatial errors on geometries excluded from fitting and checkpoint selection. This section runs without the external research archive.

The reference is a **manufactured kinematic flow**, not CFD and not a new result for the double-step research study. Length is scaled by the channel length $L$, velocity by a common $U_0$, and streamfunction by $U_0L$. The formulas below use these dimensionless variables; the plots display the corresponding ratios explicitly. A smooth wall is
$b_a(x)=a\sin^2(\pi x)$, with $0\le x\le1$ and $b_a(x)\le y\le1$.
Use the mapped coordinate $\eta=(y-b_a)/(1-b_a)$ and the exact streamfunction
$\psi=\eta^2(3-2\eta)$. The velocity is

$$u=\frac{6\eta(1-\eta)}{1-b_a},\qquad
v=u\,b'_a(x)(1-\eta).$$

Here $a$ is the maximum wall elevation divided by $L$, $x,y$ are nondimensional physical coordinates, $\eta$ is a common wall-normal coordinate, and $u=\partial\psi/\partial y$, $v=-\partial\psi/\partial x$. This construction enforces incompressibility and zero wall velocity, but we have not solved a momentum equation or validated a CFD discretization.

**The actual model.** Training fields provide the mean and a rank-$r$ POD trunk on the mapped $(x,\eta)$ nodes. A tanh branch MLP takes only $a$ and predicts the modal coefficients:
$\widehat{\mathbf u}(a,x,\eta)=\overline{\mathbf u}(x,\eta)+\sum_{k=1}^{r}c_k(a)\,\boldsymbol\phi_k(x,\eta)$.
The trunk is learned from training snapshots and then held fixed; this is a POD-DeepONet, not a jointly optimized free-trunk network. Known geometry maps the prediction back to physical $y$.

| Item | Fixed default | What a controlled change tests |
|---|---|---|
| Training geometries | 0, .08, .16, .24, .40, .48, .56 | Available shape information; do not silently add validation/test shapes |
| Validation geometries | .12, .36 | Checkpoint selection only |
| Test geometries | .32, .52 | New instances within this manufactured family, not new topology |
| POD rank | 4 | Representation floor before branch fitting |
| Branch | 2 hidden layers, width 24, tanh | Capacity at fixed input data and optimizer |
| Optimization | 500 full-batch Adam updates, then at most 100 L-BFGS iterations | Optimization at a fixed architecture; iterations and objective evaluations are different costs |
| Loss weights | v weight 1, near-wall-band weight 1 | Redistribution of errors; report unweighted physical diagnostics as well |

The training objective is the weighted mean squared velocity error over training geometries and mapped nodes. The near-wall band is $\eta<0.15$ or $\eta>0.85$; increasing its weight emphasizes adjacent fluid points, while the velocity-component weight emphasizes $v$. The validation score is a separately reported unweighted discrete relative norm. Their curves have different units. All discrete scores use the same mapped nodes, not a claim of mesh-independent continuous error.
"""

LECTURE=INTRO.replace("## Local CPU laboratory: learn a geometry-to-field operator",
                     "## CPU operator laboratory: from geometry to a learned field")+r"""

### Follow one run before changing a control

1. Inspect the seven training walls and verify that both validation and test geometry sets are disjoint from them.
2. Fit the training-only POD trunk. Its mean, modes and coefficient scaling must not read validation or test fields.
3. Initialize the branch with the declared seed. Adam and L-BFGS optimize only the weighted training objective.
4. At declared updates, evaluate the validation geometries and save the lowest-validation-error state. Record the selected update, elapsed time and objective evaluations, including a possible step-zero selection.
5. Freeze that state, then create the two exact test references. Compare with linear interpolation between adjacent training geometries and with the POD projection of the test field. The latter uses reference data and is a representation diagnostic, not a deployable predictor.
6. Inspect the native mapped-grid speed/error panels and signed u/v profiles at a declared physical x column. Keep losses and improvements visible for both test geometries.

### Four numerical investigations

**Capacity:** keep rank, training geometry, optimizer, seed and loss weights fixed; compare depth 1, 2 and 4 at a declared width. Report parameter counts and objective evaluations as well as error. A deeper network does not automatically have a matched computational budget.

**Optimization:** freeze the architecture and compare Adam, L-BFGS and Adam followed by L-BFGS. Report updates, line-search evaluations, elapsed time, selected validation update and both test errors. An L-BFGS iteration may evaluate the objective more than once.

**Loss weighting:** compare v weights 1 and 10, then near-wall weights 1 and 5 in separate experiments. Keep the unweighted validation criterion fixed. Plot signed u and v discrepancies: an improved weak component or near-wall region can accompany deterioration elsewhere.

**Representation and generalization:** compare ranks 2 and 4 using training-only bases. Report the exact test projection error separately from branch-prediction error and the linear baseline. Once test figures have guided development, call subsequent comparisons retrospective and reserve a new manufactured geometry before making a new final-test claim.

Submit the configuration, geometry partition, loss histories, per-case error table, native field/profile comparisons and one supported recommendation. This exercise teaches learning and evaluation mechanics. The retained double-step galleries address a different geometry/topology problem and must not be merged into its score table.
"""

def cells():
    def m(key,text):return nbf.v4.new_markdown_cell(text,id='late-operator-'+key)
    def c(key,text):return nbf.v4.new_code_cell(text,id='late-operator-'+key)
    return [
      m('aim',INTRO),
      c('controls',"""from flowmllab import geometry_operator_lesson as operator_lab
from dataclasses import asdict
OPERATOR_CONFIG = operator_lab.Config(
    depth=2, width=24, rank=4, optimizer="adam_lbfgs",
    adam_steps=500, lbfgs_iterations=100, learning_rate=.01,
    v_weight=1., near_wall_weight=1., seed=17)
print(asdict(OPERATOR_CONFIG))
print("Train:",operator_lab.TRAIN,"validation:",operator_lab.VALIDATION,"test:",operator_lab.TEST)
assert not (set(operator_lab.TRAIN)&set(operator_lab.VALIDATION)
            or set(operator_lab.TRAIN)&set(operator_lab.TEST)
            or set(operator_lab.VALIDATION)&set(operator_lab.TEST))
fig,ax=plt.subplots(figsize=(10,3.8),constrained_layout=True)
for a in operator_lab.TRAIN:
    shape=operator_lab.reference(a)
    ax.plot(shape["x"],shape["Y"][0],lw=1.2,label=f"training a={a:g}")
ax.plot([0,1],[1,1],color="black",lw=1.5)
ax.set(xlabel="x/L",ylabel="wall y/L",ylim=(0,1),title="Manufactured training geometries")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.35),ncol=4,frameon=False,fontsize=8)
plt.show()
"""),
      m('fit-intro',"""### Fit on training geometries; select on validation

The full-batch objective and analytic gradient are implemented in the small NumPy/SciPy module linked by the import. No test reference is used by fit(). The default fit is bounded by the controls above. A finite optimizer budget is not a convergence certificate; keep the returned termination message."""),
      c('fit',"""operator_model=operator_lab.fit(OPERATOR_CONFIG)
print("Parameters:",len(operator_model["theta"]))
print("Optimizer objective/gradient calls:",operator_model["objective_evaluations"],"elapsed seconds:",operator_model["seconds"])
print("Additional train/validation monitoring evaluations:",operator_model["monitoring_evaluations"])
print("Selected update:",operator_model["selected_step"],operator_model["selected_phase"])
print("Validation relative discrete L2:",operator_model["validation_score"])
print("L-BFGS termination:",operator_model["lbfgs"])
operator_lab.plot_training(operator_model)
plt.show()
"""),
      m('test-intro',"""### Freeze first; compare exact test fields

Linear interpolation uses only adjacent training geometries. The POD oracle projects the exact test reference onto the frozen training basis and cannot be used as a deployable prediction. Each geometry gets its own field and signed-profile comparison; all extrema are retained. The drawn wall is the actual manufactured geometry, not a CFD mesh."""),
      c('test',"""operator_tests=operator_lab.evaluate(operator_model)
operator_rows=[]
for result in operator_tests:
    operator_rows.append({"geometry_a":result["a"],**result["scores"]})
    operator_lab.plot_case(result)
    plt.show()
display(pd.DataFrame(operator_rows))
print("All scores are discrete relative norms on mapped nodes; reference and predictions share the same nodes.")
print("Manufactured geometry interpolation only. No new CFD, research training or topology-generalization claim.")
"""),
      m('experiments',"""### Your controlled experiments

1. Compare depths 1, 2 and 4 with one fixed width, rank, seed and optimizer. Keep every run, including losses to the linear baseline; report parameters and objective evaluations.
2. Compare optimizer choices adam, lbfgs and adam_lbfgs at declared budgets. Explain why one L-BFGS iteration can cost more than one Adam update.
3. Change v_weight from 1 to 10, then separately near_wall_weight from 1 to 5. Use the same unweighted validation criterion and signed u/v profiles to identify which region or component improved and which deteriorated.
4. Compare ranks 2 and 4. Separate representation error from branch error before increasing network depth.

Submit a configuration table, the training/validation histories, errors for both test geometries, and spatial/profile comparisons. Pick a development choice from validation, not the nicest test picture. Once you have used test results to change the model, label the comparison retrospective and predeclare a new geometry for a later test. Keep this manufactured experiment separate from the retained research tables below.""")
    ]
