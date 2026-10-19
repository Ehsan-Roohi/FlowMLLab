# Optional self-study | Geometry, time and trustworthy operator evaluation

**FlowMLLab · Ehsan Roohi**

**Status:** optional bridge, outside the mandatory Weeks 1–22. This is an educational exercise in evaluation contracts. It is not a reproduction of a published neural operator, a new CFD simulation, or a new blind benchmark.

**Prerequisites:** Week 15 geometry/SDF and case-level evaluation; Week 21 autonomous prediction and phase. Week 18's causal-information discussion is useful preparation. Python, NumPy array shapes and elementary sine/cosine algebra are assumed.

**Companion:** `notebooks/optional_geometry_time_bridge/Geometry_Time_Operator_Bridge.ipynb`. Worked examples use CPU and manufactured arrays. There is no model fitting, checkpoint download, CFD execution or GPU requirement. Allow **150–180 minutes of study**, an estimate rather than measured student workload. The notebook reports its own computational elapsed time separately.

## Learning objectives

After completing this module, you should be able to:

1. distinguish a geometry-conditioned steady map from a geometry-conditioned evolution operator;
2. split geometry groups before constructing overlapping history windows, and distinguish this from family holdout;
3. compute a masked field error without broadcasting across batch entries or silently changing case weights;
4. use a future-poison counterexample to distinguish genuine autonomous evaluation from repeated one-step evaluation;
5. report amplitude, circular phase error, fit residual and undefined phase without turning a signed mean into an accuracy claim.

The existing course already treats rollout, causal inputs, POD representation error and geometry transfer. This bridge combines those ideas in one small contract audit. Its assessment rewards a supported conclusion, including a negative result.

## 1. The scientific question comes before the architecture

A steady geometry-conditioned surrogate maps a shape and operating condition to a field. A time-dependent surrogate also receives observed or predicted states. In an illustrative notation,

$$ \widehat{q}_{t+1}=F_\theta(q_{t-s+1:t},\phi_\Gamma,x,y), $$

where $s$ is the history length and $\phi_\Gamma$ is a signed distance field. This notation says what information is available; it does not establish that an implementation preserves mass, resolves a boundary layer or transfers between grids.

Rabeh et al. [1] combine a CNN history encoder with an SDF-conditioned DeepONet. The paper reports approximately 5% relative single-step field error and approximately 55% error late in an autoregressive rollout. Those are **paper-reported results**, not results reproduced here. Single-step evaluation supplies real history repeatedly. Autonomous evaluation supplies real history once and then must live with its predictions.

The paper's variable-geometry study motivates the bridge, while its limits motivate the questions: which geometries were withheld, which information enters each forecast, and which metric supports the claim? A random case split is not automatically a geometry-family split. A small signed phase average is not necessarily a small phase error. SDF input is not a boundary-condition proof.

## 2. Manufactured data: useful precisely because its limits are explicit

The companion creates six shapes, each at two oscillator rates: 12 cases, 25 frames per case and a $17\times25$ grid with two velocity-like channels. Coordinates and time are dimensionless. The oscillator rate is **not Reynolds number**. Circle and rectangle signed distances are analytic. Positive distance denotes fluid; zero and negative distance are excluded from fluid-only metrics.

The circle SDF is $\phi(x,y)=\sqrt{x^2+y^2}-r$. For a rectangle centered at the origin with half-widths $a,b$, define $d_x=|x|-a$ and $d_y=|y|-b$:

$$ \phi=\sqrt{\max(d_x,0)^2+\max(d_y,0)^2}+\min(\max(d_x,d_y),0). $$

The mask is $m=1[\phi>0]$. This is a geometry convention. It must be recorded, because reversing the SDF sign reverses the interpretation of the mask.

The generated fields are smooth oscillating arrays. They are **not Navier–Stokes solutions**. Masking a field does not impose no-slip, pressure boundary conditions or incompressibility. The examples can validate indexing and numerical diagnostics; they cannot establish CFD accuracy or operator generalization. Reusing the same field formula on different masks is deliberate: geometry changes the evaluation domain without pretending to model a geometry-induced wake.

The storage contract is:

| Quantity | Shape | Meaning |
|---|---|---|
| one case's field | `[T,H,W,C]` | time, vertical coordinate, horizontal coordinate, component |
| batched predictions | `[B,H,W,C]` | one target per batch entry |
| signed distance | `[H,W]` per case | fluid positive |
| batched fluid mask | `[B,H,W]` | Boolean; shared across the C components of that case |
| history window | `[s,H,W,C]` | observed/predicted fields in chronological order |

Every shape assertion should occur before arithmetic. A singleton dimension can be dangerous: `[B,1,P,C]` multiplied by `[B,P,1]` broadcasts to `[B,B,P,C]`, coupling each prediction to multiple masks. An array operation completing successfully does not prove its scientific meaning.

## 3. Split the object of generalization before making windows

Each case has a case ID, geometry ID and oscillator rate. Two rates for the same shape must remain in the same partition when the intended question is unseen-geometry prediction. Split geometry IDs first; then construct history/target pairs inside each case.

The worked split is train = g0, g1, g3, g4; validation = g2; test = g5. With 25 frames and history four, each case yields $25-4=21$ one-step windows. There are eight training cases, two validation cases and two test cases, hence **168 / 42 / 42 windows**. These 252 windows are not 252 independent geometries.

For target index $t$, the history is Python slice `[t-s:t]`; the target is `[t]`. For $s=4,t=4$, the input indices are 0,1,2,3 and the output index is 4. This indexing should be checked by hand before model code runs.

The worked test shape belongs to the rectangle family, which is also present in training. It is a new geometry instance, **not a held-out family**. Holding out all rectangles would pose a different question and change the training data. Neither protocol is automatically superior; each must match its stated claim.

Splitting overlapping windows randomly can place histories from one trajectory in both fitting and validation. This does not prove that an independently held-out external test file is contaminated. It does mean that the validation question differs from unseen-geometry validation. State the scope of the evidence precisely.

## 4. A masked metric is also a weighting decision

For squared errors $e_{bpc}^2$, define the MSE of case b over its fluid cells and C components:

$$ E_b=\frac{\sum_{p,c}m_{bp}e_{bpc}^2}{C\sum_pm_{bp}}. $$

The **equal-case** statistic is the mean of the $E_b$. The **pooled** statistic divides the sum of all masked squared errors by the sum of all fluid-component counts. The latter gives larger fluid regions more weight. Both can be legitimate; they answer different aggregation questions.

### Worked hand example

Case A has one fluid value with error 2 and one excluded solid value with error 100. Case B has two fluid values, both with error 3. There is one component. Case errors are $4$ and $9$. Equal-case MSE is $(4+9)/2=6.5$; pooled MSE is $(4+9+9)/3=22/3$. The solid error does not contribute. Duplicating the batch leaves either mean unchanged.

The implementation deliberately rejects an empty fluid mask and shape mismatches. It also rejects nonfinite field entries, including inside solids, rather than relying on zero times NaN. This is an explicit contract choice.

A boundary strip is a second diagnostic region, not a second physical simulation. For a width $w>0$, use $0<\phi\leq w$. Its MSE can reveal a local failure hidden by a large easy far-field region. Report the strip width and number of cells; a coarse grid can leave a strip empty.

## 5. Future poisoning: test information flow, not accuracy

The `autonomous` helper accepts a predictor, initial history and number of forecast steps. It has no argument for future truth. The predictor receives a copy of history; after predicting, the oldest state is removed and the new state appended. Truth enters only when scoring the completed trajectory.

The `teacher_forced` helper intentionally reads real history for every target. This is valid one-step evaluation. It becomes misleading only when labeled an uninterrupted autonomous forecast.

### Worked hand example

Let the observed states be 1,2,3,4,5,6,7, with history length two. Let the imperfect predictor return the latest history value plus 0.25. Autonomous predictions are **2.25, 2.50, 2.75, 3.00, 3.25**. Teacher-forced predictions are **2.25, 3.25, 4.25, 5.25, 6.25**.

Now add 1000 to every truth value after initialization, leaving 1 and 2 untouched. Autonomous predictions must remain identical. Teacher-forced predictions remain identical only for the first target; later histories contain the changed observations. Scoring the autonomous trajectory against the poisoned truth must change its error. Prediction invariance and scoring sensitivity are complementary checks.

Passing this test does not prove model accuracy. A constant predictor also passes. The deliberately teacher-forced control must fail for a nondegenerate predictor, so a weak counterexample cannot silently certify a broken evaluator.

When comparing histories of length 1,4,8 or16, use the same forecast start and future horizon. Count the true initialization frames separately. A model receiving 16 true frames has a different information budget from one receiving one. A reset at an evaluation boundary is another protocol, not a continuation of the same forecast.

## 6. Phase must have an origin, amplitude and identifiability rule

At a supplied frequency f, fit

$$ y(t)=c+a\sin(2\pi f(t-t_0))+b\cos(2\pi f(t-t_0)). $$

Here $t_0$ is the first sample time. The amplitude is $A=\sqrt{a^2+b^2}$, and phase is $\phi=\operatorname{atan2}(b,a)$. The bridge supplies f because it is known for manufactured signals; it **does not estimate shedding frequency**. Week 21's continuous-frequency fit is the next step for measured traces.

The fitted residual MSE and $R^2$ describe how well a single-frequency model represents the trace. An additional harmonic can make the residual large even when the extracted fundamental phase is meaningful. Report that mismatch rather than implying the entire signal is captured.

For two phases with a common time origin and the same frequency, use the principal circular difference:

$$ \delta=\operatorname{atan2}(\sin(\widehat\phi-\phi),\cos(\widehat\phi-\phi)). $$

### Worked hand examples

- Reference 179 degrees and prediction -179 degrees differ by **2 degrees** on the circle, not 358 degrees.
- Two errors +90 and -90 degrees have signed mean zero but mean absolute circular error **90 degrees**. Cancellation is not accuracy.
- For $2\sin(2\pi f(t-t_0)+0.3)$ the amplitude is 2 and the phase is 0.3 radians. A constant signal has zero oscillatory amplitude; its phase is **undefined**, represented by `None`, not zero.

An absolute amplitude floor must be stated in signal units. The worked value is $10^{-10}$ for these dimensionless, noise-free examples, not a universal experimental threshold. Undefined pairs remain in the count; they are not silently discarded to improve the average.

If frequencies differ, separate the initial phase offset from accumulating drift, $\Delta\phi(T)=2\pi(\widehat f-f)T$. For f = 0.200 and predicted f = 0.205 over T = 20, drift is about **0.628 radians**. Do not wrap the total drift if completed turns matter. A local signal frequency is not automatically a lift-derived Strouhal number; $St=fD/U$ requires documented D, U and time units.

## 7. Four exercises and a self-study rubric

The student notebook's worked cells run without answers. Exercise cells initially report **NOT SUBMITTED**. Fill the four functions and set `RUN_EXERCISES=True`. A skipped exercise is never marked passed. A separate instructor notebook provides reference answers for self-study after an attempt.

| Exercise | Deliverable and adversarial control | Points | Study time |
|---------------------|----------------------------------------------------------|------:|------------:|
| 1. Audit group ownership | Recover geometry ownership from window records; reject a deliberately split geometry. Explain why the example is not family holdout | 25 | 25 min |
| 2. Boundary-strip error | Form the positive-SDF strip and compute equal-case MSE; reject an empty strip; compare it with the global hand example | 25 | 25 min |
| 3. Implement feedback | Implement an autonomous history update without future truth; show unchanged forecasts and changed scores after poisoning; show the teacher-forced control fails | 25 | 35 min |
| 4. Phase and missingness | Report amplitude ratio, absolute circular phase error and both residuals; report undefined phase for a constant prediction; explain a mixed-frequency residual | 25 | 30 min |

For each exercise, award **15 points for correct numerical behavior**, including the adverse case, and **10 for interpretation** with correct scope and units. An executable answer that confuses a manufactured array with CFD evidence cannot receive the interpretation points. Add 35–65 minutes for reading, worked examples and the final claim ledger to reach the study budget above.

Suggested self-study standard: at least 70/100 overall, all four tasks attempted, and the future-poison and undefined-phase controls correctly explained. This is a proposed grading rule, not a classroom-validated assessment instrument.

## 8. A bounded claim ledger

Complete a four-row ledger: (i) a behavior proved by a small counterexample, (ii) a paper-reported result not reproduced here, (iii) a hypothesis requiring CFD data, and (iv) an unresolved access or provenance question.

Examples of appropriate conclusions are: “the evaluator does not consume future observations”; “the paper reports substantial long-horizon drift”; and “SDF conditioning may help across new shapes, which these manufactured arrays cannot establish.” Avoid claims that this module reproduced the paper's accuracy, speedup, force prediction, incompressibility or OOD generalization.

FlowBench [2] is a possible research extension, not a dependency. Its current dataset card describes the removal of the 512 by128 FPO version because geometry was insufficiently resolved. A smaller spatial grid is therefore not an innocuous replacement for the paper's dataset. Public code, a published paper and a classroom adaptation must have separate versions and claims.

The supplied LinkedIn URL could not be read during the preceding audit. No content, author claim or result from that post is used here. A social post is not substituted for the paper or its numerical evidence.

## References and evidence limits

1. Rabeh, A., Murugaiyan, S., Krishnamurthy, A., and Ganapathysubramanian, B. **Predicting time-dependent flow over complex geometries using operator networks.** *Computer Methods in Applied Mechanics and Engineering* 456 (2026), 118931. DOI: https://doi.org/10.1016/j.cma.2026.118931 . The complete 23-page supplied journal PDF was read for the audit. Relevant sections: architecture/data pp.3–4, history pp.5–6, rollout pp.9–16, future work p.17. The numbers here are reported, not reproduced.
2. Tali et al. **FlowBench: A Large Scale Benchmark for Flow Simulation over Complex Geometries.** https://arxiv.org/abs/2409.18032 . Current dataset card and license: https://huggingface.co/datasets/BGLab/FlowBench . The card is used for data availability and revision history, not as an independent CFD validation.
3. He et al. **Geom-DeepONet: A point-cloud-based deep operator network for field predictions on 3D parameterized geometries.** DOI: https://doi.org/10.1016/j.cma.2024.117130 . Geometry-conditioned branch/trunk context; this module does not train that architecture.
4. Pfaff et al. **Learning Mesh-Based Simulation with Graph Networks.** https://arxiv.org/abs/2010.03409 . Appendix A.2 discusses training noise and adjusted targets. A baseline's training recipe matters; its name alone does not define a comparison.
5. Lippe et al. **PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.** https://arxiv.org/abs/2308.05732 . Motivation for frequency-sensitive rollout analysis; its refinement results are not reproduced or transferred to these manufactured data.
6. Paper authors' public code, inspected revision: https://github.com/baskargroup/TimeDependent-DeepONet/tree/9f7608635fa4570fa2e699a447ec5a3e6ef66206 . It is a versioned source for a code-reading exercise, not a verified reconstruction of every published experiment. This module uses an independent small implementation, with no downloaded model weights.

**Evidence statement:** software tests validate the explicit contracts and hand examples. They do not validate classroom learning outcomes, full-paper reproduction, blind generalization, a CFD solver or a scientific speedup. The course's existing Week 15 and Week 21 results remain separate.
