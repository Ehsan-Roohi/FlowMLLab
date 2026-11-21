"""Apply focused early-course teaching repairs without changing retained source data."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
MARK='FLOWMLLAB_EARLY_TEACHING_20261120'

def read(relative):
    p=ROOT/relative
    return p,json.loads(p.read_text(encoding='utf-8'))
def md(text):
    return {'cell_type':'markdown','metadata':{},'source':text.strip()+'\n','id':hashlib.sha256(text.encode()).hexdigest()[:12]}
def code(text):
    return {'cell_type':'code','metadata':{},'source':text.strip()+'\n','outputs':[],'execution_count':None,'id':hashlib.sha256(text.encode()).hexdigest()[:12]}
def text(c):return ''.join(c['source'])
def find(nb,needle):return next(c for c in nb['cells'] if needle in text(c))
def after(nb,needle,*cells):
    i=next(i for i,c in enumerate(nb['cells']) if needle in text(c));nb['cells'][i+1:i+1]=list(cells)
def before(nb,needle,*cells):
    i=next(i for i,c in enumerate(nb['cells']) if needle in text(c));nb['cells'][i:i]=list(cells)
def replace(nb,needle,old,new):
    c=find(nb,needle);s=text(c)
    if old not in s:raise ValueError(needle+' missing expected text')
    c['source']=s.replace(old,new)
def save(p,nb):
    nb['cells'].append(md('<!-- '+MARK+' -->'))
    p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
def external_legends(nb):
    for c in nb['cells']:
        if c['cell_type']=='code':
            s=text(c)
            s=s.replace('.legend()',".legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)")
            s=s.replace('.legend(markerscale=2)',".legend(markerscale=2, loc='upper left', bbox_to_anchor=(1.02, 1))")
            c['source']=s

def week2():
 p,nb=read('notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb')
 if any(MARK in text(c) for c in nb['cells']):return
 replace(nb,'LEARNING_RATE = 1e-3','TRAINING_SEED = 7','TRAINING_SEED = 7\nACTIVATION = "tanh"\nassert HIDDEN_UNITS and all(isinstance(w, int) and w > 0 for w in HIDDEN_UNITS)')
 c=find(nb,'LEARNING_RATE = 1e-3');c['source']=text(c)+'''

def build_surrogate(hidden_units, activation="tanh", input_dim=4, output_dim=1):
    """The tuple controls every hidden layer: (32,) and (32,32,32) are different depths."""
    if not hidden_units or any(not isinstance(w, int) or w <= 0 for w in hidden_units):
        raise ValueError("hidden_units must contain positive integer widths")
    return keras.Sequential([layers.Input(shape=(input_dim,)),
        *[layers.Dense(width, activation=activation) for width in hidden_units],
        layers.Dense(output_dim, activation="linear")])
'''
 old='''model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(HIDDEN_UNITS[0], activation="tanh"),
    layers.Dense(HIDDEN_UNITS[1], activation="tanh"),
    layers.Dense(1, activation="linear")
])'''
 replace(nb,'training_started = time.perf_counter()',old,'model = build_surrogate(HIDDEN_UNITS, ACTIVATION)')
 after(nb,'### Training controls: change one choice at a time',md(r'''### A depth control must change the actual computational graph

`HIDDEN_UNITS=(32,)` creates one hidden layer; `(32,32)` creates two; `(32,32,32)` creates three. Each layer computes $h^{(l)}=\sigma(W^{(l)}h^{(l-1)}+b^{(l)})$. Here $h^{(0)}$ contains four **training-standardized** inputs, $W$ and $b$ are trainable weights/biases, and `ACTIVATION` selects $\sigma$. The final linear layer produces one standardized $Q^*$; inverse scaling restores its physical teaching units.

The parameter count is $\sum_l (n_{l-1}+1)n_l$, including biases. The assertion after model construction must agree with that formula. Changing width affects both approximation capacity and cost; changing depth adds compositions rather than merely adding parameters. Keep the seed, split and maximum update budget fixed for the first comparison, select with validation loss, then inspect the frozen test once. More layers can optimize worse; neither lower training loss nor a smoother curve proves better generalization.

**Engineering check:** compare `(32,)`, `(32,32)`, `(32,32,32)`. Record parameter count, actual epochs, elapsed CPU time, validation MSE and the signed fixed-condition profile error. The initial comparison is against noisy synthetic labels, while the profile uses the noise-free formula. These are different error targets. No result in this notebook validates a physical rarefied-flow solver.'''))
 external_legends(nb);save(p,nb)

def week3_lab1():
 p,nb=read('notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb')
 if any(MARK in text(c) for c in nb['cells']):return
 after(nb,'## Learning objectives',md(r'''### Route, controls and expected evidence

This CPU lab has four distinct experiments: draw a distribution; recover its moments; measure estimator convergence over independent realizations; and fit a small neural surrogate to labels with two noise levels. A point in the sampling-error plot is an average over independent particle clouds, not a time history. A point in the slip-proxy plot is a separate prescribed Knudsen condition, not a DSMC cell.

| Control | Meaning | First controlled change | Diagnostic |
|---|---|---|---|
| `N_SAMPLE` | Molecules in the displayed equilibrium cloud | 5000 to 500 | Distribution shape and moment discrepancy |
| `T_TRUE`, `U_TRUE` | Temperature [K], three-component drift [m/s] | Double temperature, retain drift | Width grows by $\sqrt{2}$, center stays fixed |
| `N_VALUES`, `N_REPEATS` | Population per cloud, independent clouds per population | Double repeats only | Less scatter in estimated RMSE, no change in estimator bias |
| `HIDDEN_UNITS`, `ACTIVATION` | Layer widths and nonlinear function | One change at a time | Label fit and signed error against analytic teaching trend |
| `MLP_ALPHA`, `MAX_ITER` | Weight penalty, L-BFGS iteration cap | Change using a separate validation design | Underfit/overfit and optimizer warning |

Save the random seeds with the answer. The worked neural comparison has **fixed, predeclared settings**; its analytic evaluation curve is public, not an untouched blind test. If you tune a model, introduce separate validation conditions and freeze the choices before a new evaluation. Keep the original controlled comparison in your report.'''))
 after(nb,'## 2. Macroscopic quantities as particle moments',md(r'''For $N$ velocities $\mathbf c_i$, define the sample drift $\widehat{\mathbf U}=N^{-1}\sum_i\mathbf c_i$ and thermal residual $\mathbf C_i=\mathbf c_i-\widehat{\mathbf U}$. The code evaluates

$$T_N=\frac{m}{3k_BN}\sum_i|\mathbf C_i|^2,\qquad p_N=n_0k_BT_N,\qquad \mathbf q_N=\frac{mn_0}{2N}\sum_i |\mathbf C_i|^2\mathbf C_i.$$

Here $m$ is the argon molecular mass, $k_B$ is Boltzmann's constant, and $n_0$ is number density [m$^{-3}$]. $T_N$, $p_N$ and $\mathbf q_N$ have units K, Pa and W/m$^2$. Subtracting the drift prevents organized motion from becoming spurious temperature. At equilibrium the exact heat-flux vector vanishes because odd thermal moments cancel; a finite sample does not cancel exactly. The norm of a noisy vector is nonnegative, so its average need not vanish.

The historical denominator $N$ is intentionally retained to expose finite-sample bias. For independent fixed-$N$ clouds, $T_{N-1}=NT_N/(N-1)$ is unbiased for $N>1$. This formula is not automatically valid for time-correlated or variably populated cavity cells.'''))
 replace(nb,'def repeated_sampling_errors','    t_errors = []','    t_errors = []\n    t_unbiased_errors = []')
 replace(nb,'def repeated_sampling_errors','        t_errors.append(Te - T_TRUE)','        t_errors.append(Te - T_TRUE)\n        t_unbiased_errors.append(n_particles*Te/(n_particles-1) - T_TRUE)')
 replace(nb,'def repeated_sampling_errors','"rmse_T": np.sqrt(np.mean(np.square(t_errors))),','"rmse_T": np.sqrt(np.mean(np.square(t_errors))),\n        "bias_T": float(np.mean(t_errors)),\n        "rmse_T_unbiased": np.sqrt(np.mean(np.square(t_unbiased_errors))),\n        "bias_T_unbiased": float(np.mean(t_unbiased_errors)),')
 replace(nb,'def repeated_sampling_errors','rows = []\nfor n in N_VALUES:','N_REPEATS = 220\nrows = []\nfor n in N_VALUES:')
 replace(nb,'def repeated_sampling_errors','repeats=220, seed=100+n','repeats=N_REPEATS, seed=100+n')
 c=find(nb,'Sampling convergence');c['source']=r'''
# Dimensionless errors allow a meaningful comparison of different moments.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
N = noise_df["N"].to_numpy()
axes[0].loglog(N, noise_df["rmse_Ux"]/sigma_true, "o-", label=r"$U_x$: RMSE/$\sigma_c$")
axes[0].loglog(N, noise_df["rmse_T"]/T_TRUE, "s-", label=r"$T_N$: RMSE/$T_0$")
axes[0].loglog(N, noise_df["rmse_T_unbiased"]/T_TRUE, "^-", label=r"$T_{N-1}$: RMSE/$T_0$")
axes[0].loglog(N, N.astype(float)**(-.5), "k--", label=r"$N^{-1/2}$")
axes[0].set(xlabel="Independent particles N", ylabel="Normalized RMSE", title="Random error and finite-N bias")
axes[1].semilogx(N, noise_df["bias_T"]/T_TRUE, "s-", label=r"Measured $T_N$ bias")
axes[1].semilogx(N, noise_df["bias_T_unbiased"]/T_TRUE, "^-", label=r"Measured $T_{N-1}$ bias")
axes[1].semilogx(N, -1/N, "k--", label=r"Exact $-1/N$ for $T_N$")
axes[1].axhline(0, color=".5", lw=.8)
axes[1].set(xlabel="Independent particles N", ylabel="Mean signed error / T0", title="Bias is not reduced by more repeats")
axes[2].loglog(N, noise_df["mean_qstar_norm"], "o-", color="tab:purple")
axes[2].set(xlabel="Independent particles N", ylabel=r"Mean $|q|/(n_0k_BT_0\sigma_c)$", title="Equilibrium heat-flux noise")
for ax in axes: ax.grid(which="both", alpha=.2)
for ax in axes[:2]: ax.legend(loc="upper center", bbox_to_anchor=(.5,-.20), fontsize=8)
plt.show()
'''
 replace(nb,'def hidden_slip_trend','def make_small_mlp(random_state=4):','HIDDEN_UNITS = (24, 24)\nACTIVATION = "tanh"\nMLP_ALPHA = 1e-3\nMAX_ITER = 3000\nMLP_SEED = 4\n\ndef make_small_mlp(random_state=MLP_SEED):')
 replace(nb,'def hidden_slip_trend','hidden_layer_sizes=(24, 24), activation="tanh"','hidden_layer_sizes=HIDDEN_UNITS, activation=ACTIVATION')
 replace(nb,'def hidden_slip_trend','alpha=1e-3, max_iter=3000','alpha=MLP_ALPHA, max_iter=MAX_ITER')
 after(nb,'## 4. ML bridge:',md(r'''The smooth slip curve below is an **analytic teaching proxy**, not a solution of the Boltzmann equation. Both networks receive the same 75 condition coordinates and have identical architecture, initialization and L-BFGS settings. Only label quality changes. `StandardScaler` is fit to the training coordinates; tanh layers map that coordinate to one scalar output, and L-BFGS uses gradients of squared label error with an L2 weight penalty.

The long-averaging labels use a smaller prescribed noise amplitude. This mimics a data-quality improvement but is not a measured DSMC averaging law. Noise is clipped at zero, which can bias observations near zero. A flexible network can interpolate noise or violate monotonicity even when its curve looks smooth. Plot signed error against the known analytic proxy and retain any optimizer warning; do not call the noisier-label model a failed physical solver.'''))
 after(nb,'# Basic physical checks',code(r'''fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
ax.plot(10**log_kn_test, pred_short-y_truth_test, label="Short-averaging labels")
ax.plot(10**log_kn_test, pred_long-y_truth_test, label="Long-averaging labels", ls="--")
ax.axhline(0, color="black", lw=.8)
ax.set(xscale="log", xlabel="Knudsen number Kn", ylabel="Prediction - analytic slip proxy")
ax.legend(loc="upper left", bbox_to_anchor=(1.02,1)); ax.grid(alpha=.2); plt.show()
print("L-BFGS accepted iterations:", model_short[-1].n_iter_, model_long[-1].n_iter_)
'''))
 # Histograms are density curves, not bar charts.
 for c in nb['cells']:
  if c['cell_type']=='code':c['source']=text(c).replace('density=True, alpha=0.65','density=True, histtype="step", linewidth=1.6')
 external_legends(nb)
 before(nb,'## Final checklist',md(r'''### Three controlled investigations to submit

1. **Bias versus variability:** at N=30,60,300 report the mean signed temperature error and RMSE for both denominators. Compare the measured mean with $-T_0/N$. Repeat with twice as many independent clouds. Explain why a noisy estimate of bias can fluctuate around its exact expectation.
2. **Signal-to-noise:** keep temperature fixed and reduce the imposed x drift from100 to10 m/s. Compare the absolute drift standard error with its value relative to the drift. Do not divide by a zero velocity component. Explain why low-speed molecular flow is costly even when the thermal distribution is well sampled.
3. **Data quality versus model capacity:** retain the default fit, then change only one width/depth control. Report iteration counts, signed error and monotonicity violations for each noise level. State whether the change learned a useful trend or fit noisy labels more closely. A new validation design is required before presenting a tuned result as an independent test.'''))
 save(p,nb)

def week3_lab2():
 p,nb=read('notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb')
 if any(MARK in text(c) for c in nb['cells']):return
 after(nb,'## 2. Configuration and model assumptions',md(r'''### Trace each control to a physical or statistical question

The square has $L=1$ and molecular component variance $T_{wall}$; thus $\bar c=\sqrt{8T_{wall}/\pi}$ and the model collision time is $\tau\approx Kn/\bar c$. `Kn` sets a teaching collision rate, not a calibrated hard-sphere cross section. The comparison with the retained dimensional HS-NTC solver above is contextual; do not score this mini solver against that different physical case.

`nx,ny` define collision/sampling cells; `ppc` defines initial particles per cell, with total $N_p=n_xn_y\,ppc$. `dt` advances molecular positions, `steps` fixes final physical time, `sample_start` discards startup, and `sample_stride` selects stored field observations. Adjacent observations can remain correlated. `seed` selects one realization. `Uwall` controls tangential drift in diffuse lid emission, and `T_wall` controls thermal velocities at every wall.

The supplied quick and better-sampled configurations change grid, particles, time step, averaging window **and seed**. Their difference is not an isolated grid-convergence or sampling-error estimate. Use them to diagnose what metadata are needed; the controlled experiments at the end change one factor and compare identical physical averaging windows.'''))
 after(nb,'## 4. Cell-based stochastic elastic collisions',md(r'''For an identical-particle pair define $\mathbf V=(\mathbf v_1+\mathbf v_2)/2$ and $g=|\mathbf v_1-\mathbf v_2|$. The code draws a unit direction $\mathbf e$ uniformly on the sphere and returns $\mathbf v'_1=\mathbf V+g\mathbf e/2$ and $\mathbf v'_2=\mathbf V-g\mathbf e/2$. Therefore pair momentum and kinetic energy are conserved independently of the random direction. Isotropy requires uniform $\cos\theta$, not uniform polar angle.

Within each cell the expected number of pairs is approximately $N_{cell}\bar c\Delta t/(2KnL)$. Stochastic rounding gives an integer count without systematically truncating rare collisions. This simplified rate ignores pair-speed-dependent acceptance and is **not the VHS-NTC algorithm** used in the optional production solver. Conservation is necessary but cannot by itself validate collision statistics.'''))
 c=find(nb,'# Verify pair conservation');c['source']=text(c)+'''\nnp.testing.assert_allclose(p_after, p_before, atol=1e-14)\nnp.testing.assert_allclose(e_after, e_before, atol=1e-14)\n'''
 after(nb,'run_table = pd.DataFrame',code(r'''# Sampling occurs after each move: the first recorded time is (sample_start+1)*dt.
window_rows = []
for cfg in (quick_cfg, better_cfg):
    sampled_steps = np.arange(cfg.sample_start, cfg.steps, cfg.sample_stride)
    assert len(sampled_steps) > 0
    window_rows.append({"case":cfg.label,"first_sample_time":(sampled_steps[0]+1)*cfg.dt,
        "last_sample_time":(sampled_steps[-1]+1)*cfg.dt,
        "sample_spacing":cfg.sample_stride*cfg.dt,"recorded_samples":len(sampled_steps),
        "final_time":cfg.steps*cfg.dt,"Uwall/mean_thermal_speed":cfg.Uwall/cfg.mean_thermal_speed})
display(pd.DataFrame(window_rows))
'''))
 # Keep original plotting functions available, use a focused native-field display in default path.
 c=find(nb,'def plot_particle_snapshot')
 s=text(c).replace('cmap="coolwarm",\n                         vmin=-2, vmax=2, alpha=0.65','cmap="turbo",\n                         vmin=float(case["velocity"][ids,0].min()),\n                         vmax=float(case["velocity"][ids,0].max()), alpha=0.8')
 start=s.index('def plot_fields(case):');stop=s.index('\n\nplot_particle_snapshot',start)
 s=s[:start]+r'''def plot_fields(case):
    """Native unsmoothed means; preserve extrema and identify the biased teaching estimator."""
    cfg = case["config"]
    x=(np.arange(cfg.nx)+.5)/cfg.nx; y=(np.arange(cfg.ny)+.5)/cfg.ny
    rho=case["count"]/case["count"].mean()
    fields=[case["u"]/cfg.Uwall, case["v"]/cfg.Uwall, case["T"]/cfg.T_wall,
            rho, rho*case["T"]/cfg.T_wall, case["count"]]
    labels=[r"$u/U_w$",r"$v/U_w$",r"$T_N/T_w$"+"\nHistorical estimator",
            r"$n/\bar n$",r"$p/(\bar n k_BT_w)$ proxy", "Mean particles per cell"]
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout="constrained")
    for ax,field,label in zip(axes.flat,fields,labels):
        lo,hi=float(field.min()),float(field.max())
        if lo==hi: hi=lo+1e-12
        im=ax.contourf(x,y,field,levels=np.linspace(lo,hi,25),cmap="turbo")
        fig.colorbar(im,ax=ax,shrink=.8,format="%.3g")
        ax.set(xlabel="x/L",ylabel="y/L",title=label,aspect="equal",xlim=(0,1),ylim=(0,1))
    fig.suptitle(f"Mini DSMC teaching model: Kn={cfg.Kn}, Uwall={cfg.Uwall}, grid {cfg.nx} x {cfg.ny}")
    plt.show()
''' + s[stop:]
 s=s.replace('plot_fields(quick)\nplot_fields(better)','plot_fields(better)\n# Quick fields remain in memory; compare their raw profiles below, not noisy contour topology.')
 c['source']=s
 # Replace the single near-centerline panel by true centerline interpolation and pressure proxy.
 c=find(nb,'def resample_field');s=text(c);start=s.index('# Compare vertical centerline profiles.')
 s=s[:start]+r'''# Average the two straddling cell centers at x/L=0.5; no wall extrapolation.
fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout="constrained")
for case,style in ((quick,"o-"),(better,"s--")):
    cfg=case["config"];x=(np.arange(cfg.nx)+.5)/cfg.nx;y=(np.arange(cfg.ny)+.5)/cfg.ny
    rho=case["count"]/case["count"].mean()
    quantities=[case["u"]/cfg.Uwall,case["T"]/cfg.T_wall,rho*case["T"]/cfg.T_wall]
    for ax,field in zip(axes,quantities):
        profile=np.array([np.interp(.5,x,row) for row in field])
        ax.plot(profile,y,style,ms=3,label=cfg.label)
for ax,label in zip(axes,[r"$u/U_w$",r"$T_N/T_w$",r"$p/(\bar n k_BT_w)$ proxy"]):
    ax.set(xlabel=label,ylabel="y/L",title="x/L = 0.5; native y samples")
    ax.grid(alpha=.2);ax.legend(loc="upper center",bbox_to_anchor=(.5,-.2))
plt.show()
print("These are unmatched-window realizations; their difference is not a confidence interval.")
''';c['source']=s
 before(nb,'## Final checklist',md(r'''### Controlled engineering experiments for the required CPU model

**A. Time resolution:** keep the quick grid, particles, seed and physical start/end sampling times fixed. Halve `dt`; set `steps_new=2*steps`, `sample_start_new=2*sample_start+1` and `sample_stride_new=2*sample_stride`. The extra one is required because step indices are zero-based and sampling occurs after the move: the recorded time is `(step+1)*dt`. Check $\Delta t/\tau$, the first/last sample times, signed centerline changes and pair conservation. Equal seed does not make the two trajectories identical because event sequences differ. Report cost with physical duration, not just steps.

**B. Particle sampling:** at fixed grid and time settings, compare `ppc=8` and `ppc=16` with at least three declared seeds each. Plot every realization's centerline and the mean; distinguish run-to-run spread from error against a reference. The mean of three curves is not a grid-independent solution.

**C. Estimator reasoning:** explain the downward finite-population tendency of `T_N` using Lab 1. Do not apply a blanket $N/(N-1)$ multiplier to an already averaged variable-population field. Identify empty/single-particle cells, correlation, and nonstationarity before proposing pooled sampling. The pressure shown here is a cellwise moment proxy, not the independently sampled wall pressure of the retained production comparison.

For each experiment, submit the complete configuration table, matched-window table, native profiles, cost, and one limitation. Increasing the averaging duration while retaining startup is not evidence of steady flow.'''))
 external_legends(nb);save(p,nb)

if __name__=='__main__':
 week2();week3_lab1();week3_lab2()
 print('Updated Week2 and both Week3 notebooks; production kernels/data unchanged.')