"""Idempotent educational revision; run after the legacy notebook builders."""
from pathlib import Path
import ast
import nbformat as nbf
from classroom_cells import md, code, cavity, unet, inverse

ROOT=Path(__file__).resolve().parents[1]
# Legacy builders can invoke this script after generating their base notebooks.
# It must remain idempotent so executed outputs survive subsequent builds.

def revise(path, transform):
    nb=nbf.read(path,4)
    if nb.metadata.get('classroom_revision') == 1:
        return
    transform(nb)
    nb.metadata['classroom_revision']=1
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[]; cell.execution_count=None
    nbf.write(nb,path)

def w72(nb):
    s=nb.cells[4].source
    s=s.replace('.12*k','4*np.pi*k/105').replace('range(100)','range(145)').replace('[:60]','[:105]').replace('[60:','[105:').replace('< .15','< .02')
    nb.cells[4].source=s
    nb.cells[3].source+='\n\nThe control uses exactly two full cycles (105 training frames); a partial-cycle mean biases a homogeneous DMD fit. The tight 2% gate tests the intended linear system, not that bias.'
    nb.cells[5].source+='\n\nInflation 1000 lies at the upper edge of the searched grid. It is the best **tested** value, not evidence of a global optimum. Extend the validation grid before making that claim. QR sensor placement follows [Manohar et al.](https://arxiv.org/abs/1701.07569).'
    nb.cells.insert(5,code('''# One explicit Kalman update, using column-state covariance conventions.
H=model0.modes[model0.sensor_indices]
x=model0.initial_mean @ model0.transition
P=model0.transition.T @ model0.initial_covariance @ model0.transition + model0.process_covariance
innovation=y0[0]-model0.mean[model0.sensor_indices]-H@x
S=H@P@H.T+model0.observation_covariance
K=np.linalg.solve(S,H@P).T
x=x+K@innovation
I_KH=np.eye(len(x))-K@H
P=I_KH@P@I_KH.T+K@model0.observation_covariance@K.T  # Joseph form
np.testing.assert_allclose(x,states0[0],atol=1e-12)
np.testing.assert_allclose(P,cov0[0],atol=1e-12)'''))

def w13(nb):
    nb.cells[0].source=nb.cells[0].source.replace('This notebook audits','First train a small CPU PINN below, then examine the longer research run. This notebook audits')
    for c in nb.cells:
        c.source=c.source.replace("q['top_corner_momentum_rms']", "np.sqrt(2)*q['top_corner_momentum_rms']")
        c.source=c.source.replace("z['top_corner_momentum_y_rms'])/np.sqrt(2)","z['top_corner_momentum_y_rms'])")
    nb.cells[3:3]=cavity()

def w11(nb):
    nb.cells[6].source+=r'''

Each output is an independent Bernoulli decision, not a softmax class.
For logits z and labels y, BCE is mean(softplus(z)-y*z), summed over the two
heads; the gradient is sigmoid(z)-y. Thus shock and vortex labels may overlap.
The visible MLPClassifier below uses Adam and binary cross-entropy with L2
regularization. Numerical control comparisons use tolerances, not bitwise zero:
finite differences and floating-point subtraction leave roundoff residuals.
'''

def w12(nb):
    nb.cells[10].source+='\n\nThe in-condition example is a false positive: score about 1.065 exceeds 1.032. The maximum of only four validation scores is not a calibrated 95% limit. Under continuous exchangeable scores it gives expected exceedance 1/5. Do not tune the frozen threshold on test examples.'
    nb.cells[11].source+="\nprint('In-condition false positives:', sum(score(c)>threshold for c in tests), '/', len(tests))"
    # Embed the actual helper implementations as executable teaching code.
    tree=ast.parse((ROOT/'flowmllab/noise2noise.py').read_text())
    funcs=[ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('patch_features','fit_patch_mlp')]
    visible='''from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
check_fields=nn.check_fields
independent_targets=nn.independent_targets
PatchMLP=nn.PatchMLP
'''+ '\n\n'.join(funcs)
    s=nb.cells[18].source.replace('fitted_n2n = nn.fit_experiment(train_fields, validation_fields)', '''mlp=fit_patch_mlp(train_fields)
candidates={s:nn.noisy_validation_score(np.stack([gaussian_filter(o,(0,s,s),mode='reflect') for o in validation_fields]),validation_fields,mlp.scale) for s in (.5,1.,1.5,2.)}
fitted_n2n={'mlp':mlp,'gaussian_width':min(candidates,key=candidates.get),
            'gaussian_validation':candidates,'gain':nn.spectral_gain(train_fields),
            'training_mean':train_fields.mean(axis=0)}''')
    nb.cells[18].source=visible+'\n\n'+s
    nb.cells[21].source+='\n\nThese seeds observe the same underlying field: the training mean is therefore an unusually strong prior. Poor qx performance is a retained failure, not evidence of general denoising skill. New flow conditions are needed to establish transfer.'

def w101(nb):
    nb.cells[9].source+="\nprint('Returned cos(chi) range:', float(pred.min()), float(pred.max()))\nassert np.all(np.isfinite(pred))"
    nb.cells[11].source+='\n\nA maximum absolute error of 1.78 alone does NOT prove an out-of-bounds prediction: two numbers in [-1,1] can differ by 2. Inspect the prediction range and the clipping policy separately. A tanh output guarantees boundedness but cannot guarantee accurate scattering near sharp features.'

def main():
  for relative,fn in [
('week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb',w72),
('week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb',w101),
('week11/W11_Shock_Vortex_Identification.ipynb',w11),
('week11/W11_Lab2_Reconstruction_and_Identification.ipynb',lambda nb: nb.cells.__setitem__(slice(2,2),unet())),
('week12/W12_DSMC_Moment_Reconstruction.ipynb',w12),
('week13/W13_Rectangular_Cavity_PINN_Research.ipynb',w13)]:
    revise(ROOT/'notebooks'/relative,fn)

if __name__=='__main__':
    main()
    print('Educational cells updated; execute notebooks before publication.')
