#!/usr/bin/env python3
"""Train, validate, and test the autonomous phase-stable cylinder decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.cylinder_phase import align_and_predict, fit_case, interpolate  # noqa: E402

DEVELOPMENT = (90, 110, 120, 140)
VALIDATION = 100
FRESH_TEST = 95
RETAINED_TEST = 105
HISTORY = 4
D, U, SNAPSHOT_STRIDE = 12.0, 0.05, 25
DT_STAR = SNAPSHOT_STRIDE * U / D


def load_case(root: Path, reynolds: int) -> dict[str, np.ndarray]:
    with np.load(root / f"cylinder_cfd_re{reynolds:03d}.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def omega(fields: np.ndarray) -> np.ndarray:
    return D * (np.gradient(fields[..., 1], axis=2) - np.gradient(fields[..., 0], axis=1))


def metrics(case: dict[str, np.ndarray], prediction: np.ndarray, predicted_st: float) -> dict:
    truth = np.stack([case[name] for name in ("u", "v", "p")], axis=-1)
    fluid = ~np.asarray(case["solid"], dtype=bool)
    exact, estimate = omega(truth[HISTORY:]), omega(prediction[HISTORY:])
    frame = np.linalg.norm((estimate - exact)[:, fluid], axis=1) / np.linalg.norm(exact[:, fluid], axis=1)
    global_error = np.linalg.norm((estimate - exact)[:, fluid]) / np.linalg.norm(exact[:, fluid])
    exact_st = float(case["strouhal"])
    return {
        "future_frames": int(len(frame)),
        "vorticity_global_relative_l2": float(global_error),
        "vorticity_max_frame_relative_l2": float(frame.max()),
        "vorticity_last_frame_relative_l2": float(frame[-1]),
        "cfd_strouhal": exact_st, "predicted_strouhal": float(predicted_st),
        "strouhal_relative_error": float(abs(predicted_st-exact_st)/exact_st),
        "passes": bool(global_error < .15 and frame.max() < .15 and abs(predicted_st-exact_st)/exact_st < .02),
    }


def predict(models, case, reynolds):
    model = interpolate(models, reynolds)
    initial = np.stack([case[name][:HISTORY] for name in ("u", "v", "p")], axis=-1)
    prediction, phase = align_and_predict(model, initial, len(case["u"]), delta_t_star=DT_STAR)
    prediction[:, np.asarray(case["solid"], bool)] = 0.0
    return model, prediction, phase


def spectra(case, prediction):
    cy, cx = 48, 108
    truth = np.asarray(case["v"][:, cy, cx], float); pred = prediction[:, cy, cx, 1]
    freq = np.fft.rfftfreq(len(truth), d=DT_STAR)
    window = np.hanning(len(truth))
    a = abs(np.fft.rfft((truth-truth.mean())*window))**2
    b = abs(np.fft.rfft((pred-pred.mean())*window))**2
    keep = (freq > .05) & (freq < .35)
    return freq[keep], a[keep]/a[keep].max(), b[keep]/b[keep].max()


def make_summary(output, products):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = [(100, "Validation"), (95, "Fresh test"), (105, "Retained test")]
    errors = [100*products[re]["metrics"]["vorticity_global_relative_l2"] for re,_ in labels]
    axes[0].bar([x[1] for x in labels], errors, color=["#4477AA", "#228833", "#AA3377"])
    axes[0].axhline(15, color="k", ls="--", lw=1); axes[0].set_ylabel("Vorticity relative L2 error (%)")
    axes[0].set_title("277-frame autonomous rollout")
    re_values=[95,100,105]
    axes[1].plot(re_values,[products[r]["metrics"]["cfd_strouhal"] for r in re_values],"o-",label="LBM")
    axes[1].plot(re_values,[products[r]["metrics"]["predicted_strouhal"] for r in re_values],"s--",label="Phase decoder")
    axes[1].set(xlabel="Reynolds number",ylabel="Strouhal number",title="Shedding frequency"); axes[1].legend()
    f,a,b=spectra(products[95]["case"],products[95]["prediction"])
    axes[2].plot(f,a,label="LBM"); axes[2].plot(f,b,"--",label="Phase decoder")
    axes[2].set(xlabel="Strouhal number",ylabel="Normalized PSD",title="Fresh Re=95 wake spectrum"); axes[2].legend()
    fig.savefig(output/"phase_stable_validation.png",dpi=180); plt.close(fig)


def make_video(output, case, prediction):
    truth=np.stack([case[name] for name in ("u","v","p")],-1); a=omega(truth); b=omega(prediction); e=b-a
    extent=[0, truth.shape[2]/D, 0, truth.shape[1]/D]; frames=range(0,len(a),2)
    fig,axes=plt.subplots(1,3,figsize=(16,5.3),constrained_layout=True)
    lim=np.percentile(abs(a[:,~np.asarray(case['solid'],bool)]),99.5); elim=np.percentile(abs(e[:,~np.asarray(case['solid'],bool)]),99)
    images=[]
    for ax,z,title,v,cmap in zip(axes,(a[0],b[0],e[0]),("LBM CFD","Phase-stable learned decoder","Error"),(lim,lim,elim),("RdBu_r","RdBu_r","magma")):
        im=ax.imshow(z,origin="lower",extent=extent,aspect="auto",cmap=cmap,vmin=-v if cmap=="RdBu_r" else 0,vmax=v)
        ax.add_patch(Circle((5,47.5/D),.5,color="white",ec="black")); ax.set(title=title,xlabel="x/D"); images.append(im)
    axes[0].set_ylabel("y/D"); title=fig.suptitle("")
    def update(i):
        for im,z in zip(images,(a[i],b[i],abs(e[i]))): im.set_data(z)
        title.set_text(f"Unseen Re=95 | autonomous frame {max(0,i-HISTORY+1)}/277")
        return [*images,title]
    animation=FuncAnimation(fig,update,frames=frames,blit=False)
    animation.save(output/"re095_phase_stable_lbm_vs_decoder.mp4",writer=FFMpegWriter(fps=20,bitrate=5000))
    animation.save(output/"re095_phase_stable_lbm_vs_decoder.webp",writer=PillowWriter(fps=16))
    fig.savefig(output/"re095_phase_stable_poster.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data",type=Path,default=ROOT/"data/cylinder_cfd"); parser.add_argument("--output",type=Path,default=ROOT/"results/cylinder_phase"); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    cases={re:load_case(args.data,re) for re in (*DEVELOPMENT,VALIDATION,FRESH_TEST,RETAINED_TEST)}
    validation_scores={}
    for harmonics in (2,3,4,5,6):
        candidate=[fit_case(cases[re],reynolds=re,harmonics=harmonics) for re in DEVELOPMENT]
        model,pred,_=predict(candidate,cases[VALIDATION],VALIDATION)
        validation_scores[harmonics]=metrics(cases[VALIDATION],pred,model.strouhal)["vorticity_global_relative_l2"]
    selected=min(validation_scores,key=validation_scores.get)
    models=[fit_case(cases[re],reynolds=re,harmonics=selected) for re in DEVELOPMENT]
    products={}
    for re in (VALIDATION,FRESH_TEST,RETAINED_TEST):
        model,pred,phase=predict(models,cases[re],re); products[re]={"case":cases[re],"prediction":pred,"metrics":metrics(cases[re],pred,model.strouhal),"phase":phase}
    report={"protocol":{"development_reynolds":list(DEVELOPMENT),"validation_reynolds":VALIDATION,"fresh_test_reynolds":FRESH_TEST,"retained_test_reynolds":RETAINED_TEST,"initial_true_frames":HISTORY,"future_cfd_inputs":0,"selection":"harmonic order minimizes validation global vorticity error"},"validation_scores":validation_scores,"selected_harmonics":selected,"validation":products[100]["metrics"],"fresh_test":products[95]["metrics"],"retained_test":products[105]["metrics"],"all_gates_pass":all(products[r]["metrics"]["passes"] for r in products)}
    (args.output/"phase_stable_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    make_summary(args.output,products); make_video(args.output,products[95]["case"],products[95]["prediction"])
    print(json.dumps(report,indent=2)); return 0 if report["all_gates_pass"] else 1

if __name__=="__main__": raise SystemExit(main())
