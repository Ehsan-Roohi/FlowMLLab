"""Common helpers for MIE 690A Week 5-6 guided projects.

The project notebooks deliberately keep the Week-4 solver fixed. Students
modify the experiment design, loss, model-reduction choice, or uncertainty
analysis rather than rewriting the CFD solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd

import w4utils
from w4utils import load_dataset, field_validation_report

W5_COMMON_VERSION = "1.2"


def set_global_seed(seed: int = 690) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def require_week4_files(dataset: str = "cavity_data.npz"):
    """Load and audit the fixed Week-4 cavity dataset."""
    data = load_dataset(dataset)
    required = {"x", "y", "Re", "u", "v", "p", "psi", "split"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"Dataset is missing arrays: {sorted(missing)}")
    if not (data["u"].shape == data["v"].shape == data["p"].shape == data["psi"].shape):
        raise ValueError("u, v, p, and psi must have identical case/grid shapes")
    return data


def re_mask(data, values):
    values = np.asarray(values, dtype=float)
    return np.isin(data["Re"].astype(float), values)


def point_samples(data, case_mask, stride=2, include_pressure=True):
    """Flatten selected fields into pointwise rows [Re,x,y] -> [u,v,p]."""
    x = data["x"][::stride]
    y = data["y"][::stride]
    Xg, Yg = np.meshgrid(x, y)
    rows, targets, case_ids = [], [], []
    selected = np.where(case_mask)[0]
    for idx in selected:
        n = Xg.size
        rows.append(np.column_stack([
            np.full(n, float(data["Re"][idx])), Xg.ravel(), Yg.ravel()
        ]))
        blocks = [data["u"][idx, ::stride, ::stride].ravel(),
                  data["v"][idx, ::stride, ::stride].ravel()]
        if include_pressure:
            blocks.append(data["p"][idx, ::stride, ::stride].ravel())
        targets.append(np.column_stack(blocks))
        case_ids.append(np.full(n, idx, dtype=int))
    return np.vstack(rows), np.vstack(targets), np.concatenate(case_ids)


@dataclass
class Standardizers:
    xmean: np.ndarray
    xstd: np.ndarray
    ymean: np.ndarray
    ystd: np.ndarray

    def transform_x(self, X):
        return (np.asarray(X) - self.xmean) / self.xstd

    def transform_y(self, Y):
        return (np.asarray(Y) - self.ymean) / self.ystd

    def inverse_y(self, Y):
        return np.asarray(Y) * self.ystd + self.ymean


def fit_standardizers(X, Y):
    xmean = np.mean(X, axis=0)
    xstd = np.std(X, axis=0) + 1.0e-12
    ymean = np.mean(Y, axis=0)
    ystd = np.std(Y, axis=0) + 1.0e-12
    return Standardizers(xmean, xstd, ymean, ystd)


def make_dense_model(input_dim=3, output_dim=3, hidden=(64,64,64),
                     activation="tanh", seed=690):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input((input_dim,))]
    layers += [tf.keras.layers.Dense(int(n), activation=activation) for n in hidden]
    layers += [tf.keras.layers.Dense(output_dim)]
    return tf.keras.Sequential(layers)


def train_pointwise_model(data, train_re, val_re, hidden=(64,64,64),
                          activation="tanh", stride=2, seed=690,
                          epochs=900, patience=70, learning_rate=1e-3,
                          sample_weight=None, verbose=0):
    """Train a coordinate DNN using a case-wise train/validation split."""
    import tensorflow as tf
    train_mask = re_mask(data, train_re)
    val_mask = re_mask(data, [val_re])
    Xtr, Ytr, _ = point_samples(data, train_mask, stride=stride)
    Xva, Yva, _ = point_samples(data, val_mask, stride=stride)
    scalers = fit_standardizers(Xtr, Ytr)
    model = make_dense_model(3, Ytr.shape[1], hidden, activation, seed)
    model.compile(tf.keras.optimizers.Adam(learning_rate), loss="mse")
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )]
    history = model.fit(
        scalers.transform_x(Xtr), scalers.transform_y(Ytr),
        validation_data=(scalers.transform_x(Xva), scalers.transform_y(Yva)),
        sample_weight=sample_weight,
        epochs=epochs, batch_size=512, verbose=verbose, callbacks=callbacks
    )
    val_curve=np.asarray(history.history.get("val_loss",[]),dtype=float)
    best_epoch=int(np.argmin(val_curve)+1) if val_curve.size else int(len(history.history.get("loss",[])))
    return {"model": model, "scalers": scalers, "history": history.history,
            "best_epoch": best_epoch,
            "train_re": list(map(float, train_re)), "val_re": float(val_re),
            "hidden": tuple(hidden), "activation": activation, "seed": int(seed)}


def train_pointwise_fixed_epochs(data, train_re, hidden=(64,64,64),
                                 activation="tanh", stride=2, seed=690,
                                 epochs=300, learning_rate=1e-3,
                                 sample_weight=None, verbose=0):
    """Retrain a frozen architecture on every permitted development case.

    This helper intentionally uses no validation case and no early stopping.
    The epoch count must be frozen beforehand from a separate validation-only
    experiment.  It is used in Track 1 to give the neural model and the field-
    interpolation baseline exactly the same Reynolds-number development set.
    """
    import tensorflow as tf
    mask=re_mask(data,train_re)
    Xtr,Ytr,_=point_samples(data,mask,stride=stride)
    scalers=fit_standardizers(Xtr,Ytr)
    model=make_dense_model(3,Ytr.shape[1],hidden,activation,seed)
    model.compile(tf.keras.optimizers.Adam(learning_rate),loss="mse")
    history=model.fit(scalers.transform_x(Xtr),scalers.transform_y(Ytr),
                      sample_weight=sample_weight,epochs=int(epochs),
                      batch_size=512,verbose=verbose)
    return {"model":model,"scalers":scalers,"history":history.history,
            "best_epoch":int(epochs),"train_re":list(map(float,train_re)),
            "val_re":None,"hidden":tuple(hidden),"activation":activation,
            "seed":int(seed)}


def predict_case(bundle, re_value, x, y):
    Xg, Yg = np.meshgrid(x, y)
    X = np.column_stack([
        np.full(Xg.size, float(re_value)), Xg.ravel(), Yg.ravel()
    ])
    pred_s = bundle["model"].predict(bundle["scalers"].transform_x(X), verbose=0)
    pred = bundle["scalers"].inverse_y(pred_s)
    shape = Xg.shape
    outputs = [pred[:,i].reshape(shape) for i in range(pred.shape[1])]
    if len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs[0], outputs[1], outputs[2] - np.mean(outputs[2])


def interpolate_case(data, re_value, train_re):
    train_re = np.asarray(sorted(map(float, train_re)))
    r = float(re_value)
    if r < train_re.min() or r > train_re.max():
        # Linear extrapolation using the nearest pair. This is intentionally
        # allowed for a failure test but must be identified as extrapolation.
        if r < train_re.min(): pair = train_re[:2]
        else: pair = train_re[-2:]
    else:
        hi = int(np.searchsorted(train_re, r))
        if hi == 0: pair = train_re[:2]
        elif hi == len(train_re): pair = train_re[-2:]
        else: pair = train_re[hi-1:hi+1]
    r0, r1 = pair
    w = (r-r0)/(r1-r0)
    idx0 = int(np.where(data["Re"].astype(float)==r0)[0][0])
    idx1 = int(np.where(data["Re"].astype(float)==r1)[0][0])
    out = []
    for name in ("u","v","p"):
        field = (1-w)*data[name][idx0] + w*data[name][idx1]
        if name == "p": field = field - field.mean()
        out.append(field)
    return tuple(out)


def evaluate_prediction(data, re_value, pred):
    idx = int(np.where(data["Re"].astype(float)==float(re_value))[0][0])
    u,v,p = pred
    report = field_validation_report(
        data["x"], data["y"], data["u"][idx], data["v"][idx],
        u, v, data["p"][idx], p
    )
    return report


def results_frame(rows):
    df = pd.DataFrame(rows)
    preferred = [c for c in ["variant","method","Re","relative_L2_uv",
        "centerline_rel_L2_u","centerline_rel_L2_v","relative_L2_p",
        "relative_L2_p_interior","wall_rms_error","div_l2_pred",
        "training_seconds","seed"] if c in df.columns]
    return df[preferred + [c for c in df.columns if c not in preferred]]


def plot_case_evidence(data, re_value, predictions, title=None):
    """Plot truth, predictions, velocity error, and centerlines."""
    import matplotlib.pyplot as plt
    idx = int(np.where(data["Re"].astype(float)==float(re_value))[0][0])
    x,y = data["x"],data["y"]
    Xg,Yg = np.meshgrid(x,y)
    tu,tv,tp = data["u"][idx],data["v"][idx],data["p"][idx]
    n = len(predictions)
    fig, axes = plt.subplots(n+1, 4, figsize=(15, 3.6*(n+1)))
    axes = np.atleast_2d(axes)
    axes[0,0].streamplot(Xg,Yg,tu,tv,density=1.0)
    axes[0,0].set_title("CFD streamlines")
    im=axes[0,1].contourf(Xg,Yg,tp,28); fig.colorbar(im,ax=axes[0,1]); axes[0,1].set_title("CFD pressure")
    mid=len(x)//2
    axes[0,2].plot(tu[:,mid],y,label="u"); axes[0,2].plot(tv[mid,:],x,label="v"); axes[0,2].legend(); axes[0,2].set_title("CFD centerlines")
    axes[0,3].axis("off")
    for row,(name,(u,v,p)) in enumerate(predictions.items(),start=1):
        axes[row,0].streamplot(Xg,Yg,u,v,density=1.0); axes[row,0].set_title(f"{name}: streamlines")
        err=np.hypot(u-tu,v-tv); im=axes[row,1].contourf(Xg,Yg,err,28); fig.colorbar(im,ax=axes[row,1]); axes[row,1].set_title(f"{name}: vector error")
        axes[row,2].plot(tu[:,mid],y,"k",label="CFD u"); axes[row,2].plot(u[:,mid],y,"r--",label="model u"); axes[row,2].legend(fontsize=8)
        axes[row,3].plot(x,tv[mid,:],"k",label="CFD v"); axes[row,3].plot(x,v[mid,:],"r--",label="model v"); axes[row,3].legend(fontsize=8)
    for ax in axes.ravel():
        if ax.has_data(): ax.grid(alpha=.2)
    if title: fig.suptitle(title)
    fig.tight_layout(); return fig


# ------------------------------- POD helpers -------------------------------
def make_separate_pod(data, mask):
    pod={}
    for name in ("u","v","p"):
        F=data[name][mask].reshape(mask.sum(),-1)
        mean=F.mean(axis=0); centered=F-mean
        _,s,modes=np.linalg.svd(centered,full_matrices=False)
        energy=np.cumsum(s*s)/(np.sum(s*s)+1e-30)
        pod[name]={"mean":mean,"modes":modes,"singular_values":s,"energy":energy}
    return pod


def separate_coefficients(data,pod,mask,rank):
    blocks=[]
    for name in ("u","v","p"):
        F=data[name][mask].reshape(mask.sum(),-1)
        blocks.append((F-pod[name]["mean"])@pod[name]["modes"][:rank].T)
    return np.concatenate(blocks,axis=1)


def reconstruct_separate(data,pod,coeff,rank):
    ny,nx=data["u"].shape[1:]
    out={}; off=0
    for name in ("u","v","p"):
        c=coeff[off:off+rank]; off+=rank
        out[name]=(pod[name]["mean"]+c@pod[name]["modes"][:rank]).reshape(ny,nx)
    out["p"]-=out["p"].mean()
    return out["u"],out["v"],out["p"]


def make_shared_pod(data,mask):
    blocks=[]; scales={}
    for name in ("u","v","p"):
        F=data[name][mask].reshape(mask.sum(),-1)
        scale=float(np.std(F))+1e-12; scales[name]=scale
        blocks.append(F/scale)
    Fcat=np.concatenate(blocks,axis=1)
    mean=Fcat.mean(axis=0); centered=Fcat-mean
    _,s,modes=np.linalg.svd(centered,full_matrices=False)
    energy=np.cumsum(s*s)/(np.sum(s*s)+1e-30)
    return {"mean":mean,"modes":modes,"s":s,"energy":energy,"scales":scales,
            "nfield":data["u"].shape[1]*data["u"].shape[2]}


def shared_coefficients(data,pod,mask,rank):
    blocks=[]
    for name in ("u","v","p"):
        F=data[name][mask].reshape(mask.sum(),-1)/pod["scales"][name]
        blocks.append(F)
    Fcat=np.concatenate(blocks,axis=1)
    return (Fcat-pod["mean"])@pod["modes"][:rank].T


def reconstruct_shared(data,pod,coeff,rank):
    ny,nx=data["u"].shape[1:]; n=pod["nfield"]
    vec=pod["mean"]+coeff@pod["modes"][:rank]
    u=(vec[:n]*pod["scales"]["u"]).reshape(ny,nx)
    v=(vec[n:2*n]*pod["scales"]["v"]).reshape(ny,nx)
    p=(vec[2*n:3*n]*pod["scales"]["p"]).reshape(ny,nx); p-=p.mean()
    return u,v,p


def train_branch(re_values,coefficients,hidden=(32,32),alpha=1e-5,seed=690):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    xs=StandardScaler().fit(np.asarray(re_values).reshape(-1,1))
    ys=StandardScaler().fit(coefficients)
    model=MLPRegressor(hidden_layer_sizes=hidden,activation="tanh",solver="lbfgs",
        alpha=alpha,max_iter=5000,tol=1e-10,random_state=seed)
    Xs=xs.transform(np.asarray(re_values).reshape(-1,1))
    Ys=ys.transform(coefficients)
    # MLPRegressor expects a one-dimensional target for one-output regression
    # and a two-dimensional target for multi-output regression.
    fit_target=Ys.ravel() if Ys.shape[1] == 1 else Ys
    model.fit(Xs,fit_target)
    return {"model":model,"xs":xs,"ys":ys,"hidden":hidden,"alpha":alpha,"seed":seed}


def branch_predict(bundle,re_value):
    cs=np.asarray(bundle["model"].predict(
        bundle["xs"].transform([[float(re_value)]])
    ))
    # scikit-learn returns shape (1,) for a one-output regressor and
    # shape (1,n_outputs) for multi-output regression. StandardScaler expects
    # a 2-D array in both cases.
    if cs.ndim == 1:
        cs = cs.reshape(1,-1)
    return bundle["ys"].inverse_transform(cs)[0]


def write_project_card(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
