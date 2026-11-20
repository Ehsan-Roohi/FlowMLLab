"""A disclosed validation-only amendment after the public baseline was viewed."""
import os, sys, json, hashlib, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import tensorflow as tf
from flowmllab.cavity_lab2 import NetworkConfig, case_split, point_samples, fit_network, predict_field, field_metrics
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
OUT = ROOT/"results/week04_lab2"
parser = argparse.ArgumentParser()
parser.add_argument("--lbfgs-steps", type=int, default=600)
args = parser.parse_args()
baseline_files = ("frozen_models.json", "frozen_weights.npz", "native_test_predictions.npz", "field_metrics.csv", "training_histories.json")
baseline_dir = OUT/"product_lifting_baseline"
baseline_dir.mkdir(exist_ok=True)
for name in baseline_files:
    dest = baseline_dir/name
    if not dest.exists():
        dest.write_bytes((OUT/name).read_bytes())
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256((OUT/name).read_bytes()).hexdigest()
data = dict(np.load(ROOT/"data/cavity_data.npz", allow_pickle=False))
fit, validation, test = case_split(data)
rmin, rmax = data["Re"][fit].min(), data["Re"][fit].max()
X, Q = point_samples(data, fit, rmin, rmax, stride=2)
Xv, Qv = point_samples(data, validation, rmin, rmax, stride=1)
config = NetworkConfig(widths=(48,48,48), adam_steps=400, lbfgs_steps=args.lbfgs_steps, boundary_aware=True,
                       boundary_lifting="corner", seed=690)
model, history, meta = fit_network(X, Q, Xv, Qv, config)
baseline = json.loads((baseline_dir/"frozen_models.json").read_text())["field_models"]["Boundary-aware DNN"]
metrics = []
for k in np.flatnonzero(validation):
    up, vp = predict_field(model, data["Re"][k], data, rmin, rmax)
    metrics.append({"Re": float(data["Re"][k]), **field_metrics(data["x"],data["y"],data["u"][k],data["v"][k],up,vp)})
receipt = {"protocol": "Validation-only comparison after public baseline test results were observed; later public-test rescoring is disclosed, not a new blind test.",
           "baseline_validation_mse": baseline["selected"]["validation"], "candidate": meta,
           "validation_metrics": metrics, "test_metrics_evaluated_in_this_script": False,
           "candidate_selected": meta["selected"]["validation"] < baseline["selected"]["validation"]}
prior_path = OUT/"corner_validation.json"
comparison_path = OUT/"corner_validation_comparison.json"
prior = json.loads(comparison_path.read_text()) if comparison_path.exists() else []
if prior_path.exists():
    old_receipt = json.loads(prior_path.read_text())
    if not any(row["candidate"]["config"]["lbfgs_steps"] == old_receipt["candidate"]["config"]["lbfgs_steps"] for row in prior):
        prior.append(old_receipt)
prior = [row for row in prior if row["candidate"]["config"]["lbfgs_steps"] != args.lbfgs_steps]
prior.append(receipt)
comparison_path.write_text(json.dumps(prior,indent=2))
prior_path.write_text(json.dumps(receipt,indent=2))
np.savez_compressed(OUT/"corner_validation_weights.npz", **{f"weight_{j}":w for j,w in enumerate(model.get_weights())})
(OUT/"corner_validation_history.json").write_text(json.dumps(history,indent=2))
print(json.dumps(receipt,indent=2))
