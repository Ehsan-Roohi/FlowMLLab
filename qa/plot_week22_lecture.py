"""Readable lecture layout of the unchanged retained transfer protocol."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build(root: Path) -> Path:
    """Replot exact seed means/ranges; do not run a fit or change evidence."""
    source = root / "results/transformer_course_v3/transfer.json"
    original = source.read_bytes()
    rows = json.loads(original)["rows"]
    out = root / "results/late_course_teaching/week22_transfer_lecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = ("scratch", "pretrained", "matched-steps", "target-POD-MLP", "DMD")
    markers = ("o", "s", "^", "D", "v")
    styles = ("-", "--", "-.", ":", (0, (4, 2)))
    counts = (44, 84, 154)
    ledger = []
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.2), sharex=True)
        for row_index, reynolds in enumerate(sorted({r["target_re"] for r in rows})):
            for arm_index, arm in enumerate(arms):
                subset = [r for r in rows if r["arm"] == arm and r["target_re"] == reynolds]
                for column, key in enumerate(("field_relative_l2", "in_subspace_relative_l2")):
                    groups = [[r["metrics"][key] for r in subset
                               if r["labels"]["total_target_labels"] == n] for n in counts]
                    assert all(groups), (reynolds, arm, key)
                    mean = np.array([np.mean(v) for v in groups])
                    low = np.array([min(v) for v in groups])
                    high = np.array([max(v) for v in groups])
                    axes[row_index, column].errorbar(
                        counts, mean, yerr=np.maximum(0, [mean - low, high - mean]),
                        marker=markers[arm_index], linestyle=styles[arm_index],
                        color=f"C{arm_index}", capsize=3, linewidth=1.4, label=arm)
                    ledger.append({"Re": reynolds, "arm": arm, "metric": key,
                                   "labels": list(counts), "mean": mean.tolist(),
                                   "min": low.tolist(), "max": high.tolist()})
            source_floor = next(r["metrics"]["representation_floor"] for r in rows
                                if r["target_re"] == reynolds and r["arm"] == "pretrained")
            floors = [np.mean([r["metrics"]["representation_floor"] for r in rows
                               if r["target_re"] == reynolds and r["arm"] == "target-POD-MLP"
                               and r["labels"]["total_target_labels"] == n]) for n in counts]
            axes[row_index, 0].axhline(source_floor, color="black", linestyle="--",
                                      label="source-POD floor (source-basis arms)")
            axes[row_index, 0].plot(counts, floors, ":", color="black",
                                   label="target-POD floor (target-basis arm)")
            ledger.append({"Re": reynolds, "source_POD_floor": source_floor,
                           "target_POD_floor_mean": floors})
            for column, title in enumerate(("field error", "in-subspace error")):
                axes[row_index, column].set(title=f"Re = {reynolds}: {title}",
                                            ylabel="relative L2 error", xticks=counts)
                axes[row_index, column].grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("all target labels")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
                   ncol=2, frameon=False)
        fig.subplots_adjust(left=.10, right=.98, bottom=.07, top=.84, hspace=.31, wspace=.28)
        fig.savefig(out, dpi=180, facecolor="white")
        plt.close(fig)
    assert source.read_bytes() == original, "Retained transfer evidence changed"
    proof = {"source": source.relative_to(root).as_posix(),
             "source_sha256": hashlib.sha256(original).hexdigest(),
             "figure_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
             "meaning": "Arithmetic seed means and full min-max seed ranges from unchanged retained rows; no new training.",
             "points": ledger}
    out.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1]))
