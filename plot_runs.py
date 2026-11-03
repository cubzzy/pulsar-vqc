import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import re

# two local circular and full and the same, so we excluded full. 
RUNS_DIR = Path("[ADD YOUR PATH HERE]")

# don't change these
PLOT_DATA_DIR = RUNS_DIR / "plot_data"
OUT_DIR = RUNS_DIR / "run_plots"

def parse_run(name):
    """'ZZFeatureMapTwoLocalcircular' -> ('TwoLocal', 'circular', '')"""
    m = re.fullmatch(r"ZZFeatureMap(EfficientSU2|RealAmplitudes|TwoLocal)(linear|circular|full)\s*(\d*)", name)
    return m.group(1), m.group(2), m.group(3)

def load_column(path):
    return pd.read_csv(path).iloc[:, 0].to_numpy()

def load_runs(runs_dir):
    runs = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "loss.csv").exists()):
        ansatz, ent, rep = parse_run(d.name)
        fpr = load_column(d / "roc_fpr_df.csv")
        tpr = load_column(d / "roc_tpr_df.csv")
        runs.append({
            "label": f"{ansatz} {ent}" + (f" (run {rep})" if rep else ""),
            "loss": load_column(d / "loss.csv"),
            "fpr": fpr,
            "tpr": tpr,
            "auc": np.trapezoid(tpr, fpr),
        })
    # one distinct color per run (solid lines only); tab20 if tab10 would run out
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(runs):
        r["color"] = cmap(i)
    return runs

def line_style(r, best):
    """Best run: thick and opaque. Everything else: thin and faded."""
    if r is best:
        return {"color": r["color"], "linewidth": 1.6, "alpha": 1.0, "zorder": 3}
    return {"color": r["color"], "linewidth": 1.3, "alpha": 0.3, "zorder": 2}

def legend_best_first(ax, **kwargs):
    """Runs are plotted best-first, so the best is the first entry: bold, with fully opaque swatches."""
    leg = ax.legend(**kwargs)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    leg.get_texts()[0].set_fontweight("bold")

def plot_loss(runs, path):
    best = max(runs, key=lambda r: r["auc"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        ax.plot(r["loss"], label=r["label"], **line_style(r, best))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    legend_best_first(ax, fontsize=8)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

def plot_roc(runs, path):
    best = max(runs, key=lambda r: r["auc"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in runs:
        ax.plot(r["fpr"], r["tpr"], label=f"{r['label']} (AUC {r['auc']:.3f})", **line_style(r, best))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, label="Chance", zorder=1)  # last, so it is last in the legend
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    legend_best_first(ax, fontsize=7, loc="lower right")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    runs = sorted(load_runs(PLOT_DATA_DIR), key=lambda r: -r["auc"])  # best AUC first
    plot_loss(runs, OUT_DIR / "loss_all_runs.png")
    plot_roc(runs, OUT_DIR / "roc_all_runs.png")
    for r in runs:
        print(f"{r['label']:35s} AUC {r['auc']:.3f}  final loss {r['loss'][-1]:.3f}  iters {len(r['loss'])}")