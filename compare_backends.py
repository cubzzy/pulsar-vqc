"""
compare_backends.py

Compares per-circuit class-1 probabilities (p1) from two runs of ionq_inference.py on the SAME
test rows: the ideal Aer run vs the IonQ noisy simulator. Reads the probs_*.csv files
(columns: test_row, true, pred, p1) and prints the numbers that go on the slides.
"""

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==================================================================
# you can change these variables
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_outputs")
N_TEST = 300
# ==================================================================

AER = f"{REPO}/probs_aer_n{N_TEST}.csv"
IONQ = f"{REPO}/probs_ionq_sim_n{N_TEST}.csv"


def load_pair():
    """Return one DataFrame with one row per circuit: true, pred_aer, p1_aer, pred_ionq, p1_ionq."""
    _aer = pd.read_csv(AER)
    _ionq = pd.read_csv(IONQ)
    df = _aer.merge(_ionq, on = "test_row", suffixes = ("_aer", "_ionq"))
    assert len(df) == N_TEST
    assert (df["true_aer"] == df["true_ionq"]).all()
    df["true"] = df["true_aer"]
    return df


def label_flips(df):
    """Number of circuits where the two backends predict different classes."""
    return int((df["pred_aer"] != df["pred_ionq"]).sum())


def confidence_slope(df):
    """Slope of (ionq - .5) vs (aer - .5) through the origin; < 1 means IonQ is pulled toward .5."""
    x = df["p1_aer"] - 0.5
    y = df["p1_ionq"] - 0.5
    return float((x * y).sum() / (x * x).sum())


def plot_scatter(df, path):
    """IonQ p1 vs Aer p1, one dot per circuit. Points on the dashed y = x line agree exactly."""
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    # (true class, label, categorical color, marker): shape repeats the color so it isn't color-alone
    classes = [(0, "non-pulsar", "#2a78d6", "o"), (1, "pulsar", "#eb6834", "^")]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 1], [0, 1], "--", color=muted, linewidth=1, label="y = x (identical)")
    ax.axhline(0.5, color=muted, linewidth=0.5, alpha=0.5)
    ax.axvline(0.5, color=muted, linewidth=0.5, alpha=0.5)
    for true, label, color, marker in classes:
        sub = df[df["true"] == true]
        ax.scatter(sub["p1_aer"], sub["p1_ionq"], s=36, color=color, marker=marker,
                   edgecolor=surface, linewidth=1, label=f"{label} (n={len(sub)})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("p1, Aer (ideal)", color=ink)
    ax.set_ylabel("p1, IonQ aria-1 simulator (noisy)", color=ink)
    ax.set_title(f"Per-circuit class-1 probability, {len(df)} circuits, "
                 f"{label_flips(df)} label flips", color=ink, fontsize=11)
    ax.tick_params(colors=muted)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
    ax.legend(frameon=False, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)


if __name__ == "__main__":
    df = load_pair()
    plot_scatter(df, f"{REPO}/aer_vs_ionq_p1_n{N_TEST}.png")
    print(f"rows: {len(df)}")
    print(f"label flips: {label_flips(df)}")
    print(f"p1 correlation: {df['p1_aer'].corr(df['p1_ionq']):.3f}")
    print(f"confidence slope: {confidence_slope(df):.3f}")
    for name in ("aer", "ionq"):
        by_class = df.groupby("true")[f"p1_{name}"].mean()
        print(f"{name}: mean p1 non-pulsar {by_class[0]:.3f}, pulsar {by_class[1]:.3f}, "
              f"AUC {roc_auc_score(df['true'], df[f'p1_{name}']):.3f}")
