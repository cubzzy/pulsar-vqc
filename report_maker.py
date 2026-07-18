"""
report_maker.py

Builds the full output for a single VQC run: the confusion matrix, loss
curve, AUC curve, and circuit diagram plots, plus a markdown report tying
them together with the run's configuration and metrics. Everything for one
run+config combination lands in its own folder under reports_root.

config keys: feature_map, ansatz, entanglement, loss, n_samples, n_qubits,
             train_size, test_size, train_nonpulsars, train_pulsars,
             test_nonpulsars, test_pulsars
metrics keys: accuracy, precision, recall, f1, fpr, mcc, TP, FP, TN, FN
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def _config_caption(config):
    return (
        f"{config['feature_map']} | {config['ansatz']} ({config['entanglement']}) | "
        f"loss={config['loss']} | qubits={config['n_qubits']} | samples={config['n_samples']}"
    )


def plot_confusion_matrix(report_dir, cf_matrix_norm, train_size, test_size, config):
    sns.heatmap(cf_matrix_norm, cmap="Purples", annot=True, linewidth=1, fmt=".1%")
    plt.title(f"Confusion Matrix, Training Size = {train_size}")
    plt.xlabel(f"Model prediction, Training Size = {train_size}, Testing Size = {test_size}")
    plt.ylabel("True label")
    plt.gcf().text(0.5, 0, _config_caption(config), ha='center', va='top', fontsize=8, color='gray')
    path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to: {path}")
    return path


def plot_loss_curve(report_dir, loss_values, config):
    plt.plot(range(len(loss_values)), loss_values)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.gcf().text(0.5, 0, _config_caption(config), ha='center', va='top', fontsize=8, color='gray')
    path = os.path.join(report_dir, "loss_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to: {path}")
    return path


def plot_auc_curve(report_dir, y_test, pulsar_proba, config):
    roc_fpr, roc_tpr, _ = roc_curve(y_test, pulsar_proba)
    auc_score = auc(roc_fpr, roc_tpr)
    fig, ax = plt.subplots()
    ax.set_title("AUC curve")
    ax.plot(roc_fpr, roc_tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot(roc_fpr, roc_fpr, label="Random Guessing")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.text(0.5, 0, _config_caption(config), ha='center', va='top', fontsize=8, color='gray')
    path = os.path.join(report_dir, "auc_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"AUC curve saved to: {path}")
    return path


def plot_circuit(report_dir, model, config):
    fig = model.circuit.draw(output="mpl")
    fig.text(0.5, .1, _config_caption(config), ha='center', va='top', fontsize=8, color='gray')
    path = os.path.join(report_dir, "circuit.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Circuit diagram saved to: {path}")
    return path


def write_report(report_dir, config, metrics):
    """Write report.md into report_dir. Returns the path written to."""
    report_path = os.path.join(report_dir, "report.md")

    with open(report_path, "w") as f:
        f.write("# VQC Run Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| Feature map | {config['feature_map']} |\n")
        f.write(f"| Ansatz | {config['ansatz']} |\n")
        f.write(f"| Entanglement | {config['entanglement']} |\n")
        f.write(f"| Loss function | {config['loss']} |\n")
        f.write(f"| Training samples | {config['n_samples']} |\n")
        f.write(f"| Features/qubits | {config['n_qubits']} |\n")
        f.write(f"| Training set (non-pulsar / pulsar) | {config['train_nonpulsars']} / {config['train_pulsars']} |\n")
        f.write(f"| Testing set (non-pulsar / pulsar) | {config['test_nonpulsars']} / {config['test_pulsars']} |\n\n")

        f.write("## Metrics\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {metrics['accuracy']} |\n")
        f.write(f"| Precision | {metrics['precision']} |\n")
        f.write(f"| Recall | {metrics['recall']} |\n")
        f.write(f"| F1-score | {metrics['f1']} |\n")
        f.write(f"| FPR | {metrics['fpr']} |\n")
        f.write(f"| MCC | {metrics['mcc']} |\n")
        f.write(f"| TP / FP / TN / FN | {metrics['TP']} / {metrics['FP']} / {metrics['TN']} / {metrics['FN']} |\n\n")

        f.write("## Confusion Matrix\n\n![Confusion Matrix](confusion_matrix.png)\n\n")
        f.write("## Loss Curve\n\n![Loss Curve](loss_curve.png)\n\n")
        f.write("## AUC Curve\n\n![AUC Curve](auc_curve.png)\n\n")
        f.write("## Circuit Diagram\n\n![Circuit Diagram](circuit.png)\n\n")

    return report_path


def make_report(vqc_outputs, model, cf_matrix_norm, loss_values, y_test, pulsar_proba, config, metrics):
    """
    Create this run's folder, save all four plots into it, and write
    report.md. Returns the report folder path.
    """
    report_dir = os.path.join(
        vqc_outputs, f"seed_{config['seed']}",
        f"{config['feature_map']}_{config['ansatz']}_{config['entanglement']}_{config['loss']}_{config['n_samples']}samples",
    )
    os.makedirs(report_dir, exist_ok=True)

    plot_confusion_matrix(report_dir, cf_matrix_norm, config["train_size"], config["test_size"], config)
    plot_loss_curve(report_dir, loss_values, config)
    plot_auc_curve(report_dir, y_test, pulsar_proba, config)
    plot_circuit(report_dir, model, config)

    report_path = write_report(report_dir, config, metrics)
    print(f"Report saved to: {report_path}")

    return report_dir
