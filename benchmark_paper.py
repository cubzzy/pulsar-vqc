"""
benchmark_paper.py

Replicates Souza, Cruz & Moret (2025)'s best-MCC configuration to verify
this environment produces comparable results to the paper (arXiv:2505.15600,
Table VII): 3 features, 300 training samples, FS2 feature selection,
ZZFeatureMap + EfficientSU2 + full entanglement -> reported MCC 0.670.

Uses the original default sampler (no Aer) and SLSQP, matching the paper's
methodology exactly, so any mismatch is attributable to the replication
itself rather than a different optimizer/sampler.

Expected input, created by prepare_htru2.py:
    data/FS2/f-datacut_3-features.csv

Run:
    python benchmark_paper.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split

from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import SLSQP

from qiskit.circuit.library import ZZFeatureMap, EfficientSU2


#===================================================================
# Import preprocessed data - full (non-chunked) FS2 file, 3 features

n_feat = 3

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data", "FS2", f"f-datacut_{n_feat}-features.csv")

print("Looking for:", data_file)

df = pd.read_csv(data_file)

print(df.head())
print("Loaded shape:", df.shape)


#===================================================================
# Paper's best-MCC config: 3 features, 300 samples, FS2, ZZ, EfficientSU2, full

n_samples = 300
n_qubits = n_feat
state = 42

PAPER_REPORTED = {
    "Accuracy": 0.945,
    "Precision": 0.476,
    "Recall": 1.000,
    "F1-score": 0.645,
    "MCC": 0.670,
}


#===================================================================
# Separate features and labels

y = np.ravel(df["class"].values).astype(int)
X = df.drop(columns="class")

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=state,
    stratify=y,
)

X_train = X_train_pool.iloc[:n_samples]
y_train = y_train_pool[:n_samples]

print(f"\nTraining on {n_samples} samples, testing on {X_test.shape[0]} samples.")


#===================================================================
# Build and train VQC - default sampler (no Aer), SLSQP, matching the paper

feature_map = ZZFeatureMap(feature_dimension=n_qubits, entanglement="full", reps=2, parameter_prefix="f")
ansatz = EfficientSU2(num_qubits=n_qubits, entanglement="full", reps=2, parameter_prefix="a")

model = VQC(
    num_qubits=n_qubits,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=SLSQP(),
)

print("\nTraining...")
model.fit(X_train.to_numpy(), y_train)

print("\nTesting...")
y_pred = model.predict(X_test.to_numpy()).astype(int)

print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=[0, 1]))

cf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
TN, FP, FN, TP = cf_matrix.ravel()

accuracy = round((TP + TN) / (TP + TN + FP + FN), 3)
precision = round(TP / (TP + FP), 3) if (TP + FP) else 0
recall = round(TP / (TP + FN), 3) if (TP + FN) else 0
f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0
mcc = round(matthews_corrcoef(y_test, y_pred), 3)

ours = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1, "MCC": mcc}

print("\n=== Comparison to paper (Table VII, 3 feat / 300 samples / FS2 / ZZ+EfficientSU2/full) ===")
print(f"{'Metric':<12}{'Paper':>10}{'Ours':>10}")
for metric in ["Accuracy", "Precision", "Recall", "F1-score", "MCC"]:
    print(f"{metric:<12}{PAPER_REPORTED[metric]:>10}{ours[metric]:>10}")

sns.heatmap(cf_matrix, cmap="Purples", annot=True, linewidth=1, fmt="d")
plt.xlabel("Model prediction")
plt.ylabel("True label")
plt.savefig("confusion_matrix_benchmark_paper.png", bbox_inches="tight")
plt.close()
print("\nConfusion matrix saved to: confusion_matrix_benchmark_paper.png")
