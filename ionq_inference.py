"""
ionq_inference.py

Re-runs the TRAINED pulsar VQC (ZZFeatureMap + EfficientSU2, full, 3 qubits, seed 1, n=300)
on a noisy backend, inference only. Weights come from vqc_htru2.py (np.save of model.weights).

Run in the .venv environment (needs pandas + scikit-learn + qiskit-aer + qiskit-ionq):
    BACKEND = "aer"        -> ideal local run, sanity check, no key needed
    BACKEND = "ionq_sim"   -> IonQ noisy simulator, needs IONQ_API_KEY in the environment

Start with N_TEST = 1, then 10, then 200.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map, efficient_su2

# ==================================================================
BACKEND = "aer"   # "aer" or "ionq_sim"
NOISE_MODEL = "aria-1" # only used when BACKEND == "ionq_sim" (name from the March notebook; verify it still exists)
N_TEST = 100      #number of test circuits to run. 1, then 10, then 200
SHOTS = 1024
SEED = 1               # same seed as the trained model, so the train/test split matches

REPO = "/Users/0/Repos/pulsar-vqc/report_outputs"
DATA = f"{REPO}/data/FS2/f-datacut_3-features.csv"
WEIGHTS = f"{REPO}/weights_ZZFeatureMap_EfficientSU2_full_seed1_n300.npy"
# ==================================================================


def counts_to_label(counts):
    """Predicted class (0 or 1) from a Qiskit counts dict like {'011': 348, ...}.

    VQC's default interpret function maps a measured integer x to the class x % 2,
    i.e. the class is the value of qubit 0 (the rightmost character of the bitstring).
    The predicted class is the argmax of the two class probabilities.
    """
    total_0 = 0
    total_1 = 0
    for key, count in counts.items():
        if int(key, 2) % 2 == 0:
            total_0 += count
        else:
            total_1 += count
    return np.argmax([total_0, total_1])

def counts_to_prob1(counts):
    """Probability of class 1 (pulsar) from a counts dict: the fraction of shots where qubit 0 measured 1."""
    # TODO(human): same bitstring rule as counts_to_label, but return a number between 0 and 1
    total_0 = 0
    total_1 = 0
    for key, count in counts.items():
        if int(key, 2) % 2 == 0:
            total_0 += count
        else:
            total_1 += count
    prob = np.round((total_1 / (total_0 + total_1)), 4)
    return prob

def build_circuit(x, weights):
    """Same circuit VQC builds: feature map on x, then the ansatz on the trained weights."""
    feature_map = zz_feature_map(feature_dimension=3, entanglement="full", reps=2, parameter_prefix="f")
    ansatz = efficient_su2(num_qubits=3, entanglement="full", reps=2, parameter_prefix="a")
    qc = feature_map.compose(ansatz)
    qc.measure_all()

    bind = dict(zip(feature_map.parameters, x))
    bind.update(dict(zip(ansatz.parameters, weights)))
    return qc.assign_parameters(bind)


def get_runner():
    """Return a function that takes a circuit and returns its counts."""
    if BACKEND == "aer":
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        return lambda qc: sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()

    from qiskit_ionq import IonQProvider
    backend = IonQProvider().get_backend("ionq_simulator")
    backend.set_options(noise_model=NOISE_MODEL)
    # generic basis set; if IonQ rejects a gate on the 1-circuit dry run, this line is the one to change
    return lambda qc: backend.run(
        transpile(qc, basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=0),
        shots=SHOTS,
    ).result().get_counts()


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    y = np.ravel(df["class"].values).astype(int)
    X = df.drop(columns="class")
    _, X_test, _, y_test = train_test_split(X, y, train_size=300, random_state=SEED, stratify=y)

    weights = np.load(WEIGHTS)

    # random subset of the test set, reproducible, not class-balanced (mirrors the paper's ~200-sample test sets)
    pick = np.random.default_rng(SEED).choice(len(X_test), size=N_TEST, replace=False)
    Xs, ys = X_test.to_numpy()[pick], y_test[pick]

    run = get_runner()
    y_pred, p1 = [], []
    for i, x in enumerate(Xs):
        counts = run(build_circuit(x, weights))
        y_pred.append(counts_to_label(counts))
        p1.append(counts_to_prob1(counts))
        print(f"{i + 1}/{N_TEST} true={ys[i]} pred={y_pred[-1]} p1={p1[-1]:.3f}")

    # one row per circuit; test_row lets two backends' files be matched up circuit by circuit
    out = f"{REPO}/probs_{BACKEND}_n{N_TEST}.csv"
    pd.DataFrame({"test_row": pick, "true": ys, "pred": y_pred, "p1": p1}).to_csv(out, index=False)
    print("saved", out)

    y_pred = np.array(y_pred)
    print("\nBackend:", BACKEND, "| circuits:", N_TEST, "| shots:", SHOTS, "| pulsars in subset:", int(ys.sum()))
    if N_TEST >= 10:
        # same argument order as vqc_htru2.py: (y_true, y_pred)
        TN, FP, FN, TP = confusion_matrix(ys, y_pred, labels=[0, 1]).ravel()
        print(f"TN={TN} FP={FP} FN={FN} TP={TP}")
        print("accuracy:", round((TP + TN) / len(ys), 3), "| MCC:", round(matthews_corrcoef(ys, y_pred), 3))
