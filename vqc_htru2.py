"""
vqc_htru2.py

Runs a Qiskit Variational Quantum Classifier on a preprocessed HTRU2 file.

Expected input, created by prepare_htru2.py:
    data/7-datacut_5-features.csv

Run:
    python prepare_htru2.py
    python vqc_htru2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split

from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
#from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
from qiskit.circuit.library import (
    ZZFeatureMap,
    PauliFeatureMap,
    RealAmplitudes,
    EfficientSU2,
    TwoLocal,
)


#===================================================================
# Import preprocessed data

cut = 'f'
n_feat = 3

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "data/FS2")
filename = f"{cut}-datacut_{n_feat}-features"
data_file = os.path.join(path, filename + ".csv")

print("Looking for:", data_file)

df = pd.read_csv(data_file)

print(df.head())
print("Loaded shape:", df.shape)


#===================================================================
# Further data cut

sample_sizes = [300]
n_features = df.shape[1] - 1
n_qubits = n_features


#===================================================================
# Experiment settings

feature_maps = ["ZZFeatureMap"]
ansatz_list = ["EfficientSU2"]
entanglement_options = ["full"]
loss_functions = ["cross_entropy"]
state = 42

output_file = "vqc_results.csv"
results = pd.DataFrame()
errors = []


def create_circuit(circuit_name, n_qubits, entanglement, prefix):
    """Create a Qiskit circuit from a string name."""
    if circuit_name == "ZZFeatureMap":
        return ZZFeatureMap(
            feature_dimension=n_qubits,
            entanglement=entanglement,
            reps=2,
            parameter_prefix=prefix,
        )

    if circuit_name == "PauliFeatureMap":
        return PauliFeatureMap(
            feature_dimension=n_qubits,
            entanglement=entanglement,
            reps=2,
            parameter_prefix=prefix,
        )

    #if circuit_name == "RawFeatureVector":
        #return RawFeatureVector(feature_dimension=n_qubits)

    if circuit_name == "RealAmplitudes":
        return RealAmplitudes(
            num_qubits=n_qubits,
            entanglement=entanglement,
            reps=2,
            parameter_prefix=prefix,
        )

    if circuit_name == "EfficientSU2":
        return EfficientSU2(
            num_qubits=n_qubits,
            entanglement=entanglement,
            reps=2,
            parameter_prefix=prefix,
        )

    if circuit_name == "TwoLocal":
        return TwoLocal(
            num_qubits=n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement=entanglement,
            reps=2,
            parameter_prefix=prefix,
        )

    raise ValueError(f"Unknown circuit name: {circuit_name}")


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


#===================================================================
# Train and test VQC models

for n_samples in sample_sizes:
    X_train = X_train_pool.iloc[:n_samples]
    y_train = y_train_pool[:n_samples]

    for feature_map_name in feature_maps:
        for ansatz_name in ansatz_list:
            for entanglement in entanglement_options:
                for loss in loss_functions:
                    try:
                        print("\n==============================================")
                        print("Feature map:", feature_map_name)
                        print("Ansatz:", ansatz_name)
                        print("Entanglement:", entanglement)
                        print("Loss:", loss)
                        print("Training samples:", n_samples)
                        print("Features/qubits:", n_qubits)

                        feature_map = create_circuit(
                            feature_map_name,
                            n_qubits,
                            entanglement,
                            prefix="f",
                        )

                        ansatz = create_circuit(
                            ansatz_name,
                            n_qubits,
                            entanglement,
                            prefix="a",
                        )

                        if feature_map.num_qubits != X_train.shape[1]:
                            error_message = (
                                f"Incompatible feature map '{feature_map_name}' requires "
                                f"{feature_map.num_qubits} qubits, but "
                                f"{X_train.shape[1]} features were provided."
                            )
                            print(error_message)
                            errors.append((feature_map_name, ansatz_name, error_message))
                            continue

                        AER = SamplerV2(default_shots=4096, seed = 42)
                        aer_simulator = AerSimulator()

                        model = VQC(
                            num_qubits=n_qubits,
                            feature_map=feature_map,
                            ansatz=ansatz,
                            optimizer=COBYLA(),
                            sampler=AER,
                            pass_manager=generate_preset_pass_manager(backend=aer_simulator),
                            loss=loss
                        )

                        model.fit(X_train.to_numpy(), y_train)


                        print("\nTesting the model...")
                        y_pred = model.predict(X_test.to_numpy()).astype(int)

                        print("Classification report:")
                        print(classification_report(y_test, y_pred, labels=[0, 1]))

                        cf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
                        TN, FP, FN, TP = cf_matrix.ravel()

                        accuracy = round((TP + TN) / (TP + TN + FP + FN), 3)
                        precision = round(TP / (TP + FP), 3) if (TP + FP) else 0
                        recall = round(TP / (TP + FN), 3) if (TP + FN) else 0
                        f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0
                        fpr = round(FP / (FP + TN), 3) if (FP + TN) else 0
                        mcc = round(matthews_corrcoef(y_test, y_pred), 3)

                        result_dict = {
                            "N_samples": n_samples,
                            "N_features": n_qubits,
                            "Feature_map": feature_map_name,
                            "Ansatz": ansatz_name,
                            "Entanglement": entanglement,
                            "Loss": loss,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1-score": f1,
                            "FPR": fpr,
                            "MCC": mcc,
                            "TP": TP,
                            "TN": TN,
                            "FP": FP,
                            "FN": FN,
                        }

                        print("Result:")
                        print(result_dict)

                        results = pd.concat(
                            [results, pd.DataFrame([result_dict])],
                            ignore_index=True,
                        )

                        results.to_csv(output_file, index=False)

                        print("Confusion matrix:")

                        sns.heatmap(
                            cf_matrix,
                            cmap="Purples",
                            annot=True,
                            linewidth=1,
                            fmt="d",
                        )

                        plt.xlabel("Model prediction")
                        plt.ylabel("True label")
                        plot_filename = f"confusion_matrix_{feature_map_name}_{ansatz_name}_{entanglement}_{n_samples}samples.png"
                        plt.savefig(plot_filename, bbox_inches="tight")
                        plt.close()
                        print(f"Confusion matrix saved to: {plot_filename}")

                    except Exception as e:
                        error_message = f"Error training {feature_map_name} + {ansatz_name}: {e}"
                        print(error_message)
                        errors.append((feature_map_name, ansatz_name, str(e)))
                        continue


#===================================================================
# Print errors

if errors:
    print("\nErrors found:")
    for error in errors:
        print(error)
else:
    print("\nNo errors found.")

print(f"\nResults saved to: {output_file}")
