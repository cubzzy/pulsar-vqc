"""
vqc_htru2.py

Runs a Qiskit Variational Quantum Classifier on a preprocessed HTRU2 file.

Expected input, created by prepare_htru2.py:
    f-datacut_3-features.csv

Run:
    python prepare_htru2.py
    python vqc_htru2.py
"""

from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SLSQP
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
from report_maker import make_report
from qiskit.circuit.library import (
    zz_feature_map,
    real_amplitudes,
    efficient_su2,
    n_local,
)
import pandas as pd
import numpy as np
import os

# ================================================================== 
# These are the parameters for the output. Change them at your discretion:
seeds = [1,2,3,4,5]  # change this before running to get an independent run for averaging
cut = 'f'
n_feat = 3
sample_sizes = [300]
feature_maps = ["ZZFeatureMap"]
ansatz_list = ["EfficientSU2", "RealAmplitudes", "TwoLocal"]
entanglement_options = ["full", "linear", "circular"]
loss_functions = ["cross_entropy"] # "squared_error", "absolute_error"]
num_reps = 2 # the number of reps the circuit completes
num_shots = 4096 # the number of shots SamplerV2 executes
feat_sel = "FS2" # the method that determines how the features are chosen. See prepare_htru2.py
optimizer = "COBYLA" # if sampler = "AER", the backend will be SamplerV2 from AER and the optimizer will be set to COBYLA.
# if the optimizer = "SLSQP", the backend will be QMLSampler (the default sampler) and the optimizer will be set to SLSQP.

#===================================================================
# Import preprocessed data

script_dir = os.path.dirname(os.path.abspath(__file__))
reports_root = os.path.join(script_dir, "report_outputs")
filename = f"{cut}-datacut_{n_feat}-features"
data_file = os.path.join(reports_root, "data/", feat_sel, filename + ".csv")

df = pd.read_csv(data_file)

print(df.head())


#===================================================================
# Further data cut

n_features = df.shape[1] - 1
n_qubits = n_features


#===================================================================
# Experiment settings


csv_output_file = "vqc_results.csv"
errors = []

vqc_outputs = os.path.join(reports_root, "vqc_outputs")
csv_path = os.path.join(reports_root, csv_output_file)

# Load any existing results so a run with a new seed adds to the file
# instead of overwriting seeds/configs from previous runs.
if os.path.exists(csv_path):
    results = pd.read_csv(csv_path)
    # Strip stray whitespace (e.g. from manual/spreadsheet edits) so column
    # names and values line up with the ones this script generates below --
    # otherwise pandas treats "Feature_map " and "Feature_map" as different
    # columns and the CSV ends up with a scattered, duplicated column set.
    results.columns = results.columns.str.strip()
    str_cols = results.select_dtypes(include="object").columns
    results[str_cols] = results[str_cols].apply(lambda c: c.str.strip())
else:
    results = pd.DataFrame()


def create_circuit(circuit_name, n_qubits, entanglement, prefix):
    """Create a Qiskit circuit from a string name."""
    if circuit_name == "ZZFeatureMap":
        return zz_feature_map(
            feature_dimension=n_qubits,
            entanglement=entanglement,
            reps=num_reps,
            parameter_prefix=prefix,
        )

    if circuit_name == "RealAmplitudes":
        return real_amplitudes(
            num_qubits=n_qubits,
            entanglement=entanglement,
            reps=num_reps,
            parameter_prefix=prefix,
        )

    if circuit_name == "EfficientSU2":
        return efficient_su2(
            num_qubits=n_qubits,
            entanglement=entanglement,
            reps=num_reps,
            parameter_prefix=prefix,
        )

    if circuit_name == "TwoLocal":
        return n_local(
            num_qubits=n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement=entanglement,
            reps=num_reps,
            parameter_prefix=prefix,
        )

    raise ValueError(f"Unknown circuit name: {circuit_name}")


#===================================================================
# Separate features and labels

y = np.ravel(df["class"].values).astype(int)
X = df.drop(columns="class")


#===================================================================
# Train and test VQC models
for seed in seeds:

    for n_samples in sample_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=n_samples,
            random_state=seed,
            stratify=y,
        )
        for feature_map_name in feature_maps:
            for ansatz_name in ansatz_list:
                for entanglement in entanglement_options:
                    for loss in loss_functions:
                        try:
                            algorithm_globals.random_seed = seed
                            loss_values = []
                            print("\n==============================================")
                            print("Seed:", seed)
                            print("Feature Map:", feature_map_name)
                            print("Ansatz:", ansatz_name)
                            print("Entanglement:", entanglement)
                            print("Loss:", loss)
                            print("Training Samples:", n_samples)
                            print("Features/Qubits:", n_qubits)

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

                            if optimizer == "COBYLA":
                                AER = SamplerV2(default_shots=num_shots, seed=seed)
                                aer_simulator = AerSimulator()
                                model = VQC(
                                    num_qubits=n_qubits,
                                    feature_map=feature_map,
                                    ansatz=ansatz,
                                    loss=loss,
                                    callback=lambda weight, loss_value: loss_values.append(loss_value),
                                    optimizer=COBYLA(),
                                    sampler=AER,
                                    pass_manager=generate_preset_pass_manager(backend=aer_simulator),
                                )
                            else:
                                model = VQC(
                                    num_qubits=n_qubits,
                                    feature_map=feature_map,
                                    ansatz=ansatz,
                                    loss=loss,
                                    callback=lambda weight, loss_value: loss_values.append(loss_value),
                                    optimizer = SLSQP()
                                )
                            
                            print("\nTraining the model...")
                            model.fit(X_train.to_numpy(), y_train)

                            print("\nTesting the model...")
                            y_pred = model.predict(X_test.to_numpy()).astype(int)
                            x_pred = model.predict_proba(X_test.to_numpy())
                      
                            print("\nClassification report:")
                            print(classification_report(y_test, y_pred, labels=[0, 1]))

                            cf_matrix_norm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize='true')
                            cf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
                            TN, FP, FN, TP = cf_matrix.ravel()

                            accuracy = round((TP + TN) / (TP + TN + FP + FN), 3)
                            precision = round(TP / (TP + FP), 3) if (TP + FP) else 0
                            recall = round(TP / (TP + FN), 3) if (TP + FN) else 0
                            f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0
                            fpr = round(FP / (FP + TN), 3) if (FP + TN) else 0
                            mcc = round(matthews_corrcoef(y_test, y_pred), 3)

                            result_dict = {
                                "seed": seed,
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

                            results = pd.concat(
                                [results, pd.DataFrame([result_dict])],
                                ignore_index=True,
                            )
                            # Rerunning the same seed/config replaces its old row
                            # instead of adding a duplicate.
                            results = results.drop_duplicates(
                                subset=["seed", "N_samples", "Feature_map", "Ansatz", "Entanglement", "Loss"],
                                keep="last",
                            )

                            results.to_csv(csv_path, index=False)

                            # printing the number of non-pulsars and pulsars in the training set
                            train_nonpulsars = np.unique(y_train, return_counts=True)[1][0]
                            train_pulsars = np.unique(y_train, return_counts=True)[1][1]
                            print(f"Amount of non-pulsars in training set: {train_nonpulsars}")
                            print(f"Amount of pulsars in training set: {train_pulsars}")

                            # printing the number of non-pulsars and pulsars in the testing set
                            test_nonpulsars = np.unique(y_test, return_counts=True)[1][0]
                            test_pulsars = np.unique(y_test, return_counts=True)[1][1]
                            print(f"Amount of non-pulsars in testing set: {test_nonpulsars}")
                            print(f"Amount of pulsars in testing set: {test_pulsars}")

                            config = {
                                "seed": seed,
                                "feature_map": feature_map_name,
                                "ansatz": ansatz_name,
                                "entanglement": entanglement,
                                "loss": loss,
                                "n_samples": n_samples,
                                "n_qubits": n_qubits,
                                "train_size": X_train.shape[0],
                                "test_size": X_test.shape[0],
                                "train_nonpulsars": train_nonpulsars,
                                "train_pulsars": train_pulsars,
                                "test_nonpulsars": test_nonpulsars,
                                "test_pulsars": test_pulsars,
                            }
                            metrics = {
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "fpr": fpr,
                                "mcc": mcc,
                                "TP": TP,
                                "FP": FP,
                                "TN": TN,
                                "FN": FN,
                            }

                            make_report(vqc_outputs, model, cf_matrix_norm, loss_values, y_test, x_pred[:, 1], config, metrics)

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

print(f"\nResults saved to: {csv_path}")