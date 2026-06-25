"""
prepare_htru2_from_local.py

Preprocesses a local HTRU_2.csv file for Qiskit VQC experiments.

Input:
    data/HTRU_2.csv

Outputs:
    data/7-datacut_5-features.csv
    data/f-datacut_5-features.csv
    data/FS2/7-datacut_5-features.csv
    etc.

The output files are compatible with the original filename pattern:

    filename = f"{cut}-datacut_{n_feat}-features"

The default files in data/ use FS1 = SelectKBest(f_classif).
Optional FS2 files are placed in data/FS2/.
"""

import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


RANDOM_STATE = 42

COLUMNS = [
    "profile_mean",
    "profile_stdev",
    "profile_excess_kurtosis",
    "profile_skewness",
    "dm_mean",
    "dm_stdev",
    "dm_excess_kurtosis",
    "dm_skewness",
    "class",
]


def load_local_htru2(path="data/HTRU_2.csv"):
    """
    Load local HTRU2 data.

    Works whether the file already has headers or is the original no-header CSV.
    """
    df = pd.read_csv(path)

    # If the CSV had no header, pandas interpreted the first data row as headers.
    # In that case, reload with header=None.
    if "class" not in df.columns or df.shape[1] != 9:
        df = pd.read_csv(path, header=None, names=COLUMNS)

    # If it has numeric string headers from a no-header read, normalize them.
    if "class" not in df.columns and df.shape[1] == 9:
        df.columns = COLUMNS

    return df


def select_features(df, n_features, method):
    X = df.drop(columns="class")
    y = df["class"].astype(int)

    if method == "FS1":
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()

    if method == "FS2":
        correlations = X.apply(lambda col: abs(col.corr(y)))
        return correlations.sort_values(ascending=False).head(n_features).index.tolist()

    raise ValueError("method must be FS1 or FS2")


def scale_to_pi(df, selected_features):
    X = df[selected_features]
    y = df["class"].astype(int).reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=selected_features,
    )

    return pd.concat([X_scaled, y], axis=1)


def save_files(df, method, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    records = []

    for n_features in range(2, 9):
        selected = select_features(df, n_features, method)
        processed = scale_to_pi(df, selected)

        full_path = os.path.join(target_dir, f"f-datacut_{n_features}-features.csv")
        processed.to_csv(full_path, index=False)

        chunks = np.array_split(processed, 9)

        for i, chunk in enumerate(chunks, start=1):
            chunk_path = os.path.join(target_dir, f"{i}-datacut_{n_features}-features.csv")
            chunk.to_csv(chunk_path, index=False)

        records.append({
            "method": method,
            "n_features": n_features,
            "selected_features": ", ".join(selected),
            "full_file": full_path,
        })

        print(f"{method}, {n_features} features: {selected}")

    return records


def main():
    df = load_local_htru2("data/HTRU_2.csv")

    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)

    print("Loaded local HTRU2.")
    print("Shape:", df.shape)
    print("Class counts:")
    print(df["class"].value_counts())
    print()

    records = []

    # FS1 goes directly into data/ to match the original code.
    records.extend(save_files(df, method="FS1", target_dir="data"))

    # FS2 is also generated for optional comparison.
    records.extend(save_files(df, method="FS2", target_dir=os.path.join("data", "FS2")))

    pd.DataFrame(records).to_csv("feature_selection_summary.csv", index=False)

    print()
    print("Done. Original-style file exists at:")
    print("data/7-datacut_5-features.csv")


if __name__ == "__main__":
    main()
