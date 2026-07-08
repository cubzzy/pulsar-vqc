# Pulsar VQC — QML IRIE Summer 2026

Quantum machine learning research applying a **Variational Quantum Classifier (VQC)** to pulsar detection using the HTRU2 dataset, conducted as part of the UMD IRIE Summer 2026 program. 

**NOTE: This is the preliminary README and is subject to change.**

---

## Overview

Pulsars are rapidly rotating neutron stars that emit periodic radio pulses. Identifying them in large radio survey datasets is a classic imbalanced binary classification problem — the HTRU2 dataset contains ~17,898 candidates, of which only ~1,639 (~9%) are confirmed pulsars.

This project builds on [Souza et al. (2025)](https://arxiv.org/abs/2505.15600), which demonstrated a Qiskit-based VQC for pulsar classification but was limited to ~180–300 training samples due to simulation cost. We extend that work by:

- Scaling to larger sample sizes using **IonQ quantum hardware**
- Benchmarking against classical ML approaches
- Testing for evidence of quantum advantage on this task

---

## Dataset

**HTRU2** — High Time Resolution Universe Survey 2  
17,898 samples × 8 features, binary labels (pulsar / non-pulsar)

| Feature | Description |
|---|---|
| `profile_mean` | Mean of the integrated pulse profile |
| `profile_stdev` | Standard deviation of the integrated pulse profile |
| `profile_excess_kurtosis` | Excess kurtosis of the integrated pulse profile |
| `profile_skewness` | Skewness of the integrated pulse profile |
| `dm_mean` | Mean of the DM-SNR curve |
| `dm_stdev` | Standard deviation of the DM-SNR curve |
| `dm_excess_kurtosis` | Excess kurtosis of the DM-SNR curve |
| `dm_skewness` | Skewness of the DM-SNR curve |

---

## Method

1. **Preprocessing** — Feature selection (SelectKBest or correlation-based) and scaling to [0, π] for quantum angle encoding
2. **Circuit** — ZZFeatureMap (feature encoding) + RealAmplitudes (variational ansatz)
3. **Training** — COBYLA optimizer via Qiskit's `VQC` class
4. **Evaluation** — Accuracy, Precision, Recall, F1, FPR, MCC, confusion matrix

---

## Repository Structure

```
[add later]
```

---

## Quickstart

```bash
# 1. Place HTRU_2.csv into a data/ folder next to the scripts (it's already in the repo)
# 2. Generate preprocessed files
python prepare_htru2.py

# 3. Run the VQC experiment
python vqc_htru2.py
```

**Dependencies:** `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `qiskit`, `qiskit-machine-learning`, `qiskit-algorithms`, `qiskit-aer`

---

## Preliminary Results

Early run on a small subset (25 training samples, ~994 test samples):

| Metric | Value |
|---|---|
| Accuracy | 0.914 |
| Precision | 0.542 |
| Recall | 0.356 |
| F1-score | 0.430 |
| FPR | 0.030 |
| MCC | 0.396 |

> Low recall reflects the class imbalance — a core challenge this project aims to address.

---

## Team

| Name | Role |
|---|---|
| Ethan Martin | [add later] |
| Haedin Hilton | [add later] |
| Aiden Precht | [add later] |
| Brian Ocotlan-Urbano | [add later] |
| Ella Chen | [add later] |

**Supervisor:** Dr. Shabnam Jabeen, University of Maryland  
**Collaborators:** [add later]

---

## References

[add later]
