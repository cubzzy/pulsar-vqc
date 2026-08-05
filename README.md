# Pulsar VQC — QML IRIE Summer 2026

Quantum machine learning research applying a **Variational Quantum Classifier (VQC)** to pulsar detection using the HTRU2 dataset, conducted as part of the UMD IRIE Summer 2026 program. 

**NOTE: This is the preliminary README and is subject to change.**

---

## Overview

Pulsars are rapidly rotating neutron stars that emit periodic radio pulses. Finding them in large radio survey datasets is a classification problem at massive scale. For example, the HTRU2 dataset (High Time Resolution Universe 2) contains almost 18,000 candidates, of which only around 9% are confirmed pulsars.

This project builds on [Souza et al. (2025)](https://arxiv.org/abs/2505.15600), which demonstrated a Qiskit-based VQC for pulsar classification but was limited to ~180–300 training samples due to simulation cost. We hope to extend that work through: 

- Scaling to larger sample sizes using **IonQ quantum hardware**
- Benchmarking against classical ML approaches
- Overall, testing for evidence of quantum advantage on this task

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

1. **Preprocessing** — Feature selection: SelectKBest or correlation-based (nicknamed FS1 and FS2) and scaling to [0, π] for quantum angle encoding
2. **Circuit** — Our script is capable of running loops of multiple configurations like ZZFeatureMap + EfficientSU2/RealAmplitudes/TwoLocal + Full/Linear/Circular Entanglement
3. **Training** — COBYLA optimizer + AER backend (which uses SamplerV2) or the SLSQP optimizer and the native QMLSampler backend. SLSQP does not work with the AER backend
4. **Evaluation** — Accuracy, Precision, Recall, F1, FPR, MCC, Confusion Matrix, Loss Curve, AUC Curve, Qiskit Circuit Visualization

---

## Repository Structure

```
Team-Pulsar-Github-Repo/
├── htru2/                   # Raw HTRU2 dataset (source files)
│   ├── HTRU_2.csv
│   ├── HTRU_2.arff
│   └── htru2.zip
├── report_outputs/          # Contains previous runs, new runs get added here
│   ├── data/FS1/, FS2/       # prepare_htru2.py output
│   └── vqc_outputs/seed_*/   # vqc_htru2.py output: plots + report.md per seed/config
├── .gitignore
├── prepare_htru2.py         # Preprocesses HTRU_2 data → FS1/FS2 feature-selected CSVs
├── README.md
├── report_maker.py          # Builds per-run plots + report.md; called internally by vqc_htru2.py
├── requirements.txt
└── vqc_htru2.py             # Main experiment: sweeps seeds/feature maps/ansätze/entanglement, trains + evaluates VQC
```

---

## Quickstart

```bash
# 1. Ensure you have a fresh venv and have installed everything in requirements.txt
# 2. Place HTRU_2.csv into the htru2 folder (it's already in the repo)
# 3. Generate preprocessed files
python prepare_htru2.py

# 4. Run the VQC experiment
python vqc_htru2.py
# We have verified this code to work on Mac and Windows. 
# Expect it to take several hours for a full multi-seed run.
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `qiskit`, `qiskit-machine-learning`, `qiskit-algorithms`, `qiskit-aer`,
`pylatexenc`

---

## Preliminary Results
**Outputs Across Seeds 1-5 of Best Config: ZZFeatureMap, EfficientSU2, Full Entanglement**
|Seed|Accuracy|Precision|Recall|F1-Score|FPR|MCC|
|---|---|---|---|---|---|---|
|1|0.95|0.856|0.543|0.664|0.009|0.658|
|2|0.947|0.857|0.504|0.635|0.008|0.633|
|3|0.945|0.856|0.478|0.613|0.008|0.615|
|4|0.939|0.782|0.457|0.577|0.013|0.569|
|5|0.948|0.836|0.533|0.651|0.011|0.643|

---

## Our Team Poster
[View the full poster (PDF)](poster.pdf)

## References

Souza, A., Cruz, C., & Moret, M. A. (2025). Qiskit Variational Quantum Classifier on
the Pulsar Classification Problem. arXiv preprint arXiv:2505.15600.

R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty years of
pulsar candidate selection: from simple filters to a new principled real-time
classification approach, Monthly Notices of the Royal Astronomical Society, Volume
459, Issue 1, 11 June 2016, Pages 1104–
1123, https://doi.org/10.1093/mnras/stw656