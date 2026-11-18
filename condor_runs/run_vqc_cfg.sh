#!/bin/bash
# usage: run_vqc_cfg.sh <seed> <optimizer> <ansatz> <entanglement>
SEED=$1
OPT=$2
ANSATZ=$3
ENT=$4
REPO=$HOME/pulsar-vqc   # assumes the repo and its .venv live at ~/pulsar-vqc
OUT=$REPO/runs/${OPT}_${ANSATZ}_${ENT}_seed${SEED}
mkdir -p "$OUT"
# 1 CPU per job on this cluster: keep numpy/Aer from spawning extra threads; plot headless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
{
  hostname
  "$REPO/.venv/bin/python" -u "$REPO/vqc_htru2.py" --seed "$SEED" --optimizer "$OPT" --ansatz "$ANSATZ" --entanglement "$ENT" --out-dir "$OUT"
} > "$OUT/run.log" 2>&1
