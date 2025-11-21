#!/usr/bin/env bash
set -e

echo "========================================="
echo " HYBRID IL + RL PIPELINE (Linux/Mac)"
echo "========================================="

echo "[1/5] Collecting expert data..."
python -m src.collect_expert --episodes 300 --out outputs/expert_data.npz

echo "[2/5] Training Behavioral Cloning..."
python -m src.imitation.behavioral_cloning --data outputs/expert_data.npz --save outputs/bc_policy.pth --epochs 120
python -m src.rl.train_rl --timesteps 200000 --bc outputs/bc_policy.pth --out outputs/rl_model.zip

echo "[3/5] Training PPO with BC initialization..."
python -m src.rl.train_rl --timesteps 200000 --bc outputs/bc_policy.pth --out outputs/rl_model.zip

echo "[4/5] Evaluating PID, BC, RL..."
python -m src.evaluate --rl outputs/rl_model.zip --bc outputs/bc_policy.pth --episodes 10 --outdir outputs/plots

echo "[5/5] Plotting final comparison graph..."
plot_all_one_by_one.bat

echo "========================================="
echo " Pipeline COMPLETE!"
echo " Plots saved in outputs/plots/"
echo "========================================="
