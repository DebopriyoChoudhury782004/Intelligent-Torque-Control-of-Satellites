#!/usr/bin/env bash
set -e
python -m src.collect_expert --episodes 300 --out outputs/expert_data.npz
python -m src.imitation.behavioral_cloning --data outputs/expert_data.npz --save outputs/bc_policy.pth --epochs 120
python -m src.rl.train_rl --timesteps 200000 --bc outputs/bc_policy.pth --out outputs/rl_model.zip
python -m src.evaluate --rl outputs/rl_model.zip --bc outputs/bc_policy.pth --episodes 10
python -m src.plot_results
