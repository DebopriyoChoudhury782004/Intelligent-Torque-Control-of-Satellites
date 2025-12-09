# INTELLIGENT_TORQUE_CONTROL_OF_SATELLITES

> **Hybrid PID + Imitation Learning + Reinforcement Learning for Satellite Detumbling**

This repository implements and evaluates an **intelligent torque control framework** for satellite _detumbling_ (reducing angular rates to near-zero) using a combination of:

- Classical **PD (PID) control** as an expert,
- **Behavioral Cloning (BC)** to imitate the expert from demonstration data, and
- **Reinforcement Learning (RL)** with **PPO** (via Stable-Baselines3), optionally initialized from the BC policy.

A secondary sub-project, **`TORQUE_EXPERIMENT/`**, contains self-contained scripts for generating plots, running experiments, and reproducing results.

---

## 1. Abstract

Detumbling and attitude stabilization are critical tasks in satellite attitude control systems. Traditional PD/PID controllers are robust and interpretable, but may be suboptimal under complex dynamics or changing mission requirements. Learning-based controllers, on the other hand, can adapt and optimize performance but typically require large amounts of trial-and-error interaction with the environment.

This project proposes a **hybrid pipeline**:

1. Use a **PD expert controller** to generate high-quality demonstration trajectories.
2. Train a **Behavioral Cloning (BC) policy** to imitate the expert using supervised learning.
3. Initialize a **PPO-based RL policy** using the BC weights and continue training with reinforcement learning to improve upon the expert performance.

The entire stack is implemented in Python using **Gymnasium**, **NumPy**, **PyTorch**, and **Stable-Baselines3**, and tested in a custom **satellite detumbling environment**.

---

## 2. Problem Formulation

We consider a simplified rigid-body satellite detumbling problem.

### 2.1 State and Action Spaces

The environment `AttitudeDetumbleEnv` (in `src/envs/attitude_env.py`) defines:

- **State / Observation**  
  \[
  s = [ω_x, ω_y, ω_z, θ_x, θ_y, θ_z]
  \]
  where:

  - ω = (ω_x, ω_y, ω_z) is the angular velocity vector

  - θ = (θ_x, θ_y, θ_z) is a small-angle approximation of the attitude

- **Action**  
  \[
  a = [τ_x, τ_y, τ_z]
  \]
  where:

  - τ = (τ_x, τ_y, τ_z) is the control torque vector

  - Bounded by |τ_i| ≤ 1.0

### 2.2 Dynamics

The environment uses a simple discrete-time update with damping:

```python
damping = -0.05 * self.w
self.w = self.w + (action + damping) * dt
self.angle = self.angle + self.w * dt
```

where:

- `self.w` is the angular velocity vector,
- `self.angle` is the attitude vector,
- `dt` is the simulation time-step (default 0.1 s).

### 2.3 Reward Function

The control objective is to minimize angular velocity while keeping control effort small:

\[
r(s,a) = -(\|\omega\|^2 + 0.01\|a\|^2)
\]

Episodes terminate when:

- \(\|\omega\|^2 < 0.02\) (successful detumbling), or
- The maximum number of steps (`max_steps`, default 500) is reached.

---

## 3. Methodology

The overall pipeline is:

**PD Expert → Data Collection → Behavioral Cloning → RL (PPO) Fine-Tuning**

### 3.1 PD Expert Controller

The expert controller in `src/expert/expert_pid.py` is a PD-style controller:

```python
torque = - Kp * angle - Kd * w
torque = clip(torque, -max_torque, max_torque)
```

with default gains:

- `Kp = 2.0`, `Kd = 0.8`,
- `max_torque = 1.0`.

This expert serves as a teacher that provides stable, near-optimal demonstrations for detumbling, mapping the observation vector directly to control torques.

### 3.2 Expert Data Collection

The script `src/collect_expert.py` (invoked via shell/batch scripts) runs the PD expert in the `AttitudeDetumbleEnv` and stores transitions into an `.npz` file:

`outputs/expert_data.npz`

The expected dataset format (used by BC) is:

- `obs`: array of shape (N, 6) — states,
- `actions`: array of shape (N, 3) — expert torques.

### 3.3 Behavioral Cloning (BC)

The Behavioral Cloning module is in `src/imitation/behavioral_cloning.py`.

#### Network Architecture

`BCNet` is a fully-connected MLP:

```python
class BCNet(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3, hidden=(128, 128)):
        ...
        # [obs_dim] → 128 → ReLU → 128 → ReLU → [act_dim]
```

- Input: 6D state vector,
- Output: 3D torque vector,
- Hidden sizes: (128, 128) with ReLU non-linearities.

#### Training Objective

- Loss function: Mean Squared Error (MSE) between predicted and expert torques,
- Optimizer: Adam (`lr = 1e-3`),
- Batch size: 64,
- Epochs: default 80.

The training loop iterates over the demonstration dataset and periodically logs the average loss. After training, the network weights are saved to:

`outputs/bc_policy.pth`

You can run BC training as:

```bash
python -m src.imitation.behavioral_cloning \
    --data outputs/expert_data.npz \
    --save outputs/bc_policy.pth \
    --epochs 80
```

(or simply `python src/imitation/behavioral_cloning.py` with defaults, depending on your Python path).

### 3.4 Reinforcement Learning with PPO

RL training is implemented in `src/rl/train_rl.py` using Stable-Baselines3 PPO:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

env = DummyVecEnv([make_env])  # vectorized AttitudeDetumbleEnv
policy_kwargs = dict(net_arch=[dict(pi=[128,128], vf=[128,128])])
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")
```

Key characteristics:

- Actor and critic share a two-layer MLP with 128 units each (for both policy and value networks).
- Training horizon `total_timesteps` is configurable (default: 200_000).
- Other PPO hyperparameters use Stable-Baselines3 defaults unless changed in the code.

#### 3.4.1 Initializing PPO from Behavioral Cloning

The code includes two helper functions to copy BC weights into the PPO policy:

- `copy_weights_by_shape(src_state, dest_model)`
- `copy_weights_by_order(src_state, dest_model)`

The process:

1. Load the BC model weights from `outputs/bc_policy.pth`.
2. Attempt to align them with `model.policy.state_dict()` using key and shape matching.
3. If that fails, fall back to order-based matching (matching parameters by shape and order).

This provides a warm-start for the RL policy:

```python
model = PPO(...)
if os.path.exists(bc_path):
    src_state = torch.load(bc_path, map_location="cpu")
    updated = copy_weights_by_shape(src_state, model)
    if updated == 0:
        updated = copy_weights_by_order(src_state, model)
    print(f"Copied {updated} parameters from BC into SB3 policy (if compatible).")
else:
    print("No BC policy found; training from scratch.")
```

After initialization, PPO continues training on the environment and saves the final model to:

`outputs/rl_model.zip`

Example invocation:

```bash
python -m src.rl.train_rl \
    --timesteps 200000 \
    --bc outputs/bc_policy.pth \
    --out outputs/rl_model.zip
```

---

## 4. Repository Structure

High-level layout:

```
INTELLIGENT_TORQUE_CONTROL_OF_SATELLITES/
│
├── .venv/                     # Local virtual environment (optional / local)
│
├── outputs/
│   ├── bc_policy.pth          # Trained BC model
│   ├── expert_data.npz        # Expert trajectories (obs, actions)
│   └── rl_model.zip           # Trained PPO model
│
├── run_scripts/
│   ├── bc_pretrain.sh         # Convenience script: train BC
│   ├── collect_expert.sh      # Script: collect expert data
│   ├── evaluate.sh            # Script: evaluate trained controllers
│   └── rl_train.sh            # Script: train RL (PPO)
│
├── src/
│   ├── envs/
│   │   ├── attitude_env.py    # Custom detumbling environment (Gymnasium)
│   │   └── __init__.py
│   │
│   ├── evaluate/
│   │   ├── evaluate_policy.py # Evaluation utilities (rollouts, metrics)
│   │   └── __init__.py
│   │
│   ├── expert/
│   │   ├── expert_pid.py      # PD-based expert policy
│   │   └── __init__.py
│   │
│   ├── imitation/
│   │   ├── behavioral_cloning.py  # BCNet + training loop
│   │   └── __init__.py
│   │
│   ├── rl/
│   │   ├── train_rl.py        # PPO training with optional BC initialization
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── replay_buffer.py   # (Optional) replay structures/utilities
│   │   └── __init__.py
│   │
│   ├── collect_expert.py      # Entrypoint for generating expert_data.npz
│   └── __init__.py
│
├── TORQUE_EXPERIMENT/         # Separate experiment/plotting sub-project
│   ├── outputs/
│   │   ├── plots/
│   │   │   ├── all_controllers_omega_3axis.png
│   │   │   ├── bc_only_omega_3axis.png
│   │   │   ├── comparison_omega_3axis.png
│   │   │   ├── comparison_omega.png
│   │   │   ├── pid_only_omega_3axis.png
│   │   │   ├── rl_only_omega_3axis.png
│   │   │   └── trajectories.npz
│   │   ├── bc_policy.pth
│   │   ├── expert_data.npz
│   │   └── rl_model.zip
│   │
│   ├── src/
│   │   ├── envs/              # torque_env.py: environment variant for experiments
│   │   ├── expert/            # pid_expert.py: expert in this subproject
│   │   ├── imitation/         # behavioral_cloning.py: local BC implementation
│   │   ├── rl/                # train_rl.py: local RL training
│   │   ├── collect_expert.py
│   │   ├── evaluate.py
│   │   ├── plot_controller.py
│   │   └── plot_results.py
│   │
│   ├── README.md              # Sub-project specific usage and explanation
│   ├── requirements.txt
│   ├── run_all.bat
│   ├── run_all.sh
│   └── start.bat
│
├── extrainfo.txt              # Additional notes / metadata
├── README.md                  # (You are here) Root project README
├── requirements.txt           # Dependencies for main project
├── run_all.bat                # Top-level pipeline runner (Windows)
└── start.bat                  # Quick-start launcher (Windows)
```

---

## 5. Setup & Installation

### 5.1 Clone and Environment

```bash
git clone <this-repo-url>
cd INTELLIGENT_TORQUE_CONTROL_OF_SATELLITES
python -m venv .venv
source .venv/bin/activate   # on Linux/macOS
# or
.\.venv\Scripts\activate    # on Windows
```

### 5.2 Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Ensure your `PYTHONPATH` includes the project root so that `src.*` imports work, for example:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)          # Linux/macOS
set PYTHONPATH=%cd%                           # Windows (PowerShell/CMD)
```

---

## 6. Usage: End-to-End Pipeline

### 6.1 Quick Start (Scripts)

Depending on how the `.bat` / `.sh` scripts are configured, you can typically use:

```bash
# On Linux/macOS
bash run_scripts/collect_expert.sh
bash run_scripts/bc_pretrain.sh
bash run_scripts/rl_train.sh
bash run_scripts/evaluate.sh
```

or the top-level helpers:

```bash
# Windows
start.bat
run_all.bat

# Linux/macOS inside TORQUE_EXPERIMENT or root (depending on script config)
bash run_all.sh
```

These scripts generally:

1. Collect expert data → `outputs/expert_data.npz`
2. Train BC policy → `outputs/bc_policy.pth`
3. Train PPO RL policy → `outputs/rl_model.zip`
4. Run evaluation and/or plotting scripts.

If you modify paths or filenames, ensure that `expert_npz`, `bc_path`, and `save_path` arguments in the Python scripts match.

### 6.2 Manual Step-by-Step

#### 6.2.1 Collect Expert Demonstrations

```bash
python -m src.collect_expert          # or python src/collect_expert.py
# Check that outputs/expert_data.npz is created
```

#### 6.2.2 Train Behavioral Cloning Policy

```bash
python -m src.imitation.behavioral_cloning \
    --data outputs/expert_data.npz \
    --save outputs/bc_policy.pth \
    --epochs 80
```

#### 6.2.3 Train PPO RL Policy (with BC Initialization)

```bash
python -m src.rl.train_rl \
    --timesteps 200000 \
    --bc outputs/bc_policy.pth \
    --out outputs/rl_model.zip
```

To train from scratch (no imitation pretraining), either delete the BC file or pass an invalid path:

```bash
python -m src.rl.train_rl \
    --timesteps 200000 \
    --bc "" \
    --out outputs/rl_model.zip
```

#### 6.2.4 Evaluate Trained Policies

Use `src/evaluate/evaluate_policy.py` (or the scripts in `TORQUE_EXPERIMENT/src/evaluate.py` and `plot_*.py`) to:

- Roll out episodes using:
  - PID expert,
  - BC-only policy,
  - RL-only policy,
  - BC-initialized RL policy.
- Compare convergence time, angular velocity decay, and control effort.

---

## 7. Experiments & Plots (TORQUE_EXPERIMENT/)

The `TORQUE_EXPERIMENT/` folder is a self-contained experiment suite that:

1. Reimplements the environment (as `torque_env.py`) and controllers for reproducible experiments.
2. Provides plotting utilities to visualize and compare controllers.

Example plots:

- `pid_only_omega_3axis.png` — PD expert detumbling performance.
- `bc_only_omega_3axis.png` — Behavioral Cloning-only performance.
- `rl_only_omega_3axis.png` — PPO RL-only performance.
- `all_controllers_omega_3axis.png` — Combined comparison across controllers.

You can run the full experimental pipeline there via:

```bash
cd TORQUE_EXPERIMENT
pip install -r requirements.txt
bash run_all.sh          # or run_all.bat on Windows
```

Consult `TORQUE_EXPERIMENT/README.md` for detailed instructions specific to that sub-project.

---

## 8. Extending the Project

Possible directions for further research:

- **More realistic dynamics**  
  Incorporate full rigid-body rotational dynamics with inertia tensors, external disturbances, and actuator saturation models.

- **Partial observability**  
  Limit the state to sensor-like measurements (e.g., gyro + star tracker) and explore recurrent or belief-state policies.

- **Domain Randomization**  
  Randomize inertia, damping, and initial conditions to study robustness and generalization.

- **Alternative RL algorithms**  
  Experiment with SAC, TD3, or offline RL algorithms using the collected expert data.

- **Multi-objective reward shaping**  
  Include penalties/constraints for pointing accuracy, slew rate limits, or fuel/energy consumption.

---

## 9. Citation (Example)

If you use this codebase or ideas in an academic context, you can cite it generically as:

```
Choudhury, D. (2025). Intelligent Torque Control of Satellites via Hybrid
Imitation and Reinforcement Learning. GitHub repository.
```

(Adjust the citation format to your venue's requirements.)

---

## 10. Contact

For questions or collaboration:

- **Author**: Debopriyo Choudhury
- **Email**: sridebopriyo@gmail.com

Feel free to open an issue or pull request if you extend the framework or find bugs.
