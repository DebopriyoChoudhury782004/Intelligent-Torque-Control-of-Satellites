# src/rl/train_rl.py
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
from src.envs.torque_env import TorqueDetumbleEnv
from src.imitation.behavioral_cloning import BCNetwork
import numpy as np


def make_env():
    return TorqueDetumbleEnv()


def load_bc_into_sb3(model, bc_path):
    # best-effort weight copy by matching keys / shapes
    import torch
    bc = BCNetwork(obs_dim=3, act_dim=3)
    sd = torch.load(bc_path, map_location="cpu")
    bc.load_state_dict(sd)
    try:
        model_state = model.policy.state_dict()
        bc_state = bc.state_dict()
        copied = 0
        for k, v in bc_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                copied += 1
        model.policy.load_state_dict(model_state)
        print(f"Loaded BC weights into SB3 policy (copied {copied} tensors).")
    except Exception as e:
        print("Warning: could not copy all weights into SB3 policy:", e)


def train(timesteps=200_000, bc_path=None, save_path="outputs/rl_model.zip", n_envs=4):
    env = make_vec_env(make_env, n_envs=n_envs)
    model = PPO("MlpPolicy", env, verbose=1)
    if bc_path is not None:
        load_bc_into_sb3(model, bc_path)
    model.learn(total_timesteps=timesteps)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print("Saved RL model to", save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--bc", default=None)
    p.add_argument("--out", default="outputs/rl_model.zip")
    p.add_argument("--n_envs", type=int, default=4)
    args = p.parse_args()
    train(timesteps=args.timesteps, bc_path=args.bc,
          save_path=args.out, n_envs=args.n_envs)
