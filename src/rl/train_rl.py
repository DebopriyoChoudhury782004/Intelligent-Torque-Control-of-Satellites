# src/rl/train_rl.py
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from src.envs.attitude_env import AttitudeDetumbleEnv
from src.imitation.behavioral_cloning import BCNet
import argparse

def copy_weights_by_shape(src_state, dest_model):
    """
    Copy weights from a plain PyTorch state_dict (src_state) into SB3 policy (dest_model.policy).
    We iterate through dest_model.policy.state_dict() keys and replace values when shapes match.
    """
    dest_state = dest_model.policy.state_dict()
    updated = 0
    for d_key, d_val in dest_state.items():
        if d_key in src_state and src_state[d_key].shape == d_val.shape:
            dest_state[d_key] = src_state[d_key].clone()
            updated += 1
    dest_model.policy.load_state_dict(dest_state)
    print(f"[weight_copy] updated {updated} parameters (by exact key match + shape).")
    return updated

def copy_weights_by_order(src_state, dest_model):
    """
    Fallback copying: match parameters by order when exact keys differ.
    """
    dest_state = dest_model.policy.state_dict()
    s_items = [v for k,v in src_state.items()]
    d_keys = list(dest_state.keys())
    updated = 0
    j = 0
    for i, k in enumerate(d_keys):
        if j >= len(s_items):
            break
        if s_items[j].shape == dest_state[k].shape:
            dest_state[k] = s_items[j].clone()
            updated += 1
            j += 1
    dest_model.policy.load_state_dict(dest_state)
    print(f"[weight_copy_order] updated {updated} params (by order+shape).")
    return updated

def make_env():
    return AttitudeDetumbleEnv()

def main(total_timesteps=200_000, bc_path="outputs/bc_policy.pth", save_path="outputs/rl_model.zip"):
    env = DummyVecEnv([make_env])
    # Policy net architecture -- keep it small for CPU
    policy_kwargs = dict(net_arch=[dict(pi=[128,128], vf=[128,128])])
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")

    # Load BC weights into a BCNet and try to copy into SB3
    if bc_path and os.path.exists(bc_path):
        src_state = torch.load(bc_path, map_location="cpu")
        # Try matching keys first
        updated = copy_weights_by_shape(src_state, model)
        if updated == 0:
            # fallback to order-based copy
            updated = copy_weights_by_order(src_state, model)

        print(f"Copied {updated} parameters from BC into SB3 policy (if compatible).")
    else:
        print("No BC policy found; training from scratch.")

    # Train RL
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Saved RL model to", save_path)
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--bc", default="outputs/bc_policy.pth")
    p.add_argument("--out", default="outputs/rl_model.zip")
    args = p.parse_args()
    main(total_timesteps=args.timesteps, bc_path=args.bc, save_path=args.out)
