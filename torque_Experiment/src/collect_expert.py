# src/collect_expert.py
import numpy as np
import argparse
from pathlib import Path
from tqdm import trange

from src.envs.torque_env import TorqueDetumbleEnv
from src.expert.pid_expert import PIDExpert


def collect(expert, env, episodes=200, out_path="outputs/expert_data.npz", seed=0):
    """
    Roll out the PID expert policy and save (obs, action) pairs to a dataset.
    Observations and actions are 3-element vectors (wx,wy,wz) and (tx,ty,tz).
    """
    obs_list = []
    act_list = []
    rng = np.random.default_rng(seed)

    for ep in trange(episodes, desc="Collecting expert trajectories"):
        # Randomize initial condition for each episode (magnitude)
        env.omega_init = float(rng.uniform(2.0, 6.0))
        obs, _ = env.reset()
        expert.reset()
        done = False

        while not done:
            action = expert.act(obs)
            # ensure consistent shapes (3,)
            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(3,))
            act_list.append(np.asarray(action, dtype=np.float32).reshape(3,))

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

    obs_arr = np.vstack(obs_list) if len(
        obs_list) > 0 else np.zeros((0, 3), dtype=np.float32)
    act_arr = np.vstack(act_list) if len(
        act_list) > 0 else np.zeros((0, 3), dtype=np.float32)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs_arr, acts=act_arr)
    print(f"Saved {len(obs_arr)} samples to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200,
                   help="Number of expert episodes to collect")
    p.add_argument("--out", type=str,
                   default="outputs/expert_data.npz", help="Output .npz path")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()

    env = TorqueDetumbleEnv()
    expert = PIDExpert()
    collect(expert, env, episodes=args.episodes,
            out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
