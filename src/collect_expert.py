# src/collect_expert.py
import argparse
import numpy as np
from src.envs.attitude_env import AttitudeDetumbleEnv
from src.expert.expert_pid import ExpertPID
from src.utils.replay_buffer import SimpleBuffer

def main(episodes: int = 50, out_path: str = "outputs/expert_data.npz", seed: int = 0):
    # create env and expert
    env = AttitudeDetumbleEnv()
    np.random.seed(seed)
    buffer = SimpleBuffer()
    expert = ExpertPID(Kp=3.0, Kd=1.2)

    for ep in range(episodes):
        # gymnasium reset returns (obs, info)
        obs, _ = env.reset()
        done = False
        while not done:
            action = expert.act(obs)          # expert returns a numpy array action
            buffer.add(obs, action)           # store (obs, action)
            # gymnasium step returns (obs, reward, terminated, truncated, info)
            next_obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            obs = next_obs

        # optional: print progress
        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            print(f"Collected episode {ep + 1}/{episodes}  buffer_size={len(buffer.obs)}")

    # save dataset
    buffer.save(out_path)
    print(f"Saved expert data to {out_path}. total samples: {len(buffer.obs)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50, help="Number of expert episodes to collect")
    p.add_argument("--out", dest="out_path", type=str, default="outputs/expert_data.npz", help="Output .npz path")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()
    main(episodes=args.episodes, out_path=args.out_path, seed=args.seed)
