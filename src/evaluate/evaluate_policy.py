# src/evaluate/evaluate_policy.py
import numpy as np
from stable_baselines3 import PPO
from src.envs.attitude_env import AttitudeDetumbleEnv

def eval_sb3(model_path: str, episodes: int = 50):
    env = AttitudeDetumbleEnv()
    model = PPO.load(model_path, device="cpu")

    rewards = []
    for ep in range(episodes):
        # gymnasium reset -> (obs, info)
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # model.predict expects a numpy array (not a tuple)
            action, _ = model.predict(obs, deterministic=True)
            # gymnasium step -> (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            obs = next_obs
            ep_ret += float(reward)

        rewards.append(ep_ret)
        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            print(f"Eval episode {ep+1}/{episodes}: return={ep_ret:.3f}")

    rewards = np.array(rewards, dtype=np.float32)
    mean = float(rewards.mean())
    std = float(rewards.std())
    print(f"Evaluation over {episodes} episodes -> mean_return: {mean:.3f}, std: {std:.3f}")
    return mean, std

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to SB3 saved model (.zip)")
    p.add_argument("--episodes", type=int, default=50)
    args = p.parse_args()
    eval_sb3(args.model, args.episodes)
