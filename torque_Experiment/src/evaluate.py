# src/evaluate.py
import argparse
import numpy as np
from src.envs.torque_env import TorqueDetumbleEnv
from src.expert.pid_expert import PIDExpert
from pathlib import Path
from stable_baselines3 import PPO


def run_episode(env, policy_fn, max_steps=400):
    obs, _ = env.reset()
    traj = []
    done = False
    step = 0
    while not done and step < max_steps:
        action = policy_fn(obs)
        traj.append(np.asarray(obs, dtype=np.float32).reshape(3,))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    return np.stack(traj) if len(traj) > 0 else np.zeros((0, 3), dtype=np.float32)


def policy_pid_factory(expert):
    def f(obs):
        return expert.act(obs)
    return f


def policy_bc_factory(bc_path):
    import torch
    from src.imitation.behavioral_cloning import BCNetwork
    device = "cpu"
    net = BCNetwork(obs_dim=3, act_dim=3)
    net.load_state_dict(torch.load(bc_path, map_location=device))
    net.eval()

    def f(obs):
        x = torch.from_numpy(obs.astype("float32")).unsqueeze(0)
        with torch.no_grad():
            a = net(x).cpu().numpy().reshape(-1)
        return a
    return f


def policy_rl_factory(rl_path):
    model = PPO.load(rl_path)

    def f(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return f


def compare(rl_model=None, bc_model=None, episodes=10, outdir="outputs/plots"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = TorqueDetumbleEnv()
    expert = PIDExpert()
    pid_policy = policy_pid_factory(expert)

    trajectories = {"pid": [], "bc": [], "rl": []}
    for ep in range(episodes):
        env.omega_init = float(np.random.uniform(2.0, 6.0))
        trajectories["pid"].append(run_episode(env, pid_policy))
        if bc_model:
            env.omega_init = float(np.random.uniform(2.0, 6.0))
            trajectories["bc"].append(run_episode(
                env, policy_bc_factory(bc_model)))
        if rl_model:
            env.omega_init = float(np.random.uniform(2.0, 6.0))
            trajectories["rl"].append(run_episode(
                env, policy_rl_factory(rl_model)))

    # Save ragged lists as object arrays to preserve variable lengths
    np.savez_compressed(outdir / "trajectories.npz",
                        pid=np.array(trajectories["pid"], dtype=object),
                        bc=np.array(trajectories["bc"], dtype=object),
                        rl=np.array(trajectories["rl"], dtype=object))
    print("Saved trajectories to", outdir / "trajectories.npz")
    print("counts:", {k: len(trajectories[k]) for k in trajectories})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rl", default=None)
    p.add_argument("--bc", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--outdir", default="outputs/plots")
    args = p.parse_args()
    compare(rl_model=args.rl, bc_model=args.bc,
            episodes=args.episodes, outdir=args.outdir)
