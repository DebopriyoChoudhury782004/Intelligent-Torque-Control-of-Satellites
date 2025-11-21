# src/envs/torque_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TorqueDetumbleEnv(gym.Env):
    """
    3-axis satellite detumble env (simple linear dynamics).
    State: omega = [wx, wy, wz]
    Action: torque = [tx, ty, tz]
    Dynamics: I * domega/dt = torque - damping * omega
    Return: observation (omega,), reward scalar, terminated, truncated, info
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, I=None, damping=0.1, dt=0.05, max_torque=2.0, max_steps=400, omega_init=5.0, tol=0.02):
        super().__init__()
        # allow different inertias per axis
        if I is None:
            self.I = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.I = np.array(I, dtype=np.float32)
        self.damping = float(damping)
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.max_steps = int(max_steps)
        self.omega_init = float(omega_init)
        self.tol = float(tol)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # randomize initial angular velocity vector (each axis)
        rng = np.random.default_rng(
            seed) if seed is not None else np.random.default_rng()
        self.omega = rng.uniform(-self.omega_init,
                                 self.omega_init, size=(3,)).astype(np.float32)
        self.step_count = 0
        return self.omega.astype(np.float32), {}

    def step(self, action):
        # ensure action is 3-element vector
        torque = np.clip(np.asarray(action, dtype=np.float32).reshape(
            3,), -self.max_torque, self.max_torque)
        # element-wise dynamics
        omega_dot = (torque - self.damping * self.omega) / self.I
        self.omega = (self.omega + omega_dot * self.dt).astype(np.float32)
        self.step_count += 1
        # reward: penalize magnitude of angular velocity and small control cost
        reward = -np.linalg.norm(self.omega) - 0.01 * np.linalg.norm(torque)**2
        terminated = np.all(np.abs(self.omega) < self.tol)
        truncated = self.step_count >= self.max_steps
        return self.omega.astype(np.float32), float(reward), bool(terminated), bool(truncated), {}
