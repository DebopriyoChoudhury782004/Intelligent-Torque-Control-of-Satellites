import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AttitudeDetumbleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.1, max_steps=500):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.w = None
        self.angle = None
        self.max_torque = 1.0

        # observation = [wx, wy, wz, ax, ay, az]
        obs_low = -np.inf * np.ones(6, dtype=np.float32)
        obs_high = np.inf * np.ones(6, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # action = torque vector (3,)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.w = np.random.uniform(-1.5, 1.5, size=3).astype(np.float32)
        self.angle = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([self.w, self.angle]).astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque).astype(np.float32)

        # dynamics
        damping = -0.05 * self.w
        self.w = self.w + (action + damping) * self.dt
        self.angle = self.angle + self.w * self.dt
        self.step_count += 1

        reward = -(np.linalg.norm(self.w) + 0.01 * np.linalg.norm(action))

        terminated = np.linalg.norm(self.w) < 0.02
        truncated = self.step_count >= self.max_steps

        obs = np.concatenate([self.w, self.angle]).astype(np.float32)
        info = {"angular_velocity_norm": float(np.linalg.norm(self.w))}

        return obs, reward, terminated, truncated, info
