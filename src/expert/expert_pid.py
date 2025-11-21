# src/expert/expert_pid.py
import numpy as np

class ExpertPID:
    def __init__(self, Kp=2.0, Kd=0.8, max_torque=1.0):
        self.Kp = Kp
        self.Kd = Kd
        self.max_torque = max_torque

    def act(self, obs):
        # obs: [w_x, w_y, w_z, angle_x, angle_y, angle_z]
        w = obs[:3]
        angle = obs[3:]
        torque = - self.Kp * angle - self.Kd * w
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return torque.astype(np.float32)
