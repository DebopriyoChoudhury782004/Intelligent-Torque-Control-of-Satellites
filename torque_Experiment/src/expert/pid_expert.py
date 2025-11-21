# src/expert/pid_expert.py
import numpy as np


class PIDExpert:
    """
    Simple per-axis PD controller (no integral).
    Returns 3-element torque array.
    """

    def __init__(self, Kp=None, Kd=None, max_torque=2.0):
        # default gains per axis
        self.Kp = np.array(Kp if Kp is not None else [
                           2.0, 2.0, 2.0], dtype=np.float32)
        self.Kd = np.array(Kd if Kd is not None else [
                           0.5, 0.5, 0.5], dtype=np.float32)
        self.prev_omega = None
        self.max_torque = float(max_torque)

    def reset(self):
        self.prev_omega = None

    def act(self, obs):
        # obs is 3-element array-like
        omega = np.asarray(obs).reshape(3,)
        if self.prev_omega is None:
            domega = np.zeros_like(omega)
        else:
            domega = omega - self.prev_omega
        self.prev_omega = omega.copy()
        torque = - self.Kp * omega - self.Kd * domega
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return torque.astype(np.float32)
