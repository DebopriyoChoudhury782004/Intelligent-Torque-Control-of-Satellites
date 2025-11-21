# src/utils/replay_buffer.py
import numpy as np

class SimpleBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []

    def add(self, o, a):
        self.obs.append(o.copy())
        self.actions.append(a.copy())

    def save(self, path):
        np.savez_compressed(path, obs=np.array(self.obs), actions=np.array(self.actions))

    @staticmethod
    def load(path):
        data = np.load(path)
        return data['obs'], data['actions']
