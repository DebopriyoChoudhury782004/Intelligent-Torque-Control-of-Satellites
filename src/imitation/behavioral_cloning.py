# src/imitation/behavioral_cloning.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

class BCNet(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3, hidden=(128,128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(expert_npz="outputs/expert_data.npz", epochs=80, batch_size=64, lr=1e-3, save_path="outputs/bc_policy.pth", device="cpu"):
    data = np.load(expert_npz)
    obs = data['obs'].astype(np.float32)
    acts = data['actions'].astype(np.float32)

    ds = TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BCNet(obs_dim=obs.shape[1], act_dim=acts.shape[1], hidden=(128,128)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * xb.shape[0]
        avg = total_loss / len(ds)
        if (ep+1) % 10 == 0:
            print(f"[BC] Epoch {ep+1}/{epochs} avg_loss={avg:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved BC model to {save_path}")
    return model

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/expert_data.npz")
    p.add_argument("--save", default="outputs/bc_policy.pth")
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()
    train(expert_npz=args.data, epochs=args.epochs, save_path=args.save)
