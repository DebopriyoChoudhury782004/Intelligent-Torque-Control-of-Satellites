# src/imitation/behavioral_cloning.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCNetwork(nn.Module):
    def __init__(self, obs_dim=3, act_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x):
        return self.net(x)


def train(data_path, save_path, epochs=50, batch_size=64, lr=1e-3, device="cpu"):
    print(f"Loading data from {data_path} ...")
    d = np.load(data_path)
    obs = d["obs"].astype(np.float32)
    acts = d["acts"].astype(np.float32)
    print(f"obs shape: {obs.shape}, acts shape: {acts.shape}")
    dataset = TensorDataset(torch.from_numpy(obs), torch.from_numpy(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BCNetwork(obs_dim=obs.shape[1], act_dim=acts.shape[1])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
        avg = total / len(dataset)
        if ep % max(1, epochs//10) == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep}/{epochs} loss={avg:.6f}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved BC model to", save_path)
    return model


def parse_args_and_run():
    p = argparse.ArgumentParser(description="Behavioral cloning trainer")
    p.add_argument("--data", default="outputs/expert_data.npz",
                   help="Path to expert .npz (obs, acts)")
    p.add_argument("--save", default="outputs/bc_policy.pth",
                   help="Where to save BC weights")
    p.add_argument("--epochs", type=int, default=80,
                   help="Number of training epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--device", default="cpu", help="Torch device")
    args = p.parse_args()
    train(args.data, args.save, epochs=args.epochs,
          batch_size=args.batch, lr=args.lr, device=args.device)


if __name__ == "__main__":
    parse_args_and_run()
