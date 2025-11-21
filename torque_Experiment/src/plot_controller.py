# src/plot_controller.py
"""
Plot PID, BC, RL, or all controllers from outputs/plots/trajectories.npz.

Examples:
  python -m src.plot_controller --which rl
  python -m src.plot_controller --which bc --raw
  python -m src.plot_controller --which all
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def ensure_2d(arr, target_channels=3):
    a = np.asarray(arr)
    if a.size == 0:
        return np.zeros((0, target_channels), dtype=np.float32)
    if a.ndim == 1:
        if a.size % target_channels == 0:
            return a.reshape(-1, target_channels).astype(np.float32)
        return np.expand_dims(a.astype(np.float32), 1).repeat(target_channels, axis=1)
    if a.ndim == 2:
        T, C = a.shape
        if C == target_channels:
            return a.astype(np.float32)
        if C < target_channels:
            if C == 0:
                return np.zeros((T, target_channels), dtype=np.float32)
            last = np.expand_dims(a[:, -1], 1)
            pads = np.repeat(last, target_channels - C, axis=1)
            return np.concatenate([a.astype(np.float32), pads], axis=1)
        return a[:, :target_channels].astype(np.float32)
    return np.zeros((0, target_channels), dtype=np.float32)


def pad_and_stats(episodes, target_channels=3):
    normed = [ensure_2d(e, target_channels) for e in episodes]
    if len(normed) == 0:
        return None, None, 0
    maxlen = max(e.shape[0] for e in normed)
    if maxlen == 0:
        return None, None, len(normed)
    padded = []
    for a in normed:
        T, C = a.shape
        if T == 0:
            padded_a = np.zeros((maxlen, target_channels), dtype=np.float32)
        elif T < maxlen:
            last = a[-1:, :]
            reps = np.repeat(last, maxlen - T, axis=0)
            padded_a = np.concatenate([a, reps], axis=0)
        else:
            padded_a = a[:maxlen, :]
        padded.append(padded_a)
    stacked = np.stack(padded, axis=0)  # (N, maxlen, C)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std, len(normed)


def plot_single(mean, std, label_prefix, outpath, raw_episodes=None):
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(11, 5))
    axis_names = ["wx", "wy", "wz"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    T = mean.shape[0]
    t = np.arange(T)
    # optionally plot raw episodes faintly
    if raw_episodes is not None:
        for ep in raw_episodes:
            ep2 = ensure_2d(ep)
            if ep2.shape[0] == 0:
                continue
            # pad/truncate to T
            if ep2.shape[0] < T:
                last = ep2[-1:, :]
                reps = np.repeat(last, T-ep2.shape[0], axis=0)
                ep_pad = np.concatenate([ep2, reps], axis=0)
            else:
                ep_pad = ep2[:T, :]
            for i in range(3):
                ax.plot(t, ep_pad[:, i], color=colors[i],
                        alpha=0.12, linewidth=0.8)

    for i, name in enumerate(axis_names):
        ax.plot(
            t, mean[:, i], label=f"{name} {label_prefix}", color=colors[i], linewidth=1.8)
        ax.fill_between(t, mean[:, i]-std[:, i], mean[:, i] +
                        std[:, i], alpha=0.14, color=colors[i])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title(f"{label_prefix} — angular velocity (mean ± std)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print("Saved plot to:", outpath)
    try:
        plt.show()
    except Exception:
        pass


def plot_all(controllers_dict, outpath, raw=False):
    # controllers_dict: {"pid": list_of_eps, "bc":..., "rl":...}
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(12, 6))
    axis_names = ["wx", "wy", "wz"]
    axis_colors = {"wx": "tab:blue", "wy": "tab:orange", "wz": "tab:green"}
    linestyles = {"pid": "-", "bc": "--", "rl": ":"}
    labels = {"pid": "PID", "bc": "BC", "rl": "RL"}

    any_plotted = False
    for i, axis in enumerate(axis_names):
        for key in ("pid", "bc", "rl"):
            eps = controllers_dict.get(key, [])
            if not eps or len(eps) == 0:
                continue
            mean, std, n = pad_and_stats(eps)
            if mean is None:
                continue
            T = mean.shape[0]
            t = np.arange(T)
            ax.plot(t, mean[:, i], label=f"{axis} {labels[key]}", linestyle=linestyles.get(
                key, "-"), color=axis_colors[axis])
            ax.fill_between(t, mean[:, i]-std[:, i], mean[:, i] +
                            std[:, i], alpha=0.12, color=axis_colors[axis])
            any_plotted = True
            if raw:
                # overlay raw episodes faintly
                for ep in eps:
                    ep2 = ensure_2d(ep)
                    if ep2.shape[0] == 0:
                        continue
                    if ep2.shape[0] < T:
                        last = ep2[-1:, :]
                        reps = np.repeat(last, T-ep2.shape[0], axis=0)
                        ep_pad = np.concatenate([ep2, reps], axis=0)
                    else:
                        ep_pad = ep2[:T, :]
                    ax.plot(
                        np.arange(T), ep_pad[:, i], color=axis_colors[axis], alpha=0.06, linewidth=0.6)
    if not any_plotted:
        print("No data found for any controller. Aborting.")
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("Controllers comparison — angular velocity (mean ± std)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print("Saved plot to:", outpath)
    try:
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["pid", "bc", "rl", "all"],
                        default="all", help="Which controller(s) to plot")
    parser.add_argument(
        "--npz", default="outputs/plots/trajectories.npz", help="Trajectories npz")
    parser.add_argument("--outdir", default="outputs/plots", help="Output dir")
    parser.add_argument("--raw", action="store_true",
                        help="Overlay raw episode traces")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        print("Trajectories file not found:", npz_path)
        return

    data = np.load(npz_path, allow_pickle=True)
    controllers = {}
    for key in ("pid", "bc", "rl"):
        if key in data.files:
            controllers[key] = list(data[key])
        else:
            controllers[key] = []

    if args.which == "all":
        out_file = outdir / "all_controllers_omega_3axis.png"
        plot_all(controllers, out_file, raw=args.raw)
        return

    # single controller
    eps = controllers.get(args.which, [])
    if len(eps) == 0:
        print(f"No episodes found for {args.which}. Nothing to plot.")
        return
    mean, std, n = pad_and_stats(eps)
    if mean is None:
        print(f"No valid episodes for {args.which}.")
        return
    label = {"pid": "PID", "bc": "BC", "rl": "RL"}[args.which]
    out_file = outdir / f"{args.which}_only_omega_3axis.png"
    raw_eps = eps if args.raw else None
    plot_single(mean, std, label, out_file, raw_episodes=raw_eps)


if __name__ == "__main__":
    main()
