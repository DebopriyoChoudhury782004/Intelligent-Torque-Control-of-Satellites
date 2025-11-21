# src/plot_results.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def normalize_episode(arr, target_channels=3):
    """
    Ensure arr is a 2D array of shape (T, C) where C == target_channels.
    Handles:
      - empty arrays
      - 1D arrays that can be reshaped to (-1, target_channels)
      - 1D arrays that represent a single channel (T,)
      - 2D arrays with fewer/more channels (pad/truncate channels to target_channels)
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros((0, target_channels), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % target_channels == 0:
            return arr.reshape(-1, target_channels).astype(np.float32)
        else:
            # treat as single-channel time series, repeat into channels
            return np.expand_dims(arr.astype(np.float32), axis=1).repeat(target_channels, axis=1)

    if arr.ndim == 2:
        T, C = arr.shape
        if C == target_channels:
            return arr.astype(np.float32)
        elif C < target_channels:
            # pad channels by repeating the last column
            if C == 0:
                pads = np.zeros((T, target_channels), dtype=np.float32)
                return pads
            last = np.expand_dims(arr[:, -1], axis=1)
            pads = np.repeat(last, target_channels - C, axis=1)
            return np.concatenate([arr.astype(np.float32), pads], axis=1)
        else:  # C > target_channels
            return arr[:, :target_channels].astype(np.float32)

    # Fallback
    return np.zeros((0, target_channels), dtype=np.float32)


def avg_ragged(list_of_arrays, target_channels=3):
    """
    Given a list of episode arrays (variable lengths), normalize them to (T,C),
    pad in time (by repeating final step) to the longest T, and return mean,std.
    """
    normed = [normalize_episode(a, target_channels=target_channels)
              for a in list_of_arrays]
    if len(normed) == 0:
        return np.array([]), np.array([])

    maxlen = max(a.shape[0] for a in normed)
    if maxlen == 0:
        return np.zeros((0, target_channels)), np.zeros((0, target_channels))

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

    stacked = np.stack(padded, axis=0)  # shape (N, maxlen, C)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std


def plot_trajs(npz_path="outputs/plots/trajectories.npz", outdir="outputs/plots"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(npz_path)
    if not npz_path.exists():
        print("No trajectories file found at", npz_path)
        return

    data = np.load(npz_path, allow_pickle=True)

    # collect controllers present
    controllers = []
    for key in ("pid", "bc", "rl"):
        if key in data.files:
            controllers.append((key, list(data[key])))

    if not controllers:
        print("No controllers found in", npz_path)
        return

    # Use a safe built-in style (some environment builds may not have seaborn styles)
    plt.style.use("bmh")

    fig, ax = plt.subplots(figsize=(11, 5))

    axis_names = ["wx", "wy", "wz"]
    axis_colors = {"wx": "tab:blue", "wy": "tab:orange", "wz": "tab:green"}
    linestyles = {"pid": "-", "bc": "--", "rl": ":"}
    labels = {"pid": "PID", "bc": "BC", "rl": "RL"}

    plotted_any = False
    for i, axis in enumerate(axis_names):
        for key, arrs in controllers:
            if len(arrs) == 0:
                continue
            mean, std = avg_ragged(arrs, target_channels=3)
            if mean.size == 0:
                continue
            T = mean.shape[0]
            t = np.arange(T)
            ax.plot(t, mean[:, i],
                    label=f"{axis} {labels[key]}",
                    linestyle=linestyles.get(key, "-"),
                    color=axis_colors[axis])
            ax.fill_between(t, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                            alpha=0.12, color=axis_colors[axis])
            plotted_any = True

    if not plotted_any:
        print("No valid episode data found to plot.")
        return

    ax.set_xlabel("timestep")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("Angular velocity convergence (mean ± std)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    out_file = outdir / "comparison_omega_3axis.png"
    plt.savefig(out_file, dpi=200)
    print("Saved plot to", out_file)
    try:
        plt.show()
    except Exception:
        pass


def inspect_npz(npz_path="outputs/plots/trajectories.npz", max_print=3):
    """
    Helper to print shapes of the first few episodes for debugging.
    """
    if not Path(npz_path).exists():
        print("File not found:", npz_path)
        return
    d = np.load(npz_path, allow_pickle=True)
    for k in d.files:
        arrs = list(d[k])
        print(f"{k}: {len(arrs)} episodes")
        for i, a in enumerate(arrs[:max_print]):
            a = np.asarray(a)
            print("  ", k, "ep", i, "shape:", a.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", default="outputs/plots/trajectories.npz", help="Path to trajectories npz")
    parser.add_argument("--outdir", default="outputs/plots",
                        help="Output directory for plots")
    parser.add_argument("--inspect", action="store_true",
                        help="Only inspect npz and print episode shapes")
    args = parser.parse_args()

    if args.inspect:
        inspect_npz(args.npz)
    else:
        plot_trajs(npz_path=args.npz, outdir=args.outdir)
