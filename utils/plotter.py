import os
import glob
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Smoothing helper ──────────────────────────────────────────────────────────

def _smooth(data: List[float], window: int = 50) -> np.ndarray:
    arr = np.array(data, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    pad      = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


# ── Training plot ─────────────────────────────────────────────────────────────

def plot_training_results(
    team1_rewards: List[float],
    team2_rewards: List[float],
    team1_wins: List[float],
    team2_wins: List[float],
    policy_losses: List[float],
    value_losses: List[float],
    save_path: str,
):
    """
    3-panel training figure:
      Panel 1 — smoothed episode rewards (rolling window 50)
      Panel 2 — rolling win rate over last 100 episodes (%)
      Panel 3 — smoothed policy loss and value loss
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    eps = np.arange(1, len(team1_rewards) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MAPPO CTF — Training Results", fontsize=14, fontweight="bold")

    # Panel 1: Rewards
    ax = axes[0]
    ax.plot(eps, team1_rewards, alpha=0.15, color="#1565C0", linewidth=0.5)
    ax.plot(eps, team2_rewards, alpha=0.15, color="#B71C1C", linewidth=0.5)
    ax.plot(eps, _smooth(team1_rewards), color="#1565C0", linewidth=1.8, label="Team 1")
    ax.plot(eps, _smooth(team2_rewards), color="#B71C1C", linewidth=1.8, label="Team 2")
    ax.set_title("Episode Rewards (smoothed)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Win rates
    ax = axes[1]
    ax.plot(eps, team1_wins, color="#1565C0", linewidth=1.8, label="Team 1 Win %")
    ax.plot(eps, team2_wins, color="#B71C1C", linewidth=1.8, label="Team 2 Win %")
    ax.fill_between(eps, team1_wins, alpha=0.1, color="#1565C0")
    ax.fill_between(eps, team2_wins, alpha=0.1, color="#B71C1C")
    ax.set_title("Win Rate (last 100 eps)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Losses
    ax = axes[2]
    ax.plot(eps, policy_losses, alpha=0.15, color="#6A1B9A", linewidth=0.5)
    ax.plot(eps, value_losses,  alpha=0.15, color="#E65100", linewidth=0.5)
    ax.plot(eps, _smooth(policy_losses), color="#6A1B9A", linewidth=1.8, label="Policy Loss")
    ax.plot(eps, _smooth(value_losses),  color="#E65100", linewidth=1.8, label="Value Loss")
    ax.set_title("Training Losses (smoothed)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Evaluation plot ───────────────────────────────────────────────────────────

def plot_eval_results(
    team1_rewards: List[float],
    team2_rewards: List[float],
    episode_lengths: List[int],
    team1_wins: int,
    team2_wins: int,
    draws: int,
    save_path: str,
):
    """
    3-panel evaluation figure:
      Panel 1 — pie chart of win distribution
      Panel 2 — per-episode rewards for both teams
      Panel 3 — episode lengths with mean line
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    eps = np.arange(1, len(team1_rewards) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MAPPO CTF — Evaluation Results", fontsize=14, fontweight="bold")

    # Panel 1: Win pie
    ax = axes[0]
    labels, sizes, colors = [], [], []
    if team1_wins > 0:
        labels.append(f"Team 1\n({team1_wins})")
        sizes.append(team1_wins)
        colors.append("#1565C0")
    if team2_wins > 0:
        labels.append(f"Team 2\n({team2_wins})")
        sizes.append(team2_wins)
        colors.append("#B71C1C")
    if draws > 0:
        labels.append(f"Draw\n({draws})")
        sizes.append(draws)
        colors.append("#90A4AE")
    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    ax.set_title("Win Distribution", fontweight="bold")

    # Panel 2: Per-episode rewards
    ax = axes[1]
    ax.plot(eps, team1_rewards, color="#1565C0", linewidth=1.0, marker="o",
            markersize=2, label="Team 1", alpha=0.8)
    ax.plot(eps, team2_rewards, color="#B71C1C", linewidth=1.0, marker="o",
            markersize=2, label="Team 2", alpha=0.8)
    ax.axhline(np.mean(team1_rewards), color="#1565C0", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"T1 mean={np.mean(team1_rewards):.1f}")
    ax.axhline(np.mean(team2_rewards), color="#B71C1C", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"T2 mean={np.mean(team2_rewards):.1f}")
    ax.set_title("Per-Episode Rewards", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Episode lengths
    ax = axes[2]
    mean_len = float(np.mean(episode_lengths))
    ax.bar(eps, episode_lengths, color="#26A69A", alpha=0.6, width=0.8)
    ax.axhline(mean_len, color="#00695C", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_len:.1f} steps")
    ax.set_title("Episode Lengths", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── GIF generator ─────────────────────────────────────────────────────────────

def create_episode_gif(frames_dir: str, output_path: str, fps: int = 4):
    """
    Stitch all step_XXXX.png files in frames_dir into an animated GIF.

    Args:
        frames_dir  : directory containing step_XXXX.png frame images
        output_path : path for the output .gif file
        fps         : frames per second
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [GIF] Pillow not installed — skipping GIF generation. Run: pip install Pillow")
        return

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "step_*.png")))
    if not frame_files:
        print(f"  [GIF] No frames found in {frames_dir}")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    frames = [Image.open(f).convert("RGBA") for f in frame_files]
    duration_ms = max(1, int(1000 / fps))

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  [GIF] {len(frames)} frames → {output_path}")
