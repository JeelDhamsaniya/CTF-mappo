"""
train.py — Main training entry point for MAPPO Capture The Flag.

Usage:
    python train.py
"""

import os
import sys
import time

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.ctf_env import CaptureTheFlagEnv
from training.trainer import MAPPOTrainer


def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("  MAPPO Capture The Flag — Configuration")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<28}: {v}")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    progress_coef = config.get("progress_reward_coef", 1.0)
    env           = CaptureTheFlagEnv(progress_reward_coef=progress_coef)
    trainer       = MAPPOTrainer(env=env, config=config, device=device)

    t0 = time.time()
    trainer.train(num_episodes=config["num_episodes"])
    elapsed = time.time() - t0

    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\n[Total Training Time] {h:02d}h {m:02d}m {s:02d}s")


if __name__ == "__main__":
    main()
