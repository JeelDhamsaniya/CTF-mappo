"""
evaluate.py — Evaluation entry point for MAPPO Capture The Flag.

Usage:
    python evaluate.py                            # auto-finds latest checkpoint
    python evaluate.py --checkpoint checkpoints   # specify checkpoint dir
    python evaluate.py --episodes 50              # run 50 evaluation episodes
    python evaluate.py --terminal 5               # show ASCII board for first 5 episodes
    python evaluate.py --gif 5                    # save animated GIF for first 5 episodes
    python evaluate.py --delay 0.2               # slow down board display (seconds)
"""

import os
import sys
import glob
import argparse

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.ctf_env import CaptureTheFlagEnv
from mappo.agent import MAPPOAgent
from evaluation.evaluator import MAPPOEvaluator


def _find_checkpoints(ckpt_dir: str):
    """Return (team1_path, team2_path) preferring final, else highest episode."""
    t1_final = os.path.join(ckpt_dir, "team1_final.pt")
    t2_final = os.path.join(ckpt_dir, "team2_final.pt")
    if os.path.isfile(t1_final) and os.path.isfile(t2_final):
        return t1_final, t2_final

    t1_files = sorted(glob.glob(os.path.join(ckpt_dir, "team1_ep*.pt")),
                      key=lambda p: int(p.split("ep")[-1].replace(".pt", "")))
    t2_files = sorted(glob.glob(os.path.join(ckpt_dir, "team2_ep*.pt")),
                      key=lambda p: int(p.split("ep")[-1].replace(".pt", "")))

    if not t1_files or not t2_files:
        raise FileNotFoundError(
            f"No checkpoints found in '{ckpt_dir}'.\n"
            "Please run  python train.py  first."
        )
    return t1_files[-1], t2_files[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO agents on CTF.")
    parser.add_argument("--checkpoint", type=str,  default=None,
                        help="Checkpoint directory (default: auto-detect in checkpoints/)")
    parser.add_argument("--episodes",   type=int,  default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--terminal",   type=int,  default=3,
                        help="Show ASCII board for first N episodes (default: 3)")
    parser.add_argument("--gif",        type=int,  default=3,
                        help="Generate animated GIF for first N episodes (default: 3)")
    parser.add_argument("--delay",      type=float, default=0.12,
                        help="Seconds between board display steps (default: 0.12)")
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ckpt_dir  = args.checkpoint or os.path.join(base_dir, "checkpoints")
    t1_path, t2_path = _find_checkpoints(ckpt_dir)
    print(f"[Checkpoint] Team 1 → {t1_path}")
    print(f"[Checkpoint] Team 2 → {t2_path}\n")

    progress_coef = config.get("progress_reward_coef", 1.0)
    env           = CaptureTheFlagEnv(progress_reward_coef=progress_coef)
    obs_shape     = (env.OBS_SIZE,)
    action_size   = env.action_space.n

    team1_agent = MAPPOAgent(
        agent_ids=env.team1_ids, obs_shape=obs_shape, action_size=action_size,
        lr_actor=config["lr_actor"], lr_critic=config["lr_critic"],
    ).to(device)

    team2_agent = MAPPOAgent(
        agent_ids=env.team2_ids, obs_shape=obs_shape, action_size=action_size,
        lr_actor=config["lr_actor"], lr_critic=config["lr_critic"],
    ).to(device)

    team1_agent.load(t1_path)
    team2_agent.load(t2_path)
    print("[Weights loaded successfully]\n")

    # Set eval mode
    for actor in team1_agent.actors.values():
        actor.eval()
    team1_agent.critic.eval()
    for actor in team2_agent.actors.values():
        actor.eval()
    team2_agent.critic.eval()

    evaluator = MAPPOEvaluator(
        env=env,
        team1_agent=team1_agent,
        team2_agent=team2_agent,
        device=device,
    )

    evaluator.evaluate(
        num_episodes=args.episodes,
        render_terminal_first_n=args.terminal,
        render_gif_first_n=args.gif,
        step_delay=args.delay,
    )


if __name__ == "__main__":
    main()
