import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.ctf_env import CaptureTheFlagEnv
from mappo.agent import MAPPOAgent
from utils.plotter import plot_eval_results, create_episode_gif


class MAPPOEvaluator:
    """
    Evaluates trained MAPPO agents with near-greedy (Boltzmann, T=0.1) action selection.
    Supports:
      - Terminal ASCII board display (print_board)
      - Matplotlib frame-saving → animated GIF per episode
    """

    def __init__(
        self,
        env: CaptureTheFlagEnv,
        team1_agent: MAPPOAgent,
        team2_agent: MAPPOAgent,
        device,
    ):
        self.env          = env
        self.team1_agent  = team1_agent
        self.team2_agent  = team2_agent
        self.device       = device

    # ── Single episode ────────────────────────────────────────────────────────

    def run_episode(
        self,
        render_terminal: bool = False,
        render_gif: bool = False,
        gif_dir: str = "logs/renders",
        ep_idx: int = 1,
        step_delay: float = 0.12,
    ) -> dict:
        """
        Run one evaluation episode.

        Args:
            render_terminal : print ASCII board to stdout each step
            render_gif      : save matplotlib frames for GIF generation
            gif_dir         : directory to save frame PNG files
            ep_idx          : episode index (used for labelling GIF frames)
            step_delay      : seconds to pause between displayed steps
        Returns:
            dict with team1_reward, team2_reward, winner, episode_length, scores
        """
        obs_dict    = self.env.reset()
        total_rew   = {aid: 0.0 for aid in self.env.agent_ids}
        ep_len      = 0
        frames_dir  = os.path.join(gif_dir, f"ep_{ep_idx:04d}")

        if render_gif:
            os.makedirs(frames_dir, exist_ok=True)

        if render_terminal:
            print(f"\n{'='*52}")
            print(f"  Episode {ep_idx} — watch the agents play!")
            print(f"  Team1 agents: A (team1_0)  B (team1_1)")
            print(f"  Team2 agents: C (team2_0)  D (team2_1)")
            print(f"  Legend:  F1=Team1 flag  F2=Team2 flag  ##=obstacle")
            print(f"{'='*52}")
            self.env.print_board(step_count=0, delay=step_delay)

        for step in range(200):
            actions = {}
            for aid in self.env.team1_ids:
                actions[aid] = self.team1_agent.select_action_greedy(aid, obs_dict[aid], temperature=0.1)
            for aid in self.env.team2_ids:
                actions[aid] = self.team2_agent.select_action_greedy(aid, obs_dict[aid], temperature=0.1)

            next_obs, rewards, dones, info = self.env.step(actions)
            done   = dones["__all__"]
            ep_len += 1

            for aid in self.env.agent_ids:
                total_rew[aid] += rewards[aid]

            if render_terminal:
                self.env.print_board(step_count=step + 1, delay=step_delay)
                # Show step rewards inline
                t1_step = sum(rewards[a] for a in self.env.team1_ids)
                t2_step = sum(rewards[a] for a in self.env.team2_ids)
                print(f"  Step reward → T1: {t1_step:+.1f}   T2: {t2_step:+.1f}")

            if render_gif:
                fig = self.env.render(step_count=step + 1)
                fig.savefig(os.path.join(frames_dir, f"step_{step + 1:04d}.png"), dpi=72)
                plt.close(fig)

            obs_dict = next_obs
            if done:
                break

        scores = self.env.scores
        if scores["team1"] > scores["team2"]:
            winner = "team_1"
        elif scores["team2"] > scores["team1"]:
            winner = "team_2"
        else:
            winner = "draw"

        if render_terminal:
            print(f"\n  ★  Episode {ep_idx} result: {winner.upper()}")
            print(f"     T1 total reward: {sum(total_rew[a] for a in self.env.team1_ids):.1f}")
            print(f"     T2 total reward: {sum(total_rew[a] for a in self.env.team2_ids):.1f}")
            print(f"     Steps: {ep_len}")

        if render_gif:
            gif_path = os.path.join(gif_dir, f"episode_{ep_idx:04d}.gif")
            create_episode_gif(frames_dir, gif_path)
            print(f"  [GIF] Saved → {gif_path}")

        return {
            "team1_reward":   sum(total_rew[a] for a in self.env.team1_ids),
            "team2_reward":   sum(total_rew[a] for a in self.env.team2_ids),
            "winner":         winner,
            "episode_length": ep_len,
            "scores":         dict(scores),
        }

    # ── Multi-episode evaluation ──────────────────────────────────────────────

    def evaluate(
        self,
        num_episodes: int = 100,
        render_terminal_first_n: int = 3,
        render_gif_first_n: int = 3,
        step_delay: float = 0.12,
    ):
        """
        Run num_episodes evaluation episodes and print a summary table.

        Args:
            num_episodes            : total episodes to run
            render_terminal_first_n : show ASCII board for the first N episodes
            render_gif_first_n      : generate animated GIF for the first N episodes
            step_delay              : seconds between displayed steps
        """
        print("\n" + "=" * 60)
        print("  MAPPO CTF — Evaluation")
        print(f"  Episodes: {num_episodes}")
        print(f"  Terminal render: first {render_terminal_first_n} episodes")
        print(f"  GIF render: first {render_gif_first_n} episodes")
        print("=" * 60)

        t1_rewards: List[float] = []
        t2_rewards: List[float] = []
        ep_lengths: List[int]   = []
        t1_wins = t2_wins = draws = 0

        for ep in range(1, num_episodes + 1):
            do_terminal = ep <= render_terminal_first_n
            do_gif      = ep <= render_gif_first_n

            result = self.run_episode(
                render_terminal=do_terminal,
                render_gif=do_gif,
                gif_dir="logs/renders",
                ep_idx=ep,
                step_delay=step_delay,
            )

            t1_rewards.append(result["team1_reward"])
            t2_rewards.append(result["team2_reward"])
            ep_lengths.append(result["episode_length"])

            if result["winner"] == "team_1":
                t1_wins += 1
                wstr = "T1 ✓"
            elif result["winner"] == "team_2":
                t2_wins += 1
                wstr = "T2 ✓"
            else:
                draws += 1
                wstr = "draw"

            if not do_terminal:
                print(
                    f"  Ep {ep:4d} | {wstr:5} | "
                    f"T1 Rwd:{result['team1_reward']:8.1f}  "
                    f"T2 Rwd:{result['team2_reward']:8.1f} | "
                    f"Len:{result['episode_length']:4d}"
                )

        # Summary
        total = num_episodes
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  {'Metric':<28} {'Team 1':>10} {'Team 2':>10}")
        print(f"  {'-'*48}")
        print(f"  {'Wins':<28} {t1_wins:>10}  {t2_wins:>10}")
        print(f"  {'Draws':<28} {draws:>10}")
        print(f"  {'Win Rate (%)':<28} {t1_wins/total*100:>10.1f}  {t2_wins/total*100:>10.1f}")
        print(f"  {'Avg Reward':<28} {np.mean(t1_rewards):>10.2f}  {np.mean(t2_rewards):>10.2f}")
        print(f"  {'Avg Episode Length':<28} {np.mean(ep_lengths):>10.1f}")
        print("=" * 60)

        os.makedirs("logs", exist_ok=True)
        plot_eval_results(
            team1_rewards=t1_rewards, team2_rewards=t2_rewards,
            episode_lengths=ep_lengths,
            team1_wins=t1_wins, team2_wins=t2_wins, draws=draws,
            save_path=os.path.join("logs", "eval_results.png"),
        )
        print("\n[Plot] Evaluation chart saved → logs/eval_results.png")
