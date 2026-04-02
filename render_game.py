"""
render_game.py — Live graphical game viewer for trained MAPPO CTF agents.

Opens a real matplotlib window showing the game being played with colors,
animated agent movement, flags, obstacles, and live score tracking.

Usage:
    python render_game.py                        # 5 episodes, default speed
    python render_game.py --episodes 3           # watch 3 episodes
    python render_game.py --speed 0.4            # seconds per step
    python render_game.py --checkpoint ckpt_dir  # custom checkpoint path
"""

import os
import sys
import glob
import argparse
import time

# ── Set interactive backend BEFORE any other matplotlib imports ────────────────
import matplotlib
for _backend in ["MacOSX", "TkAgg", "Qt5Agg", "WXAgg"]:
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import torch
from env.ctf_env import CaptureTheFlagEnv
from mappo.agent import MAPPOAgent

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d0d1a",
    "grid_bg":   "#111128",
    "cell":      "#1a1a35",
    "grid_line": "#252545",
    "obstacle":  "#3a3a5a",
    "t1_fill":   "#42aaf5",    # Team 1 agent body (blue)
    "t1_edge":   "#1565c0",    # Team 1 agent border
    "t1_glow":   "#4fc3f7",
    "t2_fill":   "#66bb6a",    # Team 2 agent body (green)
    "t2_edge":   "#2e7d32",    # Team 2 agent border
    "t2_glow":   "#a5d6a7",
    "flag1":     "#1565c0",    # Team 1 flag (blue)
    "flag2":     "#c62828",    # Team 2 flag (red)
    "text":      "#e8e8ff",
    "sub_text":  "#7777aa",
    "gold":      "#ffd700",
    "panel_bg":  "#0a0a1e",
}


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _rounded_rect(ax, x, y, w, h, r=0.08, **kw):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}", **kw
    ))


def _is_defending(env, aid):
    """Return True if agent is in own defense zone and an enemy is also in the zone."""
    own_flag   = env.TEAM1_FLAG_POS if "team1" in aid else env.TEAM2_FLAG_POS
    enemy_ids  = env.team2_ids if "team1" in aid else env.team1_ids
    in_zone = env._manhattan(env.positions.get(aid, (-99,-99)), own_flag) <= env.DEFENSE_ZONE_RADIUS
    enemy_near = any(env._manhattan(env.positions.get(e, (-99,-99)), own_flag) <= env.DEFENSE_ZONE_RADIUS + 1
                     for e in enemy_ids)
    return in_zone and enemy_near


def draw_board(ax, env: CaptureTheFlagEnv):
    """Render the 10x10 game board on ax."""
    ax.cla()
    G = env.GRID_SIZE
    ax.set_facecolor(C["grid_bg"])
    ax.set_xlim(-0.55, G - 0.45)
    ax.set_ylim(-0.55, G - 0.45)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Cell tiles ────────────────────────────────────────────────────────────
    for r in range(G):
        for c in range(G):
            _rounded_rect(ax, c - 0.46, r - 0.46, 0.92, 0.92, r=0.06,
                          facecolor=C["cell"], edgecolor=C["grid_line"],
                          linewidth=0.7, zorder=1)

    # ── Obstacles ─────────────────────────────────────────────────────────────
    for r, c in env.OBSTACLES:
        _rounded_rect(ax, c - 0.46, r - 0.46, 0.92, 0.92, r=0.04,
                      facecolor=C["obstacle"], edgecolor="#5a5a7e",
                      linewidth=1.2, zorder=2)
        ax.text(c, r, "▓", ha="center", va="center",
                fontsize=14, color="#6a6a8e", zorder=3)

    # ── Defense zones (glowing rings around flags) ───────────────────────────
    dz = env.DEFENSE_ZONE_RADIUS + 0.5   # slightly wider than actual zone
    r1, c1 = env.TEAM1_FLAG_POS
    ax.add_patch(plt.Circle((c1, r1), dz, color=C["flag1"], alpha=0.07, zorder=0, linewidth=0))
    ax.add_patch(plt.Circle((c1, r1), dz, fill=False, edgecolor=C["flag1"],
                             linewidth=1.0, linestyle="--", alpha=0.4, zorder=0))

    r2, c2 = env.TEAM2_FLAG_POS
    ax.add_patch(plt.Circle((c2, r2), dz, color=C["flag2"], alpha=0.07, zorder=0, linewidth=0))
    ax.add_patch(plt.Circle((c2, r2), dz, fill=False, edgecolor=C["flag2"],
                             linewidth=1.0, linestyle="--", alpha=0.4, zorder=0))

    # ── Team 1 flag — blue circle ─────────────────────────────────────────────
    ax.add_patch(plt.Circle((c1, r1), 0.40, color=C["flag1"], alpha=0.25, zorder=2))
    ax.add_patch(plt.Circle((c1, r1), 0.33, color=C["flag1"], zorder=3))
    ax.text(c1, r1, "F1", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)

    # ── Team 2 flag — red circle ──────────────────────────────────────────────
    ax.add_patch(plt.Circle((c2, r2), 0.40, color=C["flag2"], alpha=0.25, zorder=2))
    ax.add_patch(plt.Circle((c2, r2), 0.33, color=C["flag2"], zorder=3))
    ax.text(c2, r2, "F2", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)

    # ── Team 1 agents — blue squares (A=team1_0, B=team1_1) ──────────────────
    agent_cfg = {
        "team1_0": (C["t1_fill"], C["t1_edge"], C["t1_glow"], "A"),
        "team1_1": ("#81d4fa",    "#01579b",    "#b3e5fc",    "B"),
        "team2_0": (C["t2_fill"], C["t2_edge"], C["t2_glow"], "C"),
        "team2_1": ("#a5d6a7",    "#1b5e20",    "#c8e6c9",    "D"),
    }

    # ── Build cell→agent map to detect overlaps (enemy tagging) ──────────────
    cell_agents = {}
    for aid in agent_cfg:
        if aid in env.positions:
            cell_agents.setdefault(env.positions[aid], []).append(aid)

    drawn_tags = set()   # cells where we already drew the TAG! flash

    for aid, (fill, edge, glow, sym) in agent_cfg.items():
        if aid not in env.positions:
            continue
        r, c     = env.positions[aid]
        guarding = _is_defending(env, aid)

        # RED override when actively defending
        if guarding:
            fill = "#ef5350"
            edge = "#b71c1c"
            glow = "#ffcdd2"
            extra_glow_color = "#ef9a9a"
        else:
            extra_glow_color = glow

        # Offset when multiple agents share a cell (enemy tagging)
        cohabitants = cell_agents.get((r, c), [aid])
        if len(cohabitants) > 1:
            idx    = cohabitants.index(aid)
            dc_off = -0.22 if idx == 0 else 0.22
            dr_off = 0.0
            # Draw TAG! flash once per shared cell
            if (r, c) not in drawn_tags:
                ax.text(c, r - 0.45, "TAG!", ha="center", va="center",
                        fontsize=8, color="#ff1744", fontweight="bold", zorder=10)
                drawn_tags.add((r, c))
        else:
            dc_off, dr_off = 0.0, 0.0

        draw_c = c + dc_off
        draw_r = r + dr_off

        # Extra outer glow ring when guarding
        if guarding:
            ax.add_patch(plt.Circle((draw_c, draw_r), 0.54, color=extra_glow_color,
                                    alpha=0.30, zorder=3))
        # Normal glow
        sz = 0.36 if len(cohabitants) > 1 else 0.46
        ax.add_patch(plt.Circle((draw_c, draw_r), sz, color=glow, alpha=0.20, zorder=3))

        # Agent body — smaller when sharing cell, thicker when defending
        bsz = 0.32 if len(cohabitants) > 1 else 0.38
        lw  = 3.2 if guarding else 2.2
        _rounded_rect(ax, draw_c - bsz, draw_r - bsz, bsz * 2, bsz * 2, r=0.09,
                      facecolor=fill, edgecolor=edge, linewidth=lw, zorder=4)

        # Label
        label = f"{sym}D" if guarding else sym
        fs    = 7 if len(cohabitants) > 1 else (8 if guarding else 11)
        ax.text(draw_c, draw_r, label, ha="center", va="center",
                fontsize=fs, color="white" if guarding else edge,
                fontweight="bold", zorder=5)



    # ── Column / row axis labels ───────────────────────────────────────────────
    for i in range(G):
        ax.text(i, -0.54, str(i), ha="center", va="center",
                fontsize=7, color=C["sub_text"])
        ax.text(-0.54, i, str(i), ha="center", va="center",
                fontsize=7, color=C["sub_text"])


def draw_panel(ax_p, env: CaptureTheFlagEnv,
               episode: int, step: int,
               t1_r: float, t2_r: float,
               winner: str = None):
    """Draw the right-side info / score panel."""
    ax_p.cla()
    ax_p.set_facecolor(C["panel_bg"])
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1)
    ax_p.axis("off")

    def txt(x, y, s, **kw):
        ax_p.text(x, y, s, transform=ax_p.transAxes, **kw)

    # ── Title ─────────────────────────────────────────────────────────────────
    txt(0.5, 0.97, "CAPTURE", ha="center", va="top",
        fontsize=17, color=C["text"], fontweight="bold")
    txt(0.5, 0.90, "THE FLAG", ha="center", va="top",
        fontsize=17, color=C["gold"], fontweight="bold")

    # ── Episode / step ────────────────────────────────────────────────────────
    ax_p.axhline(0.83, xmin=0.05, xmax=0.95, color="#252545", linewidth=1)
    txt(0.5, 0.81, f"Episode  {episode}", ha="center", va="top",
        fontsize=11, color=C["sub_text"])
    txt(0.5, 0.75, f"Step  {step}", ha="center", va="top",
        fontsize=11, color=C["sub_text"])

    ax_p.axhline(0.70, xmin=0.05, xmax=0.95, color="#252545", linewidth=1)

    # ── Team 1 score card ─────────────────────────────────────────────────────
    _rounded_rect(ax_p, 0.04, 0.57, 0.92, 0.11, r=0.02,
                  facecolor="#0d2440", edgecolor=C["t1_edge"], linewidth=1.8,
                  transform=ax_p.transAxes, zorder=1)
    txt(0.10, 0.645, "●", ha="left", va="center", fontsize=15, color=C["t1_fill"])
    txt(0.22, 0.645, "TEAM 1", ha="left", va="center",
        fontsize=9, color=C["t1_fill"], fontweight="bold")
    txt(0.55, 0.645, "(A, B)", ha="left", va="center",
        fontsize=8, color=C["sub_text"])
    txt(0.94, 0.645, f"Score: {env.scores['team1']}",
        ha="right", va="center", fontsize=10, color=C["text"], fontweight="bold")

    # ── Team 2 score card ─────────────────────────────────────────────────────
    _rounded_rect(ax_p, 0.04, 0.43, 0.92, 0.11, r=0.02,
                  facecolor="#0d2e14", edgecolor=C["t2_edge"], linewidth=1.8,
                  transform=ax_p.transAxes, zorder=1)
    txt(0.10, 0.505, "●", ha="left", va="center", fontsize=15, color=C["t2_fill"])
    txt(0.22, 0.505, "TEAM 2", ha="left", va="center",
        fontsize=9, color=C["t2_fill"], fontweight="bold")
    txt(0.55, 0.505, "(C, D)", ha="left", va="center",
        fontsize=8, color=C["sub_text"])
    txt(0.94, 0.505, f"Score: {env.scores['team2']}",
        ha="right", va="center", fontsize=10, color=C["text"], fontweight="bold")

    ax_p.axhline(0.39, xmin=0.05, xmax=0.95, color="#252545", linewidth=1)

    # ── Episode rewards ───────────────────────────────────────────────────────
    txt(0.5, 0.37, "Episode Reward", ha="center", va="top",
        fontsize=8, color=C["sub_text"])
    txt(0.5, 0.31, f"T1 :  {t1_r:+.0f}", ha="center", va="top",
        fontsize=11, color=C["t1_fill"], fontweight="bold")
    txt(0.5, 0.25, f"T2 :  {t2_r:+.0f}", ha="center", va="top",
        fontsize=11, color=C["t2_fill"], fontweight="bold")

    ax_p.axhline(0.19, xmin=0.05, xmax=0.95, color="#252545", linewidth=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [
        ("F1 = Team 1 flag (blue)", C["flag1"]),
        ("F2 = Team 2 flag (red)",  C["flag2"]),
        ("## = Obstacle",           C["obstacle"]),
        ("XD = Agent defending",    "#ef5350"),   # red when defending
        ("-- = Defense zone edge",  C["sub_text"]),
    ]
    for i, (label, col) in enumerate(legend):
        txt(0.06, 0.17 - i * 0.042, label, ha="left", va="top",
            fontsize=7, color=col)


    # ── Winner overlay ────────────────────────────────────────────────────────
    if winner:
        wc  = C["t1_fill"] if "1" in winner else (C["t2_fill"] if "2" in winner else C["gold"])
        msg = "TEAM 1 WINS!" if "1" in winner else ("TEAM 2 WINS!" if "2" in winner else "DRAW!")
        _rounded_rect(ax_p, 0.03, 0.30, 0.94, 0.48, r=0.03,
                      facecolor="#000018", edgecolor=wc, linewidth=3,
                      transform=ax_p.transAxes, zorder=8)
        txt(0.5, 0.72, "[ WIN ]", ha="center", va="center",
            fontsize=18, color=C["gold"], fontweight="bold", zorder=9)
        txt(0.5, 0.60, msg, ha="center", va="center",
            fontsize=12, color=wc, fontweight="bold", zorder=9)
        txt(0.5, 0.50, f"in {step} steps", ha="center", va="center",
            fontsize=9, color=C["sub_text"], zorder=9)
        txt(0.5, 0.42, f"T1: {t1_r:.0f}   T2: {t2_r:.0f}",
            ha="center", va="center", fontsize=8, color=C["sub_text"], zorder=9)


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def find_checkpoints(ckpt_dir: str):
    t1 = os.path.join(ckpt_dir, "team1_final.pt")
    t2 = os.path.join(ckpt_dir, "team2_final.pt")
    if os.path.isfile(t1) and os.path.isfile(t2):
        return t1, t2
    t1s = sorted(glob.glob(os.path.join(ckpt_dir, "team1_ep*.pt")),
                 key=lambda p: int(p.split("ep")[-1].replace(".pt", "")))
    t2s = sorted(glob.glob(os.path.join(ckpt_dir, "team2_ep*.pt")),
                 key=lambda p: int(p.split("ep")[-1].replace(".pt", "")))
    if not t1s or not t2s:
        raise FileNotFoundError(
            f"No checkpoints in '{ckpt_dir}'. Run python train.py first."
        )
    return t1s[-1], t2s[-1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live CTF game viewer.")
    parser.add_argument("--episodes",   type=int,   default=5,    help="Episodes to watch")
    parser.add_argument("--speed",      type=float, default=0.25, help="Seconds per step")
    parser.add_argument("--checkpoint", type=str,   default=None, help="Checkpoint directory")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir  = args.checkpoint or os.path.join(base, "checkpoints")
    t1_path, t2_path = find_checkpoints(ckpt_dir)

    print(f"[Device] {device}")
    print(f"[Team 1] {t1_path}")
    print(f"[Team 2] {t2_path}\n")

    prog_coef = config.get("progress_reward_coef", 1.0)
    env       = CaptureTheFlagEnv(progress_reward_coef=prog_coef)

    t1_agent = MAPPOAgent(env.team1_ids, (env.OBS_SIZE,), env.action_space.n,
                          lr_actor=config["lr_actor"], lr_critic=config["lr_critic"]).to(device)
    t2_agent = MAPPOAgent(env.team2_ids, (env.OBS_SIZE,), env.action_space.n,
                          lr_actor=config["lr_actor"], lr_critic=config["lr_critic"]).to(device)
    t1_agent.load(t1_path)
    t2_agent.load(t2_path)
    for a in t1_agent.actors.values(): a.eval()
    for a in t2_agent.actors.values(): a.eval()
    t1_agent.critic.eval()
    t2_agent.critic.eval()

    # ── Figure setup ──────────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(13, 8), facecolor=C["bg"])
    fig.canvas.manager.set_window_title("MAPPO — Capture The Flag")

    # Two columns: 75% board | 25% info panel
    ax_board = fig.add_axes([0.01, 0.02, 0.70, 0.95])   # [left, bottom, w, h]
    ax_panel = fig.add_axes([0.72, 0.02, 0.27, 0.95])
    ax_board.set_facecolor(C["grid_bg"])
    ax_panel.set_facecolor(C["panel_bg"])

    print("=" * 55)
    print("  MAPPO CTF — Live Viewer")
    print(f"  Episodes: {args.episodes}   Speed: {args.speed}s/step")
    print("  Close the window at any time to stop.")
    print("=" * 55 + "\n")

    for ep in range(1, args.episodes + 1):
        obs   = env.reset()
        t1_r  = t2_r = 0.0
        done  = False

        print(f"--- Episode {ep} ---")

        for step in range(200):
            actions = {}
            for aid in env.team1_ids:
                actions[aid] = t1_agent.select_action_greedy(aid, obs[aid], temperature=0.15)
            for aid in env.team2_ids:
                actions[aid] = t2_agent.select_action_greedy(aid, obs[aid], temperature=0.15)

            obs, rewards, dones, _ = env.step(actions)
            done = dones["__all__"]

            t1_r += sum(rewards[a] for a in env.team1_ids)
            t2_r += sum(rewards[a] for a in env.team2_ids)

            # Redraw
            draw_board(ax_board, env)
            draw_panel(ax_panel, env, ep, step + 1, t1_r, t2_r)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(args.speed)

            if not plt.fignum_exists(fig.number):
                print("Window closed — exiting.")
                return

            if done:
                break

        # ── Determine winner ──────────────────────────────────────────────────
        s = env.scores
        if s["team1"] > s["team2"]:
            winner = "team_1"
        elif s["team2"] > s["team1"]:
            winner = "team_2"
        else:
            winner = "draw"

        print(f"  Result: {winner.upper()}  |  T1 reward: {t1_r:.0f}  T2 reward: {t2_r:.0f}  |  Steps: {step+1}")

        # Show winner overlay for 2 seconds
        draw_board(ax_board, env)
        draw_panel(ax_panel, env, ep, step + 1, t1_r, t2_r, winner=winner)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(2.5)

        if not plt.fignum_exists(fig.number):
            return

    print("\nAll episodes finished. Close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
