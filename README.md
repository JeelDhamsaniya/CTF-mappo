# 🚩 MAPPO — Capture The Flag

A robust **Multi-Agent Reinforcement Learning (MARL)** system implementing **MAPPO** (Multi-Agent Proximal Policy Optimization) for a 10x10 grid-world Capture The Flag game.

![CTF Game Preview](https://github.com/JeelDhamsaniya/CTF-mappo/blob/main/logs/training_results.png?raw=true)

## 🚀 Key Features

*   **CTDE Architecture**: Centralized Training, Decentralized Execution.
*   **Centralized Critic**: A shared critic that observes the concatenated states of all teammates for stable value estimation.
*   **Residual Actor Networks**: Advanced policy networks with **Residual Blocks** and **LayerNorm** for faster, more stable convergence.
*   **Ego-Centric Observations**: Symmetric observation logic ensures both teams learn from the same data distribution, leading to balanced win rates.
*   **Tactical Gameplay**:
    *   **Defense Zones**: Agents earn rewards for guarding their flag when enemies are near.
    *   **Coordination Bonuses**: Rewards for simultaneous attacking and defending roles.
    *   **Tagging Mechanics**: Defenders can "tag" intruders, teleporting them back to their starting quadrant.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/JeelDhamsaniya/CTF-mappo.git
cd CTF-mappo
```

### 2. Set up environment
It is highly recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏃 How to Run

### 1. Training Agents
Start the MAPPO training process. This will run for 5,000 episodes by default and save checkpoints in the `checkpoints/` folder.
```bash
python train.py
```
*   **Logs**: Check `logs/` for training plots and metrics.

### 2. Live Visualization (Watch them play!)
Use the interactive Matplotlib visualizer to watch the trained agents in action.
```bash
python render_game.py --speed 0.25 --episodes 5
```
*   **Blue Agents (A, B)**: Team 1
*   **Green Agents (C, D)**: Team 2
*   **Red "D" Indicator**: Active Defender (earning guard rewards!)
*   **Trophy Icon**: Winning team.

### 3. Evaluation & GIF Generation
Run a batch of episodes to calculate win rates and generate animated GIFs of the gameplay.
```bash
python evaluate.py --episodes 100 --gif 5
```
*   GIFs are saved in `logs/renders/`.

---

## 📂 Project Structure

```text
ctf_mappo/
├── env/               # Custom Gymnasium Environment
├── mappo/             # MAPPO Algorithm (Actor/Critic/Buffer)
├── training/          # Training Loop logic
├── configs/           # Hyperparameters (config.yaml)
├── checkpoints/       # Saved .pt models
└── logs/              # Plots and Rendered GIFs
```

## 🧠 Technical Details

*   **Algorithm**: MAPPO (Multi-Agent PPO)
*   **Observation Space**: 102 (Grid + Team Identity)
*   **Action Space**: Discrete(5) — [Up, Down, Left, Right, Stay]
*   **Framework**: PyTorch & Gymnasium
