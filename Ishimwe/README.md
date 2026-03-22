# DQN Agent — Demon Attack (Atari)

A Deep Q-Network (DQN) agent trained to play **Demon Attack** using Stable Baselines3 and Gymnasium.

---

## Project Structure

```
demon_attack_dqn/
├── train.py                  # Train the DQN agent (10 experiments)
├── play.py                   # Load model and run evaluation episodes
├── requirements.txt          # Python dependencies
├── demon_attack_dqn.ipynb    # Notebook that I used training models
├── dqn_model.zip             # Saved trained model
├── training_rewards.png      # Final training curve plot
├── all_experiments.png       # All 10 experiments comparison plot
├── README.md                 # This file 
└── experiment_results.txt    # Hyperparameter tuning results table
```

---

## Setup & Installation

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Atari ROMs
AutoROM --accept-license
```

---

## Running

### Train the agent
```bash
python train.py
```
- Runs 10 hyperparameter experiments (100,000 steps each)
- Automatically picks the best config and trains final model (500,000 steps)
- Saves `dqn_model.zip`, `training_rewards.png`, `all_experiments.png`, `experiment_results.txt`

### Watch the agent play
```bash
python play.py
```
- Loads `dqn_model.zip` and plays 5 episodes
- Uses greedy policy (`deterministic=True`) — always picks highest Q-value action
- Renders game in real time via `render_mode="human"`

---

## Policy Comparison: CNN vs MLP

| Policy | Notes |
|--------|-------|
| **CnnPolicy**  | Processes raw 84×84 grayscale frames stacked 4 deep. Captures spatial patterns — ideal for visual Atari games. **Used in this project.** |
| **MlpPolicy** | Flattens pixel input into a vector. Loses spatial structure — not suitable for Atari pixel-based games. |

> **Conclusion:** CnnPolicy is the correct choice for Demon Attack because the game's sprites, movement, and visual patterns are inherently spatial. MlpPolicy would need significantly more steps to achieve comparable performance.

---

## Hyperparameter Tuning Results

> 10 experiments conducted on Google Colab (Tesla T4 GPU), 100,000 steps each.

| # | Experiment | LR | Gamma | Batch | ε Start | ε End | ε Decay | Mean Reward | Best Reward | Episodes | Noted Behavior |
|---|-----------|-----|-------|-------|---------|-------|---------|-------------|-------------|----------|----------------|
| 1 | Baseline | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 7.70 | 39.00 | 1442 | Solid baseline. Stable learning with moderate exploration. Agent learns to shoot enemies consistently. |
| 2 | High LR | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 5.55 | 44.00 | 1375 | LR too high — Q-values become unstable. Lower mean reward despite higher best single episode. |
| 3 | Low LR | 5e-05 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | 4.30 | 33.00 | 1470 | LR too low — convergence too slow. Agent barely improves within 100k steps. Worst performer. |
| 4 | Low Gamma | 0.0001 | 0.95 | 32 | 1.0 | 0.01 | 0.10 | 4.75 | 45.00 | 1444 | Lower gamma makes agent myopic — ignores long-term survival. Below baseline performance. |
| 5 | High Gamma | 0.0001 | 0.999 | 32 | 1.0 | 0.01 | 0.10 | 5.25 | 39.00 | 1315 | Agent plans too far ahead. Slower early learning. Slight improvement over low gamma but below baseline. |
| 6 | Large Batch | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | 9.05 | 44.00 | 1249 | Larger batch = more stable gradient updates. Clear jump in mean reward over baseline. |
| 7 | Larger Batch | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 0.10 | 10.80 | 56.00 | 1219 | Best reward of any single experiment. Very stable learning curve. Highest best reward (56.00). |
| 8 | High Epsilon End | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 10.65 | 43.00 | 1358 | More residual exploration. Strong mean reward but lower best — agent keeps trying suboptimal actions. |
| 9 | Fast Epsilon Decay | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.20 | 6.05 | 34.00 | 1500 | Exploits too early — gets stuck in local optima. Poor performance despite most episodes completed. |
| 10 | **Best Combo** ✅ | **0.0001** | **0.99** | **64** | **1.0** | **0.01** | **0.15** | **11.20** | **42.00** | **1332** | **Best mean reward (11.20). Balanced epsilon decay + larger batch = most consistent improvement. Selected for final training.** |

### Final Training Results (Best Config — Exp 10)

| Metric | Value |
|--------|-------|
| Config | lr=0.0001, gamma=0.99, batch=64, ε: 1.0→0.01, decay=0.15 |
| Total Steps | 500,000 |
| Total Episodes | **5,308** |
| Mean Reward (last 20 eps) | **9.70** |
| Best Episode Reward | **85.00** 🏆 |
| Training Time | **47.5 min** (Tesla T4 GPU) |

---

## Key Insights from Hyperparameter Tuning

- **Learning Rate:** `0.0001` is the sweet spot. `0.001` causes instability; `5e-05` is too slow to converge.
- **Gamma:** `0.99` works best for Demon Attack. Lower values make the agent shortsighted; higher values slow early learning.
- **Batch Size:** Larger batches (`64–128`) produce more stable gradient estimates and consistently better performance. Batch 128 achieved the highest best reward (56.00).
- **Epsilon Decay:** A moderate decay of `0.10–0.15` gives the agent enough exploration time. Fast decay (`0.20`) causes early exploitation and poor long-term performance.
- **Best Config:** Combining `batch=64` with `eps_decay=0.15` gave the highest mean reward (11.20) — the balance between stable updates and adequate exploration was key.

---

## Agent Gameplay

> *(https://drive.google.com/file/d/1Npr4AmQnrSO78M6_10__DziMD-xBCQav/view?usp=sharing)*

---

## References
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://ale.farama.org/environments/complete_list/)
- [Demon Attack Environment](https://ale.farama.org/environments/demon_attack/)
- [DQN Paper — Mnih et al. 2015](https://www.nature.com/articles/nature14236)
