"""
train.py — DQN Agent Training Script for Demon Attack (Atari)
Environment : ALE/DemonAttack-v5
Framework   : Stable Baselines3 + Gymnasium

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN ON GOOGLE COLAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cell 1 — Install:
    !pip install stable-baselines3[extra] gymnasium[atari,accept-rom-license] ale-py autorom[accept-rom-license] matplotlib
    !AutoROM --accept-license

Cell 2 — Upload train.py then run:
    !python train.py

Cell 3 — Download model:
    from google.colab import files
    files.download('dqn_model.zip')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Fix ALE registration (must be before other imports) ──────────────────────
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works on Colab/servers
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────
# GPU Check & Force
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"[GPU] ✔ CUDA available — {torch.cuda.get_device_name(0)}")
else:
    print("[GPU] ✘ No GPU — running on CPU (slow!)")
    print("[GPU]   Go to Runtime → Change runtime type → T4 GPU")
print(f"[GPU] Using device: {device}\n")


# ─────────────────────────────────────────────
# Global Settings
# ─────────────────────────────────────────────
ENV_ID           = "ALE/DemonAttack-v5"
POLICY           = "CnnPolicy"
EXPERIMENT_STEPS = 100_000      # steps per experiment (fast but enough to compare)
FINAL_STEPS      = 1_000_000    # steps for the best model final training
MODEL_PATH       = "dqn_model"  # saved as dqn_model.zip
LOG_DIR          = "./logs/"
RESULTS_PATH     = "experiment_results.txt"
PLOT_PATH        = "training_rewards.png"


# ─────────────────────────────────────────────
# 10 Hyperparameter Experiments
# ─────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "Exp 1 - Baseline",
        "lr": 1e-4, "gamma": 0.99, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 2 - High LR",
        "lr": 1e-3, "gamma": 0.99, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 3 - Low LR",
        "lr": 5e-5, "gamma": 0.99, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 4 - Low Gamma",
        "lr": 1e-4, "gamma": 0.95, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 5 - High Gamma",
        "lr": 1e-4, "gamma": 0.999, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 6 - Large Batch",
        "lr": 1e-4, "gamma": 0.99, "batch": 64,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 7 - Larger Batch",
        "lr": 1e-4, "gamma": 0.99, "batch": 128,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
    },
    {
        "name": "Exp 8 - High Epsilon End",
        "lr": 1e-4, "gamma": 0.99, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 0.10,
    },
    {
        "name": "Exp 9 - Fast Epsilon Decay",
        "lr": 1e-4, "gamma": 0.99, "batch": 32,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.20,
    },
    {
        "name": "Exp 10 - Best Combo",
        "lr": 1e-4, "gamma": 0.99, "batch": 64,
        "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.15,
    },
]


# ─────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────
class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)
                if self.verbose > 0:
                    print(
                        f"  Ep {len(self.episode_rewards):>4d} | "
                        f"Reward: {r:>8.2f} | "
                        f"Length: {l:>6d}"
                    )
        return True


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
def make_env(env_id, render_mode=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(env_id, render_mode=None):
    envs = DummyVecEnv([make_env(env_id, render_mode)])
    envs = VecFrameStack(envs, n_stack=4)
    return envs


# ─────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────
def plot_training(rewards, lengths, save_path, title="Training Curves"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(rewards, alpha=0.4, color="steelblue", label="Episode reward")
    if len(rewards) >= 10:
        w = max(10, len(rewards) // 20)
        avg = np.convolve(rewards, np.ones(w) / w, mode="valid")
        axes[0].plot(range(w - 1, len(rewards)), avg,
                     color="darkblue", linewidth=2, label=f"Moving avg ({w})")
    axes[0].set_title("Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(lengths, alpha=0.4, color="darkorange", label="Episode length")
    if len(lengths) >= 10:
        w = max(10, len(lengths) // 20)
        avg = np.convolve(lengths, np.ones(w) / w, mode="valid")
        axes[1].plot(range(w - 1, len(lengths)), avg,
                     color="saddlebrown", linewidth=2, label=f"Moving avg ({w})")
    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved → {save_path}")


# ─────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────
def run_experiment(exp, steps, verbose=0):
    print(f"\n{'='*60}")
    print(f"  {exp['name']}")
    print(f"  lr={exp['lr']}  gamma={exp['gamma']}  batch={exp['batch']}")
    print(f"  ε: {exp['eps_start']}→{exp['eps_end']}  decay={exp['eps_decay']}")
    print(f"  Steps: {steps:,}  |  Device: {device}")
    print(f"{'='*60}")

    vec_env = build_vec_env(ENV_ID)

    model = DQN(
        POLICY, vec_env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        batch_size=exp["batch"],
        buffer_size=100_000,
        exploration_fraction=exp["eps_decay"],
        exploration_initial_eps=exp["eps_start"],
        exploration_final_eps=exp["eps_end"],
        target_update_interval=1_000,
        learning_starts=10_000,
        train_freq=4,
        device=device,          # ← forces GPU usage
        tensorboard_log=LOG_DIR,
        verbose=0,
    )

    logger = TrainingLogger(verbose=verbose)
    start = time.time()
    model.learn(total_timesteps=steps, callback=logger,
                tb_log_name=exp["name"].replace(" ", "_"))
    elapsed = time.time() - start

    mean_r  = np.mean(logger.episode_rewards[-20:]) if logger.episode_rewards else 0.0
    best_r  = max(logger.episode_rewards) if logger.episode_rewards else 0.0
    n_eps   = len(logger.episode_rewards)

    print(f"\n  ✔ Done in {elapsed/60:.1f} min")
    print(f"  Episodes     : {n_eps}")
    print(f"  Mean reward  : {mean_r:.2f}  (last 20 eps)")
    print(f"  Best reward  : {best_r:.2f}")

    vec_env.close()

    return {
        "exp": exp,
        "mean_reward": mean_r,
        "best_reward": best_r,
        "n_episodes": n_eps,
        "elapsed_min": elapsed / 60,
        "rewards": logger.episode_rewards,
        "lengths": logger.episode_lengths,
        "model": model,
    }


# ─────────────────────────────────────────────
# Save results table
# ─────────────────────────────────────────────
def save_results_table(results):
    lines = []
    lines.append("=" * 100)
    lines.append("HYPERPARAMETER TUNING RESULTS — DEMON ATTACK DQN")
    lines.append("=" * 100)
    lines.append(f"{'#':<4} {'Experiment':<28} {'LR':<8} {'Gamma':<7} {'Batch':<7} "
                 f"{'ε Start':<8} {'ε End':<7} {'ε Decay':<8} "
                 f"{'Mean R':<10} {'Best R':<10} {'Episodes':<10}")
    lines.append("-" * 100)

    for i, r in enumerate(results, 1):
        e = r["exp"]
        lines.append(
            f"{i:<4} {e['name']:<28} {e['lr']:<8} {e['gamma']:<7} {e['batch']:<7} "
            f"{e['eps_start']:<8} {e['eps_end']:<7} {e['eps_decay']:<8} "
            f"{r['mean_reward']:<10.2f} {r['best_reward']:<10.2f} {r['n_episodes']:<10}"
        )

    lines.append("=" * 100)

    # Find best
    best = max(results, key=lambda x: x["mean_reward"])
    lines.append(f"\n✔ BEST EXPERIMENT: {best['exp']['name']}")
    lines.append(f"  Mean Reward: {best['mean_reward']:.2f}  |  Best Reward: {best['best_reward']:.2f}")

    table = "\n".join(lines)
    print("\n" + table)

    with open(RESULTS_PATH, "w") as f:
        f.write(table)
    print(f"\n[INFO] Results saved → {RESULTS_PATH}")

    return best


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  DEMON ATTACK DQN — 10 EXPERIMENT HYPERPARAMETER TUNING")
    print(f"  Each experiment: {EXPERIMENT_STEPS:,} steps")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Phase 1: Run all 10 experiments ─────────────────────────
    all_results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, steps=EXPERIMENT_STEPS, verbose=0)
        all_results.append(result)

    # ── Phase 2: Print results table & find best ─────────────────
    best = save_results_table(all_results)

    # ── Phase 3: Plot all experiment rewards ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in all_results:
        if r["rewards"]:
            ax.plot(r["rewards"], alpha=0.6, label=r["exp"]["name"])
    ax.set_title("All 10 Experiments — Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("all_experiments.png", dpi=150)
    plt.close()
    print("[INFO] Comparison plot saved → all_experiments.png")

    # ── Phase 4: Full training with best config ───────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING — {best['exp']['name']}")
    print(f"  Steps: {FINAL_STEPS:,}")
    print(f"{'='*60}")

    final = run_experiment(best["exp"], steps=FINAL_STEPS, verbose=1)

    # Save final model
    final["model"].save(MODEL_PATH)
    print(f"\n[INFO] ✔ Final model saved → {MODEL_PATH}.zip")

    # Plot final training curves
    plot_training(
        final["rewards"], final["lengths"],
        PLOT_PATH,
        title=f"Final Training — {best['exp']['name']}"
    )

    print(f"\n[SUMMARY] Final training complete!")
    print(f"[SUMMARY] Episodes     : {final['n_episodes']}")
    print(f"[SUMMARY] Mean reward  : {final['mean_reward']:.2f}")
    print(f"[SUMMARY] Best reward  : {final['best_reward']:.2f}")
    print(f"\n[INFO] Files ready to download:")
    print(f"  → dqn_model.zip")
    print(f"  → training_rewards.png")
    print(f"  → all_experiments.png")
    print(f"  → experiment_results.txt")


if __name__ == "__main__":
    main()
