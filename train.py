"""
train.py — ALE/DemonAttack-v5
──────────────────────────────
Vectorized DQN training using make_atari_env + VecFrameStack.
Uses n_envs parallel environments for significant speedup.
Saves a checkpoint every --save-freq steps so play.py can load it live.

Usage:
    python3 train.py
    python3 train.py --n-envs 4 --timesteps 1000000
    python3 train.py --lr 0.0001 --gamma 0.99 --batch 32 --n-envs 4
"""

import argparse
import csv
import os
import time
import platform
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID       = "ALE/DemonAttack-v5"
CHECKPOINT   = "dqn_latest.zip"          # play.py watches this file
ACTION_NAMES = ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]


# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────
class TrainLogger(BaseCallback):
    def __init__(self, log_path: str, save_freq: int = 10_000,
                 print_freq: int = 5):
        super().__init__()
        self.log_path   = log_path
        self.save_freq  = save_freq
        self.print_freq = print_freq
        self.ep_rewards = []
        self.ep_lengths = []
        self._ep_count  = 0
        self._last_save = 0
        self._start     = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode","reward","length","elapsed_s","steps"])

    def _on_step(self) -> bool:
        # periodic checkpoint for live play
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(CHECKPOINT)
            self._last_save = self.num_timesteps

        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                self._ep_count += 1
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self._ep_count, r, l,
                        round(time.time() - self._start, 1),
                        self.num_timesteps,
                    ])
                if self._ep_count % self.print_freq == 0:
                    recent = self.ep_rewards[-self.print_freq:]
                    sps = self.num_timesteps / max(time.time()-self._start, 1)
                    print(f"  Ep {self._ep_count:>5} | "
                          f"Steps {self.num_timesteps:>9,} | "
                          f"SPS {sps:>6.0f} | "
                          f"Reward  mean={np.mean(recent):>8.1f}  "
                          f"best={max(self.ep_rewards):>7.1f}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_training(rewards, lengths, out_dir, tag):
    if not rewards:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4),
                                    facecolor="#0d0d1a")
    xs = list(range(1, len(rewards)+1))
    for ax, data, col, title in [
        (ax1, rewards, "#ff3860", "Episode Score"),
        (ax2, lengths, "#8b5cf6", "Episode Length"),
    ]:
        ax.set_facecolor("#0a0a20")
        ax.tick_params(colors="#7a7ab0", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e40")
        ax.fill_between(xs, data, alpha=0.2, color=col)
        ax.plot(xs, data, color=col, lw=1.2)
        if len(data) >= 10:
            w  = min(20, len(data))
            ma = np.convolve(data, np.ones(w)/w, mode="valid")
            ax.plot(xs[w-1:], ma, color="#00e5b0", lw=2, label=f"avg({w})")
            ax.legend(fontsize=8, facecolor="#12122a", labelcolor="#e0e0e0")
        ax.set_title(title, color="#dde4ff", fontsize=11, pad=6)
        ax.set_xlabel("Episode", color="#7a7ab0")
    fig.suptitle(f"DemonAttack-v5  |  {tag}", color="#00e5b0", fontsize=12)
    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f"curves_{tag}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Build vectorized environment
# ──────────────────────────────────────────────────────────────────────────────
def make_env(n_envs: int):
    """
    make_atari_env applies: NoopReset, MaxAndSkip, EpisodicLife,
    FireReset, ClipReward, WarpFrame(84x84 grayscale), ScaledFloat.
    VecFrameStack stacks 4 consecutive frames → obs: (84,84,4).
    This shrinks memory ~10x vs raw (210,160,3) and speeds up CNN.
    """
    # macOS Python 3.12 uses 'spawn' by default — SubprocVecEnv works
    # but requires the guard below. DummyVecEnv is safer for GUI usage.
    vec_cls = DummyVecEnv
    if n_envs > 1 and platform.system() != "Darwin":
        vec_cls = SubprocVecEnv

    env = make_atari_env(ENV_ID, n_envs=n_envs,
                         vec_env_cls=vec_cls, seed=42)
    env = VecFrameStack(env, n_stack=4)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", tag)
    os.makedirs(out_dir, exist_ok=True)

    sep = "=" * 64
    print(f"\n{sep}")
    print("  DQN TRAINING  |  ALE/DemonAttack-v5  (Vectorized)")
    print(sep)
    print(f"  Policy      : {args.policy}")
    print(f"  n_envs      : {args.n_envs}  (parallel environments)")
    print(f"  Timesteps   : {args.timesteps:,}")
    print(f"  lr          : {args.lr}  |  gamma : {args.gamma}  |  batch : {args.batch}")
    print(f"  epsilon     : {args.eps_start} -> {args.eps_end}  (decay={args.eps_decay})")
    print(f"  Obs shape   : (84, 84, 4)  grayscale+stacked  [10x smaller than raw]")
    print(f"  Checkpoint  : saved every {args.save_freq:,} steps -> {CHECKPOINT}")
    print(sep + "\n")

    env = make_env(args.n_envs)

    model = DQN(
        args.policy,
        env,
        learning_rate           = args.lr,
        gamma                   = args.gamma,
        batch_size              = args.batch,
        exploration_initial_eps = args.eps_start,
        exploration_final_eps   = args.eps_end,
        exploration_fraction    = args.eps_decay,
        buffer_size             = 50_000,    # safe: 84x84x4 frames are tiny
        learning_starts         = 5_000,
        target_update_interval  = 1_000,
        train_freq              = 4,
        optimize_memory_usage   = False,
        verbose                 = 0,
    )

    log_csv = os.path.join(out_dir, "episode_log.csv")
    cb = TrainLogger(log_csv, save_freq=args.save_freq, print_freq=5)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb,
                log_interval=None)
    elapsed = time.time() - t0

    sps = args.timesteps / max(elapsed, 1)
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)  |  {sps:.0f} steps/sec")

    model.save(os.path.join(out_dir, "dqn_model"))
    model.save("dqn_model")
    model.save(CHECKPOINT)
    print("  Model saved -> ./dqn_model.zip  +  ./dqn_latest.zip")

    if cb.ep_rewards:
        plot_training(cb.ep_rewards, cb.ep_lengths, out_dir, tag)
        last20 = float(np.mean(cb.ep_rewards[-20:]))
        print(f"  Mean score (last 20 eps): {last20:.1f}")
    else:
        last20 = 0.0

    # log to experiments CSV
    fp     = "hyperparameter_experiments.csv"
    is_new = not os.path.exists(fp)
    with open(fp, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["run_tag","member","env","policy","lr","gamma",
                        "batch_size","eps_start","eps_end","eps_decay",
                        "n_envs","timesteps","mean_score_last20",
                        "elapsed_s","noted_behavior"])
        w.writerow([tag, "--", ENV_ID, args.policy,
                    args.lr, args.gamma, args.batch,
                    args.eps_start, args.eps_end, args.eps_decay,
                    args.n_envs, args.timesteps,
                    round(last20, 1), round(elapsed, 1),
                    "-- fill in manually --"])
    print(f"  Logged -> {fp}\n")
    env.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Vectorized DQN training on ALE/DemonAttack-v5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--policy",     default="CnnPolicy",
                   choices=["CnnPolicy","MlpPolicy"])
    p.add_argument("--n-envs",     type=int,   default=4,   dest="n_envs",
                   help="Parallel environments (4 = ~3-4x speedup)")
    p.add_argument("--timesteps",  type=int,   default=1_000_000)
    p.add_argument("--lr",         type=float, default=0.0001)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--eps-start",  type=float, default=1.0,  dest="eps_start")
    p.add_argument("--eps-end",    type=float, default=0.01, dest="eps_end")
    p.add_argument("--eps-decay",  type=float, default=0.10, dest="eps_decay")
    p.add_argument("--save-freq",  type=int,   default=10_000, dest="save_freq",
                   help="Save checkpoint every N steps for live play")
    args = p.parse_args()
    train(args)