"""
run_experiments.py — ALE/DemonAttack-v5
────────────────────────────────────────
Runs all 10 hyperparameter experiments automatically.

RESUME:   If it crashes, just re-run the same command.
          Completed experiments are saved to experiments_checkpoint.json
          and automatically skipped on restart.

SPEED:    Uses make_atari_env + VecFrameStack + n_envs parallel envs.
          With n_envs=4 expect ~3-4x speedup over single env.

LIVE:     Saves dqn_latest.zip every 10k steps.
          Run play.py in another terminal to watch the agent learn.

Usage:
    python3 run_experiments.py                      # full run
    python3 run_experiments.py --timesteps 200000   # quick test (~30 min)
    python3 run_experiments.py --start 4 --end 7    # run subset
    python3 run_experiments.py --member
    python3 run_experiments.py --reset              # clear checkpoint, start fresh
"""

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID      = "ALE/DemonAttack-v5"
CHECKPOINT  = "dqn_latest.zip"
PROGRESS_F  = "experiments_checkpoint.json"
CSV_PATH    = "hyperparameter_experiments.csv"

# name to log with csv
MEMBER_NAME = "Gatwaza"

EXPERIMENTS = [
    {
        "id": 1, "label": "Baseline",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.99,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Balanced default — reference point for all other runs",
    },
    {
        "id": 2, "label": "High LR",
        "policy": "CnnPolicy", "lr": 0.001, "gamma": 0.99,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "10x higher lr — faster early gains but may oscillate",
    },
    {
        "id": 3, "label": "Low LR",
        "policy": "CnnPolicy", "lr": 0.00001, "gamma": 0.99,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Very slow convergence — low reward expected",
    },
    {
        "id": 4, "label": "Low Gamma",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.90,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Short-sighted agent — prefers immediate kills over survival",
    },
    {
        "id": 5, "label": "High Gamma",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.999,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Far-sighted — values long-term bunker preservation",
    },
    {
        "id": 6, "label": "Large Batch",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.99,
        "batch": 128, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Smoother gradients — more stable but slower per update",
    },
    {
        "id": 7, "label": "Small Batch",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.99,
        "batch": 16, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.10,
        "hypothesis": "Noisy gradients — high variance, may diverge",
    },
    {
        "id": 8, "label": "High Epsilon End",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.99,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.10, "eps_decay": 0.10,
        "hypothesis": "10% random forever — less exploitation, lower peak score",
    },
    {
        "id": 9, "label": "Slow Epsilon Decay",
        "policy": "CnnPolicy", "lr": 0.0001, "gamma": 0.99,
        "batch": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.50,
        "hypothesis": "Explores for half of training — broader search space",
    },
    {
        "id": 10, "label": "Best Guess Combined",
        "policy": "CnnPolicy", "lr": 0.0005, "gamma": 0.995,
        "batch": 64, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.15,
        "hypothesis": "Tuned combo — moderate lr, near-infinite horizon, large batch",
    },
]


# Checkpoint / resume helpers
def load_progress() -> dict:
    if os.path.exists(PROGRESS_F):
        with open(PROGRESS_F) as f:
            return json.load(f)
    return {"completed": [], "results": {}}


def save_progress(progress: dict):
    with open(PROGRESS_F, "w") as f:
        json.dump(progress, f, indent=2)


def mark_done(progress: dict, exp_id: int, result: dict):
    if exp_id not in progress["completed"]:
        progress["completed"].append(exp_id)
    progress["results"][str(exp_id)] = result
    save_progress(progress)


# Vectorized environment builder
def make_env(n_envs: int):
    vec_cls = DummyVecEnv
    if n_envs > 1 and platform.system() != "Darwin":
        vec_cls = SubprocVecEnv
    env = make_atari_env(ENV_ID, n_envs=n_envs,
                         vec_env_cls=vec_cls, seed=42)
    return VecFrameStack(env, n_stack=4)


# Callback to track rewards, save checkpoints, and print progress
class ExpCallback(BaseCallback):
    def __init__(self, total_steps: int, save_freq: int = 10_000):
        super().__init__()
        self.total_steps = total_steps
        self.save_freq   = save_freq
        self.ep_rewards  = []
        self.ep_lengths  = []
        self._last_save  = 0
        self._start      = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(CHECKPOINT)
            self._last_save = self.num_timesteps
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(float(info["episode"]["r"]))
                self.ep_lengths.append(int(info["episode"]["l"]))
        return True

    @property
    def sps(self):
        return self.num_timesteps / max(time.time()-self._start, 1)


# Chart helpers

def save_exp_chart(rewards, exp_id, label, out_dir):
    if not rewards:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e1e40")
    xs = list(range(1, len(rewards)+1))
    ax.fill_between(xs, rewards, alpha=0.2, color="#ff3860")
    ax.plot(xs, rewards, color="#ff3860", lw=1.2, label="Score")
    if len(rewards) >= 10:
        w  = min(10, len(rewards))
        ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
        ax.plot(xs[w-1:], ma, color="#00e5b0", lw=2.2, label=f"avg({w})")
    ax.set_title(f"Exp {exp_id}: {label}", color="#dde4ff", fontsize=11, pad=8)
    ax.set_xlabel("Episode", color="#7a7ab0")
    ax.set_ylabel("Score",   color="#7a7ab0")
    ax.legend(fontsize=8, facecolor="#12122a", labelcolor="#e0e0e0")
    fig.tight_layout(pad=1.2)
    path = os.path.join(out_dir, f"exp_{exp_id:02d}_{label.replace(' ','_')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_summary_chart(results: list, out_dir: str):
    if len(results) < 2:
        return
    ids    = [r["id"]     for r in results]
    means  = [r["mean"]   for r in results]
    bests  = [r["best"]   for r in results]
    last20 = [r["last20"] for r in results]
    labels = [f"#{r['id']}\n{r['label'][:12]}" for r in results]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e1e40")
    x = np.arange(len(ids))
    w = 0.28
    ax.bar(x-w, means,  width=w, color="#8b5cf6", alpha=0.85, label="Mean (all)")
    ax.bar(x,   last20, width=w, color="#00e5b0", alpha=0.85, label="Mean (last 20)")
    ax.bar(x+w, bests,  width=w, color="#ff3860", alpha=0.85, label="Best")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#aaaacc", fontsize=7.5)
    ax.set_ylabel("Score", color="#7a7ab0")
    ax.set_title("DemonAttack-v5  |  All Experiments Comparison",
                 color="#dde4ff", fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#12122a", labelcolor="#e0e0e0")
    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, "summary_all_experiments.png")
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Summary chart -> {path}")


# Run one experiment

def run_one(exp: dict, timesteps: int, n_envs: int,
            out_dir: str, member: str) -> dict:
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  EXP {exp['id']:>2}/10  |  {exp['label']}")
    print(sep)
    print(f"  lr={exp['lr']}  gamma={exp['gamma']}  batch={exp['batch']}")
    print(f"  eps: {exp['eps_start']} -> {exp['eps_end']}  decay={exp['eps_decay']}")
    print(f"  Hypothesis: {exp['hypothesis']}\n")

    env   = make_env(n_envs)
    model = DQN(
        exp["policy"], env,
        learning_rate           = exp["lr"],
        gamma                   = exp["gamma"],
        batch_size              = exp["batch"],
        exploration_initial_eps = exp["eps_start"],
        exploration_final_eps   = exp["eps_end"],
        exploration_fraction    = exp["eps_decay"],
        buffer_size             = 50_000,
        learning_starts         = 5_000,
        target_update_interval  = 1_000,
        train_freq              = 4,
        optimize_memory_usage   = False,
        verbose                 = 0,
    )

    cb = ExpCallback(timesteps)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, log_interval=None)
    elapsed = time.time() - t0

    rewards = cb.ep_rewards
    n_ep    = len(rewards)
    mean_all = round(float(np.mean(rewards)),        1) if rewards else 0.0
    best     = round(float(max(rewards)),            1) if rewards else 0.0
    last20   = round(float(np.mean(rewards[-20:])),  1) if len(rewards) >= 20 \
               else round(float(np.mean(rewards)),   1) if rewards else 0.0

    sps = timesteps / max(elapsed, 1)
    print(f"  Done {elapsed:.0f}s ({elapsed/60:.1f}m) | {sps:.0f} sps")
    print(f"  Episodes={n_ep}  mean={mean_all}  best={best}  last20={last20}")

    # save model
    model_path = os.path.join(out_dir, f"exp_{exp['id']:02d}_model")
    model.save(model_path)
    model.save(CHECKPOINT)   # latest checkpoint for play.py

    # chart
    chart = save_exp_chart(rewards, exp["id"], exp["label"], out_dir)
    if chart:
        print(f"  Chart -> {chart}")

    # csv log
    auto_note = (
        f"n_ep={n_ep}, mean={mean_all}, best={best}, last20={last20}. "
        f"{exp['hypothesis']}"
    )
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["exp_id","label","member","env","policy",
                        "lr","gamma","batch_size",
                        "eps_start","eps_end","eps_decay",
                        "n_envs","timesteps","n_episodes",
                        "mean_score","best_score","mean_last20",
                        "elapsed_s","noted_behavior"])
        w.writerow([exp["id"], exp["label"], member, ENV_ID, exp["policy"],
                    exp["lr"], exp["gamma"], exp["batch"],
                    exp["eps_start"], exp["eps_end"], exp["eps_decay"],
                    n_envs, timesteps, n_ep,
                    mean_all, best, last20,
                    round(elapsed, 1), auto_note])

    env.close()
    return {"id": exp["id"], "label": exp["label"],
            "mean": mean_all, "best": best, "last20": last20,
            "elapsed": round(elapsed, 1)}



# Main

def main():
    p = argparse.ArgumentParser(
        description="10 DemonAttack-v5 DQN experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timesteps", type=int,   default=500_000,
                   help="Steps per experiment. Use 200000 for quick test.")
    p.add_argument("--n-envs",    type=int,   default=4, dest="n_envs",
                   help="Parallel envs per experiment (speedup ~3-4x)")
    p.add_argument("--start",     type=int,   default=1)
    p.add_argument("--end",       type=int,   default=10)
    p.add_argument("--member",    default=MEMBER_NAME)
    p.add_argument("--reset",     action="store_true",
                   help="Clear checkpoint and restart from experiment 1")
    args = p.parse_args()

    # handle reset
    if args.reset and os.path.exists(PROGRESS_F):
        os.remove(PROGRESS_F)
        print("  Checkpoint cleared — starting fresh.\n")

    progress = load_progress()
    done_ids = set(progress["completed"])

    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", f"experiments_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    exps = [e for e in EXPERIMENTS
            if args.start <= e["id"] <= args.end
            and e["id"] not in done_ids]

    already_done = [e for e in EXPERIMENTS
                    if e["id"] in done_ids
                    and args.start <= e["id"] <= args.end]

    print("\n" + "=" * 62)
    print("  DEMON ATTACK  |  EXPERIMENTS RUNNER")
    print("=" * 62)
    print(f"  Member      : {args.member}")
    print(f"  Env         : {ENV_ID}")
    print(f"  Timesteps   : {args.timesteps:,} per experiment")
    print(f"  n_envs      : {args.n_envs}  (~{args.n_envs}x speedup)")
    print(f"  Est. time   : ~{len(exps)*args.timesteps/args.n_envs/60_000:.0f} min")
    print(f"  To run      : {len(exps)} experiments")
    print(f"  Already done: {len(already_done)} — {[e['id'] for e in already_done]}")
    print(f"  Checkpoint  : {PROGRESS_F}  (resume if crash)")
    print(f"  Live play   : python3 play.py")
    print("=" * 62)

    if not exps:
        print("\n  All experiments in range are already complete!")
        print(f"  Use --reset to start over, or --start/--end to run new ones.")
        print("=" * 62 + "\n")
        return

    results = list(progress["results"].values())
    total_start = time.time()

    for exp in exps:
        try:
            result = run_one(exp, args.timesteps, args.n_envs,
                             out_dir, args.member)
            results.append(result)
            mark_done(progress, exp["id"], result)
            remaining = sum(1 for e in exps if e["id"] > exp["id"])
            if remaining:
                avg_t = (time.time()-total_start) / (exp["id"]-args.start+1)
                print(f"\n  {remaining} experiments left — "
                      f"est. {remaining*avg_t/60:.0f} min remaining")

        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.")
            print(f"  Progress saved to {PROGRESS_F}")
            print("  Re-run same command to resume.\n")
            break
        except Exception as e:
            print(f"\n  ERROR in experiment {exp['id']}: {e}")
            print("  Skipping — progress saved, continuing...\n")
            continue

    # final summary
    total_elapsed = time.time() - total_start
    all_results = [r for r in results if isinstance(r, dict) and "id" in r]
    all_results.sort(key=lambda x: x["id"])

    print("\n" + "=" * 62)
    print("  EXPERIMENTS COMPLETE")
    print("=" * 62)
    print(f"  Total time : {total_elapsed/60:.1f} min")
    print(f"\n  {'#':>3}  {'Label':<28}  {'Mean':>7}  {'Best':>7}  {'Last20':>7}")
    print(f"  {'─'*3}  {'─'*28}  {'─'*7}  {'─'*7}  {'─'*7}")
    for r in all_results:
        if isinstance(r, dict) and "mean" in r:
            print(f"  {r['id']:>3}  {r['label']:<28}  "
                  f"{r['mean']:>7.1f}  {r['best']:>7.1f}  {r['last20']:>7.1f}")

    if len(all_results) >= 2:
        best_exp = max(all_results, key=lambda x: x.get("last20", 0))
        print(f"\n  Best experiment : #{best_exp['id']} — {best_exp['label']}")
        print(f"  Last-20 mean    : {best_exp['last20']:.1f}")
        save_summary_chart(all_results, out_dir)

    print(f"\n  CSV log     -> {CSV_PATH}")
    print(f"  Latest model-> {CHECKPOINT}  (use with play.py)")
    print(f"  Checkpoint  -> {PROGRESS_F}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()