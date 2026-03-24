"""
eval.py — ALE/DemonAttack-v5  |  Gatwaza
──────────────────────────────────────────
Headless post-training evaluation of all saved experiment models.
Runs N episodes per model silently (no GUI) and produces:
  - Terminal ranking table
  - eval_results.json
  - eval_comparison.png  (bar chart)
  - Per-model score distribution plot

This reveals how models actually behave at inference time (GreedyQPolicy)
vs what they scored during training — they can differ significantly.

Usage:
    python3 eval.py                        # evaluates all found models, 10 eps each
    python3 eval.py --episodes 20          # more episodes = more reliable estimate
    python3 eval.py --model dqn_best.zip   # evaluate one specific model only
    python3 eval.py --episodes 10 --verbose  # print every episode score
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID       = "ALE/DemonAttack-v5"
ACTION_NAMES = ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]


# ──────────────────────────────────────────────────────────────────────────────
# Discover all saved models
# ──────────────────────────────────────────────────────────────────────────────
def discover_models() -> list[dict]:
    models = []

    for fname, label in [
        ("dqn_best.zip",   "★ BEST  (dqn_best.zip)"),
        ("dqn_latest.zip", "Latest (dqn_latest.zip)"),
        ("dqn_model.zip",  "Final  (dqn_model.zip)"),
    ]:
        if os.path.exists(fname):
            models.append({"label": label, "path": fname, "exp_id": None})

    for path in sorted(glob.glob("logs/experiments_*/exp_*_model.zip")):
        fname = os.path.basename(path)
        try:
            exp_num = int(fname.split("_")[1])
        except (IndexError, ValueError):
            exp_num = 0
        label_raw = fname.replace("_model.zip","").replace("_"," ")
        label = f"Exp {exp_num:>2} — {' '.join(label_raw.split()[2:])}"
        models.append({"label": label, "path": path, "exp_id": exp_num})

    seen, unique = set(), []
    for m in models:
        if m["path"] not in seen:
            seen.add(m["path"])
            unique.append(m)
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate one model for N episodes
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(path: str, label: str, n_episodes: int,
                   verbose: bool = False) -> dict:
    try:
        model = DQN.load(path)
    except Exception as e:
        print(f"  ✗ Could not load {path}: {e}")
        return None

    model_env = VecFrameStack(
        make_atari_env(ENV_ID, n_envs=1, vec_env_cls=DummyVecEnv, seed=42),
        n_stack=4)

    rewards      = []
    ep_lengths   = []
    action_counts = {a: 0 for a in range(6)}
    t0 = time.time()

    for ep in range(n_episodes):
        obs   = model_env.reset()
        ep_r  = 0.0
        steps = 0
        done  = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.atleast_1d(action)
            obs, rew, dones, _ = model_env.step(action)
            done    = bool(np.atleast_1d(dones)[0])
            ep_r   += float(np.atleast_1d(rew)[0])
            steps  += 1
            action_counts[int(action[0])] += 1

        rewards.append(ep_r)
        ep_lengths.append(steps)

        if verbose:
            print(f"    ep {ep+1:>3}/{n_episodes}  score={ep_r:>8.1f}  "
                  f"steps={steps:>6,}")

    model_env.close()
    elapsed = time.time() - t0

    # action distribution — what % of time spent on each action
    total_actions = sum(action_counts.values())
    action_pct = {
        ACTION_NAMES[k]: round(v/total_actions*100, 1)
        for k, v in action_counts.items()
    } if total_actions > 0 else {}

    # detect dominant behaviour
    dominant_action = max(action_counts, key=action_counts.get)
    fire_pct = action_pct.get("FIRE", 0) + \
               action_pct.get("RIGHTFIRE", 0) + \
               action_pct.get("LEFTFIRE", 0)
    move_pct = action_pct.get("RIGHT", 0) + action_pct.get("LEFT", 0)
    noop_pct = action_pct.get("NOOP", 0)

    if noop_pct > 50:
        behaviour = "STUCK — mostly NOOP, not learned"
    elif fire_pct > 60:
        behaviour = "AGGRESSIVE — fires frequently"
    elif move_pct > fire_pct:
        behaviour = "EVASIVE — moves more than fires"
    else:
        behaviour = "BALANCED — mix of movement and firing"

    return {
        "label":           label,
        "path":            path,
        "n_episodes":      n_episodes,
        "mean":            round(float(np.mean(rewards)),   1),
        "std":             round(float(np.std(rewards)),    1),
        "median":          round(float(np.median(rewards)), 1),
        "best":            round(float(max(rewards)),       1),
        "worst":           round(float(min(rewards)),       1),
        "mean_ep_length":  round(float(np.mean(ep_lengths)), 1),
        "consistency":     round(1 - float(np.std(rewards)) /
                                 max(abs(float(np.mean(rewards))), 1), 3),
        "fire_pct":        round(fire_pct, 1),
        "move_pct":        round(move_pct, 1),
        "noop_pct":        round(noop_pct, 1),
        "action_dist":     action_pct,
        "behaviour":       behaviour,
        "rewards":         rewards,
        "elapsed_s":       round(elapsed, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────
def save_comparison_chart(results: list, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d0d1a")
    names  = [r["label"][:20]        for r in results]
    means  = [r["mean"]              for r in results]
    bests  = [r["best"]              for r in results]
    stds   = [r["std"]               for r in results]
    lengths = [r["mean_ep_length"]   for r in results]
    fire   = [r["fire_pct"]          for r in results]
    noop   = [r["noop_pct"]          for r in results]

    colors = ["#00e5b0" if m == max(means) else "#8b5cf6" for m in means]
    x = np.arange(len(names))

    # chart 1 — mean + best scores
    ax = axes[0]
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e40")
    bars = ax.bar(x-0.2, means, width=0.35, color=colors,
                  alpha=0.9, label="Mean score")
    ax.bar(x+0.2, bests, width=0.35, color="#ff3860",
           alpha=0.5, label="Best score")
    ax.errorbar(x-0.2, means, yerr=stds, fmt="none",
                color="white", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right",
                       color="#aaaacc", fontsize=7)
    ax.set_title("Mean & Best Score  (error bar = std dev)",
                 color="#dde4ff", fontsize=10, pad=8)
    ax.set_ylabel("Score", color="#7a7ab0")
    ax.legend(fontsize=8, facecolor="#12122a", labelcolor="#e0e0e0")

    # chart 2 — episode length (survival time)
    ax = axes[1]
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e40")
    len_colors = ["#fbbf24" if l == max(lengths) else "#38bdf8" for l in lengths]
    ax.bar(x, lengths, color=len_colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right",
                       color="#aaaacc", fontsize=7)
    ax.set_title("Mean Episode Length  (survival time)",
                 color="#dde4ff", fontsize=10, pad=8)
    ax.set_ylabel("Steps", color="#7a7ab0")

    # chart 3 — action behaviour (fire% vs noop%)
    ax = axes[2]
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e40")
    ax.bar(x-0.2, fire, width=0.35, color="#ff3860",
           alpha=0.85, label="Fire %")
    ax.bar(x+0.2, noop, width=0.35, color="#44447a",
           alpha=0.85, label="NOOP %")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right",
                       color="#aaaacc", fontsize=7)
    ax.set_title("Action Behaviour  (fire% vs noop%)",
                 color="#dde4ff", fontsize=10, pad=8)
    ax.set_ylabel("% of actions", color="#7a7ab0")
    ax.legend(fontsize=8, facecolor="#12122a", labelcolor="#e0e0e0")

    fig.suptitle(
        f"DemonAttack-v5  |  Post-Training Evaluation  |  "
        f"{results[0]['n_episodes']} episodes each",
        color="#00e5b0", fontsize=12, y=1.01)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved -> {out_path}")


def save_score_distributions(results: list, out_path: str):
    """Box plot showing score spread per model — reveals consistency."""
    n = len(results)
    fig, ax = plt.subplots(figsize=(max(10, n*1.2), 5), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a20")
    ax.tick_params(colors="#7a7ab0", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e40")

    data   = [r["rewards"] for r in results]
    labels = [r["label"][:18] for r in results]

    bp = ax.boxplot(data, patch_artist=True, labels=labels,
                    medianprops=dict(color="#fbbf24", linewidth=2),
                    whiskerprops=dict(color="#7a7ab0"),
                    capprops=dict(color="#7a7ab0"),
                    flierprops=dict(marker="o", color="#ff3860",
                                   markersize=4, alpha=0.6))
    colors = ["#00e5b0","#ff3860","#8b5cf6","#fbbf24","#38bdf8",
              "#f97316","#a3e635","#e879f9","#fb7185","#34d399"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_xticklabels(labels, rotation=45, ha="right",
                       color="#aaaacc", fontsize=7)
    ax.set_title(
        "Score Distribution per Model  "
        "(median=gold line, box=IQR, whiskers=range)",
        color="#dde4ff", fontsize=10, pad=8)
    ax.set_ylabel("Score", color="#7a7ab0")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Distribution chart -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Post-training headless evaluation of all models — Gatwaza",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes per model (more = more reliable)")
    p.add_argument("--model",   default=None,
                   help="Evaluate one specific model only")
    p.add_argument("--verbose", action="store_true",
                   help="Print every episode score during evaluation")
    args = p.parse_args()

    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", f"eval_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # build model list
    if args.model:
        if not os.path.exists(args.model):
            print(f"  Model not found: {args.model}")
            return
        models = [{"label": args.model, "path": args.model, "exp_id": None}]
    else:
        models = discover_models()

    print("\n" + "=" * 66)
    print("  DEMON ATTACK  |  POST-TRAINING EVALUATION  |  Gatwaza")
    print("=" * 66)
    print(f"  Models found  : {len(models)}")
    print(f"  Episodes each : {args.episodes}")
    print(f"  Policy        : GreedyQPolicy (deterministic=True, eps=0)")
    print(f"  Output        : {out_dir}/")
    print("=" * 66)

    results = []
    for i, m in enumerate(models, 1):
        print(f"\n  [{i}/{len(models)}]  {m['label']}")
        print(f"  Path: {m['path']}")
        result = evaluate_model(m["path"], m["label"],
                                args.episodes, args.verbose)
        if result is None:
            continue
        results.append(result)

        # inline summary
        print(f"  mean={result['mean']:>8.1f}  "
              f"std={result['std']:>7.1f}  "
              f"best={result['best']:>8.1f}  "
              f"worst={result['worst']:>8.1f}")
        print(f"  survival={result['mean_ep_length']:>6.0f} steps  |  "
              f"fire={result['fire_pct']:>5.1f}%  "
              f"noop={result['noop_pct']:>5.1f}%")
        print(f"  behaviour: {result['behaviour']}")

    if not results:
        print("\n  No models could be evaluated.")
        return

    # ── Final ranking table ───────────────────────────────────────────────────
    ranked = sorted(results, key=lambda x: x["mean"], reverse=True)

    print("\n" + "=" * 66)
    print("  EVALUATION COMPLETE — RANKING BY MEAN SCORE")
    print("=" * 66)
    print(f"\n  {'Rank':<5} {'Label':<24} {'Mean':>8} {'Std':>7} "
          f"{'Best':>8} {'Surviv':>7} {'Fire%':>6} {'Behaviour'}")
    print(f"  {'─'*5} {'─'*24} {'─'*8} {'─'*7} "
          f"{'─'*8} {'─'*7} {'─'*6} {'─'*20}")

    for rank, r in enumerate(ranked, 1):
        marker = " ★" if rank == 1 else "  "
        print(f"  {rank:<5} {r['label'][:24]:<24} "
              f"{r['mean']:>8.1f} {r['std']:>7.1f} "
              f"{r['best']:>8.1f} {r['mean_ep_length']:>7.0f} "
              f"{r['fire_pct']:>6.1f}  "
              f"{r['behaviour'][:20]}{marker}")

    best = ranked[0]
    worst = ranked[-1]
    print(f"\n  ★ Best  : {best['label']}  mean={best['mean']}")
    print(f"  ✗ Worst : {worst['label']}  mean={worst['mean']}")

    # insight about training vs eval discrepancy
    print(f"\n  INSIGHTS:")
    for r in ranked:
        notes = []
        if r["noop_pct"] > 40:
            notes.append("HIGH NOOP — model may be stuck in local minimum")
        if r["std"] > abs(r["mean"]) * 0.5:
            notes.append("HIGH VARIANCE — inconsistent play")
        if r["std"] < abs(r["mean"]) * 0.2:
            notes.append("LOW VARIANCE — consistent and reliable")
        if r["fire_pct"] > 70:
            notes.append("FIRES aggressively — good targeting behaviour")
        if r["mean_ep_length"] == max(x["mean_ep_length"] for x in results):
            notes.append("LONGEST survival — best at staying alive")
        if notes:
            print(f"  {r['label'][:28]}: {' | '.join(notes)}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # JSON
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        # remove rewards list for cleaner JSON (keep stats only)
        output = [{k: v for k, v in r.items() if k != "rewards"}
                  for r in results]
        json.dump(output, f, indent=2)
    print(f"\n  JSON results  -> {json_path}")

    # Charts
    if len(results) > 1:
        save_comparison_chart(
            ranked,
            os.path.join(out_dir, "eval_comparison.png"))
        save_score_distributions(
            ranked,
            os.path.join(out_dir, "eval_distributions.png"))
    else:
        # single model — just print score list
        print(f"  Scores: {results[0]['rewards']}")

    print("\n" + "=" * 66)
    print(f"  All outputs -> {out_dir}/")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()