"""
play.py — DQN Agent Evaluation Script for Demon Attack (Atari)
Environment: ALE/DemonAttack-v5
Framework: Stable Baselines3 + Gymnasium

Loads the best saved model and runs several episodes with greedy action
selection (GreedyQPolicy equivalent: always pick argmax Q-value).
"""

import time
import numpy as np
import ale_py
import gymnasium as gym

# ── Fix ALE registration ──────────────────────────────────────
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ENV_ID      = "ALE/DemonAttack-v5"
MODEL_PATH  = "dqn_model.zip"       # path to the saved model
NUM_EPISODES = 5                    # number of episodes to play
RENDER_DELAY = 0.01                 # seconds between frames (lower = faster)


# ─────────────────────────────────────────────
# Environment factory (with rendering)
# ─────────────────────────────────────────────
def make_env(env_id: str):
    """Creates a wrapped Atari env with human rendering enabled."""
    def _init():
        env = gym.make(env_id, render_mode="human")
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(env_id: str):
    envs = DummyVecEnv([make_env(env_id)])
    envs = VecFrameStack(envs, n_stack=4)
    return envs


# ─────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────
def greedy_play(model: DQN, vec_env, num_episodes: int = 5):
    """
    Runs `num_episodes` episodes using a greedy policy (ε = 0),
    which is the SB3 equivalent of GreedyQPolicy — the agent always
    picks the action with the highest predicted Q-value.
    """
    episode_rewards = []
    episode_lengths = []

    obs = vec_env.reset()

    for ep in range(1, num_episodes + 1):
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        print(f"\n{'─'*40}")
        print(f"  Episode {ep} / {num_episodes}")
        print(f"{'─'*40}")

        while not done:
            # deterministic=True → argmax Q-value (greedy / no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)

            ep_reward += float(reward[0])
            ep_steps  += 1
            done       = bool(done_arr[0])

            time.sleep(RENDER_DELAY)   # slow down for human viewing

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        print(f"  ✔ Reward: {ep_reward:.2f}  |  Steps: {ep_steps}")

        # reset for next episode
        obs = vec_env.reset()

    return episode_rewards, episode_lengths


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  DQN Agent — Demon Attack (Evaluation Mode)")
    print("=" * 50)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Episodes  : {NUM_EPISODES}")
    print(f"  Policy    : Greedy (deterministic=True)")
    print("=" * 50)

    # ── Load model ───────────────────────────────────────────────
    print(f"\n[INFO] Loading model from '{MODEL_PATH}' …")
    model = DQN.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # ── Build environment ────────────────────────────────────────
    print(f"[INFO] Building environment: {ENV_ID} (render_mode='human') …")
    vec_env = build_vec_env(ENV_ID)

    # ── Play episodes ────────────────────────────────────────────
    rewards, lengths = greedy_play(model, vec_env, NUM_EPISODES)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Episodes played : {NUM_EPISODES}")
    print(f"  Mean reward     : {np.mean(rewards):.2f}")
    print(f"  Best reward     : {max(rewards):.2f}")
    print(f"  Worst reward    : {min(rewards):.2f}")
    print(f"  Mean ep length  : {np.mean(lengths):.0f} steps")
    print(f"{'='*50}\n")

    vec_env.close()


if __name__ == "__main__":
    main()
