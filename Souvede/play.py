"""
DQN Agent Playing Script for Demon Attack (Atari)
This script loads a trained DQN model and plays the game with visualization.
"""

import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import DQN
import os
import time
import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


def make_atari_env(env_id: str = "ALE/DemonAttack-v5", render_mode: str = "human"):
    """
    Create and wrap the Atari environment with proper preprocessing.

    Args:
        env_id: The Gymnasium environment ID
        render_mode: "human" for GUI display, "rgb_array" for headless

    Returns:
        Wrapped environment
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        return env
    return _init


def play_game(
    model_path: str = "./dqn_model.zip",
    num_episodes: int = 5,
    deterministic: bool = True,
    render: bool = True,
    frame_delay: float = 0.01,
    verbose: bool = True
):
    """
    Load a trained DQN model and play the game.

    This function uses a greedy policy (deterministic=True) which ensures
    the agent always selects the action with the highest Q-value,
    maximizing performance during evaluation.

    Args:
        model_path: Path to the saved DQN model (.zip file)
        num_episodes: Number of episodes to play
        deterministic: If True, uses greedy Q-policy (always select max Q-value action)
        render: Whether to render the game visually
        frame_delay: Delay between frames (seconds) for better visualization
        verbose: Whether to print episode statistics

    Returns:
        Dictionary containing episode rewards and statistics
    """

    # Verify model exists
    if not os.path.exists(model_path):
        # Try without .zip extension
        if not model_path.endswith('.zip'):
            model_path_with_zip = model_path + '.zip'
            if os.path.exists(model_path_with_zip):
                model_path = model_path_with_zip
            else:
                raise FileNotFoundError(
                    f"Model not found at {model_path} or {model_path_with_zip}. "
                    "Please train a model first using train.py"
                )
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train a model first using train.py"
            )

    print("="*60)
    print("DQN AGENT - DEMON ATTACK PLAYER")
    print("="*60)
    print(f"Loading model from: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(
        f"Policy: {'Greedy (deterministic)' if deterministic else 'Stochastic'}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    print("="*60 + "\n")

    # Load the trained model
    model = DQN.load(model_path)
    print("Model loaded successfully!")

    # Create the environment with rendering
    render_mode = "human" if render else "rgb_array"
    env = DummyVecEnv(
        [make_atari_env("ALE/DemonAttack-v5", render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=4)  # Same frame stacking as training

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []

    print("\nStarting gameplay...\n")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"Episode {episode + 1}/{num_episodes} starting...")

        while not done:
            # Get action using greedy policy (deterministic=True)
            # This ensures the agent always picks the action with highest Q-value
            action, _states = model.predict(obs, deterministic=deterministic)

            # Execute action in environment
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            steps += 1

            # Add small delay for better visualization
            if render and frame_delay > 0:
                time.sleep(frame_delay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if verbose:
            print(f"Episode {episode + 1} finished!")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            print("-"*40)

    # Close environment
    env.close()

    # Calculate and display statistics
    stats = {
        "episodes": num_episodes,
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths))
    }

    print("\n" + "="*60)
    print("GAMEPLAY SUMMARY")
    print("="*60)
    print(f"Total Episodes Played: {num_episodes}")
    print(
        f"Mean Reward: {stats['mean_reward']:.2f} (+/- {stats['std_reward']:.2f})")
    print(f"Max Reward: {stats['max_reward']:.2f}")
    print(f"Min Reward: {stats['min_reward']:.2f}")
    print(f"Mean Episode Length: {stats['mean_length']:.2f} steps")
    print("="*60)

    return stats


def evaluate_model(
    model_path: str = "./dqn_model.zip",
    num_episodes: int = 20,
    verbose: bool = False
):
    """
    Evaluate the model performance without rendering (faster).

    Uses greedy Q-policy for evaluation to measure true model performance.

    Args:
        model_path: Path to the saved DQN model
        num_episodes: Number of episodes for evaluation
        verbose: Whether to print per-episode stats

    Returns:
        Evaluation statistics
    """
    return play_game(
        model_path=model_path,
        num_episodes=num_episodes,
        deterministic=True,  # Greedy policy
        render=False,
        verbose=verbose
    )


def watch_best_model(
    logs_dir: str = "./logs",
    num_episodes: int = 3
):
    """
    Find and play with the best model from training experiments.

    Args:
        logs_dir: Directory containing training logs
        num_episodes: Number of episodes to play
    """
    # Look for best_model.zip files in experiment directories
    best_model_path = None

    for root, dirs, files in os.walk(logs_dir):
        if "best_model.zip" in files:
            best_model_path = os.path.join(root, "best_model.zip")
            print(f"Found best model at: {best_model_path}")
            break

    if best_model_path is None:
        # Fall back to default model path
        if os.path.exists("./dqn_model.zip"):
            best_model_path = "./dqn_model.zip"
        else:
            print("No trained model found. Please run train.py first.")
            return

    return play_game(
        model_path=best_model_path,
        num_episodes=num_episodes,
        deterministic=True,
        render=True
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Play Demon Attack with trained DQN agent")
    parser.add_argument("--model", type=str, default="./dqn_model.zip",
                        help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering (for faster evaluation)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of greedy")
    parser.add_argument("--delay", type=float, default=0.01,
                        help="Frame delay in seconds for visualization")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation mode (no rendering, more episodes)")
    parser.add_argument("--best", action="store_true",
                        help="Find and play with the best model from logs")

    args = parser.parse_args()

    if args.best:
        # Find and play with best model
        watch_best_model(num_episodes=args.episodes)

    elif args.evaluate:
        # Evaluation mode (no rendering, more episodes)
        stats = evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            verbose=True
        )

    else:
        # Normal play mode with rendering
        stats = play_game(
            model_path=args.model,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
            render=not args.no_render,
            frame_delay=args.delay
        )
