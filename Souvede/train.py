"""
DQN Agent Training Script for Demon Attack (Atari)
This script trains a DQN agent using Stable Baselines3 on the Demon Attack environment.
"""

from datetime import datetime
import json
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import DQN
import os
import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to log training metrics including rewards and episode lengths.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        # Track rewards and episode info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    if self.verbose > 0:
                        print(f"Episode {len(self.episode_rewards)}: "
                              f"Reward = {info['episode']['r']:.2f}, "
                              f"Length = {info['episode']['l']}")
        return True

    def _on_training_end(self) -> None:
        # Save metrics to file
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
            'mean_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
        }
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n=== Training Complete ===")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(
            f"Mean Reward: {metrics['mean_reward']:.2f} (+/- {metrics['std_reward']:.2f})")
        print(f"Mean Episode Length: {metrics['mean_length']:.2f}")


def make_atari_env(env_id: str = "ALE/DemonAttack-v5", render_mode: str = None):
    """
    Create and wrap the Atari environment with proper preprocessing.
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train_dqn(
    policy: str = "CnnPolicy",
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    exploration_fraction: float = 0.1,
    buffer_size: int = 100000,
    learning_starts: int = 50000,
    target_update_interval: int = 10000,
    total_timesteps: int = 1000000,
    log_dir: str = "./logs",
    model_save_path: str = "./dqn_model",
    experiment_name: str = "default",
    verbose: int = 1
):
    """
    Train a DQN agent with specified hyperparameters.

    Args:
        policy: "CnnPolicy" or "MlpPolicy"
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        batch_size: Number of samples per gradient update
        exploration_initial_eps: Initial epsilon for exploration
        exploration_final_eps: Final epsilon for exploration
        exploration_fraction: Fraction of training for epsilon decay
        buffer_size: Size of the replay buffer
        learning_starts: How many steps before learning starts
        target_update_interval: Update target network every N steps
        total_timesteps: Total training timesteps
        log_dir: Directory for tensorboard logs
        model_save_path: Path to save the trained model
        experiment_name: Name for this experiment
        verbose: Verbosity level

    Returns:
        Trained DQN model and training metrics
    """

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Log hyperparameters
    hyperparams = {
        "policy": policy,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "batch_size": batch_size,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": exploration_fraction,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
        "total_timesteps": total_timesteps,
        "experiment_name": experiment_name
    }

    with open(os.path.join(exp_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print("\n" + "="*60)
    print(f"Training DQN Agent - Experiment: {experiment_name}")
    print("="*60)
    print(f"Policy: {policy}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch Size: {batch_size}")
    print(f"Epsilon: {exploration_initial_eps} -> {exploration_final_eps} "
          f"(decay fraction: {exploration_fraction})")
    print(f"Total Timesteps: {total_timesteps}")
    print("="*60 + "\n")

    # Create the environment
    env = DummyVecEnv([make_atari_env("ALE/DemonAttack-v5")])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for temporal info

    # Create evaluation environment
    eval_env = DummyVecEnv([make_atari_env("ALE/DemonAttack-v5")])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Initialize the DQN agent
    model = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        tensorboard_log=exp_dir,
        verbose=verbose,
        device="auto"  # Will use GPU if available
    )

    # Create callbacks
    metrics_callback = TrainingMetricsCallback(log_dir=exp_dir, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_dir,
        log_path=exp_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Checkpoint callback - saves model every 50k steps for recovery
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=exp_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback, eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save the final model
    final_model_path = f"{model_save_path}.zip"
    model.save(model_save_path)
    print(f"\nModel saved to: {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()

    return model, metrics_callback.episode_rewards, metrics_callback.episode_lengths


def is_experiment_completed(experiment_name: str, log_dir: str = "./logs") -> bool:
    """
    Check if an experiment has already been completed by looking for training_metrics.json.
    """
    pattern = os.path.join(
        log_dir, f"{experiment_name}_*", "training_metrics.json")
    matches = glob.glob(pattern)
    return len(matches) > 0


def run_single_experiment(exp: dict, resume: bool = True) -> dict:
    """
    Run a single experiment. Designed to be called by multiprocessing pool.

    Args:
        exp: Experiment configuration dictionary
        resume: Skip if already completed

    Returns:
        Result dictionary with experiment metrics or error
    """
    exp_name = exp["name"]

    # Check if already completed
    if resume and is_experiment_completed(exp_name):
        print(f"[{exp_name}] Skipping - already completed")
        return {"experiment": exp_name, "skipped": True}

    print(f"[{exp_name}] Starting experiment...")

    try:
        _, rewards, lengths = train_dqn(
            policy="CnnPolicy",
            learning_rate=exp["learning_rate"],
            gamma=exp["gamma"],
            batch_size=exp["batch_size"],
            exploration_initial_eps=exp["exploration_initial_eps"],
            exploration_final_eps=exp["exploration_final_eps"],
            exploration_fraction=exp["exploration_fraction"],
            total_timesteps=exp["total_timesteps"],
            experiment_name=exp_name,
            verbose=0  # Reduce noise in parallel mode
        )

        result = {
            "experiment": exp_name,
            "hyperparameters": exp,
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "std_reward": float(np.std(rewards)) if rewards else 0,
            "max_reward": float(np.max(rewards)) if rewards else 0,
            "total_episodes": len(rewards),
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0
        }
        print(f"[{exp_name}] Completed! Mean reward: {result['mean_reward']:.2f}")
        return result

    except Exception as e:
        print(f"[{exp_name}] Failed: {e}")
        return {"experiment": exp_name, "error": str(e)}


def aggregate_results_from_disk(log_dir: str = "./logs") -> list:
    """
    Aggregate results from all completed experiments by reading from disk.
    This allows resuming and combining results even after interruption.
    """
    results = []

    # Find all experiment directories with training_metrics.json
    pattern = os.path.join(log_dir, "*", "training_metrics.json")
    metrics_files = glob.glob(pattern)

    for metrics_path in metrics_files:
        exp_dir = os.path.dirname(metrics_path)
        exp_name = os.path.basename(exp_dir).rsplit('_', 2)[
            0]  # Remove timestamp

        # Skip single_run and cnn_policy experiments
        if exp_name in ["single_run", "cnn_policy"]:
            continue

        try:
            # Load training metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Load hyperparameters
            hyperparams_path = os.path.join(exp_dir, "hyperparameters.json")
            hyperparams = {}
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    hyperparams = json.load(f)

            # Load evaluation results if available
            eval_path = os.path.join(exp_dir, "evaluations.npz")
            best_eval_reward = None
            if os.path.exists(eval_path):
                eval_data = np.load(eval_path)
                if 'results' in eval_data:
                    best_eval_reward = float(
                        np.max(np.mean(eval_data['results'], axis=1)))

            result = {
                "experiment": exp_name,
                "exp_dir": exp_dir,
                "hyperparameters": hyperparams,
                "mean_reward": metrics.get('mean_reward', 0),
                "std_reward": metrics.get('std_reward', 0),
                "max_reward": max(metrics.get('episode_rewards', [0])) if metrics.get('episode_rewards') else 0,
                "total_episodes": len(metrics.get('episode_rewards', [])),
                "mean_episode_length": metrics.get('mean_length', 0),
                "best_eval_reward": best_eval_reward
            }
            results.append(result)

        except Exception as e:
            print(f"Warning: Could not load results from {exp_dir}: {e}")

    return results


def run_hyperparameter_experiments(resume: bool = True, parallel: bool = False, max_workers: int = 2):
    """
    Run multiple experiments with different hyperparameter configurations.
    This function tests various combinations and logs results.

    Args:
        resume: If True, skip experiments that have already been completed.
        parallel: If True, run experiments in parallel using multiprocessing.
        max_workers: Maximum number of parallel experiments (only if parallel=True).
    """

    # Define hyperparameter configurations to test
    experiments = [
        # Experiment 1: Baseline configuration
        {
            "name": "baseline",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 2: Higher learning rate
        {
            "name": "high_lr",
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 3: Lower learning rate
        {
            "name": "low_lr",
            "learning_rate": 1e-5,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 4: Lower gamma (more short-term focused)
        {
            "name": "low_gamma",
            "learning_rate": 1e-4,
            "gamma": 0.95,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 5: Higher gamma (more long-term focused)
        {
            "name": "high_gamma",
            "learning_rate": 1e-4,
            "gamma": 0.999,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 6: Larger batch size
        {
            "name": "large_batch",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 7: Smaller batch size
        {
            "name": "small_batch",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 16,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 8: Slower exploration decay
        {
            "name": "slow_exploration",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "exploration_fraction": 0.3,
            "total_timesteps": 500000
        },
        # Experiment 9: Higher final epsilon (more exploration)
        {
            "name": "high_final_eps",
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.1,
            "total_timesteps": 500000
        },
        # Experiment 10: Optimized configuration
        {
            "name": "optimized",
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.15,
            "total_timesteps": 500000
        }
    ]

    if parallel:
        # Parallel execution using ProcessPoolExecutor
        print(
            f"\nRunning experiments in PARALLEL mode with {max_workers} workers")
        print(f"Total experiments: {len(experiments)}")
        print("="*60 + "\n")

        # Filter experiments to run (skip completed if resume=True)
        experiments_to_run = []
        for exp in experiments:
            if resume and is_experiment_completed(exp['name']):
                print(f"[{exp['name']}] Already completed - will skip")
            else:
                experiments_to_run.append(exp)

        print(f"\nExperiments to run: {len(experiments_to_run)}")
        print("="*60 + "\n")

        if experiments_to_run:
            # Use spawn context on Windows for CUDA compatibility
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single_experiment, exp, resume): exp['name']
                    for exp in experiments_to_run
                }

                for future in as_completed(futures):
                    exp_name = futures[future]
                    try:
                        result = future.result()
                        print(f"[{exp_name}] Finished processing")
                    except Exception as e:
                        print(f"[{exp_name}] Exception: {e}")
    else:
        # Sequential execution (original behavior)
        for i, exp in enumerate(experiments):
            print(f"\n{'#'*60}")
            print(f"Experiment {i+1}/{len(experiments)}: {exp['name']}")
            print(f"{'#'*60}")

            # Check if experiment already completed (for resume support)
            if resume and is_experiment_completed(exp['name']):
                print(
                    f"Skipping '{exp['name']}' - already completed. Use --no-resume to force re-run.")
                continue

            try:
                _, rewards, lengths = train_dqn(
                    policy="CnnPolicy",
                    learning_rate=exp["learning_rate"],
                    gamma=exp["gamma"],
                    batch_size=exp["batch_size"],
                    exploration_initial_eps=exp["exploration_initial_eps"],
                    exploration_final_eps=exp["exploration_final_eps"],
                    exploration_fraction=exp["exploration_fraction"],
                    total_timesteps=exp["total_timesteps"],
                    experiment_name=exp["name"]
                )
                print(f"Experiment {exp['name']} completed!")

            except Exception as e:
                print(f"Experiment {exp['name']} failed: {e}")

    # Aggregate results from disk (works for both parallel and sequential)
    results = aggregate_results_from_disk()

    # Save aggregated results
    os.makedirs('./logs', exist_ok=True)
    with open('./logs/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<20} {'Mean Reward':<15} {'Best Eval':<15} {'Episodes':<10}")
    print("-"*80)

    # Sort by best eval reward (or mean reward as fallback)
    sorted_results = sorted(
        results,
        key=lambda r: r.get('best_eval_reward') or r.get('mean_reward', 0),
        reverse=True
    )

    for r in sorted_results:
        eval_str = f"{r['best_eval_reward']:.2f}" if r.get(
            'best_eval_reward') else "N/A"
        print(
            f"{r['experiment']:<20} {r['mean_reward']:<15.2f} {eval_str:<15} {r['total_episodes']:<10}")

    # Print best experiment
    if sorted_results:
        best = sorted_results[0]
        print("\n" + "="*80)
        print(f"BEST EXPERIMENT: {best['experiment']}")
        print(f"  Mean Reward: {best['mean_reward']:.2f}")
        if best.get('best_eval_reward'):
            print(f"  Best Eval Reward: {best['best_eval_reward']:.2f}")
        print(f"  Model Location: {best.get('exp_dir', 'N/A')}")
        print("="*80)

    return results


def compare_policies():
    """
    Compare MLPPolicy vs CNNPolicy performance.
    Note: For Atari games with image observations, CNNPolicy is typically better.
    MLPPolicy would require flattened observations and is not recommended for images.
    """
    print("\n" + "="*60)
    print("POLICY COMPARISON: CNNPolicy vs MlpPolicy")
    print("="*60)
    print("\nNote: For Atari games with image-based observations:")
    print("- CNNPolicy: Designed for image inputs, uses convolutional layers")
    print("- MlpPolicy: Designed for vector inputs, would require flattening images")
    print("\nCNNPolicy is the recommended choice for Atari environments.")
    print("MlpPolicy would result in poor performance due to loss of spatial information.")
    print("="*60 + "\n")

    # Train with CNNPolicy (recommended)
    print("Training with CNNPolicy (Recommended for Atari)...")
    model_cnn, rewards_cnn, lengths_cnn = train_dqn(
        policy="CnnPolicy",
        total_timesteps=200000,
        experiment_name="cnn_policy"
    )

    return {
        "cnn_policy": {
            "mean_reward": float(np.mean(rewards_cnn)) if rewards_cnn else 0,
            "max_reward": float(np.max(rewards_cnn)) if rewards_cnn else 0
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DQN Agent for Demon Attack")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "experiments", "compare"],
                        help="Training mode: single run, hyperparameter experiments, or policy comparison")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--eps-start", type=float, default=1.0,
                        help="Initial exploration epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05,
                        help="Final exploration epsilon")
    parser.add_argument("--eps-fraction", type=float, default=0.1,
                        help="Fraction of training for epsilon decay")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip completed experiments (force re-run all)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run experiments in parallel (use with caution on GPU)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers (default: 2)")
    parser.add_argument("--results", action="store_true",
                        help="Just show aggregated results from completed experiments")

    args = parser.parse_args()

    if args.mode == "single":
        # Single training run with specified hyperparameters
        print("Running single training session...")
        model, rewards, lengths = train_dqn(
            policy="CnnPolicy",
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            exploration_initial_eps=args.eps_start,
            exploration_final_eps=args.eps_end,
            exploration_fraction=args.eps_fraction,
            total_timesteps=args.timesteps,
            experiment_name="single_run"
        )

    elif args.mode == "experiments":
        # Just show results if --results flag
        if args.results:
            print("Aggregating results from completed experiments...")
            results = aggregate_results_from_disk()
            sorted_results = sorted(
                results,
                key=lambda r: r.get('best_eval_reward') or r.get(
                    'mean_reward', 0),
                reverse=True
            )
            print(f"\n{'Experiment':<20} {'Mean Reward':<15} {'Best Eval':<15}")
            print("-"*50)
            for r in sorted_results:
                eval_str = f"{r['best_eval_reward']:.2f}" if r.get(
                    'best_eval_reward') else "N/A"
                print(
                    f"{r['experiment']:<20} {r['mean_reward']:<15.2f} {eval_str:<15}")
            if sorted_results:
                print(f"\nBest: {sorted_results[0]['experiment']}")
        else:
            # Run experiments
            print("Running hyperparameter tuning experiments...")
            if not args.no_resume:
                print("(Resumable mode: completed experiments will be skipped)")
            if args.parallel:
                print(f"(Parallel mode: {args.workers} workers)")
            results = run_hyperparameter_experiments(
                resume=not args.no_resume,
                parallel=args.parallel,
                max_workers=args.workers
            )

    elif args.mode == "compare":
        # Compare policies
        print("Running policy comparison...")
        results = compare_policies()
