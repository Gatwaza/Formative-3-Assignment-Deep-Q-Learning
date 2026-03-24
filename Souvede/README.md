# DQN Agent for Demon Attack (Atari)

A Deep Q-Network (DQN) implementation using Stable Baselines3 to play the Demon Attack Atari game.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Accept Atari ROM License (First Time Only)

```bash
pip install gymnasium[accept-rom-license]
```

## Usage

### Training the Agent

```bash
# Basic training (500,000 timesteps)
python train.py --mode single --timesteps 500000

# Run hyperparameter experiments (10 configurations)
python train.py --mode experiments

# Run experiments in parallel (faster if GPU allows)
python train.py --mode experiments --parallel --workers 2

# Resume interrupted experiments (default behavior)
python train.py --mode experiments

# Force re-run all experiments
python train.py --mode experiments --no-resume

# View aggregated results without running
python train.py --mode experiments --results

# Custom training with specific hyperparameters
python train.py --mode single --lr 0.0001 --gamma 0.99 --batch-size 32 --timesteps 1000000
```

### Playing with Trained Agent

```bash
# Play 5 episodes with rendering
python play.py --episodes 5

# Load specific model
python play.py --model ./dqn_model.zip --episodes 3

# Evaluate without rendering (faster)
python play.py --evaluate --episodes 20

# Find and play with best model from experiments
python play.py --best
```

## Files

- `train.py` - Training script with hyperparameter tuning
- `play.py` - Playing script with greedy policy evaluation
- `HYPERPARAMETER_DOCUMENTATION.md` - Detailed hyperparameter analysis
- `requirements.txt` - Project dependencies

## Model Architecture

- **Policy**: CNNPolicy (Convolutional Neural Network)
- **Algorithm**: Deep Q-Network (DQN)
- **Frame Stacking**: 4 frames for temporal information
- **Preprocessing**: Atari wrappers (grayscale, resize, frame skip)

---

## Hyperparameter Experiments

We tested 10 different hyperparameter configurations to find the optimal settings for Demon Attack:

### Experiment Configurations

| #   | Experiment       | Learning Rate | Gamma | Batch Size | Epsilon (start→end) | Decay |
| --- | ---------------- | ------------- | ----- | ---------- | ------------------- | ----- |
| 1   | baseline         | 1e-4          | 0.99  | 32         | 1.0 → 0.05          | 0.1   |
| 2   | high_lr          | 5e-4          | 0.99  | 32         | 1.0 → 0.05          | 0.1   |
| 3   | low_lr           | 1e-5          | 0.99  | 32         | 1.0 → 0.05          | 0.1   |
| 4   | low_gamma        | 1e-4          | 0.95  | 32         | 1.0 → 0.05          | 0.1   |
| 5   | high_gamma       | 1e-4          | 0.999 | 32         | 1.0 → 0.05          | 0.1   |
| 6   | large_batch      | 1e-4          | 0.99  | 64         | 1.0 → 0.05          | 0.1   |
| 7   | small_batch      | 1e-4          | 0.99  | 16         | 1.0 → 0.05          | 0.1   |
| 8   | slow_exploration | 1e-4          | 0.99  | 32         | 1.0 → 0.05          | 0.3   |
| 9   | high_final_eps   | 1e-4          | 0.99  | 32         | 1.0 → 0.10          | 0.1   |
| 10  | optimized        | 2.5e-4        | 0.99  | 32         | 1.0 → 0.01          | 0.15  |

### Results (500,000 timesteps each)

| Experiment       | Mean Reward | Std Dev | Best Eval Reward | Episodes |
| ---------------- | ----------- | ------- | ---------------- | -------- |
| **high_gamma**   | 7.80        | 10.81   | **36.4**         | 6728     |
| baseline         | 7.80        | 11.04   | 23.4             | 6672     |
| large_batch      | 7.72        | 10.79   | 27.0             | 6538     |
| small_batch      | 7.26        | 10.25   | 22.4             | 6594     |
| low_gamma        | 7.21        | 10.32   | 24.0             | 7138     |
| high_lr          | 6.65        | 9.02    | 23.0             | 7360     |
| low_lr           | 4.37        | 5.04    | 10.2             | 8435     |
| slow_exploration | —           | —       | 11.0             | —        |
| high_final_eps   | —           | —       | —                | —        |
| optimized        | —           | —       | —                | —        |

> **Note**: Experiments marked with "—" did not complete fully. `slow_exploration` has partial evaluation data; `high_final_eps` and `optimized` were interrupted before completion.

**Best Configuration**: `high_gamma` (γ=0.999) achieved the highest evaluation reward of **36.4**, showing that long-term reward consideration is crucial for Demon Attack.

---

## Key Features

### Resumable Experiments

**Challenge**: Training 10 experiments sequentially takes many hours. If the terminal is killed mid-training, all progress could be lost.

**Solution**: The training script now supports resumable experiments:

- Completed experiments are automatically detected and skipped on restart
- Checkpoints are saved every 50,000 timesteps for recovery
- Best model is saved whenever evaluation improves
- Results are aggregated from disk, so any completed experiment counts

```bash
# Resume from where you left off
python train.py --mode experiments

# Force re-run even completed experiments
python train.py --mode experiments --no-resume
```

### Parallel Experiment Execution

**Challenge**: Running experiments sequentially is slow. With sufficient hardware resources, multiple experiments could run simultaneously.

**Solution**: Parallel execution using multiprocessing:

- Run N experiments simultaneously with `--parallel` flag
- Control worker count with `--workers N` (default: 2)
- Each worker handles one complete experiment
- Results still aggregated correctly from disk

```bash
# Run 2 experiments at a time
python train.py --mode experiments --parallel

# Run 3 experiments at a time (requires more GPU memory)
python train.py --mode experiments --parallel --workers 3
```

**Memory Consideration**: Each DQN model uses ~2-4GB GPU memory. Only increase workers if you have sufficient VRAM.

### Additional Features

- Automatic hyperparameter experiment logging
- TensorBoard integration for training visualization
- Best model checkpoint saving (every 10k eval steps)
- Periodic checkpoint saves (every 50k training steps)
- Greedy Q-policy for evaluation (deterministic action selection)
- Results aggregation across all completed experiments
