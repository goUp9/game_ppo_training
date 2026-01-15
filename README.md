# PPO Training for Game and Trace Environments

Train language models using PPO (Proximal Policy Optimization) with LoRA (Low-Rank Adaptation) on strategic games and code tracing tasks via the affinetes framework.

## Architecture

The training pipeline uses:
- Model with LoRA adapters for parameter-efficient training
- vLLM server for fast model inference
- affinetes containers for environment execution (OpenSpiel/Trace)
- PPO Trainer (TRL) for policy optimization

## Supported Environments

| Environment | Docker Image | Description |
|-------------|--------------|-------------|
| Game | affinefoundation/game:openspiel | Strategic board/card games |
| Trace | affinefoundation/trace:latest | Code execution prediction |

## Why Reinforcement Learning?

RL enables self-improvement beyond expert demonstrations, discovers novel strategies through exploration, and directly optimizes for task success.

## PPO Overview

1. Collect rollouts (model responses + environment rewards)
2. Compute advantages (how much better than average)
3. Update policy with clipping (stable updates)
4. Repeat

Key hyperparameters: learning_rate=1e-5, cliprange=0.2, ppo_epochs=4

## LoRA Benefits

| Metric | Full Fine-tuning | LoRA (r=16) |
|--------|------------------|-------------|
| Trainable Params | 3B (100%) | ~15M (0.5%) |
| VRAM Required | 24+ GB | 8-10 GB |
| Checkpoint Size | 6+ GB | ~50 MB |

## Games

Supported: Goofspiel, Liars Dice, Leduc Poker, Gin Rummy, Othello, Backgammon, Hex, Clobber etc ...

Task ID format: GGGGCCCCCCCC (game index + config variant)

## Trace Environment

Predict exact stdout of Python programs with injected debug prints (__DBG_N__).

## Quick Start

### 1. Install

```bash
cd game_ppo_training
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
export CHUTES_API_KEY="your-api-key"
export WANDB_API_KEY="your-wandb-key"  # optional
```

### 3. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --port 8000
```

### 4. Configure (config/env_config.json)

For OpenSpiel:
```json
{
  "env_type": "openspiel",
  "env_mode": "docker",
  "use_vllm": true,
  "vllm_base_url": "http://localhost:8000/v1"
}
```

For Trace:
```json
{
  "env_type": "trace",
  "env_mode": "docker",
  "use_vllm": true,
  "vllm_base_url": "http://localhost:8000/v1"
}
```

### 5. Train

```bash
python train_ppo.py
```

## Project Structure

```
game_ppo_training/
├── train_ppo.py          # Main training script
├── curriculum.py         # Curriculum learning
├── utils.py              # Utilities
├── config/
│   ├── train_config.json # Training hyperparameters
│   ├── env_config.json   # Environment settings
│   └── trace_config.json # Trace preset
├── requirements.txt      # Dependencies
└── README.md
```

## affinetes Usage

```python
import affinetes as af_env

env = af_env.load_env(
    image="affinefoundation/game:openspiel",
    mode="docker",
    env_vars={"CHUTES_API_KEY": "xxx"},
)

result = await env.evaluate(
    task_id=100000000,
    seed=42,
    model="Qwen/Qwen3-4B",
    base_url="http://localhost:8000/v1",
)

await env.cleanup()
```

## Curriculum Learning

Stages: Easy (random opponent) -> Medium (random) -> Hard (MCTS)
Automatic progression when success rate exceeds threshold (70%)

## Resource Requirements

- GPU VRAM: 8-10 GB (4-bit quantization)
- Docker: Running with 8GB+ memory
- vLLM Server: Separate or shared GPU
- Training Time: ~80-100 hours (10k steps)
