# Game PPO Training

Train language models to play strategic games using **PPO (Proximal Policy Optimization)** with **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

---

## Why Reinforcement Learning for Games?

Traditional supervised fine-tuning (SFT) has limitations for game-playing AI:

| Approach | Limitation |
|----------|------------|
| **SFT** | Requires expert demonstrations; model can only imitate, not improve beyond data |
| **Rule-based** | Brittle; fails on novel situations; doesn't generalize |
| **Search (MCTS)** | Computationally expensive at inference time |

**Reinforcement Learning solves these problems:**

- **Self-improvement**: Model learns from its own experience, not just expert data
- **Exploration**: Discovers novel strategies through trial and error
- **Optimization**: Directly optimizes for winning, not imitation
- **Generalization**: Learns principles, not just patterns

### Real-World Success Stories

- **AlphaGo/AlphaZero**: Defeated world champions using RL + self-play
- **OpenAI Five**: Beat professional Dota 2 teams
- **Pluribus**: Superhuman poker player

---

## PPO: Proximal Policy Optimization

PPO is the most popular RL algorithm for LLM fine-tuning (used in ChatGPT, Claude, etc.). Here's why:

### How PPO Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO Training Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. COLLECT ROLLOUTS                                             │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│     │  Model   │───▶│   Game   │───▶│ Rewards  │                │
│     │ (Policy) │    │   Env    │    │ (Win/Loss)│               │
│     └──────────┘    └──────────┘    └──────────┘                │
│                                                                  │
│  2. COMPUTE ADVANTAGES                                           │
│     A(s,a) = Q(s,a) - V(s)                                      │
│     "How much better was this action than average?"              │
│                                                                  │
│  3. UPDATE POLICY (with clipping)                                │
│     L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)                    │
│     "Improve policy but don't change too drastically"            │
│                                                                  │
│  4. REPEAT                                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key PPO Concepts

| Concept | Description |
|---------|-------------|
| **Policy (π)** | The model's strategy - maps states to action probabilities |
| **Value Function (V)** | Estimates expected future reward from a state |
| **Advantage (A)** | How much better an action is vs. the average |
| **KL Divergence** | Measures how much the policy has changed |
| **Clipping** | Prevents destructively large policy updates |

### PPO Hyperparameters

```python
# Critical hyperparameters
learning_rate = 1e-5      # Too high → unstable, too low → slow
cliprange = 0.2           # Limits policy change per update
init_kl_coef = 0.2        # KL penalty strength
ppo_epochs = 4            # Updates per batch of data
batch_size = 8            # Samples per training step
```

### Why PPO for LLMs?

1. **Stable**: Clipping prevents catastrophic forgetting
2. **Sample Efficient**: Reuses data multiple times per batch
3. **Scalable**: Works with large models and distributed training
4. **Proven**: Powers most RLHF systems today

---

## LoRA: Low-Rank Adaptation

Training a 3B+ parameter model requires 24+ GB VRAM. LoRA makes it possible on a single GPU.

### How LoRA Works

```
Original Weight Matrix W (d × k):
┌─────────────────────────────────────┐
│                                     │
│          W (frozen)                 │  ← 3B parameters, frozen
│                                     │
└─────────────────────────────────────┘

LoRA Decomposition:
┌─────────────────────────────────────┐
│                                     │
│          W (frozen)                 │
│                                     │
│    +                                │
│                                     │
│  ┌───┐   ┌─────────────────────┐   │
│  │ A │ × │         B           │   │  ← ~1M parameters, trainable
│  │r×d│   │        k×r          │   │
│  └───┘   └─────────────────────┘   │
│                                     │
└─────────────────────────────────────┘

W' = W + A × B  (r << d, k)
```

### LoRA Benefits

| Metric | Full Fine-tuning | LoRA (r=16) |
|--------|------------------|-------------|
| Trainable Params | 3B (100%) | ~15M (0.5%) |
| VRAM Required | 24+ GB | 8-10 GB |
| Training Speed | Baseline | 2-3x faster |
| Checkpoint Size | 6+ GB | ~50 MB |

### LoRA Configuration

```python
LoraConfig(
    r=16,                    # Rank - higher = more capacity
    lora_alpha=32,           # Scaling factor (usually 2×r)
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ],
    lora_dropout=0.05,       # Regularization
)
```

---

## Game Environment Features

### Supported Games (OpenSpiel)

| Game | Type | Players | Complexity |
|------|------|---------|------------|
| **Leduc Poker** | Imperfect info | 2 | Low |
| **Liar's Dice** | Imperfect info | 2 | Low |
| **Goofspiel** | Perfect info | 2 | Medium |
| **Hex** | Perfect info | 2 | Medium |
| **Clobber** | Perfect info | 2 | Medium |
| **Othello** | Perfect info | 2 | High |
| **Backgammon** | Stochastic | 2 | High |
| **Gin Rummy** | Imperfect info | 2 | High |

### Task ID System

Games are identified by 12-digit task IDs:

```
task_id = GGGGCCCCCCCC
         │   │
         │   └── Configuration variant (0-99,999,999)
         └────── Game index (0-11)

Examples:
  0              → Goofspiel, default config
  100000000      → Liar's Dice, default config
  200000042      → Leduc Poker, config variant 42
```

### Environment Features

- **Rule Enforcement**: Invalid moves rejected automatically
- **Multi-opponent**: Random or MCTS opponents
- **Configurable**: Different game variants via task ID
- **Efficient**: Local in-process execution

---

## Curriculum Learning

Training progresses through stages of increasing difficulty:

```
Stage 1: EASY                Stage 2: MEDIUM              Stage 3: HARD
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Random opponent │   →     │ Random opponent │   →     │ MCTS opponent   │
│ Learn mechanics │         │ Learn strategy  │         │ Master tactics  │
│ Target: 60%     │         │ Target: 60%     │         │ No ceiling      │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

### Automatic Progression

- Tracks rolling success rate over last 100 episodes
- Advances when success rate exceeds threshold (default: 70%)
- Prevents premature advancement to harder stages

### Failure Replay Buffer

Hard tasks are remembered and replayed:

```python
# Probability of sampling from failure buffer
failure_replay_prob = 0.3  # 30% from failures, 70% from curriculum

# Priority based on difficulty
priority = 1.0 - score  # Lower scores = higher priority
```

---

## Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd game_ppo_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install OpenSpiel (if not already installed)
pip install open_spiel
```

### 2. Configure

Edit `config/train_config.json`:

```json
{
  "model": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "use_4bit": true
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32
  },
  "ppo": {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "num_train_steps": 10000
  }
}
```

Edit `config/env_config.json`:

```json
{
  "allowed_game_indices": [0, 1, 2, 6, 7],
  "opponent": "mcts",
  "gamma": 0.99
}
```

### 3. Set Environment

```bash
# Path to OpenSpiel game agents (adjust as needed)
export OPENSPIEL_AGENTS_PATH="/path/to/openspiel/agents"

# Optional: Weights & Biases for tracking
export WANDB_API_KEY="your-key"

# Optional: Hugging Face for gated models
export HF_TOKEN="your-token"
```

### 4. Train

```bash
# Start training
python train_ppo.py

# Resume from checkpoint
python train_ppo.py --resume checkpoints/ppo_lora/checkpoint-500
```

---

## Project Structure

```
game_ppo_training/
├── train_ppo.py          # Main training script
├── curriculum.py         # Curriculum learning & failure replay
├── utils.py              # Utilities (logging, checkpoints)
├── config/
│   ├── train_config.json # Training hyperparameters
│   └── env_config.json   # Environment settings
├── requirements.txt      # Python dependencies
├── checkpoints/          # Saved model checkpoints
└── README.md             # This file
```

---

## Monitoring Training

### Console Output

```
2024-01-15 10:30:00 | INFO | Step 100/10000: Collecting rollouts...
2024-01-15 10:30:45 | INFO | Collected 24 samples from 5 games
2024-01-15 10:31:00 | INFO | Step 100 | Reward: 0.45 | Score: 0.52 | Success: 48.00%

🎓 Curriculum Progression: Stage 2/3
   Stage: medium | Opponent: random
   Success rate: 71.23%
```

### Weights & Biases

If enabled, tracks:
- Mean reward per step
- Success rate
- PPO loss components
- KL divergence
- Curriculum stage

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
"batch_size": 4,
"gradient_accumulation_steps": 8,

# Enable 4-bit quantization
"use_4bit": true,
"use_nested_quant": true
```

### Poor Performance

- **Low success rate**: Reduce learning rate to `5e-6`
- **High KL divergence**: Increase `init_kl_coef` to `0.5`
- **Unstable training**: Increase `ppo_epochs` to 5-6

### Environment Issues

```bash
# Verify OpenSpiel installation
python -c "import pyspiel; print('OK')"

# Check game agents path
echo $OPENSPIEL_AGENTS_PATH
ls $OPENSPIEL_AGENTS_PATH
```

---

## Expected Results

### Training Timeline

| Steps | Expected Behavior |
|-------|-------------------|
| 0-1000 | Learn game mechanics (30-40% success) |
| 1000-3000 | Develop basic strategy (50-60%) |
| 3000-5000 | Master easy stage, advance to medium |
| 5000-8000 | Learn medium tactics |
| 8000-10000 | Advance to hard, face MCTS |

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| GPU VRAM | 8-10 GB (4-bit) |
| Training Time | ~80-100 hours (10k steps) |
| Checkpoint Size | ~50 MB each |

---

