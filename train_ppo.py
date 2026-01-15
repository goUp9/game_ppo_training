#!/usr/bin/env python3
"""
PPO + LoRA Training for OpenSpiel Game Environments

This script trains a language model to play strategic games using:
- PPO (Proximal Policy Optimization) for policy learning
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Curriculum learning for progressive difficulty
- Failure-based replay for hard case mastery

Usage:
    python train_ppo.py                          # Start fresh training
    python train_ppo.py --resume checkpoint-100  # Resume from checkpoint
"""

import os
import sys
import json
import torch
import asyncio
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import numpy as np

from curriculum import CurriculumSampler, TaskConfig
from utils import setup_logging, save_checkpoint, load_checkpoint


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration for base LLM"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class PPOTrainingConfig:
    """PPO training hyperparameters"""
    # Batch settings
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Learning
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    
    # PPO-specific
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # Training schedule
    num_train_steps: int = 10000
    save_freq: int = 500
    eval_freq: int = 100
    log_freq: int = 10
    
    # Generation
    max_seq_length: int = 2048
    max_new_tokens: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Paths
    output_dir: str = "./checkpoints/ppo_lora"
    resume_from: Optional[str] = None
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "game-ppo-training"
    
    # Collection settings
    min_valid_samples_per_step: int = 8
    max_collection_episodes: int = 500


# =============================================================================
# OpenSpiel Episode Runner
# =============================================================================

@dataclass
class TurnSample:
    """Single turn sample for PPO training"""
    prompt_text: str
    response_text: str
    reward: float
    info: Dict[str, Any]


@dataclass
class EpisodeData:
    """Complete episode data for logging"""
    task_id: int
    seed: int
    game_name: str
    opponent: str
    final_reward: float
    total_turns: int
    valid_turns: int
    invalid_turns: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "game_name": self.game_name,
            "opponent": self.opponent,
            "final_reward": self.final_reward,
            "total_turns": self.total_turns,
            "valid_turns": self.valid_turns,
            "invalid_turns": self.invalid_turns,
        }


def _import_openspiel():
    """Import OpenSpiel components"""
    import pyspiel
    from open_spiel.python.algorithms import mcts
    from open_spiel.python.bots import uniform_random
    
    # Import game configuration (adjust path as needed)
    openspiel_path = os.environ.get("OPENSPIEL_AGENTS_PATH", "")
    if openspiel_path and openspiel_path not in sys.path:
        sys.path.insert(0, openspiel_path)
    
    from game_config import create_game
    from agents import GAME_AGENTS
    
    return pyspiel, mcts, uniform_random, create_game, GAME_AGENTS


def _build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build chat prompt from messages"""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _parse_action_id(text: str) -> Optional[int]:
    """Extract action ID from model output"""
    import re
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _create_opponent_bot(opponent: str, player_id: int, seed: int, game, agent):
    """Create opponent bot (random or MCTS)"""
    pyspiel, mcts, uniform_random, _, _ = _import_openspiel()
    
    if opponent == "random":
        return uniform_random.UniformRandomBot(player_id, np.random.RandomState(seed))
    
    if opponent == "mcts":
        mcts_config = agent.get_mcts_config()
        if mcts_config is None:
            return uniform_random.UniformRandomBot(player_id, np.random.RandomState(seed))
        
        max_simulations, n_rollouts = mcts_config
        
        class RandomRolloutEvaluator(mcts.Evaluator):
            def __init__(self, n_rollouts=1, random_state=None):
                self._n_rollouts = n_rollouts
                self._random_state = random_state or np.random.RandomState()
            
            def evaluate(self, state):
                if state.is_terminal():
                    return state.returns()
                total_returns = np.zeros(state.num_players())
                for _ in range(self._n_rollouts):
                    working_state = state.clone()
                    while not working_state.is_terminal():
                        legal = working_state.legal_actions()
                        if not legal:
                            break
                        working_state.apply_action(self._random_state.choice(legal))
                    total_returns += working_state.returns()
                return total_returns / self._n_rollouts
            
            def prior(self, state):
                legal = state.legal_actions()
                return [(a, 1.0 / len(legal)) for a in legal] if legal else []
        
        evaluator = RandomRolloutEvaluator(n_rollouts, np.random.RandomState(seed))
        return mcts.MCTSBot(game, 1.414, max_simulations, evaluator, np.random.RandomState(seed))
    
    raise ValueError(f"Unknown opponent: {opponent}")


def run_episode(
    model, tokenizer, task_id: int, seed: int,
    opponent: str = "mcts", temperature: float = 0.8,
    max_new_tokens: int = 8, max_seq_length: int = 1024,
    gamma: float = 0.99, device: str = "cuda"
) -> Tuple[List[TurnSample], Dict[str, Any], EpisodeData]:
    """
    Run a complete game episode and collect training samples.
    
    Returns:
        turn_samples: List of (prompt, response, reward) for PPO
        episode_info: Summary dict for curriculum updates
        episode_data: Complete episode data for logging
    """
    pyspiel, _, _, create_game, GAME_AGENTS = _import_openspiel()
    
    # Create game
    game, game_cfg = create_game(task_id)
    game_name = game_cfg["game_name"]
    num_players = game.num_players()
    llm_player_id = seed % num_players
    
    # Get game-specific agent
    agent_class = GAME_AGENTS.get(game_name)
    if not agent_class:
        raise ValueError(f"No agent for game: {game_name}")
    agent = agent_class()
    
    # Create opponent bots
    bots = [None] * num_players
    for pid in range(num_players):
        if pid != llm_player_id:
            bots[pid] = _create_opponent_bot(opponent, pid, seed + pid, game, agent)
    
    # Initialize game state
    state = game.new_initial_state()
    messages = [{"role": "system", "content": agent.generate_system_prompt()}]
    turn_samples: List[TurnSample] = []
    valid_count, invalid_count = 0, 0
    rng = np.random.RandomState(seed)
    
    # Play game
    while not state.is_terminal():
        cur_player = state.current_player()
        
        # Handle chance nodes
        if cur_player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choice(actions, p=probs))
            continue
        
        # Opponent turn
        if cur_player != llm_player_id:
            if bots[cur_player]:
                state.apply_action(bots[cur_player].step(state))
            else:
                legal = state.legal_actions(cur_player)
                state.apply_action(rng.choice(legal) if legal else 0)
            continue
        
        # LLM turn
        legal_actions = state.legal_actions(llm_player_id)
        user_prompt = agent.generate_user_prompt(state, llm_player_id, legal_actions)
        messages.append({"role": "user", "content": user_prompt})
        
        prompt_text = _build_chat_prompt(tokenizer, messages)
        
        # Generate action
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=max(32, max_seq_length - max_new_tokens)
        ).to(device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=0.95, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action_id = _parse_action_id(response_text)
        
        # Validate action
        is_valid = action_id is not None and action_id in legal_actions
        actual_action = action_id if is_valid else (rng.choice(legal_actions) if legal_actions else 0)
        
        state.apply_action(actual_action)
        messages.append({"role": "assistant", "content": str(actual_action)})
        
        if is_valid:
            valid_count += 1
            turn_samples.append(TurnSample(
                prompt_text=prompt_text,
                response_text=response_text.strip(),
                reward=0.0,
                info={"game_name": game_name, "task_id": task_id}
            ))
        else:
            invalid_count += 1
    
    # Compute final reward
    returns = state.returns()
    llm_return = returns[llm_player_id]
    
    try:
        min_u, max_u = game.min_utility(), game.max_utility()
        final_reward = (llm_return - min_u) / (max_u - min_u) if max_u > min_u else 0.0
    except:
        final_reward = llm_return
    
    # Assign discounted rewards
    for i in range(len(turn_samples) - 1, -1, -1):
        turn_samples[i].reward = float((gamma ** (len(turn_samples) - 1 - i)) * final_reward)
    
    episode_info = {
        "game_name": game_name, "task_id": task_id, "seed": seed,
        "opponent": opponent, "final_reward": float(final_reward),
        "valid_turns": valid_count, "invalid_turns": invalid_count
    }
    
    episode_data = EpisodeData(
        task_id=task_id, seed=seed, game_name=game_name, opponent=opponent,
        final_reward=float(final_reward), total_turns=valid_count + invalid_count,
        valid_turns=valid_count, invalid_turns=invalid_count
    )
    
    return turn_samples, episode_info, episode_data


# =============================================================================
# PPO Trainer Class
# =============================================================================

class GamePPOTrainer:
    """PPO Trainer for game environments"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        ppo_config: PPOTrainingConfig,
        env_config: Dict = None
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.ppo_config = ppo_config
        self.env_config = env_config or {}
        
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.ppo_trainer = None
        self.curriculum_sampler = None
        self.start_step = 0
        
        self._setup()
    
    def _setup(self):
        """Initialize all components"""
        self.logger.info("Setting up PPO+LoRA training...")
        
        # Initialize wandb
        if self.ppo_config.use_wandb:
            import wandb
            wandb.init(
                project=self.ppo_config.wandb_project,
                config={
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "ppo": self.ppo_config.__dict__,
                }
            )
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quant_config = None
        if self.model_config.use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_config.use_nested_quant,
            )
        
        # Load base model
        self.logger.info(f"Loading model: {self.model_config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Create model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, peft_config=peft_config)
        
        # Create frozen reference model
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_config.model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.ref_model.eval()
        
        # PPO trainer
        ppo_cfg = PPOConfig(
            batch_size=self.ppo_config.batch_size,
            mini_batch_size=self.ppo_config.mini_batch_size,
            gradient_accumulation_steps=self.ppo_config.gradient_accumulation_steps,
            learning_rate=self.ppo_config.learning_rate,
            ppo_epochs=self.ppo_config.ppo_epochs,
            init_kl_coef=self.ppo_config.init_kl_coef,
            target=self.ppo_config.target_kl,
            cliprange=self.ppo_config.cliprange,
            cliprange_value=self.ppo_config.cliprange_value,
            vf_coef=self.ppo_config.vf_coef,
            log_with="wandb" if self.ppo_config.use_wandb else None,
            use_score_scaling=True,
            use_score_norm=True,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_cfg,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Curriculum sampler
        allowed_games = self.env_config.get("allowed_game_indices", [0, 1, 2, 6, 7])
        self.curriculum_sampler = CurriculumSampler(
            allowed_game_indices=allowed_games,
            failure_buffer_size=1000,
            progression_threshold=0.7,
        )
        
        # Create output directories
        Path(self.ppo_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Setup complete!")
    
    def collect_rollouts(self, batch_size: int, step: int = 0) -> Dict:
        """Collect training samples from game episodes"""
        rollouts = {"queries": [], "responses": [], "rewards": [], "metadata": []}
        games_with_samples: Dict[str, int] = {}
        available_games = list(self.curriculum_sampler.task_pool.keys())
        
        total_attempts = 0
        max_attempts = self.ppo_config.max_collection_episodes
        
        self.logger.info(f"Collecting samples from {len(available_games)} games...")
        
        while total_attempts < max_attempts:
            total_attempts += 1
            
            # Check if we have samples from all games
            games_needing = [g for g in available_games if games_with_samples.get(g, 0) == 0]
            if not games_needing:
                break
            
            # Sample task for a game that needs samples
            target_game = games_needing[total_attempts % len(games_needing)]
            task_config = self.curriculum_sampler.sample_task_for_game(target_game)
            
            task_id = task_config.task_id
            seed = task_config.seed or int(torch.randint(0, 2**31 - 1, (1,)).item())
            opponent = task_config.opponent
            
            # Run episode
            try:
                turn_samples, episode_info, _ = run_episode(
                    model=self.model.pretrained_model if hasattr(self.model, "pretrained_model") else self.model,
                    tokenizer=self.tokenizer,
                    task_id=task_id,
                    seed=seed,
                    opponent=opponent,
                    temperature=self.ppo_config.temperature,
                    max_new_tokens=self.ppo_config.max_new_tokens,
                    max_seq_length=self.ppo_config.max_seq_length,
                    gamma=self.env_config.get("gamma", 0.99),
                    device=str(self.device),
                )
                
                # Update curriculum
                score = episode_info.get("final_reward", 0.0)
                self.curriculum_sampler.update(task_id, score, score > 0.5)
                
                # Track samples per game
                game_name = episode_info.get("game_name", "unknown")
                if episode_info.get("valid_turns", 0) > 0:
                    games_with_samples[game_name] = games_with_samples.get(game_name, 0) + 1
                
                # Add samples to rollouts
                for ts in turn_samples:
                    q = self.tokenizer.encode(
                        ts.prompt_text, return_tensors="pt",
                        max_length=self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens,
                        truncation=True
                    ).to(self.device)
                    r = self.tokenizer.encode(ts.response_text, return_tensors="pt", truncation=True).to(self.device)
                    
                    rollouts["queries"].append(q.squeeze(0))
                    rollouts["responses"].append(r.squeeze(0))
                    rollouts["rewards"].append(torch.tensor(ts.reward))
                    rollouts["metadata"].append({"score": score, "success": score > 0.5})
                    
            except Exception as e:
                self.logger.warning(f"Episode failed: {e}")
                continue
        
        self.logger.info(f"Collected {len(rollouts['queries'])} samples from {len(games_with_samples)} games")
        return rollouts
    
    def train_step(self, rollouts: Dict) -> Dict:
        """Execute one PPO training step"""
        queries = rollouts["queries"]
        responses = rollouts["responses"]
        rewards = rollouts["rewards"]
        
        if not queries:
            return {"mean_reward": 0.0, "mean_score": 0.0, "success_rate": 0.0}
        
        # Process in batches
        all_stats = []
        batch_size = self.ppo_config.batch_size
        
        for i in range(0, len(queries), batch_size):
            end = min(i + batch_size, len(queries))
            if end - i < batch_size:
                continue
            
            batch_q = queries[i:end]
            batch_r = responses[i:end]
            batch_rew = rewards[i:end]
            
            stats = self.ppo_trainer.step(batch_q, batch_r, batch_rew)
            all_stats.append(stats)
        
        # Aggregate stats
        if all_stats:
            stats = {}
            for k in all_stats[0].keys():
                vals = [s.get(k, 0) for s in all_stats]
                if isinstance(vals[0], (int, float)):
                    stats[k] = sum(vals) / len(vals)
        else:
            stats = {}
        
        stats["mean_reward"] = torch.mean(torch.stack(rewards)).item() if rewards else 0.0
        stats["mean_score"] = sum(m["score"] for m in rollouts["metadata"]) / len(rollouts["metadata"]) if rollouts["metadata"] else 0.0
        stats["success_rate"] = sum(m["success"] for m in rollouts["metadata"]) / len(rollouts["metadata"]) if rollouts["metadata"] else 0.0
        
        return stats
    
    async def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training...")
        
        for step in range(self.start_step, self.ppo_config.num_train_steps):
            # Collect rollouts
            self.logger.info(f"Step {step + 1}/{self.ppo_config.num_train_steps}: Collecting rollouts...")
            rollouts = self.collect_rollouts(self.ppo_config.min_valid_samples_per_step, step + 1)
            
            if not rollouts["queries"]:
                self.logger.warning(f"Step {step + 1}: No samples collected, skipping")
                continue
            
            # Train
            stats = self.train_step(rollouts)
            
            # Log
            if (step + 1) % self.ppo_config.log_freq == 0:
                self.logger.info(
                    f"Step {step + 1} | Reward: {stats['mean_reward']:.4f} | "
                    f"Score: {stats['mean_score']:.4f} | Success: {stats['success_rate']:.2%}"
                )
                
                if self.ppo_config.use_wandb:
                    import wandb
                    wandb.log(stats, step=step + 1)
            
            # Save checkpoint
            if (step + 1) % self.ppo_config.save_freq == 0:
                save_path = Path(self.ppo_config.output_dir) / f"checkpoint-{step + 1}"
                self.logger.info(f"Saving checkpoint: {save_path}")
                save_checkpoint(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    save_path=save_path,
                    curriculum_state=self.curriculum_sampler.get_state(),
                    training_args={"step": step + 1},
                )
            
            # Evaluate
            if (step + 1) % self.ppo_config.eval_freq == 0:
                self.logger.info("Running evaluation...")
                eval_stats = await self.evaluate()
                self.logger.info(f"Eval | Score: {eval_stats['eval_mean_score']:.4f}")
        
        # Final save
        final_path = Path(self.ppo_config.output_dir) / "final"
        save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            save_path=final_path,
            curriculum_state=self.curriculum_sampler.get_state(),
            training_args={"step": self.ppo_config.num_train_steps},
        )
        
        self.logger.info("Training complete!")
    
    async def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        results = []
        self.model.eval()
        
        for _ in range(num_episodes):
            task_config = self.curriculum_sampler.sample_task(eval_mode=True)
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            
            try:
                _, episode_info, _ = run_episode(
                    model=self.model.pretrained_model if hasattr(self.model, "pretrained_model") else self.model,
                    tokenizer=self.tokenizer,
                    task_id=task_config.task_id,
                    seed=seed,
                    opponent="random",
                    temperature=0.1,
                    max_new_tokens=self.ppo_config.max_new_tokens,
                    device=str(self.device),
                )
                results.append({"score": episode_info["final_reward"], "success": episode_info["final_reward"] > 0.5})
            except:
                results.append({"score": 0.0, "success": False})
        
        self.model.train()
        
        return {
            "eval_mean_score": sum(r["score"] for r in results) / len(results) if results else 0.0,
            "eval_success_rate": sum(r["success"] for r in results) / len(results) if results else 0.0,
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """Load JSON config file"""
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def main():
    """Main entry point"""
    config_dir = Path("config")
    
    # Load configs
    train_cfg = load_config(config_dir / "train_config.json")
    env_cfg = load_config(config_dir / "env_config.json")
    
    # Create config objects
    model_config = ModelConfig(**train_cfg.get("model", {}))
    lora_config = LoRAConfig(**train_cfg.get("lora", {}))
    ppo_config = PPOTrainingConfig(**train_cfg.get("ppo", {}))
    
    # Create trainer
    trainer = GamePPOTrainer(
        model_config=model_config,
        lora_config=lora_config,
        ppo_config=ppo_config,
        env_config=env_cfg,
    )
    
    # Train
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
