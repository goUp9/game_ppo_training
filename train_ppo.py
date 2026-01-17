#!/usr/bin/env python3
"""
PPO + LoRA Training for Game/Trace Environments

This script trains a language model using:
- PPO (Proximal Policy Optimization) for policy learning
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Curriculum learning for progressive difficulty
- Failure-based replay for hard case mastery

Supported Environments (via affinetes):
- OpenSpiel Games: Strategic board/card games
- Trace: Code execution output prediction

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

# Import affinetes as installed Python module
import affinetes as af_env

from curriculum import CurriculumSampler, TaskConfig
from utils import setup_logging, save_checkpoint, load_checkpoint


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration for base LLM"""
    model_name: str = "Qwen/Qwen3-4B"
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
    max_new_tokens: int = 512  # Trace needs longer outputs
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
# Episode Data Structures
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
    env_name: str  # "openspiel" or "trace"
    game_name: str  # Game name for openspiel, "trace" for trace env
    final_reward: float
    total_turns: int
    valid_turns: int
    invalid_turns: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "env_name": self.env_name,
            "game_name": self.game_name,
            "final_reward": self.final_reward,
            "total_turns": self.total_turns,
            "valid_turns": self.valid_turns,
            "invalid_turns": self.invalid_turns,
        }


# =============================================================================
# Environment Wrapper using affinetes module
# =============================================================================

class AffinetesEnvironment:
    """
    Wrapper for affinetes environments (OpenSpiel games, Trace, etc.)
    
    Uses the affinetes Python module to load and interact with containerized
    environments via Docker or Basilica mode.
    """
    
    # Docker images for each environment type
    ENV_IMAGES = {
        "openspiel": "affinefoundation/game:openspiel",
        "trace": "affinefoundation/trace:latest",
    }
    
    def __init__(
        self,
        env_type: str = "openspiel",
        mode: str = "docker",
        api_key: Optional[str] = None,
    ):
        """
        Initialize affinetes environment.
        
        Args:
            env_type: Environment type ("openspiel" or "trace")
            mode: Execution mode ("docker" or "basilica")
            api_key: API key for LLM service (uses CHUTES_API_KEY env var if not provided)
        """
        self.env_type = env_type
        self.mode = mode
        self.api_key = api_key or os.environ.get("CHUTES_API_KEY", "")
        
        if env_type not in self.ENV_IMAGES:
            raise ValueError(f"Unknown env_type: {env_type}. Must be one of {list(self.ENV_IMAGES.keys())}")
        
        self._env = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the affinetes environment (lazy loading)"""
        if self._initialized:
            return
        
        image = self.ENV_IMAGES[self.env_type]
        
        self._env = af_env.load_env(
            image=image,
            mode=self.mode,
            env_vars={"CHUTES_API_KEY": self.api_key},
            mem_limit="8g" if self.env_type == "openspiel" else "4g",
            pull=True,
        )
        
        self._initialized = True
    
    async def evaluate(
        self,
        task_id: int,
        seed: int,
        model: str,
        base_url: str,
        temperature: float = 0.7,
        timeout: int = 600,
        api_key: str = "dummy-key",  # vLLM doesn't validate API keys
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation on the environment.
        
        Args:
            task_id: Task identifier
            seed: Random seed for reproducibility
            model: LLM model name
            base_url: LLM API base URL
            temperature: Generation temperature
            timeout: Evaluation timeout in seconds
            api_key: API key for LLM service (vLLM doesn't validate)
            **kwargs: Additional environment-specific parameters
        
        Returns:
            Evaluation result dictionary with score, success, extra, etc.
        """
        await self.initialize()
        
        result = await self._env.evaluate(
            task_id=task_id,
            seed=seed,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            api_key=api_key,
            _timeout=timeout + 60,  # Proxy timeout
            **kwargs
        )
        
        return result
    
    async def cleanup(self):
        """Clean up the environment"""
        if self._env is not None:
            await self._env.cleanup()
            self._initialized = False


# =============================================================================
# Local Episode Runners (for direct model inference without vLLM server)
# =============================================================================

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
    """Extract action ID from model output (for OpenSpiel)"""
    import re
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


# =============================================================================
# PPO Trainer Class
# =============================================================================

class GamePPOTrainer:
    """
    PPO Trainer for game/trace environments using affinetes.
    
    Supports two modes:
    1. Local inference: Model runs locally, environment via affinetes container
    2. Remote inference: Both model (vLLM) and environment run in containers
    """
    
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
        
        # Environment configuration
        self.env_type = self.env_config.get("env_type", "openspiel")
        self.env_mode = self.env_config.get("env_mode", "docker")
        
        # affinetes environment (for remote evaluation)
        self.affinetes_env: Optional[AffinetesEnvironment] = None
        
        # vLLM server configuration (for remote inference)
        self.use_vllm = self.env_config.get("use_vllm", False)
        self.vllm_base_url = self.env_config.get("vllm_base_url", "http://localhost:8000/v1")
        self.api_key = self.env_config.get("api_key", "dummy-key")
        
        self._setup()
    
    def _setup(self):
        """Initialize all components"""
        self.logger.info(f"Setting up PPO+LoRA training for {self.env_type}...")
        
        # Initialize wandb
        if self.ppo_config.use_wandb:
            import wandb
            wandb.init(
                project=self.ppo_config.wandb_project,
                config={
                    "model": self.model_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "ppo": self.ppo_config.__dict__,
                    "env_type": self.env_type,
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
        
        # PPO trainer - batch_size must be >= mini_batch_size * gradient_accumulation_steps
        # We use mini_batch_size=1 and gradient_accumulation_steps=1 for maximum flexibility
        ppo_cfg = PPOConfig(
            batch_size=self.ppo_config.mini_batch_size,
            mini_batch_size=self.ppo_config.mini_batch_size,
            gradient_accumulation_steps=1,  # Handle accumulation in our training loop
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
            ratio_threshold=20.0,  # Increase threshold for early training
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_cfg,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Curriculum sampler
        if self.env_type == "openspiel":
            allowed_games = self.env_config.get("allowed_game_indices", list(range(8)))
            self.curriculum_sampler = CurriculumSampler(
                allowed_game_indices=allowed_games,
                failure_buffer_size=1000,
                progression_threshold=0.7,
            )
        else:
            # For Trace, use simple curriculum sampler
            self.curriculum_sampler = CurriculumSampler(
                allowed_game_indices=[0],
                failure_buffer_size=1000,
                progression_threshold=0.7,
            )
        
        # Initialize affinetes environment (lazy)
        self.affinetes_env = AffinetesEnvironment(
            env_type=self.env_type,
            mode=self.env_mode,
            api_key=os.environ.get("CHUTES_API_KEY"),
        )
        
        # Create output directories
        Path(self.ppo_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Setup complete!")
    
    async def collect_rollouts_via_affinetes(self, batch_size: int, step: int = 0) -> Dict:
        """
        Collect rollouts using affinetes containerized environment.
        
        This method uses the affinetes module to run episodes in Docker/Basilica containers.
        The model inference happens via vLLM server (must be running separately).
        """
        rollouts = {"queries": [], "responses": [], "rewards": [], "metadata": []}
        
        if not self.use_vllm:
            self.logger.warning("vLLM not enabled. Set use_vllm=true in env_config for affinetes mode.")
            return rollouts
        
        self.logger.info(f"Collecting {batch_size} samples via affinetes ({self.env_type})...")
        
        # Retry settings
        max_attempts = batch_size * 3  # Try up to 3x to get enough samples
        attempts = 0
        successful_episodes = 0
        failed_episodes = 0
        
        while len(rollouts["queries"]) < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Sample task
            if self.env_type == "openspiel":
                task_config = self.curriculum_sampler.sample_task()
                task_id = task_config.task_id
            else:
                task_id = int(torch.randint(0, 10000, (1,)).item())
            
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            
            try:
                # Run evaluation via affinetes
                result = await self.affinetes_env.evaluate(
                    task_id=task_id,
                    seed=seed,
                    model=self.model_config.model_name,
                    base_url=self.vllm_base_url,
                    temperature=self.ppo_config.temperature,
                    timeout=self.env_config.get("timeout", 180),
                    api_key=self.api_key,
                    opponent=self.env_config.get("opponent", "random") if self.env_type == "openspiel" else None,
                )
                
                # Check for errors in result
                if result.get("error"):
                    self.logger.warning(f"Episode {attempts} returned error: {result.get('error')}")
                    failed_episodes += 1
                    continue
                
                score = float(result.get("score", 0.0))
                success = bool(result.get("success", False))
                extra = result.get("extra", {})
                
                # Update curriculum
                self.curriculum_sampler.update(task_id, score, success)
                
                # Extract conversation for PPO samples
                conversation = extra.get("conversation", [])
                
                if not conversation:
                    self.logger.debug(f"Episode {attempts}: No conversation in result, skipping")
                    failed_episodes += 1
                    continue
                
                # Extract all assistant responses (for multi-turn games)
                samples_from_episode = 0
                prompt_so_far = []
                
                for msg in conversation:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "assistant" and content:
                        # We have a prompt and response pair
                        if prompt_so_far:
                            prompt_text = _build_chat_prompt(self.tokenizer, prompt_so_far)
                            response_text = content.strip()
                            
                            q = self.tokenizer.encode(
                                prompt_text, return_tensors="pt",
                                max_length=self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens,
                                truncation=True
                            ).to(self.device)
                            r = self.tokenizer.encode(response_text, return_tensors="pt", truncation=True).to(self.device)
                            
                            rollouts["queries"].append(q.squeeze(0))
                            rollouts["responses"].append(r.squeeze(0))
                            # Rewards must be float tensors on CPU for PPO trainer
                            rollouts["rewards"].append(torch.tensor(score, dtype=torch.float32))
                            rollouts["metadata"].append({"score": score, "success": success, "task_id": task_id})
                            samples_from_episode += 1
                        
                        # Add assistant message to history for next turn
                        prompt_so_far.append(msg)
                    else:
                        # Add non-assistant messages to prompt
                        prompt_so_far.append(msg)
                
                if samples_from_episode > 0:
                    successful_episodes += 1
                    self.logger.debug(f"Episode {attempts}: Extracted {samples_from_episode} samples (score={score:.2f})")
                else:
                    failed_episodes += 1
                    self.logger.debug(f"Episode {attempts}: No valid samples extracted")
                
            except Exception as e:
                self.logger.warning(f"Episode {attempts} failed with exception: {e}")
                failed_episodes += 1
                continue
        
        self.logger.info(
            f"Collected {len(rollouts['queries'])} samples from {successful_episodes} episodes "
            f"({failed_episodes} failed, {attempts} total attempts)"
        )
        
        # Warn if we couldn't collect enough samples
        if len(rollouts["queries"]) < batch_size:
            self.logger.warning(
                f"Only collected {len(rollouts['queries'])}/{batch_size} samples. "
                f"Check: 1) vLLM server running? 2) Docker container healthy? 3) API key set?"
            )
        
        return rollouts
    
    async def collect_rollouts_local(self, batch_size: int, step: int = 0) -> Dict:
        """
        Collect rollouts using local model inference.
        
        This method runs the model locally and uses affinetes only for environment state.
        Faster for development but requires local GPU.
        """
        rollouts = {"queries": [], "responses": [], "rewards": [], "metadata": []}
        
        self.logger.info(f"Collecting {batch_size} samples with local inference...")
        
        # For local mode, we generate responses locally and evaluate via affinetes
        for i in range(batch_size):
            # Sample task
            if self.env_type == "openspiel":
                task_config = self.curriculum_sampler.sample_task()
                task_id = task_config.task_id
            else:
                task_id = int(torch.randint(0, 10000, (1,)).item())
            
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            
            try:
                # For now, use affinetes evaluation (model inference happens in container)
                # TODO: Implement true local inference with environment state extraction
                result = await self.affinetes_env.evaluate(
                    task_id=task_id,
                    seed=seed,
                    model=self.model_config.model_name,
                    base_url=self.vllm_base_url if self.use_vllm else "http://localhost:8000/v1",
                    temperature=self.ppo_config.temperature,
                    timeout=self.env_config.get("timeout", 180),
                    api_key=self.api_key,
                )
                
                score = float(result.get("score", 0.0))
                success = bool(result.get("success", False))
                extra = result.get("extra", {})
                
                self.curriculum_sampler.update(task_id, score, success)
                
                # Extract samples from conversation
                conversation = extra.get("conversation", [])
                if conversation:
                    prompt_messages = [m for m in conversation if m.get("role") != "assistant"]
                    response_text = ""
                    for m in conversation:
                        if m.get("role") == "assistant":
                            response_text = m.get("content", "")
                            break
                    
                    if prompt_messages and response_text:
                        prompt_text = _build_chat_prompt(self.tokenizer, prompt_messages)
                        
                        q = self.tokenizer.encode(
                            prompt_text, return_tensors="pt",
                            max_length=self.ppo_config.max_seq_length - self.ppo_config.max_new_tokens,
                            truncation=True
                        ).to(self.device)
                        r = self.tokenizer.encode(response_text, return_tensors="pt", truncation=True).to(self.device)
                        
                        rollouts["queries"].append(q.squeeze(0))
                        rollouts["responses"].append(r.squeeze(0))
                        rollouts["rewards"].append(torch.tensor(score, dtype=torch.float32))
                        rollouts["metadata"].append({"score": score, "success": success})
                
            except Exception as e:
                self.logger.warning(f"Episode {i} failed: {e}")
                continue
        
        self.logger.info(f"Collected {len(rollouts['queries'])} samples")
        return rollouts
    
    async def collect_rollouts(self, batch_size: int, step: int = 0) -> Dict:
        """Collect training samples from episodes"""
        if self.use_vllm:
            return await self.collect_rollouts_via_affinetes(batch_size, step)
        else:
            return await self.collect_rollouts_local(batch_size, step)
    
    def train_step(self, rollouts: Dict) -> Dict:
        """Execute one PPO training step"""
        queries = rollouts["queries"]
        responses = rollouts["responses"]
        rewards = rollouts["rewards"]
        
        if not queries:
            return {"mean_reward": 0.0, "mean_score": 0.0, "success_rate": 0.0}
        
        all_stats = []
        mini_batch_size = self.ppo_config.mini_batch_size
        
        # Process all samples in mini-batches
        num_samples = len(queries)
        
        for i in range(0, num_samples, mini_batch_size):
            end = min(i + mini_batch_size, num_samples)
            actual_batch_size = end - i
            
            # Only process complete mini-batches
            if actual_batch_size < mini_batch_size:
                self.logger.debug(f"Skipping incomplete batch with {actual_batch_size} samples")
                continue
            
            batch_q = queries[i:end]
            batch_r = responses[i:end]
            batch_rew = rewards[i:end]
            
            try:
                stats = self.ppo_trainer.step(batch_q, batch_r, batch_rew)
                all_stats.append(stats)
                self.logger.debug(f"PPO step completed for mini-batch {i//mini_batch_size + 1}")
            except Exception as e:
                self.logger.warning(f"PPO step failed for mini-batch {i//mini_batch_size + 1}: {e}")
                continue
        
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
        stats["num_samples"] = num_samples
        
        return stats
    
    async def train(self):
        """Main training loop"""
        self.logger.info(f"Starting PPO training for {self.env_type}...")
        self.logger.info(f"Config: batch_size={self.ppo_config.batch_size}, "
                        f"mini_batch_size={self.ppo_config.mini_batch_size}, "
                        f"lr={self.ppo_config.learning_rate}")
        
        for step in range(self.start_step, self.ppo_config.num_train_steps):
            self.logger.info(f"Step {step + 1}/{self.ppo_config.num_train_steps}: Collecting rollouts...")
            rollouts = await self.collect_rollouts(self.ppo_config.min_valid_samples_per_step, step + 1)
            
            num_samples = len(rollouts["queries"])
            if num_samples == 0:
                self.logger.warning(f"Step {step + 1}: No samples collected, skipping")
                continue
            
            self.logger.info(f"Step {step + 1}: Training on {num_samples} samples...")
            stats = self.train_step(rollouts)
            
            # Log every step for debugging
            self.logger.info(
                f"Step {step + 1} | Samples: {stats.get('num_samples', num_samples)} | "
                f"Reward: {stats['mean_reward']:.4f} | Score: {stats['mean_score']:.4f} | "
                f"Success: {stats['success_rate']:.2%}"
            )
            
            if self.ppo_config.use_wandb:
                import wandb
                wandb.log(stats, step=step + 1)
            
            # Remove the duplicate log_freq check since we log every step now
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
            
            if (step + 1) % self.ppo_config.eval_freq == 0:
                self.logger.info("Running evaluation...")
                eval_stats = await self.evaluate()
                self.logger.info(f"Eval | Score: {eval_stats['eval_mean_score']:.4f}")
        
        # Cleanup affinetes environment
        if self.affinetes_env:
            await self.affinetes_env.cleanup()
        
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
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            
            try:
                if self.env_type == "openspiel":
                    task_config = self.curriculum_sampler.sample_task(eval_mode=True)
                    task_id = task_config.task_id
                else:
                    task_id = int(torch.randint(0, 10000, (1,)).item())
                
                result = await self.affinetes_env.evaluate(
                    task_id=task_id,
                    seed=seed,
                    model=self.model_config.model_name,
                    base_url=self.vllm_base_url if self.use_vllm else "http://localhost:8000/v1",
                    temperature=0.1,  # Lower temperature for evaluation
                    timeout=self.env_config.get("timeout", 180),
                    api_key=self.api_key,
                )
                
                score = float(result.get("score", 0.0))
                results.append({"score": score, "success": score > 0.5})
                
            except Exception as e:
                self.logger.warning(f"Eval episode failed: {e}")
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
    
    train_cfg = load_config(config_dir / "train_config.json")
    env_cfg = load_config(config_dir / "env_config.json")
    
    model_config = ModelConfig(**train_cfg.get("model", {}))
    lora_config = LoRAConfig(**train_cfg.get("lora", {}))
    ppo_config = PPOTrainingConfig(**train_cfg.get("ppo", {}))
    
    trainer = GamePPOTrainer(
        model_config=model_config,
        lora_config=lora_config,
        ppo_config=ppo_config,
        env_config=env_cfg,
    )
    
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
