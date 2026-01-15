#!/usr/bin/env python3
"""
Utility functions for PPO+LoRA training
"""

import logging
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
    
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("game_ppo_training")


def save_checkpoint(
    model,
    tokenizer,
    save_path: Path,
    curriculum_state: Optional[Dict] = None,
    optimizer_state: Optional[Dict] = None,
    training_args: Optional[Dict] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        save_path: Checkpoint directory path
        curriculum_state: Curriculum sampler state
        optimizer_state: Optimizer state dict
        training_args: Training arguments
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model (LoRA adapters)
    model_path = save_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    
    # Save tokenizer
    tokenizer_path = save_path / "tokenizer"
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save metadata
    metadata = {"model_type": "ppo_lora"}
    
    if curriculum_state:
        metadata["curriculum_state"] = curriculum_state
    
    if optimizer_state:
        torch.save(optimizer_state, save_path / "optimizer.pt")
    
    if training_args:
        metadata["training_args"] = training_args
    
    with open(save_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Checkpoint saved: {save_path}")


def load_checkpoint(
    model,
    tokenizer,
    load_path: Path,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load adapters into
        tokenizer: Tokenizer (unused, for compatibility)
        load_path: Checkpoint directory path
    
    Returns:
        Metadata dictionary
    """
    load_path = Path(load_path)
    
    # Load LoRA adapters
    model.load_adapter(load_path / "model")
    
    # Load metadata
    metadata = {}
    if (load_path / "metadata.json").exists():
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
    
    # Load optimizer state if exists
    optimizer_path = load_path / "optimizer.pt"
    if optimizer_path.exists():
        metadata["optimizer_state"] = torch.load(optimizer_path)
    
    print(f"✅ Checkpoint loaded: {load_path}")
    return metadata


def count_parameters(model) -> Dict[str, int]:
    """
    Count trainable vs total parameters.
    
    Args:
        model: Model to analyze
    
    Returns:
        Parameter count statistics
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": 100 * trainable / total if total > 0 else 0,
    }


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


class MovingAverage:
    """Moving average tracker for metrics"""
    
    def __init__(self, window_size: int = 100):
        from collections import deque
        self.values = deque(maxlen=window_size)
    
    def update(self, value: float):
        self.values.append(value)
    
    def get(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    def reset(self):
        self.values.clear()


class EarlyStopping:
    """Early stopping helper"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
