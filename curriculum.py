#!/usr/bin/env python3
"""
Curriculum Learning and Failure-Based Sampling for Game RL Training

Features:
- Multi-stage curriculum (easy → hard progression)
- Automatic stage advancement based on success rate
- Failure replay buffer for hard case mastery
- Game-specific task sampling
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from pathlib import Path


@dataclass
class TaskConfig:
    """Configuration for a single training task"""
    task_id: int
    difficulty: str  # "easy", "medium", "hard"
    opponent: str    # "random", "mcts"
    seed: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class CurriculumStage:
    """A stage in the curriculum"""
    name: str
    opponent: str
    weight: float = 1.0
    min_success_rate: float = 0.6


class FailureBuffer:
    """
    Replay buffer for failed/low-scoring tasks.
    Prioritizes difficult tasks for more practice.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.task_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
    
    def add(self, task_id: int, score: float, success: bool):
        """Add task result to buffer"""
        self.task_stats[task_id]["attempts"] += 1
        if success:
            self.task_stats[task_id]["successes"] += 1
        
        # Add to buffer if failed or low score
        if not success or score < 0.5:
            self.buffer.append({
                "task_id": task_id,
                "score": score,
                "priority": 1.0 - score,
            })
    
    def sample(self, k: int = 1) -> List[int]:
        """Sample task IDs with priority weighting"""
        if not self.buffer:
            return []
        
        priorities = np.array([item["priority"] for item in self.buffer])
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(k, len(self.buffer)),
            replace=False,
            p=probs
        )
        
        return [self.buffer[i]["task_id"] for i in indices]
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        total_attempts = sum(s["attempts"] for s in self.task_stats.values())
        total_successes = sum(s["successes"] for s in self.task_stats.values())
        
        return {
            "buffer_size": len(self.buffer),
            "unique_tasks": len(self.task_stats),
            "total_attempts": total_attempts,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
        }


class CurriculumSampler:
    """
    Curriculum-based task sampler with failure replay.
    
    Implements:
    1. Multi-stage curriculum (easy → hard)
    2. Automatic progression based on success rate
    3. Failure-based replay for difficult tasks
    4. Game-specific sampling for balanced training
    """
    
    # Game name to index mapping (matches OpenSpiel game_config.py)
    GAME_NAME_TO_INDEX = {
        "goofspiel": 0,
        "liars_dice": 1,
        "leduc_poker": 2,
        "gin_rummy": 3,
        "othello": 4,
        "backgammon": 5,
        "hex": 6,
        "clobber": 7,
        "hearts": 8,
        "euchre": 9,
        "dots_and_boxes": 10,
        "go": 11,
    }
    
    GAME_BLOCK_SIZE = 100_000_000
    
    def __init__(
        self,
        allowed_game_indices: Optional[List[int]] = None,
        failure_buffer_size: int = 1000,
        failure_replay_prob: float = 0.3,
        progression_threshold: float = 0.7,
        eval_window: int = 100,
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            allowed_game_indices: List of game indices to train on
            failure_buffer_size: Max size of failure replay buffer
            failure_replay_prob: Probability of sampling from failure buffer
            progression_threshold: Success rate to advance to next stage
            eval_window: Window size for computing success rate
        """
        self.allowed_game_indices = list(allowed_game_indices) if allowed_game_indices else list(range(8))
        self.failure_replay_prob = failure_replay_prob
        self.progression_threshold = progression_threshold
        self.eval_window = eval_window
        
        # Curriculum stages
        self.stages = [
            CurriculumStage(name="easy", opponent="random"),
            CurriculumStage(name="medium", opponent="random"),
            CurriculumStage(name="hard", opponent="mcts"),
        ]
        self.current_stage_idx = 0
        
        # Failure buffer
        self.failure_buffer = FailureBuffer(max_size=failure_buffer_size)
        
        # Recent performance tracking
        self.recent_results = deque(maxlen=eval_window)
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "stage_changes": 0,
            "failure_replays": 0,
            "curriculum_samples": 0,
        }
    
    @property
    def task_pool(self) -> Dict[str, int]:
        """Get available games as {name: index} dict"""
        index_to_name = {v: k for k, v in self.GAME_NAME_TO_INDEX.items()}
        return {index_to_name[idx]: idx for idx in self.allowed_game_indices if idx in index_to_name}
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    def sample_task(self, eval_mode: bool = False) -> TaskConfig:
        """
        Sample a task using curriculum + failure replay.
        
        Args:
            eval_mode: If True, sample uniformly for evaluation
        
        Returns:
            TaskConfig for the sampled task
        """
        self.stats["total_samples"] += 1
        
        if eval_mode:
            return self._sample_from_curriculum()
        
        # Training: mix curriculum and failure replay
        if random.random() < self.failure_replay_prob and self.failure_buffer.buffer:
            task_ids = self.failure_buffer.sample(k=1)
            if task_ids and self._is_task_allowed(task_ids[0]):
                self.stats["failure_replays"] += 1
                return TaskConfig(
                    task_id=task_ids[0],
                    difficulty=self.current_stage.name,
                    opponent=self.current_stage.opponent,
                    seed=random.randint(0, 2**31 - 1),
                    metadata={"source": "failure_buffer"}
                )
        
        self.stats["curriculum_samples"] += 1
        return self._sample_from_curriculum()
    
    def _is_task_allowed(self, task_id: int) -> bool:
        """Check if task belongs to an allowed game"""
        game_idx = task_id // self.GAME_BLOCK_SIZE
        return game_idx in set(self.allowed_game_indices)
    
    def _sample_from_curriculum(self) -> TaskConfig:
        """Sample task from current curriculum stage"""
        game_idx = random.choice(self.allowed_game_indices)
        config_id = random.randint(0, self.GAME_BLOCK_SIZE - 1)
        task_id = game_idx * self.GAME_BLOCK_SIZE + config_id
        
        return TaskConfig(
            task_id=task_id,
            difficulty=self.current_stage.name,
            opponent=self.current_stage.opponent,
            seed=random.randint(0, 2**31 - 1),
            metadata={"source": "curriculum", "stage": self.current_stage.name}
        )
    
    def sample_task_for_game(self, game_name: str) -> TaskConfig:
        """
        Sample a task for a specific game.
        
        Args:
            game_name: Name of the game (e.g., "liars_dice", "hex")
        
        Returns:
            TaskConfig for the specified game
        """
        game_idx = self.GAME_NAME_TO_INDEX.get(game_name)
        if game_idx is None:
            return self._sample_from_curriculum()
        
        config_id = random.randint(0, self.GAME_BLOCK_SIZE - 1)
        task_id = game_idx * self.GAME_BLOCK_SIZE + config_id
        
        self.stats["curriculum_samples"] += 1
        self.stats["total_samples"] += 1
        
        return TaskConfig(
            task_id=task_id,
            difficulty=self.current_stage.name,
            opponent=self.current_stage.opponent,
            seed=random.randint(0, 2**31 - 1),
            metadata={"source": "targeted_game", "game_name": game_name}
        )
    
    def update(self, task_id: int, score: float, success: bool):
        """
        Update sampler with task result.
        
        Args:
            task_id: Task ID
            score: Task score (0.0-1.0)
            success: Whether task was successful
        """
        self.failure_buffer.add(task_id, score, success)
        self.recent_results.append(success)
        
        if len(self.recent_results) >= self.eval_window:
            self._check_stage_progression()
    
    def _check_stage_progression(self):
        """Check if should advance to next curriculum stage"""
        success_rate = sum(self.recent_results) / len(self.recent_results)
        
        if success_rate >= self.progression_threshold and self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.stats["stage_changes"] += 1
            self.recent_results.clear()
            
            print(f"🎓 Curriculum Progression: Stage {self.current_stage_idx + 1}/{len(self.stages)}")
            print(f"   Stage: {self.current_stage.name} | Opponent: {self.current_stage.opponent}")
            print(f"   Success rate: {success_rate:.2%}")
    
    def get_state(self) -> Dict:
        """Get state for checkpointing"""
        return {
            "current_stage_idx": self.current_stage_idx,
            "stats": self.stats,
            "buffer_stats": self.failure_buffer.get_stats(),
            "recent_results": list(self.recent_results),
        }
    
    def load_state(self, state: Dict):
        """Load state from checkpoint"""
        self.current_stage_idx = state.get("current_stage_idx", 0)
        self.stats = state.get("stats", self.stats)
        
        recent = state.get("recent_results", [])
        self.recent_results.clear()
        self.recent_results.extend(recent)
    
    def get_info(self) -> Dict:
        """Get current curriculum information"""
        return {
            "stage_idx": self.current_stage_idx,
            "stage_name": self.current_stage.name,
            "opponent": self.current_stage.opponent,
            "total_stages": len(self.stages),
            "recent_success_rate": sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.0,
            "progression_threshold": self.progression_threshold,
        }


def save_curriculum_checkpoint(sampler: CurriculumSampler, path: Path):
    """Save curriculum state to file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(sampler.get_state(), f, indent=2)


def load_curriculum_checkpoint(sampler: CurriculumSampler, path: Path):
    """Load curriculum state from file"""
    if not path.exists():
        return
    with open(path, 'r') as f:
        sampler.load_state(json.load(f))
