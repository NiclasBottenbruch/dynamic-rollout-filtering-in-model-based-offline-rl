import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

        # per-dimension min/max/P1/P99 of observations and rewards
        self.dimwise_stats_up_to_date = False
        self.dimwise_stats = {
            "observations": {
                "min": None,
                "max": None,
                "p1": None,
                "p99": None,
            },
            "rewards": {
                "min": None,
                "max": None,
                "p1": None,
                "p99": None,
            }
        }


    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
        self.dimwise_stats_up_to_date = False
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        update_dimwise_stats: bool = False
    ) -> None:
        """
            Add a batch of transitions to the replay buffer.
            When buffer is full, it will overwrite oldest transitions (circular buffer).
        """
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
        self.dimwise_stats_up_to_date = False
        if update_dimwise_stats:
            self.update_dimwise_stats()

    def load_dataset(self, dataset: Dict[str, np.ndarray], update_dimwise_stats: bool = True) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)

        self.dimwise_stats_up_to_date = False
        if update_dimwise_stats:
            self.update_dimwise_stats()

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        self.dimwise_stats_up_to_date = False
        return obs_mean, obs_std

    def update_dimwise_stats(self, force: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        if not self.dimwise_stats_up_to_date or force:
            # Combine observations and next_observations for stats calculation
            combined_obs = np.concatenate([self.observations, self.next_observations], axis=0) # shape: (2*size, obs_dim)
            self.dimwise_stats["observations"]["min"] = combined_obs.min(0) # shape: (obs_dim,)
            self.dimwise_stats["observations"]["max"] = combined_obs.max(0) # shape: (obs_dim,)
            self.dimwise_stats["observations"]["p1"] = np.percentile(combined_obs, 1, axis=0) # shape: (obs_dim,)
            self.dimwise_stats["observations"]["p99"] = np.percentile(combined_obs, 99, axis=0) # shape: (obs_dim,)

            self.dimwise_stats["rewards"]["min"] = self.rewards.min(0) # shape: (1,)
            self.dimwise_stats["rewards"]["max"] = self.rewards.max(0) # shape: (1,)
            self.dimwise_stats["rewards"]["p1"] = np.percentile(self.rewards, 1, axis=0) # shape: (1,)
            self.dimwise_stats["rewards"]["p99"] = np.percentile(self.rewards, 99, axis=0) # shape: (1,)

            self.dimwise_stats_up_to_date = True
        return self.dimwise_stats

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }