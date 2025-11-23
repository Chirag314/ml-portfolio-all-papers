from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    capacity: int
    obs_dim: int


class ReplayBuffer:
    """
    Basic cyclic replay buffer used by DQN.
    Stores (obs, action, reward, next_obs, done) tuples.
    """

    def __init__(self, cfg: ReplayBufferConfig):
        self.capacity = int(cfg.capacity)
        self.obs_buf = np.zeros((self.capacity, cfg.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, cfg.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.idx] = obs
        self.next_obs_buf[self.idx] = next_obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.done_buf[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs_buf[idxs], device=device)
        actions = torch.as_tensor(self.act_buf[idxs], device=device)
        rewards = torch.as_tensor(self.rew_buf[idxs], device=device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=device)
        dones = torch.as_tensor(self.done_buf[idxs], device=device)

        return obs, actions, rewards, next_obs, dones
