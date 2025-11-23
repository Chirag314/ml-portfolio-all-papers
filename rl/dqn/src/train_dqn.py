import argparse
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import yaml

from .path_setup import REPO_ROOT
from common.seed import set_seed
from common.logging import init_loggers
from .q_network import QNetwork, QNetworkConfig
from .replay_buffer import ReplayBuffer, ReplayBufferConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_mlflow", action="store_true")
    p.add_argument("--run_name", type=str, default="dqn_cartpole")
    return p.parse_args()


def linear_epsilon_decay(steps, eps_start, eps_end, eps_decay_steps, step):
    if step >= eps_decay_steps:
        return eps_end
    frac = step / eps_decay_steps
    return eps_start + frac * (eps_end - eps_start)


def evaluate(env, q_net, device, n_episodes: int = 5) -> float:
    q_net.eval()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        tot_rew = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = q_net(obs_t)
                action = q_values.argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            tot_rew += reward
        rewards.append(tot_rew)
    return float(np.mean(rewards))


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env_id = cfg["env"]["id"]
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net_cfg = QNetworkConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg["network"]["hidden_dim"],
    )
    q_net = QNetwork(net_cfg).to(device)
    target_net = QNetwork(net_cfg).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    buffer_cfg = ReplayBufferConfig(
        capacity=cfg["dqn"]["buffer_capacity"],
        obs_dim=obs_dim,
    )

    replay_buffer = ReplayBuffer(buffer_cfg)
