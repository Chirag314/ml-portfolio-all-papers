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

from .path_setup import REPO_ROOT  # noqa: F401
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


def linear_epsilon_decay(step, eps_start, eps_end, eps_decay_steps):
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
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
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

    optimizer = optim.Adam(q_net.parameters(), lr=float(cfg["dqn"]["lr"]))
    loss_fn = nn.MSELoss()

    logger = init_loggers(
        project="ml-portfolio",
        run_name=args.run_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        tags={"paper": "mnih-2015-dqn", **cfg},
    )

    history = {
        "steps": [],
        "train_rewards": [],
        "eval_rewards": [],
        "eps": [],
    }

    ckpt_path = Path(cfg["logging"]["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(cfg["logging"]["history_path"])
    reward_plot_path = Path(cfg["logging"]["reward_plot_path"])

    gamma = float(cfg["dqn"]["gamma"])
    batch_size = int(cfg["dqn"]["batch_size"])
    start_training = int(cfg["dqn"]["start_training"])
    train_freq = int(cfg["dqn"]["train_freq"])
    target_update_freq = int(cfg["dqn"]["target_update_freq"])
    max_steps = int(cfg["dqn"]["max_steps"])
    eps_start = float(cfg["dqn"]["eps_start"])
    eps_end = float(cfg["dqn"]["eps_end"])
    eps_decay_steps = int(cfg["dqn"]["eps_decay_steps"])

    eval_interval_episodes = int(cfg["logging"]["eval_interval_episodes"])
    eval_episodes = int(cfg["logging"]["eval_episodes"])

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    recent_rewards = deque(maxlen=100)

    for step in range(1, max_steps + 1):
        eps = linear_epsilon_decay(step, eps_start, eps_end, eps_decay_steps)

        # epsilon-greedy policy
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            with torch.no_grad():
                q_values = q_net(obs_t)
                action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward

        # if episode ends
        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1

            if episode_count % eval_interval_episodes == 0:
                eval_mean = evaluate(eval_env, q_net, device, n_episodes=eval_episodes)
                mean_100 = float(np.mean(recent_rewards)) if recent_rewards else 0.0

                history["steps"].append(step)
                history["train_rewards"].append(mean_100)
                history["eval_rewards"].append(eval_mean)
                history["eps"].append(eps)

                print(
                    f"[step {step}] episodes={episode_count} "
                    f"mean_100={mean_100:.1f} eval_mean={eval_mean:.1f} eps={eps:.3f}"
                )

                if logger:
                    logger.log_metrics(
                        {
                            "train/mean_100_reward": mean_100,
                            "eval/mean_reward": eval_mean,
                            "explore/epsilon": eps,
                        },
                        step=step,
                    )

                # save checkpoint
                torch.save(
                    {"q_net_state_dict": q_net.state_dict(), "config": cfg},
                    ckpt_path,
                )

            obs, _ = env.reset()
            episode_reward = 0.0

        # DQN update
        if (
            (step >= start_training)
            and (step % train_freq == 0)
            and replay_buffer.can_sample(batch_size)
        ):
            q_net.train()
            obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(
                batch_size, device
            )

            # Q(s, a)
            q_values = q_net(obs_b)
            q_sa = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)

            # target Q
            with torch.no_grad():
                next_q_values = target_net(next_obs_b)
                next_q_max = next_q_values.max(dim=1).values
                target = rew_b + gamma * (1.0 - done_b) * next_q_max

            loss = loss_fn(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logger:
                logger.log_metrics({"train/td_loss": float(loss.item())}, step=step)

        # target network update
        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    # Save history & plot
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if history["steps"]:
        plt.figure()
        plt.plot(history["steps"], history["train_rewards"], label="train_mean_100")
        plt.plot(history["steps"], history["eval_rewards"], label="eval_mean")
        plt.xlabel("environment steps")
        plt.ylabel("reward")
        plt.title("DQN on CartPole-v1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_plot_path)
        plt.close()

    if logger:
        logger.finish()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
