# Deep Q-Network (DQN) – CartPole-v1

This subproject implements **Deep Q-Network (DQN)**, inspired by:

> Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015.

To keep it lightweight, we apply DQN to `CartPole-v1` instead of Atari,
but keep the key algorithmic ideas:

- Q-network parameterizing Q(s, a)
- Experience replay buffer
- Target network for stable bootstrapping
- ε-greedy exploration with linear decay

---

## How to run

From repo root:

```bash
cd rl/dqn

# (optional) activate root venv
# D:\Machine_Learning\ml-portfolio-all-papers> .\venv\Scripts\activate

pip install -r ../../requirements.txt
pip install gymnasium[classic-control]
