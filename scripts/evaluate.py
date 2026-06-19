from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_bdqn.envs.search_track_env import SearchTrackEnv, EnvConfig
from uav_bdqn.agents.bdqn_agent import BDQNAgent, BDQNConfig


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = resolve_device(args.device)
    env = SearchTrackEnv(EnvConfig(seed=args.seed))
    agent = BDQNAgent(BDQNConfig(obs_shape=env.observation_shape, action_dim=env.action_dim, device=device))
    agent.load(args.checkpoint)

    rewards, completed, detected = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs, sample=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        completed.append(info["completed"])
        detected.append(info["detected"])
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, detected={info['detected']}/4, completed={info['completed']}/4")

    print("---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average detected: {np.mean(detected):.2f}/4")
    print(f"Average completed: {np.mean(completed):.2f}/4")


if __name__ == "__main__":
    main()
