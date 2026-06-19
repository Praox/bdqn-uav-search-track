from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_bdqn.envs.search_track_env import SearchTrackEnv, EnvConfig
from uav_bdqn.agents.replay_buffer import ReplayBuffer
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
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--sensor-radius", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=str, default="runs")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    env_cfg = EnvConfig(
        grid_size=args.grid_size,
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    env = SearchTrackEnv(env_cfg)
    agent_cfg = BDQNConfig(
        obs_shape=env.observation_shape,
        action_dim=env.action_dim,
        device=device,
    )
    agent = BDQNAgent(agent_cfg)
    replay = ReplayBuffer(args.buffer_size, env.observation_shape, seed=args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(exist_ok=True)

    global_step = 0
    recent_rewards = []
    pbar = trange(args.episodes, desc="Training BDQN")
    for ep in pbar:
        obs, info = env.reset()
        agent.resample_q_function()  # one sampled Q-function per episode = deep exploration
        ep_reward = 0.0
        ep_completed = 0
        ep_detected = 0

        for _ in range(args.max_steps):
            if global_step < args.warmup:
                action = np.random.randint(env.action_dim)
            else:
                action = agent.act(obs, sample=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            global_step += 1
            ep_completed = info["completed"]
            ep_detected = info["detected"]

            if replay.size >= args.batch_size and global_step >= args.warmup:
                batch = replay.sample(args.batch_size)
                agent.train_step(batch)

            if done:
                break

        recent_rewards.append(ep_reward)
        recent_rewards = recent_rewards[-50:]
        pbar.set_postfix({
            "avg_reward": f"{np.mean(recent_rewards):.2f}",
            "detected": ep_detected,
            "completed": ep_completed,
        })

        if (ep + 1) % 100 == 0:
            agent.save(str(run_dir / f"bdqn_ep{ep+1}.pt"))
            agent.save(str(run_dir / "latest.pt"))

    agent.save(str(run_dir / "latest.pt"))
    print(f"Saved checkpoint to {run_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
