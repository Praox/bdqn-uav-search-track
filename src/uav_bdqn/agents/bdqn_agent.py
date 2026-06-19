from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from uav_bdqn.agents.replay_buffer import Batch
from uav_bdqn.models.networks import GridFeatureNet


@dataclass
class BDQNConfig:
    obs_shape: tuple[int, int, int] = (5, 20, 20)
    action_dim: int = 10
    feature_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-4
    target_update_period: int = 500
    blr_lambda: float = 1.0
    blr_noise_var: float = 1.0
    posterior_update_period: int = 100
    device: str = "cpu"


class BayesianLinearHead:
    """Per-action Bayesian linear regression posterior over final-layer weights.

    For each action a:
        Q(s,a) = w_a^T phi(s)
        w_a ~ N(mu_a, Sigma_a)

    We maintain conjugate BLR sufficient statistics using feature vectors and TD targets.
    """

    def __init__(self, action_dim: int, feature_dim: int, lam: float = 1.0, noise_var: float = 1.0):
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lam = lam
        self.noise_var = noise_var
        self.reset()

    def reset(self):
        d = self.feature_dim
        self.precision = np.stack([self.lam * np.eye(d, dtype=np.float64) for _ in range(self.action_dim)])
        self.b = np.zeros((self.action_dim, d), dtype=np.float64)
        self.mu = np.zeros((self.action_dim, d), dtype=np.float64)
        self.cov = np.stack([(1.0 / self.lam) * np.eye(d, dtype=np.float64) for _ in range(self.action_dim)])

    def update(self, features: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        for a in range(self.action_dim):
            mask = actions == a
            if not np.any(mask):
                continue
            phi = features[mask].astype(np.float64)
            y = targets[mask].astype(np.float64)
            self.precision[a] += (phi.T @ phi) / self.noise_var
            self.b[a] += (phi.T @ y) / self.noise_var
            self.cov[a] = np.linalg.inv(self.precision[a] + 1e-6 * np.eye(self.feature_dim))
            self.mu[a] = self.cov[a] @ self.b[a]

    def sample_weights(self) -> np.ndarray:
        weights = np.zeros_like(self.mu)
        for a in range(self.action_dim):
            cov = self.cov[a] + 1e-6 * np.eye(self.feature_dim)
            weights[a] = np.random.multivariate_normal(self.mu[a], cov)
        return weights.astype(np.float32)

    def mean_weights(self) -> np.ndarray:
        return self.mu.astype(np.float32)


class BDQNAgent:
    def __init__(self, cfg: BDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.feature_net = GridFeatureNet(cfg.obs_shape[0], cfg.feature_dim).to(self.device)
        self.target_feature_net = copy.deepcopy(self.feature_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.feature_net.parameters(), lr=cfg.lr)
        self.blr = BayesianLinearHead(cfg.action_dim, cfg.feature_dim, cfg.blr_lambda, cfg.blr_noise_var)
        self.sampled_w = self.blr.sample_weights()
        self.train_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, sample: bool = True) -> int:
        x = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        phi = self.feature_net(x).cpu().numpy()[0]
        w = self.sampled_w if sample else self.blr.mean_weights()
        q = w @ phi
        return int(np.argmax(q))

    def resample_q_function(self):
        self.sampled_w = self.blr.sample_weights()

    def train_step(self, batch: Batch) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        mean_w = torch.as_tensor(self.blr.mean_weights(), dtype=torch.float32, device=self.device)

        phi = self.feature_net(obs)
        q_all = phi @ mean_w.T
        q_sa = q_all.gather(1, actions[:, None]).squeeze(1)

        with torch.no_grad():
            next_phi_online = self.feature_net(next_obs)
            next_actions = torch.argmax(next_phi_online @ mean_w.T, dim=1)
            next_phi_target = self.target_feature_net(next_obs)
            next_q = (next_phi_target @ mean_w.T).gather(1, next_actions[:, None]).squeeze(1)
            target = rewards + cfg.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.feature_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % cfg.target_update_period == 0:
            self.target_feature_net.load_state_dict(self.feature_net.state_dict())

        if self.train_steps % cfg.posterior_update_period == 0:
            self.update_posterior(batch)
            self.resample_q_function()

        return {"loss": float(loss.item()), "q_mean": float(q_sa.mean().item())}

    @torch.no_grad()
    def update_posterior(self, batch: Batch):
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        mean_w = torch.as_tensor(self.blr.mean_weights(), dtype=torch.float32, device=self.device)

        phi = self.feature_net(obs).cpu().numpy()
        next_phi = self.target_feature_net(next_obs)
        next_q = torch.max(next_phi @ mean_w.T, dim=1).values
        targets = (rewards + self.cfg.gamma * (1.0 - dones) * next_q).cpu().numpy()
        self.blr.update(phi, batch.actions, targets)

    def save(self, path: str):
        torch.save({
            "cfg": self.cfg.__dict__,
            "feature_net": self.feature_net.state_dict(),
            "target_feature_net": self.target_feature_net.state_dict(),
            "blr_mu": self.blr.mu,
            "blr_cov": self.blr.cov,
            "blr_precision": self.blr.precision,
            "blr_b": self.blr.b,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.feature_net.load_state_dict(ckpt["feature_net"])
        self.target_feature_net.load_state_dict(ckpt["target_feature_net"])
        self.blr.mu = ckpt["blr_mu"]
        self.blr.cov = ckpt["blr_cov"]
        self.blr.precision = ckpt["blr_precision"]
        self.blr.b = ckpt["blr_b"]
        self.resample_q_function()
