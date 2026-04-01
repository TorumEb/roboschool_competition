from __future__ import annotations

import torch


class HistoryWrapper:
    def __init__(self, env):
        self.env = env
        self.history_length = int(getattr(env.cfg.env, "num_observation_history", 1))
        self.base_obs_dim = int(env.num_obs)
        self.num_obs = self.base_obs_dim * self.history_length
        self.num_envs = env.num_envs
        self.num_privileged_obs = env.num_privileged_obs
        self.device = env.device
        self.obs_history = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)

    def _push(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.view(self.num_envs, self.base_obs_dim)
        if self.history_length > 1:
            self.obs_history = torch.roll(self.obs_history, shifts=-self.base_obs_dim, dims=-1)
        self.obs_history[:, -self.base_obs_dim :] = obs
        return self.obs_history

    def get_observations(self):
        return self._push(self.env.get_observations())

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def reset_idx(self, env_ids):
        result = self.env.reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0
        return result

    def reset(self):
        obs, privileged_obs = self.env.reset()
        self.obs_history.zero_()
        if obs is not None:
            self._push(obs)
        return self.obs_history, privileged_obs

    def step(self, actions):
        obs, privileged_obs, rew, done, info = self.env.step(actions)
        stacked = self._push(obs)
        return stacked, privileged_obs, rew, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)
