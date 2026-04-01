from __future__ import annotations

import numpy as np


class RewardThresholdCurriculum:
    def __init__(self, seed: int, **key_ranges):
        self.rng = np.random.RandomState(seed)
        self.keys = list(key_ranges.keys())
        self.cfg = {}
        self.weights = None
        grids = []
        idx_grids = []
        for key, v_range in key_ranges.items():
            low, high, bins = v_range
            if bins <= 0:
                raise ValueError(f"Invalid bin count for {key}: {bins}")
            bin_size = (high - low) / bins if bins > 0 else 0.0
            self.cfg[key] = np.linspace(low + 0.5 * bin_size, high - 0.5 * bin_size, bins) if bins > 1 else np.array([(low + high) / 2.0])
            grids.append(self.cfg[key])
            idx_grids.append(np.arange(len(self.cfg[key])))

        mesh = np.meshgrid(*grids, indexing="ij")
        idx_mesh = np.meshgrid(*idx_grids, indexing="ij")
        self.grid = np.stack(mesh).reshape(len(self.keys), -1).T
        self.idx_grid = np.stack(idx_mesh).reshape(len(self.keys), -1).T
        self.weights = np.ones(len(self.grid), dtype=np.float64)
        self.weights /= self.weights.sum()

    def sample(self, batch_size: int):
        inds = self.rng.choice(len(self.grid), size=batch_size, p=self.weights)
        samples = self.grid[inds]
        return samples, inds

    def update(self, bin_inds, rewards, thresholds):
        if len(bin_inds) == 0:
            return
        if len(thresholds) == 0:
            return
        success = np.ones(len(bin_inds), dtype=bool)
        for reward, threshold in zip(rewards, thresholds):
            reward = np.asarray(reward)
            success &= reward > threshold
        if not np.any(success):
            return
        selected = np.asarray(bin_inds)[success]
        self.weights[selected] = np.clip(self.weights[selected] + 0.2, 0.0, 1.0)
        self.weights /= self.weights.sum()
