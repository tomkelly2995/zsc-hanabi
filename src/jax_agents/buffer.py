# src/jax_agents/buffer.py
# Numpy-backed reservoir sampling buffer — identical semantics to the PyTorch
# ReservoirBuffer but stores numpy arrays (not torch tensors).
#
# AdvantageBuffer entries: (obs_raw, partner_acts_padded, prefix_len, target, iter_t)
#   obs_raw            : np.ndarray (OBS_DIM,) float32 — raw observation (no GRU)
#   partner_acts_padded: np.ndarray (MAX_PARTNER_TURNS,) int32 — full episode partner sequence
#   prefix_len         : int — how many partner actions preceded this self-turn
#   target             : np.ndarray (8,) float32 — Q-targets (taken action = score)
#   iter_t             : float — CFR iteration index (for recency weighting)
#
# PolicyBuffer entries: (obs_h, sigma)
#   obs_h : np.ndarray (148,) float32
#   sigma : np.ndarray (8,) float32 — regret-matched strategy

import numpy as np


class ReservoirBuffer:
    """
    Fixed-capacity uniform-random-sample buffer (reservoir sampling).
    Items beyond max_size are kept with probability max_size / total_seen,
    replacing a uniformly random existing entry.
    """

    def __init__(self, max_size: int):
        self._max = max_size
        self._data: list = []
        self._total: int = 0

    def add(self, item) -> None:
        self._total += 1
        if len(self._data) < self._max:
            self._data.append(item)
        else:
            idx = np.random.randint(0, self._total)
            if idx < self._max:
                self._data[idx] = item

    def sample(self, batch_size: int) -> list:
        n = min(batch_size, len(self._data))
        idxs = np.random.choice(len(self._data), size=n, replace=False)
        return [self._data[i] for i in idxs]

    def partial_reset(self, retention: float = 0.2) -> None:
        """
        Keep a uniform random subset of size floor(len * retention).
        Resets the total-seen counter to match the new size so that
        future adds are weighted correctly against the retained entries.
        """
        n_keep = max(0, int(len(self._data) * retention))
        if n_keep > 0 and len(self._data) > 0:
            idxs = np.random.choice(len(self._data), size=n_keep, replace=False)
            self._data = [self._data[i] for i in idxs]
        else:
            self._data = []
        self._total = len(self._data)

    def __len__(self) -> int:
        return len(self._data)
