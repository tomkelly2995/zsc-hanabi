# src/networks/gru_encoder.py
# GRUEncoder: encodes partner action sequences into h_oppo (64-dim).
# Embedding dim: 16, GRU hidden size: 64  (from spec Section 3.3).
#
# Public interface uses flat (hidden_dim,) tensors for h_oppo throughout —
# the GRU's internal (num_layers, batch, hidden) shape is kept internal.
#
# Two use cases:
#   1. Profile creation / lazy re-encoding:
#        h_oppo = encoder.encode_sequences(raw_sequences)
#   2. Online adaptation during traversal or live play:
#        h = encoder.initial_hidden()
#        for action in observed_actions:
#            h = encoder.step(h, action)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    """
    GRU-based encoder for partner action sequences.

    encode_sequences(): encodes a batch of variable-length episodes and
        returns the mean of their final hidden states — the cached_h_oppo
        stored in a BehaviourProfile.

    step(): advances the hidden state by one action — used online during
        ODCFR traversal and live play to maintain a running h_oppo.
    """

    def __init__(self, num_actions: int = 8, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_actions, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

    def encode_sequences(self, raw_sequences: list) -> torch.Tensor:
        """
        Encode a list of variable-length action-ID sequences.

        For each episode, embeds the action sequence and runs it through the
        GRU; takes the final hidden state. Returns the mean over all episodes.
        Empty sequences contribute a zero hidden state to the mean (preserving
        original semantics).

        All non-empty sequences are processed in a single batched GRU forward
        pass via pack_padded_sequence, replacing the previous serial loop.

        Args:
            raw_sequences: list of lists of int, e.g. [[0,1,2], [3,2,1,0]].
                           Must contain at least one non-empty sequence.

        Returns:
            torch.Tensor of shape (hidden_dim,)  — the cached_h_oppo value.
        """
        n_total = len(raw_sequences)
        device = next(self.parameters()).device
        if n_total == 0:
            return torch.zeros(self.hidden_dim, device=device)

        non_empty = [seq for seq in raw_sequences if len(seq) > 0]
        if not non_empty:
            return torch.zeros(self.hidden_dim, device=device)

        n = len(non_empty)
        lengths = [len(s) for s in non_empty]
        max_len = max(lengths)
        device = next(self.parameters()).device

        # Build padded action tensor on the encoder's device: (n, max_len)
        padded = torch.zeros(n, max_len, dtype=torch.long, device=device)
        for j, seq in enumerate(non_empty):
            padded[j, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        embedded = self.embedding(padded)                                # (n, max_len, embed_dim)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)                                        # h_n: (1, n, hidden_dim)
        h_finals = h_n.squeeze(0)                                        # (n, hidden_dim)

        # Divide by n_total so empty sequences contribute implicit zeros to the
        # mean, matching the original per-sequence averaging semantics.
        return h_finals.sum(dim=0) / n_total                             # (hidden_dim,)

    def step(self, h: torch.Tensor, action: int) -> torch.Tensor:
        """
        Advance the hidden state by one observed action.

        Args:
            h:      Current hidden state, shape (hidden_dim,).
            action: Observed partner action UID (int).

        Returns:
            New hidden state, shape (hidden_dim,).
        """
        device = next(self.parameters()).device
        action_t = torch.tensor([action], dtype=torch.long, device=device)  # (1,)
        embedded = self.embedding(action_t).unsqueeze(0)                    # (1, 1, embed_dim)
        h_in = h.unsqueeze(0).unsqueeze(0)                                  # (1, 1, hidden_dim)
        _, h_out = self.gru(embedded, h_in)                                 # (1, 1, hidden_dim)
        return h_out.squeeze(0).squeeze(0)                                  # (hidden_dim,)

    def initial_hidden(self) -> torch.Tensor:
        """
        Return a zero hidden state for starting a new sequence.
        Used at the beginning of each traversal and each live episode.
        Placed on the same device as the encoder weights automatically.

        Returns:
            torch.Tensor of shape (hidden_dim,), all zeros.
        """
        device = next(self.parameters()).device
        return torch.zeros(self.hidden_dim, device=device)
