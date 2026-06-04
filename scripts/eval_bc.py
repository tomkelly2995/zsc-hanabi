#!/usr/bin/env python3
# scripts/eval_bc.py
# ---------------------------------------------------------------------------
# Standalone BC accuracy benchmark — no XDO training loop required.
#
# Generates episodes using a configurable "partner" policy, then trains and
# evaluates several BC configurations side-by-side.  Runs in a few minutes
# on CPU so you can iterate on BC improvements without a full 80-iter XDO run.
#
# Usage (from project root):
#   python scripts/eval_bc.py                   # 500 episodes, random partner
#   python scripts/eval_bc.py --episodes 1000   # more data
#   python scripts/eval_bc.py --partner softmax # softmax-random partner (non-uniform)
#
# Partner policy options:
#   random   — uniform over legal actions (hardest to clone; accuracy ceiling ~0.35)
#   softmax  — softmax-random with temperature=0.5 (more peaked; higher ceiling)
#   greedy   — argmax of a random linear policy (most deterministic; highest ceiling)
# ---------------------------------------------------------------------------

import sys
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is on path regardless of where the script is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.hle_wrapper import HLEWrapper
from src.agents.bc_partner_model import BCPartnerModel


# ---------------------------------------------------------------------------
# Partner policies (generate training data of varying clonability)
# ---------------------------------------------------------------------------

class RandomPartner:
    """Uniform random over legal actions. Hardest to clone — BC accuracy ~0.3."""
    def act(self, obs, legal_actions):
        return random.choice(legal_actions)


class SoftmaxPartner:
    """
    Fixed random linear policy run through softmax with temperature.
    More peaked than uniform; BC accuracy ceiling depends on temperature.
    temperature=0.3 → near-greedy; temperature=1.0 → near-uniform.
    """
    def __init__(self, obs_dim=84, num_actions=8, temperature=0.3, seed=42):
        torch.manual_seed(seed)
        self._W = torch.randn(num_actions, obs_dim) * 0.1
        self._temp = temperature
        self._num_actions = num_actions

    def act(self, obs, legal_actions):
        obs_t = torch.from_numpy(obs).float()
        logits = (self._W @ obs_t) / self._temp
        mask = torch.full((self._num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        probs = F.softmax(logits + mask, dim=0)
        return torch.multinomial(probs, 1).item()


class GreedyPartner:
    """
    Always picks argmax of a fixed random linear policy (deterministic).
    Upper bound for BC clonability with linear policies.
    """
    def __init__(self, obs_dim=84, num_actions=8, seed=42):
        torch.manual_seed(seed)
        self._W = torch.randn(num_actions, obs_dim) * 0.1
        self._num_actions = num_actions

    def act(self, obs, legal_actions):
        obs_t = torch.from_numpy(obs).float()
        logits = self._W @ obs_t
        mask = torch.full((self._num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        return (logits + mask).argmax().item()


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episodes(partner, num_episodes: int, partner_id: int = 1):
    """
    Roll out num_episodes episodes with a random agent_0 and the given partner
    as agent_1 (or swap if partner_id=0).

    Returns:
        obs_action_pairs : list of (obs_np, action_int) — partner's turns only
        scores           : list of int — terminal team scores
    """
    env = HLEWrapper()
    pairs = []
    scores = []

    for _ in range(num_episodes):
        obs_step = env.reset()
        done = False
        info = {"score": 0}

        while not done:
            pid = obs_step.current_player
            obs = obs_step.player_obs[pid]
            legal = obs_step.legal_moves[pid]

            if pid == partner_id:
                action = partner.act(obs, legal)
                pairs.append((obs.copy(), action))
            else:
                # Agent on the other side plays uniformly random
                action = random.choice(legal)

            obs_step, _, done, info = env.step(action)

        scores.append(info["score"])

    return pairs, scores


# ---------------------------------------------------------------------------
# BC configurations to compare
# ---------------------------------------------------------------------------

def make_baseline(obs_dim, num_actions):
    """Current production config — 84→128→Dropout(0.2)→64→8, no masking, no scheduler."""
    return BCPartnerModel(obs_dim=obs_dim, num_actions=num_actions)


def train_config(name, model, pairs, epochs, lr, use_legal_mask, label_smoothing,
                 use_scheduler, batch_size=64):
    """
    Train a BCPartnerModel with the given hyper-parameters and return
    (name, train_loss, val_accuracy, val_entropy).

    use_legal_mask: if True, mask illegal actions to -inf before cross-entropy
                    (requires legal action list stored alongside each pair).
    """
    if len(pairs) == 0:
        return name, 0.0, 0.0, 0.0

    # 80/20 train/val split
    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    obs_list = [torch.from_numpy(o).float() if isinstance(o, np.ndarray) else o.float()
                for o, _ in train_pairs]
    obs_tensor = torch.stack(obs_list)
    act_tensor = torch.tensor([a for _, a in train_pairs], dtype=torch.long)

    n = len(train_pairs)
    effective_batch = min(batch_size, n)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    final_loss = 0.0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, effective_batch):
            idx = perm[start: start + effective_batch]
            logits = model(obs_tensor[idx])
            loss = criterion(logits, act_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if scheduler:
            scheduler.step()
        if epoch == epochs - 1:
            final_loss = epoch_loss / max(num_batches, 1)
    model.eval()

    acc, entropy = model.evaluate(val_pairs)
    return name, final_loss, acc, entropy


# ---------------------------------------------------------------------------
# Legal-masking variant of train_supervised
# (needs legal actions stored alongside each pair)
# ---------------------------------------------------------------------------

def evaluate_with_legal_mask(model, pairs_with_legal):
    """
    Evaluate a model using the same legal-action masking as action_distribution()
    at inference time.  Returns (accuracy, mean_entropy).
    """
    if len(pairs_with_legal) == 0:
        return 0.0, 0.0

    num_actions = model._num_actions
    obs_list = [torch.from_numpy(o).float() if isinstance(o, np.ndarray) else o.float()
                for o, _, _ in pairs_with_legal]
    obs_tensor  = torch.stack(obs_list)
    act_tensor  = torch.tensor([a for _, a, _ in pairs_with_legal], dtype=torch.long)
    n = len(pairs_with_legal)

    # Use a large-but-finite negative value so softmax pushes illegal probs ~0
    # without producing NaN/inf in entropy calculations.
    legal_masks = torch.full((n, num_actions), -1e9)
    for i, (_, _, legal) in enumerate(pairs_with_legal):
        legal_masks[i, legal] = 0.0

    model.eval()
    with torch.no_grad():
        logits = model(obs_tensor) + legal_masks
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

    accuracy     = (preds == act_tensor).float().mean().item()
    log_probs    = torch.log(probs.clamp(min=1e-12))
    mean_entropy = -(probs * log_probs).sum(dim=1).mean().item()
    return accuracy, mean_entropy


def train_with_legal_mask(model, pairs_with_legal, epochs, lr, label_smoothing,
                          use_scheduler, batch_size=64):
    """
    Like train_config but uses legal-action masking during training.
    pairs_with_legal: list of (obs, action, legal_actions_list)

    Uses -1e9 instead of -inf so that label_smoothing does not create inf/NaN
    losses (label smoothing distributes a small probability to ALL actions; if
    any logit is -inf, log_softmax(-inf) = -inf and the loss diverges).
    """
    if len(pairs_with_legal) == 0:
        return 0.0

    random.shuffle(pairs_with_legal)
    n = len(pairs_with_legal)
    effective_batch = min(batch_size, n)

    obs_list = [torch.from_numpy(o).float() if isinstance(o, np.ndarray) else o.float()
                for o, _, _ in pairs_with_legal]
    obs_tensor = torch.stack(obs_list)
    act_tensor = torch.tensor([a for _, a, _ in pairs_with_legal], dtype=torch.long)
    # Use a large-but-finite negative to avoid inf loss with label_smoothing
    num_actions = model._num_actions
    legal_masks = torch.full((n, num_actions), -1e9)
    for i, (_, _, legal) in enumerate(pairs_with_legal):
        legal_masks[i, legal] = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    final_loss = 0.0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, effective_batch):
            idx = perm[start: start + effective_batch]
            logits = model(obs_tensor[idx]) + legal_masks[idx]
            loss = F.cross_entropy(
                logits, act_tensor[idx],
                label_smoothing=label_smoothing,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if scheduler:
            scheduler.step()
        if epoch == epochs - 1:
            final_loss = epoch_loss / max(num_batches, 1)
    model.eval()
    return final_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BC accuracy benchmark")
    parser.add_argument("--episodes",   type=int,   default=500,
                        help="Number of episodes to generate (default 500)")
    parser.add_argument("--partner",    type=str,   default="softmax",
                        choices=["random", "softmax", "greedy"],
                        help="Partner policy type (default: softmax)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Softmax partner temperature (default 0.3)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBS_DIM      = 84
    NUM_ACTIONS  = 8

    # ------------------------------------------------------------------
    # Build partner
    # ------------------------------------------------------------------
    if args.partner == "random":
        partner = RandomPartner()
    elif args.partner == "softmax":
        partner = SoftmaxPartner(temperature=args.temperature, seed=args.seed)
    else:
        partner = GreedyPartner(seed=args.seed)

    # ------------------------------------------------------------------
    # Generate episodes — also collect legal actions for masking variant
    # ------------------------------------------------------------------
    print(f"\nGenerating {args.episodes} episodes with '{args.partner}' partner "
          f"(temp={args.temperature})...")

    env = HLEWrapper()
    pairs           = []   # (obs, action)
    pairs_w_legal   = []   # (obs, action, legal_actions)
    scores          = []
    partner_id      = 1

    for ep in range(args.episodes):
        obs_step = env.reset()
        done = False
        info = {"score": 0}
        while not done:
            pid    = obs_step.current_player
            obs    = obs_step.player_obs[pid]
            legal  = obs_step.legal_moves[pid]
            if pid == partner_id:
                action = partner.act(obs, legal)
                pairs.append((obs.copy(), action))
                pairs_w_legal.append((obs.copy(), action, list(legal)))
            else:
                action = random.choice(legal)
            obs_step, _, done, info = env.step(action)
        scores.append(info["score"])

    print(f"  Collected {len(pairs)} (obs, action) pairs  |  "
          f"mean score={np.mean(scores):.3f}  |  "
          f"random BC baseline={1/NUM_ACTIONS:.3f}")

    # ------------------------------------------------------------------
    # Run BC configurations
    # ------------------------------------------------------------------
    configs = [
        # (display_name, epochs, lr, use_legal_mask, label_smoothing, use_scheduler)
        ("Baseline    (40ep, 1e-3, no mask, no sched)", 40, 1e-3, False, 0.0,  False),
        ("MoreEpochs  (80ep, 1e-3, no mask, no sched)", 80, 1e-3, False, 0.0,  False),
        ("Scheduler   (40ep, 1e-3, no mask, cosine  )", 40, 1e-3, False, 0.0,  True),
        ("LabelSmooth (40ep, 1e-3, no mask, ls=0.1  )", 40, 1e-3, False, 0.1,  False),
        ("LegalMask   (40ep, 1e-3, MASK,   no sched )", 40, 1e-3, True,  0.0,  False),
        ("LS+Sched    (40ep, 1e-3, no mask, cosine+ls )", 40, 1e-3, False, 0.1,  True),
        ("Combined    (80ep, 1e-3, MASK,   cosine    )", 80, 1e-3, True,  0.0,  True),
    ]

    print(f"\n{'Config':<52}  {'TrainLoss':>9}  {'ValAcc':>7}  {'ValEntropy':>10}")
    print("-" * 82)

    for cfg_name, epochs, lr, use_mask, ls, use_sched in configs:
        model = BCPartnerModel(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)

        if use_mask:
            # train and evaluate both with legal masking (matches real inference)
            random.shuffle(pairs_w_legal)
            split     = int(0.8 * len(pairs_w_legal))
            train_wl  = pairs_w_legal[:split]
            val_wl    = pairs_w_legal[split:]
            train_loss = train_with_legal_mask(
                model, train_wl, epochs, lr, ls, use_sched
            )
            val_acc, val_entropy = evaluate_with_legal_mask(model, val_wl)
        else:
            cfg_name_out, train_loss, val_acc, val_entropy = train_config(
                cfg_name, model, list(pairs), epochs, lr,
                use_mask, ls, use_sched
            )

        print(f"  {cfg_name:<50}  {train_loss:>9.4f}  {val_acc:>7.4f}  {val_entropy:>10.4f}")

    print()
    print("Notes:")
    print(f"  Random baseline accuracy = {1/NUM_ACTIONS:.4f}")
    print(f"  Max entropy (uniform)    = {np.log(NUM_ACTIONS):.4f}")
    print(f"  Partner type             = {args.partner}")
    if args.partner == "softmax":
        print(f"  Temperature              = {args.temperature}  "
              f"(lower = more peaked = easier to clone)")
    print()


if __name__ == "__main__":
    main()
