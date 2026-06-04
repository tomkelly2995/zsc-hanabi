#!/usr/bin/env python3
"""
scripts/evaluate_crossplay.py

Cross-play (ZSC) evaluation for trained RPPO agents.

Pairs agent A from one checkpoint directory with agent B from another,
runs joint episodes, and reports the mean team score.  This is the standard
"XP score" from the ZSC literature — measuring whether independently trained
agents using the same method can coordinate at test time.

Evaluation modes
────────────────
  default   — pair A from run1 with B from run2 (and A2+B1 for symmetry)
  self      — pair A and B from the same run (sanity check, should be high)
  sweep     — evaluate all checkpoint iterations across two runs

Usage
─────
  # Cross-play between two runs:
  python scripts/evaluate_crossplay.py \\
      --run1 runs/seed42 --run2 runs/seed123

  # Self-play sanity check (same run):
  python scripts/evaluate_crossplay.py --run1 runs/seed42 --self

  # Sweep all saved checkpoints:
  python scripts/evaluate_crossplay.py \\
      --run1 runs/seed42 --run2 runs/seed123 --sweep
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import numpy as np
import jax

from src.utils.checkpointing import load_agent, load_metadata
from main_jax import run_joint_rppo_episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint(run_dir: str, agent_id: str, tag: str = "final") -> str:
    """
    Find a checkpoint file in run_dir matching agent_id and tag.

    tag can be "final" or "iter###".  Raises FileNotFoundError if not found.
    """
    pattern = os.path.join(run_dir, f"{agent_id}_{tag}.pkl")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching '{pattern}' in {run_dir}.\n"
            f"Available files: {os.listdir(run_dir)}"
        )
    return sorted(matches)[-1]


def _find_all_checkpoints(run_dir: str, agent_id: str) -> list[tuple[str, str]]:
    """Return (tag, path) pairs for all checkpoints of agent_id, sorted by iter."""
    pattern = os.path.join(run_dir, f"{agent_id}_*.pkl")
    paths   = sorted(glob.glob(pattern))
    result  = []
    for p in paths:
        filename = os.path.basename(p)
        tag = filename[len(agent_id) + 1 : -4]   # strip "agent_X_" prefix and ".pkl"
        result.append((tag, p))
    return result


def evaluate_pair(path_a: str, path_b: str, n_episodes: int, seed: int,
                  device=None) -> dict:
    """
    Run n_episodes with agent A playing as player 0 and agent B as player 1.

    Returns a dict with mean, std, and per-episode scores.
    """
    device  = device or jax.devices()[0]
    pol_a   = load_agent(path_a, device=device)
    pol_b   = load_agent(path_b, device=device)

    key      = jax.random.PRNGKey(seed)
    episodes = run_joint_rppo_episodes(pol_a, pol_b, n_episodes, key=key, device=device)
    scores   = np.array([ep["score"] for ep in episodes], dtype=np.float32)

    return {
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores)),
        "scores": scores.tolist(),
        "path_a": path_a,
        "path_b": path_b,
    }


def xp_score(result_ab: dict, result_ba: dict) -> tuple[float, float]:
    """
    Symmetric XP score = ½(score(A1,B2) + score(A2,B1)).

    Returns (xp_mean, xp_std) pooling both directions.
    """
    all_scores = result_ab["scores"] + result_ba["scores"]
    return float(np.mean(all_scores)), float(np.std(all_scores))


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def run_self_eval(run_dir: str, n_episodes: int, seed: int, device) -> None:
    """Pair A and B from the same run — sanity check for within-run coordination."""
    sep = "=" * 60
    print(sep)
    print(f"Self-evaluation: {run_dir}")
    print(sep)

    path_a = _find_checkpoint(run_dir, "agent_A")
    path_b = _find_checkpoint(run_dir, "agent_B")

    meta_a = load_metadata(path_a)
    meta_b = load_metadata(path_b)
    print(f"  agent_A: iter={meta_a['iteration']}  pool={len(meta_a.get('profile_scores', []))}")
    print(f"  agent_B: iter={meta_b['iteration']}  pool={len(meta_b.get('profile_scores', []))}")

    result = evaluate_pair(path_a, path_b, n_episodes, seed, device)
    print(f"\n  Score (A+B same run, {n_episodes} eps): "
          f"{result['mean']:.3f} ± {result['std']:.3f}")
    print(f"  Score distribution: " +
          "  ".join(f"{s}→{result['scores'].count(s)}" for s in range(5)))


def run_crossplay_eval(run1_dir: str, run2_dir: str, n_episodes: int,
                       seed: int, device) -> None:
    """Standard cross-play: A from run1 vs B from run2, and vice versa."""
    sep = "=" * 60
    print(sep)
    print("Cross-play evaluation (ZSC)")
    print(f"  Run 1: {run1_dir}")
    print(f"  Run 2: {run2_dir}")
    print(sep)

    path_a1 = _find_checkpoint(run1_dir, "agent_A")
    path_b1 = _find_checkpoint(run1_dir, "agent_B")
    path_a2 = _find_checkpoint(run2_dir, "agent_A")
    path_b2 = _find_checkpoint(run2_dir, "agent_B")

    # Direction 1: A from run1, B from run2
    print(f"\n  [A1 + B2]  {n_episodes} episodes …", end="", flush=True)
    r_ab = evaluate_pair(path_a1, path_b2, n_episodes, seed, device)
    print(f"  {r_ab['mean']:.3f} ± {r_ab['std']:.3f}")

    # Direction 2: A from run2, B from run1
    print(f"  [A2 + B1]  {n_episodes} episodes …", end="", flush=True)
    r_ba = evaluate_pair(path_a2, path_b1, n_episodes, seed + 1, device)
    print(f"  {r_ba['mean']:.3f} ± {r_ba['std']:.3f}")

    xp_mean, xp_std = xp_score(r_ab, r_ba)
    print(f"\n  XP score (symmetric): {xp_mean:.3f} ± {xp_std:.3f}")

    # Self-play baselines for comparison
    print(f"\n  [self-play baselines]")
    r_self1 = evaluate_pair(path_a1, path_b1, n_episodes, seed + 2, device)
    r_self2 = evaluate_pair(path_a2, path_b2, n_episodes, seed + 3, device)
    print(f"  Run 1 self: {r_self1['mean']:.3f} ± {r_self1['std']:.3f}")
    print(f"  Run 2 self: {r_self2['mean']:.3f} ± {r_self2['std']:.3f}")

    sp_mean = 0.5 * (r_self1['mean'] + r_self2['mean'])
    print(f"\n  XP / SP ratio: {xp_mean / sp_mean:.3f}  "
          f"(1.0 = perfect ZSC, <1.0 = convention mismatch)")


def run_sweep(run1_dir: str, run2_dir: str, n_episodes: int,
              seed: int, device) -> None:
    """Evaluate cross-play at every saved checkpoint iteration."""
    sep = "=" * 60
    print(sep)
    print("Checkpoint sweep")
    print(f"  Run 1: {run1_dir}")
    print(f"  Run 2: {run2_dir}")
    print(sep)

    ckpts_a1 = _find_all_checkpoints(run1_dir, "agent_A")
    ckpts_b2 = _find_all_checkpoints(run2_dir, "agent_B")

    if len(ckpts_a1) != len(ckpts_b2):
        print(f"  Warning: different number of checkpoints "
              f"({len(ckpts_a1)} vs {len(ckpts_b2)}). "
              f"Evaluating matching tags only.")

    tags_a = {tag for tag, _ in ckpts_a1}
    tags_b = {tag for tag, _ in ckpts_b2}
    shared = sorted(tags_a & tags_b)

    print(f"\n  {'tag':>10}  {'XP A1+B2':>10}  {'self A1+B1':>12}")
    print("  " + "-" * 38)

    for tag in shared:
        path_a1 = dict(ckpts_a1)[tag]
        path_b1 = _find_checkpoint(run1_dir, "agent_B", tag)
        path_b2 = dict(ckpts_b2)[tag]

        r_xp   = evaluate_pair(path_a1, path_b2, n_episodes, seed,     device)
        r_self = evaluate_pair(path_a1, path_b1, n_episodes, seed + 1, device)
        print(f"  {tag:>10}  {r_xp['mean']:>10.3f}  {r_self['mean']:>12.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Cross-play ZSC evaluation for trained RPPO agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run1",       type=str, required=True,
                   help="Checkpoint directory for run 1")
    p.add_argument("--run2",       type=str, default=None,
                   help="Checkpoint directory for run 2 (omit with --self)")
    p.add_argument("--self",       action="store_true",
                   help="Self-evaluation within --run1 only")
    p.add_argument("--sweep",      action="store_true",
                   help="Sweep all checkpoint iterations")
    p.add_argument("--n_episodes", type=int, default=500,
                   help="Episodes per evaluation pair")
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()

    device = jax.devices()[0]
    print(f"JAX device: {device}")

    if args.self:
        run_self_eval(args.run1, args.n_episodes, args.seed, device)
    elif args.sweep:
        if args.run2 is None:
            p.error("--sweep requires --run2")
        run_sweep(args.run1, args.run2, args.n_episodes, args.seed, device)
    else:
        if args.run2 is None:
            p.error("cross-play requires --run2 (or use --self)")
        run_crossplay_eval(args.run1, args.run2, args.n_episodes, args.seed, device)
