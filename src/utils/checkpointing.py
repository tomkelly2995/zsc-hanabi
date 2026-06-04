# src/utils/checkpointing.py
# Save and load trained XDO-RPPO agents for cross-play evaluation.
#
# File format — save_agent (pickle):
#   params, player_id, meta_strategy, profile_scores, iteration, config, profile_pool
#
# File format — save_br (pickle):
#   params, player_id, iteration, target_profile_idx

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from src.population.rppo_policy_jax import RPPOPolicy


def save_agent(solver, save_dir: str, agent_id: str, config: dict,
               iteration: int | None = None) -> str:
    """
    Save a solver's oracle policy to disk.

    Converts JAX arrays to numpy for portability — the saved file has no
    JAX or GPU dependency and can be loaded on any machine.

    Args:
        solver    : XDORPPOHanabiSolverJax instance.
        save_dir  : Directory to write the checkpoint into.
        agent_id  : "agent_A" or "agent_B" — used for the filename.
        config    : Dict of training hyperparameters to store alongside weights.
        iteration : XDO iteration number; None saves as "final".

    Returns:
        Path to the written checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)

    tag      = f"iter{iteration:03d}" if iteration is not None else "final"
    filename = f"{agent_id}_{tag}.pkl"
    path     = os.path.join(save_dir, filename)

    payload = {
        "params":         jax.tree_util.tree_map(np.array, solver.oracle.params),
        "player_id":      solver.oracle.player_id,
        "meta_strategy":  np.array(solver.meta_strategy),
        "profile_scores": list(solver._profile_scores),
        "iteration":      solver._iteration,
        "config":         config,
        "profile_pool":   [
            jax.tree_util.tree_map(np.array, p.bc_model_params)
            for p in solver.profile_pool
        ],
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    return path


def load_agent(path: str, device=None) -> RPPOPolicy:
    """
    Load a saved oracle policy and return it as an RPPOPolicy ready for
    episode collection.

    Args:
        path   : Path to a .pkl checkpoint written by save_agent().
        device : JAX device to place the params on.  None → jax.devices()[0].

    Returns:
        RPPOPolicy with ac_params on `device` and the saved player_id.
    """
    device = device or jax.devices()[0]

    with open(path, "rb") as f:
        payload = pickle.load(f)

    params = jax.device_put(
        jax.tree_util.tree_map(jnp.array, payload["params"]),
        device,
    )
    return RPPOPolicy(ac_params=params, player_id=payload["player_id"])


def load_metadata(path: str) -> dict:
    """Return the non-parameter fields from a checkpoint (config, scores, etc.)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return {k: v for k, v in payload.items() if k != "params"}


def save_br(solver, save_dir: str, agent_id: str, iteration: int) -> str:
    """
    Save the current oracle as a lightweight per-iteration best-response checkpoint.

    Each BR_k is the oracle trained specifically against BC profile k.  These are
    used by evaluate_ll_inference.py: log-likelihood inference identifies which BC
    profile best explains a new partner's behaviour, then deploys the corresponding
    BR_k rather than the final mixed-strategy oracle.

    Lighter than save_agent — stores only params, not the full profile pool.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{agent_id}_br{iteration:03d}.pkl"
    path     = os.path.join(save_dir, filename)

    payload = {
        "params":             jax.tree_util.tree_map(np.array, solver.oracle.params),
        "player_id":          solver.oracle.player_id,
        "iteration":          iteration,
        "target_profile_idx": iteration,   # BR_k trained against profile_k
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return path


def save_run(solver_a, solver_b, save_dir: str, config: dict,
             iteration: int | None = None) -> None:
    """
    Convenience wrapper — saves both agents from a run in one call.

    Creates:
        <save_dir>/agent_A_<tag>.pkl
        <save_dir>/agent_B_<tag>.pkl
        <save_dir>/config.json          (written once; not overwritten if present)
    """
    path_a = save_agent(solver_a, save_dir, "agent_A", config, iteration)
    path_b = save_agent(solver_b, save_dir, "agent_B", config, iteration)
    print(f"  [ckpt] saved → {path_a}")
    print(f"  [ckpt] saved → {path_b}")

    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
