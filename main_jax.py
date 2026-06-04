# main_jax.py
# Joint training loop for Cooperative XDO on tiny Hanabi — JAX/Flax version.
# Uses pure-JAX environment (src/jax_env/tiny_hanabi.py) and JAX solvers.
#
# Usage:
#   python main_jax.py [--xdo_iterations N] [--episodes_per_iter E]
#                      [--cfr_min T] [--cfr_max T] [--k_simulations K]
#                      [--seed S]

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax
import jax.numpy as jnp

from src.jax_env.tiny_hanabi import (
    reset      as _env_reset,
    step       as _env_step,
    get_obs    as _env_get_obs,
    legal_mask as _env_legal_mask,
)
from src.xdo.xdo_solver_jax import XDOHanabiSolverJax
from src.xdo.xdo_rppo_solver_jax import XDORPPOHanabiSolverJax
from src.population.hanabi_policy_jax import HanabiPolicyJax
from src.jax_agents.simulation import run_N_joint_episodes, joint_results_to_episode_dicts
from src.jax_agents.rppo_agent_jax import run_N_joint_rppo_episodes
from src.utils.logging_jax import TensorboardLoggerJax
from src.utils.checkpointing import save_run


# ---------------------------------------------------------------------------
# Episode collection helpers
# ---------------------------------------------------------------------------

def _run_random_episodes(n: int, seed: int = 0) -> list:
    """
    Collect n episodes with fully random legal-action play.
    Used for the bootstrap batch before any policies are trained.
    """
    episodes = []
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    for _ in range(n):
        key, k_reset, k_steps = jax.random.split(key, 3)
        state = _env_reset(k_reset)
        prev_state = state
        prev_act = jnp.array(0, dtype=jnp.int32)
        steps = []
        done = False

        step_key = k_steps
        while not done:
            step_key, k_act = jax.random.split(step_key)
            cur = int(jnp.argmax(state.cur_player_idx))

            obs0 = np.array(_env_get_obs(state, prev_state, prev_act, 0))
            obs1 = np.array(_env_get_obs(state, prev_state, prev_act, 1))
            legal = np.array(_env_legal_mask(state))
            legal_ids = [i for i, v in enumerate(legal) if v]

            action = int(rng.choice(legal_ids))
            steps.append({
                "current_player": cur,
                "player_obs": {0: obs0, 1: obs1},
                "action": action,
            })
            act_jnp = jnp.array(action, dtype=jnp.int32)
            new_state, _, done_jnp = _env_step(state, act_jnp)
            done = bool(done_jnp)
            prev_state = state
            prev_act = act_jnp
            state = new_state

        episodes.append({"steps": steps, "score": int(state.score)})
    return episodes


def run_joint_episodes(
    policy_a: HanabiPolicyJax,
    policy_b: HanabiPolicyJax,
    n: int,
    key: jax.Array,
) -> list:
    """
    Collect n joint episodes via vmap — all episodes run in one JIT call.

    Returns list of episode dicts in XDOHanabiSolverJax format.
    """
    keys = jax.random.split(key, n)
    results = run_N_joint_episodes(
        keys,
        policy_a.pol_params, policy_a.gru_params,
        policy_b.pol_params, policy_b.gru_params,
    )
    return joint_results_to_episode_dicts(results)


def run_joint_rppo_episodes(policy_a, policy_b, n: int, key: jax.Array,
                            device=None) -> list:
    """
    Collect n joint episodes where both players use RPPOActorCritic.

    Mirrors run_joint_episodes but calls run_N_joint_rppo_episodes.
    Returns list of episode dicts compatible with extract_behaviour_profile().

    Both agents' params are moved to `device` before dispatch so the JIT
    kernel runs on a single device even when A and B were trained on
    different GPUs.  Defaults to jax.devices()[0].
    """
    device = device or jax.devices()[0]
    keys       = jax.device_put(jax.random.split(key, n), device)
    ac_params_0 = jax.device_put(policy_a.ac_params, device)
    ac_params_1 = jax.device_put(policy_b.ac_params, device)
    results = run_N_joint_rppo_episodes(keys, ac_params_0, ac_params_1)
    return joint_results_to_episode_dicts(results)


# ---------------------------------------------------------------------------
# Metric helpers (identical logic to main.py)
# ---------------------------------------------------------------------------

def _swap_regret(meta_strategy: np.ndarray, profile_scores: list) -> float:
    if len(profile_scores) < 2:
        return 0.0
    scores = np.array(profile_scores)
    return float(np.dot(meta_strategy, np.maximum(0.0, scores.max() - scores)))


def _metastrategy_entropy(meta_strategy: np.ndarray) -> float:
    if len(meta_strategy) < 2:
        return 0.0
    probs = meta_strategy[meta_strategy > 0]
    return float(-np.sum(probs * np.log(probs)))


def _top_convention_score(meta_strategy: np.ndarray, profile_scores: list) -> float:
    if len(profile_scores) < 2:
        return float(profile_scores[0]) if profile_scores else 0.0
    return float(profile_scores[int(np.argmax(meta_strategy))])


# ---------------------------------------------------------------------------
# Per-agent RPPO training step — runs in its own thread on its pinned device
# ---------------------------------------------------------------------------

def _train_rppo_agent(
    solver,
    shared_batch: list,
    partner_scores_snapshot: list,
    ppo_updates: int,
    n_episodes: int,
    bc_rounds: int,
    device,
):
    """
    One full XDO inner-loop step for a single RPPO agent.

    Runs solve_meta_game → request_new_policy → extract_behaviour_profile
    entirely inside a jax.default_device context so that BC training and any
    incidental JAX array creation land on the correct GPU.

    Called from ThreadPoolExecutor so A and B execute concurrently on T4 x2.
    """
    with jax.default_device(device):
        solver.solve_meta_game(partner_scores=partner_scores_snapshot)

        t0 = time.perf_counter()
        policy = solver.request_new_policy(
            ppo_updates=ppo_updates,
            n_episodes=n_episodes,
        )
        policy_secs = time.perf_counter() - t0

        t1 = time.perf_counter()
        solver.extract_behaviour_profile(
            shared_batch, bc_epochs=40, bc_rounds=bc_rounds
        )
        profile_secs = time.perf_counter() - t1

    return policy, policy_secs, profile_secs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cooperative XDO training on tiny Hanabi (JAX/Flax)")
    parser.add_argument("--xdo_iterations",      type=int,   default=20)
    parser.add_argument("--episodes_per_iter",   type=int,   default=50)
    parser.add_argument("--cfr_min",             type=int,   default=200)
    parser.add_argument("--cfr_max",             type=int,   default=400)
    parser.add_argument("--cfr_exponent",        type=float, default=0.75)
    parser.add_argument("--k_simulations",       type=int,   default=16)
    parser.add_argument("--k_max",               type=int,   default=96)
    parser.add_argument("--k_exponent",          type=float, default=0.75)
    parser.add_argument("--adv_train_steps",     type=int,   default=50)
    parser.add_argument("--pol_train_steps",     type=int,   default=200)
    parser.add_argument("--early_stop_min_iters", type=int,  default=250)
    parser.add_argument("--early_stop_delta",    type=float, default=0.01)
    parser.add_argument("--partner_temperature", type=float, default=0.3)
    parser.add_argument("--aux_loss_weight",     type=float, default=0.2)
    parser.add_argument("--gru_lr",              type=float, default=5e-5,
                        help="GRU joint-update learning rate. Lowered from 1e-4 after "
                             "diagnostic showed IS-weighted targets caused Q explosion "
                             "and cos_sim instability at higher LRs.")
    parser.add_argument("--gru_grad_clip",       type=float, default=0.3,
                        help="Global-norm gradient clip applied to GRU optimizer. "
                             "Lowered from 0.5 for stability with IS-weighted training.")
    parser.add_argument("--explore_eps",         type=float, default=0.06,
                        help="ε-on-policy exploration at self-turns (0=off, 0.06=default). "
                             "Prevents action collapse (hint_rank never selected).")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--bc_rounds",           type=int,   default=3,
                        help="Iterative BC bootstrapping rounds per profile build. "
                             "Each round rebuilds teacher-forced h_bc pairs with "
                             "current params then trains for bc_epochs//bc_rounds epochs. "
                             "bc_rounds=1 reproduces the old single-pass behaviour.")
    parser.add_argument("--rescore_topk",         type=int,   default=3,
                        help="Number of highest-weight profiles to re-score with the "
                             "current oracle policy at the start of each solve_meta_game "
                             "call. Keeps the meta-strategy tracking real coordination "
                             "quality rather than stale creation-time probe scores. "
                             "Set to 0 to disable re-scoring.")
    parser.add_argument("--rescore_full_every",   type=int,   default=5,
                        help="Re-score ALL profiles every N outer XDO iterations. "
                             "Counteracts convention pool stagnation by letting low-weight "
                             "profiles resurface if the oracle has learned to coordinate "
                             "with them. On full-rescore iterations the top-k partial "
                             "rescore is skipped (all scores are already fresh). "
                             "Set to 0 to disable. Default 5.")
    parser.add_argument("--coupling_alpha",       type=float, default=0.3,
                        help="Blend weight for partner probe scores in solve_meta_game. "
                             "Each agent's score for profile k is blended as: "
                             "(1-α)*own_score[k] + α*partner_score[k] before the "
                             "multiplicative-weights update. Bounds L1 divergence between "
                             "agents' meta-strategies while preserving role-asymmetric "
                             "diversity (player 0 vs 1 naturally differ). "
                             "Useful range is [0.0, 0.5]: α=0.0 disables coupling "
                             "(independent, can reach L1=2.0); α=0.5 forces identical "
                             "meta-strategies (pure shared average, L1=0); "
                             "α=0.3 is a good default giving bounded moderate diversity. "
                             "Values above 0.5 swap the asymmetry and are not useful.")
    parser.add_argument("--log_dir",             type=str,   default="runs/xdo_jax",
                        help="Directory for TensorboardX logs. "
                             "View with: tensorboard --logdir <log_dir>")
    parser.add_argument("--save_dir",             type=str,   default=None,
                        help="Directory to write agent checkpoints. "
                             "If omitted, no checkpoints are saved.")
    parser.add_argument("--checkpoint_every",     type=int,   default=5,
                        help="Save a checkpoint every N XDO iterations "
                             "(only used when --save_dir is set).")
    parser.add_argument("--br_save_dir",          type=str,   default=None,
                        help="Directory to save per-iteration BR_k checkpoints "
                             "for log-likelihood inference evaluation. "
                             "Each iteration k saves agent_A_br{k:03d}.pkl and "
                             "agent_B_br{k:03d}.pkl. If omitted, BRs are not saved.")
    parser.add_argument("--temp_schedule",        type=str,   default=None,
                        help="Comma-separated partner temperatures to cycle through "
                             "across oracle training iterations, e.g. '0.1,0.3,0.5,1.0'. "
                             "Low temp → deterministic partner; high temp → exploratory. "
                             "Cycling creates behaviourally diverse BC profiles, which "
                             "makes log-likelihood inference discriminative. "
                             "If omitted, uses constant --partner_temperature.")
    parser.add_argument("--sequential",           action="store_true",
                        help="Force sequential A-then-B training even when multiple "
                             "GPUs are available. Avoids IOStream timeouts and memory "
                             "pressure from simultaneous JAX kernels on Kaggle.")
    # Solver selection
    parser.add_argument("--solver",              type=str,   default="odcfr",
                        choices=["odcfr", "rppo"],
                        help="Inner-loop solver: 'odcfr' (default) or 'rppo'.")
    # RPPO-specific hyperparameters (only used when --solver rppo)
    parser.add_argument("--ppo_updates",         type=int,   default=1000,
                        help="PPO collect+update iterations per oracle call (RPPO only).")
    parser.add_argument("--n_episodes_rppo",     type=int,   default=256,
                        help="Episodes collected per PPO update (RPPO only).")
    parser.add_argument("--minibatch_size",      type=int,   default=128,
                        help="PPO minibatch size. Must divide n_episodes_rppo × MAX_SELF_TURNS (RPPO only).")
    parser.add_argument("--ppo_epochs",          type=int,   default=4,
                        help="PPO gradient epochs per collected batch (RPPO only).")
    parser.add_argument("--clip_eps",            type=float, default=0.2,
                        help="PPO clipping ratio ε. Lower → more conservative updates (RPPO only).")
    parser.add_argument("--ent_coef",            type=float, default=0.05,
                        help="Entropy coefficient for PPO loss (RPPO only).")
    parser.add_argument("--vf_coef",             type=float, default=0.5,
                        help="Value function loss coefficient for PPO (RPPO only).")
    parser.add_argument("--rppo_lr",             type=float, default=3e-4,
                        help="Learning rate for RPPOActorCritic optimizer (RPPO only).")
    parser.add_argument("--gamma",               type=float, default=0.99,
                        help="Discount factor (RPPO only).")
    parser.add_argument("--gae_lambda",          type=float, default=0.95,
                        help="GAE λ parameter (RPPO only).")
    parser.add_argument("--max_grad_norm",       type=float, default=0.5,
                        help="Global gradient norm clip (RPPO only).")
    args = parser.parse_args()

    # Parse temperature schedule (comma-separated string → list of floats)
    _temp_schedule = (
        [float(t) for t in args.temp_schedule.split(",")]
        if args.temp_schedule else None
    )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Log JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # --- Device selection ------------------------------------------------
    _all_devices = jax.devices()
    _n_devices   = len(_all_devices)
    device_a     = _all_devices[0]
    device_b     = _all_devices[1] if _n_devices > 1 else _all_devices[0]
    _parallel    = _n_devices > 1 and args.solver == "rppo" and not args.sequential
    print(f"Device A: {device_a}  Device B: {device_b}  "
          f"Parallel A‖B: {_parallel}"
          + ("  (--sequential override)" if args.sequential else ""))

    # --- Logger ----------------------------------------------------------
    logger = TensorboardLoggerJax(args.log_dir)
    print(f"Logging to:  {args.log_dir}")

    # --- Initialisation ------------------------------------------------------
    print(f"Solver: {args.solver}")
    print("Initialising solvers …")

    if args.solver == "odcfr":
        solver_a = XDOHanabiSolverJax(
            "agent_A",
            K_simulations=args.k_simulations,
            partner_temperature=args.partner_temperature,
            adv_train_steps=args.adv_train_steps,
            pol_train_steps=args.pol_train_steps,
            aux_loss_weight=args.aux_loss_weight,
            gru_lr=args.gru_lr,
            gru_grad_clip=args.gru_grad_clip,
            explore_eps=args.explore_eps,
            rescore_topk=args.rescore_topk,
            rescore_full_every=args.rescore_full_every,
            coupling_alpha=args.coupling_alpha,
            seed=args.seed,
        )
        solver_b = XDOHanabiSolverJax(
            "agent_B",
            K_simulations=args.k_simulations,
            partner_temperature=args.partner_temperature,
            adv_train_steps=args.adv_train_steps,
            pol_train_steps=args.pol_train_steps,
            aux_loss_weight=args.aux_loss_weight,
            gru_lr=args.gru_lr,
            gru_grad_clip=args.gru_grad_clip,
            explore_eps=args.explore_eps,
            rescore_topk=args.rescore_topk,
            rescore_full_every=args.rescore_full_every,
            coupling_alpha=args.coupling_alpha,
            seed=args.seed + 1,
        )
    else:  # rppo
        solver_a = XDORPPOHanabiSolverJax(
            "agent_A",
            partner_temperature=args.partner_temperature,
            lr=args.rppo_lr,
            clip_eps=args.clip_eps,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            aux_loss_weight=args.aux_loss_weight,
            ppo_updates=args.ppo_updates,
            n_episodes=args.n_episodes_rppo,
            rescore_topk=args.rescore_topk,
            rescore_full_every=args.rescore_full_every,
            coupling_alpha=args.coupling_alpha,
            seed=args.seed,
            device=device_a,
            temp_schedule=_temp_schedule,
            br_save_dir=args.br_save_dir,
        )
        solver_b = XDORPPOHanabiSolverJax(
            "agent_B",
            partner_temperature=args.partner_temperature,
            lr=args.rppo_lr,
            clip_eps=args.clip_eps,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            aux_loss_weight=args.aux_loss_weight,
            ppo_updates=args.ppo_updates,
            n_episodes=args.n_episodes_rppo,
            rescore_topk=args.rescore_topk,
            rescore_full_every=args.rescore_full_every,
            coupling_alpha=args.coupling_alpha,
            seed=args.seed + 1,
            device=device_b,
            temp_schedule=_temp_schedule,
            br_save_dir=args.br_save_dir,
        )

    init_batch = _run_random_episodes(args.episodes_per_iter, seed=args.seed)
    solver_a.extract_behaviour_profile(init_batch, bc_epochs=100, bc_rounds=args.bc_rounds)
    solver_b.extract_behaviour_profile(init_batch, bc_epochs=100, bc_rounds=args.bc_rounds)

    if args.solver == "odcfr":
        init_early_stop_min = min(args.early_stop_min_iters, int(args.cfr_min * 0.80))
        policy_a = solver_a.request_new_policy(
            cfr_iterations=args.cfr_min,
            early_stop_min_iters=init_early_stop_min,
            early_stop_delta=args.early_stop_delta,
        )
        policy_b = solver_b.request_new_policy(
            cfr_iterations=args.cfr_min,
            early_stop_min_iters=init_early_stop_min,
            early_stop_delta=args.early_stop_delta,
        )
    else:  # rppo — A and B train simultaneously on separate GPUs if available
        def _rppo_init_a():
            with jax.default_device(device_a):
                return solver_a.request_new_policy(
                    ppo_updates=args.ppo_updates,
                    n_episodes=args.n_episodes_rppo,
                )

        def _rppo_init_b():
            with jax.default_device(device_b):
                return solver_b.request_new_policy(
                    ppo_updates=args.ppo_updates,
                    n_episodes=args.n_episodes_rppo,
                )

        if _parallel:
            print("  Init: training A‖B in parallel …")
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(_rppo_init_a)
                fut_b = pool.submit(_rppo_init_b)
                policy_a = fut_a.result()
                policy_b = fut_b.result()
        else:
            policy_a = _rppo_init_a()
            policy_b = _rppo_init_b()
    print("Initialisation complete.")

    key = jax.random.PRNGKey(args.seed + 1000)  # separate stream from solver seeds
    run_t0 = time.perf_counter()  # wall-clock start for total-elapsed printing

    # Config snapshot stored alongside every checkpoint for reproducibility.
    _ckpt_config = vars(args)

    # --- XDO outer loop ------------------------------------------------------
    for iteration in range(1, args.xdo_iterations + 1):
        iter_t0 = time.perf_counter()

        t0 = time.perf_counter()
        key, k_joint = jax.random.split(key)
        if args.solver == "odcfr":
            shared_batch = run_joint_episodes(
                policy_a, policy_b, args.episodes_per_iter, key=k_joint
            )
        else:
            shared_batch = run_joint_rppo_episodes(
                policy_a, policy_b, args.episodes_per_iter, key=k_joint
            )
        episodes_secs = time.perf_counter() - t0
        mean_score = float(np.mean([ep["score"] for ep in shared_batch]))

        if args.solver == "odcfr":
            progress = (iteration - 1) / max(args.xdo_iterations - 1, 1)
            k_raw = args.k_simulations + int(
                (args.k_max - args.k_simulations) * progress ** args.k_exponent
            )
            k = max((k_raw // 32) * 32, args.k_simulations)
            cfr_iterations = args.cfr_min + int(
                (args.cfr_max - args.cfr_min) * progress ** args.cfr_exponent
            )
            effective_early_stop_min = min(
                args.early_stop_min_iters, int(cfr_iterations * 0.80)
            )

        if args.solver == "odcfr":
            solver_a.solve_meta_game(partner_scores=solver_b._profile_scores)
            t0 = time.perf_counter()
            policy_a = solver_a.request_new_policy(
                cfr_iterations=cfr_iterations,
                k_simulations=k,
                early_stop_min_iters=effective_early_stop_min,
                early_stop_delta=args.early_stop_delta,
            )
            policy_a_secs = time.perf_counter() - t0

            t0 = time.perf_counter()
            solver_a.extract_behaviour_profile(shared_batch, bc_epochs=40, bc_rounds=args.bc_rounds)
            profile_a_secs = time.perf_counter() - t0

            solver_b.solve_meta_game(partner_scores=solver_a._profile_scores)
            t0 = time.perf_counter()
            policy_b = solver_b.request_new_policy(
                cfr_iterations=cfr_iterations,
                k_simulations=k,
                early_stop_min_iters=effective_early_stop_min,
                early_stop_delta=args.early_stop_delta,
            )
            policy_b_secs = time.perf_counter() - t0

            t0 = time.perf_counter()
            solver_b.extract_behaviour_profile(shared_batch, bc_epochs=40, bc_rounds=args.bc_rounds)
            profile_b_secs = time.perf_counter() - t0

        else:  # rppo — snapshot scores first so each agent reads a stable view
            scores_a_snap = list(solver_a._profile_scores)
            scores_b_snap = list(solver_b._profile_scores)

            if _parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_a = pool.submit(
                        _train_rppo_agent,
                        solver_a, shared_batch, scores_b_snap,
                        args.ppo_updates, args.n_episodes_rppo, args.bc_rounds,
                        device_a,
                    )
                    fut_b = pool.submit(
                        _train_rppo_agent,
                        solver_b, shared_batch, scores_a_snap,
                        args.ppo_updates, args.n_episodes_rppo, args.bc_rounds,
                        device_b,
                    )
                    policy_a, policy_a_secs, profile_a_secs = fut_a.result()
                    policy_b, policy_b_secs, profile_b_secs = fut_b.result()
            else:
                policy_a, policy_a_secs, profile_a_secs = _train_rppo_agent(
                    solver_a, shared_batch, scores_b_snap,
                    args.ppo_updates, args.n_episodes_rppo, args.bc_rounds,
                    device_a,
                )
                policy_b, policy_b_secs, profile_b_secs = _train_rppo_agent(
                    solver_b, shared_batch, scores_a_snap,
                    args.ppo_updates, args.n_episodes_rppo, args.bc_rounds,
                    device_b,
                )

        iter_secs   = time.perf_counter() - iter_t0
        elapsed_hrs = (time.perf_counter() - run_t0) / 3600
        sr_a  = _swap_regret(solver_a.meta_strategy, solver_a._profile_scores)
        tcs_a = _top_convention_score(solver_a.meta_strategy, solver_a._profile_scores)
        ent_a = _metastrategy_entropy(solver_a.meta_strategy)
        sr_b  = _swap_regret(solver_b.meta_strategy, solver_b._profile_scores)
        tcs_b = _top_convention_score(solver_b.meta_strategy, solver_b._profile_scores)
        ent_b = _metastrategy_entropy(solver_b.meta_strategy)
        l1 = (
            float(np.sum(np.abs(solver_a.meta_strategy - solver_b.meta_strategy)))
            if len(solver_a.meta_strategy) == len(solver_b.meta_strategy)
            else float("nan")
        )

        if args.solver == "odcfr":
            oa, ob = solver_a.oracle, solver_b.oracle
            print(
                f"[iter {iteration:3d}/{args.xdo_iterations}] "
                f"score={mean_score:.3f}  "
                f"SR_A={sr_a:.4f}  "
                f"top={tcs_a:.3f}  "
                f"H={ent_a:.3f}  "
                f"|π|={len(solver_a.profile_pool)}  "
                f"K={k}  cfr={cfr_iterations}  "
                f"L1={l1:.4f}  iter={iter_secs:.1f}s  total={elapsed_hrs:.2f}h"
            )
            print(
                f"  A: eps={episodes_secs:.2f}s  "
                f"iters={oa.last_cfr_iterations_run}/{cfr_iterations}  "
                f"Δ={oa.last_probe_delta:.4f}  "
                f"regret={oa.last_mean_regret:.3f}  "
                f"sim={oa.last_train_sim_secs:.2f}s  "
                f"adv={oa.last_train_adv_secs:.2f}s  "
                f"gru={oa.last_train_gru_secs:.2f}s  "
                f"pol={oa.last_train_pol_secs:.2f}s  "
                f"prof={profile_a_secs:.2f}s"
            )
            print(
                f"  B: "
                f"iters={ob.last_cfr_iterations_run}/{cfr_iterations}  "
                f"Δ={ob.last_probe_delta:.4f}  "
                f"regret={ob.last_mean_regret:.3f}  "
                f"sim={ob.last_train_sim_secs:.2f}s  "
                f"adv={ob.last_train_adv_secs:.2f}s  "
                f"gru={ob.last_train_gru_secs:.2f}s  "
                f"pol={ob.last_train_pol_secs:.2f}s  "
                f"prof={profile_b_secs:.2f}s"
            )
        else:  # rppo
            print(
                f"[iter {iteration:3d}/{args.xdo_iterations}] "
                f"score={mean_score:.3f}  "
                f"SR_A={sr_a:.4f}  "
                f"top={tcs_a:.3f}  "
                f"H={ent_a:.3f}  "
                f"|π|={len(solver_a.profile_pool)}  "
                f"ppo_updates={args.ppo_updates}  "
                f"L1={l1:.4f}  iter={iter_secs:.1f}s  total={elapsed_hrs:.2f}h"
            )
            print(
                f"  A: eps={episodes_secs:.2f}s  "
                f"actor_loss={solver_a.last_actor_loss:.4f}  "
                f"value_loss={solver_a.last_value_loss:.4f}  "
                f"entropy={solver_a.last_entropy:.4f}  "
                f"score={solver_a.last_mean_score:.3f}  "
                f"train={solver_a.last_train_secs:.1f}s  "
                f"prof={profile_a_secs:.2f}s"
            )
            print(
                f"  B: "
                f"actor_loss={solver_b.last_actor_loss:.4f}  "
                f"value_loss={solver_b.last_value_loss:.4f}  "
                f"entropy={solver_b.last_entropy:.4f}  "
                f"score={solver_b.last_mean_score:.3f}  "
                f"train={solver_b.last_train_secs:.1f}s  "
                f"prof={profile_b_secs:.2f}s"
            )

        # --- TensorboardX logging ----------------------------------------
        # Outer-loop quality metrics (same for both solvers)
        logger.log_iteration(
            iteration=iteration,
            agent_id="agent_A",
            mean_joint_score=mean_score,
            swap_regret=sr_a,
            metastrategy_entropy=ent_a,
            top_convention_score=tcs_a,
            probe_score=solver_a.last_probe_score,
            pool_size=len(solver_a.profile_pool),
        )
        logger.log_iteration(
            iteration=iteration,
            agent_id="agent_B",
            mean_joint_score=mean_score,
            swap_regret=sr_b,
            metastrategy_entropy=ent_b,
            top_convention_score=tcs_b,
            probe_score=solver_b.last_probe_score,
            pool_size=len(solver_b.profile_pool),
        )

        if args.solver == "odcfr":
            oa, ob = solver_a.oracle, solver_b.oracle
            logger.log_profile_metrics(
                iteration=iteration,
                agent_id="agent_A",
                bc_loss=solver_a.last_bc_loss,
                bc_accuracy=solver_a.last_bc_accuracy,
                cfr_iters_run=oa.last_cfr_iterations_run,
                probe_delta=oa.last_probe_delta,
                adv_loss=oa.last_adv_loss,
                gru_loss=oa.last_gru_loss,
                mean_regret=oa.last_mean_regret,
            )
            logger.log_profile_metrics(
                iteration=iteration,
                agent_id="agent_B",
                bc_loss=solver_b.last_bc_loss,
                bc_accuracy=solver_b.last_bc_accuracy,
                cfr_iters_run=ob.last_cfr_iterations_run,
                probe_delta=ob.last_probe_delta,
                adv_loss=ob.last_adv_loss,
                gru_loss=ob.last_gru_loss,
                mean_regret=ob.last_mean_regret,
            )
            logger.log_timings(
                iteration=iteration,
                agent_id="agent_A",
                episodes_secs=episodes_secs,
                policy_train_secs=policy_a_secs,
                profile_build_secs=profile_a_secs,
                sim_secs=oa.last_train_sim_secs,
                adv_secs=oa.last_train_adv_secs,
                gru_secs=oa.last_train_gru_secs,
                pol_secs=oa.last_train_pol_secs,
            )
            logger.log_timings(
                iteration=iteration,
                agent_id="agent_B",
                episodes_secs=episodes_secs,
                policy_train_secs=policy_b_secs,
                profile_build_secs=profile_b_secs,
                sim_secs=ob.last_train_sim_secs,
                adv_secs=ob.last_train_adv_secs,
                gru_secs=ob.last_train_gru_secs,
                pol_secs=ob.last_train_pol_secs,
            )
        else:  # rppo — log available metrics; skip ODCFR-only ones
            # BC loss/accuracy are solver-agnostic
            logger.log_profile_metrics(
                iteration=iteration,
                agent_id="agent_A",
                bc_loss=solver_a.last_bc_loss,
                bc_accuracy=solver_a.last_bc_accuracy,
                cfr_iters_run=0,
                probe_delta=float("nan"),
                adv_loss=solver_a.last_actor_loss,
                gru_loss=solver_a.last_value_loss,
                mean_regret=float("nan"),
            )
            logger.log_profile_metrics(
                iteration=iteration,
                agent_id="agent_B",
                bc_loss=solver_b.last_bc_loss,
                bc_accuracy=solver_b.last_bc_accuracy,
                cfr_iters_run=0,
                probe_delta=float("nan"),
                adv_loss=solver_b.last_actor_loss,
                gru_loss=solver_b.last_value_loss,
                mean_regret=float("nan"),
            )
            logger.log_timings(
                iteration=iteration,
                agent_id="agent_A",
                episodes_secs=episodes_secs,
                policy_train_secs=policy_a_secs,
                profile_build_secs=profile_a_secs,
                sim_secs=solver_a.last_train_secs,
                adv_secs=float("nan"),
                gru_secs=float("nan"),
                pol_secs=float("nan"),
            )
            logger.log_timings(
                iteration=iteration,
                agent_id="agent_B",
                episodes_secs=episodes_secs,
                policy_train_secs=policy_b_secs,
                profile_build_secs=profile_b_secs,
                sim_secs=solver_b.last_train_secs,
                adv_secs=float("nan"),
                gru_secs=float("nan"),
                pol_secs=float("nan"),
            )

        # Log cross-agent divergence (l1 already computed above for printing)
        logger.log_metastrategy_divergence(iteration, l1)

        # --- Periodic checkpoint ---------------------------------------------
        if (args.save_dir and args.solver == "rppo"
                and iteration % args.checkpoint_every == 0):
            save_run(solver_a, solver_b, args.save_dir, _ckpt_config, iteration)

    # --- Final evaluation ----------------------------------------------------
    print("\nEvaluation …")
    if args.solver == "odcfr":
        eval_batch = run_joint_episodes(policy_a, policy_b, n=200,
                                        key=jax.random.PRNGKey(9999))
    else:
        eval_batch = run_joint_rppo_episodes(policy_a, policy_b, n=200,
                                             key=jax.random.PRNGKey(9999))
    joint_score = float(np.mean([ep["score"] for ep in eval_batch]))
    print(f"  Joint score (200 eps): {joint_score:.3f}")

    if len(solver_a.meta_strategy) == len(solver_b.meta_strategy):
        l1 = float(np.sum(np.abs(solver_a.meta_strategy - solver_b.meta_strategy)))
        print(f"  MetaStrategy L1 divergence: {l1:.4f}")

    logger.log_eval(joint_score)
    logger.close()
    print(f"\nLogs written to: {args.log_dir}")

    # --- Save final agents ---------------------------------------------------
    if args.save_dir and args.solver == "rppo":
        save_run(solver_a, solver_b, args.save_dir, _ckpt_config, iteration=None)
        print(f"Checkpoints written to: {args.save_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
