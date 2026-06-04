# Zero-Shot Coordination in Cooperative Multi-Agent RL

**MSc Artificial Intelligence — Munster Technological University, 2026**
Thomas Kelly

---

## Overview

This repository contains the implementation for my MSc thesis: *Zero-Shot Coordination in Cooperative Multi-Agent Reinforcement Learning via Meta-Strategy Alignment and Test-Time Convention Inference*.

The central problem is **zero-shot coordination (ZSC)**: how do you get two independently trained agents to cooperate effectively when they have never interacted during training? Standard self-play produces agents that develop private conventions — strategies that work within a training run but collapse when paired with an agent from a different run.

This work proposes and evaluates two contributions:

1. **Meta-strategy coupling** — a training-time mechanism that blends each agent's probe scores with its partner's during XDO, forcing both agents' multiplicative-weights updates to stay aligned and preventing convention divergence.
2. **Convention-conditioned belief (CB) blending** — a test-time adaptation method that maintains a belief distribution over the partner's convention and blends the oracle policy with the most likely BC policy, achieving strong cross-play performance without any retraining.

Experiments are conducted on **Tiny Hanabi**, a tractable cooperative benchmark that preserves the full zero-shot coordination challenge.

For full methodology, results, and analysis see the [thesis](./R00152085_Thesis.pdf).

---

## Repository Structure

```
├── src/                    # Core implementation
│   ├── jax_agents/         # RPPO oracle and BC agents
│   ├── jax_env/            # Tiny Hanabi JAX environment
│   ├── jax_networks/       # GRU-based actor-critic networks
│   ├── xdo/                # XDO solver and meta-strategy logic
│   ├── population/         # BC profile and population management
│   └── utils/              # Shared utilities
├── scripts/
│   ├── evaluate_crossplay.py       # Self-play and cross-play evaluation
│   ├── evaluate_ll_inference.py    # Test-time adaptation evaluation
│   ├── plot_training_diagnostics.py
│   └── plot_cb_trajectory.py
├── configs/
│   └── tiny_hanabi.json    # Environment and training configuration
├── runs/                   # Trained model checkpoints
│   ├── seed42_alpha05/     # Coupled run, seed 42
│   ├── seed123_alpha05/    # Coupled run, seed 123
│   ├── seed42_alpha0/      # Uncoupled run, seed 42
│   └── seed123_alpha0/     # Uncoupled run, seed 123
├── figures/                # All result figures
├── tests/                  # Test suite
└── main_jax.py             # Training entry point
```

---

## Installation

**Prerequisites:** Python 3.10+, JAX

```bash
pip install -r requirements.txt
```

This project also requires the [Hanabi Learning Environment](https://github.com/google-deepmind/hanabi-learning-environment):

```bash
git clone https://github.com/google-deepmind/hanabi-learning-environment.git
cd hanabi-learning-environment
pip install .
```

---

## Usage

**Evaluate cross-play (coupled vs uncoupled):**

```bash
python scripts/evaluate_crossplay.py --run1 runs/seed42_alpha05 --run2 runs/seed123_alpha05
python scripts/evaluate_crossplay.py --run1 runs/seed42_alpha0 --run2 runs/seed123_alpha0
```

**Evaluate test-time adaptation:**

```bash
python scripts/evaluate_ll_inference.py --run1 runs/seed42_alpha05 --run2 runs/seed123_alpha05
```

**Run training:**

```bash
python main_jax.py [--xdo_iterations N] [--episodes_per_iter E] [--cfr_min T] [--cfr_max T] [--k_simulations K] [--seed S]
```

---

## Results

Full results and analysis are in the [thesis](./thesis.pdf). Key findings:

- Meta-strategy coupling (α=0.5) reduces L1 divergence between agents to near-zero and raises the XP/SP ratio from ~0.52 to **0.759**
- CB blending at β=0.2 achieves a cross-play score of **3.094** — 90.2% of the self-play ceiling — with no retraining required

---

## Acknowledgements

Built on top of [JaxMARL](https://github.com/FLAIROx/JaxMARL) and the [Hanabi Learning Environment](https://github.com/google-deepmind/hanabi-learning-environment).
