"""
Generate thesis training diagnostic figures from TensorBoard CSV exports.

Figures produced (all saved to figures/):
  1. joint_score.pdf/.png           — stochastic mean joint score per XDO iteration
  2. rollout_score.pdf/.png         — greedy eval rollout score per XDO iteration
  3. l1_divergence_coupled.pdf/.png — L1 divergence under coupling (α=0.5)
  4. l1_comparison.pdf/.png         — L1 divergence: α=0 vs α=0.5
  5. entropy.pdf/.png               — meta-strategy entropy per iteration
  6. bc_accuracy.pdf/.png           — BC profile accuracy per iteration
  7. swap_regret.pdf/.png           — swap regret per iteration

CSV files expected in ~/Downloads/:
  10_05_seed42_alpha05_xdo_rppo 2.csv         → mean joint score, seed 42
  10_05_seed123_alpha05_xdo_rppo 2.csv        → mean joint score, seed 123
  10_05_seed42_alpha05_xdo_rppo.csv           → greedy rollout score, seed 42
  10_05_seed123_alpha05_xdo_rppo.csv          → greedy rollout score, seed 123
  10_05_seed42_alpha05_xdo_rppo (1).csv       → L1 divergence α=0.5, seed 42
  10_05_seed123_alpha05_xdo_rppo (1).csv      → L1 divergence α=0.5, seed 123
  10_05_seed42_alpha0_xdo_rppo.csv            → L1 divergence α=0, seed 42
  10_05_seed123_alpha0_xdo_rppo.csv           → L1 divergence α=0, seed 123
  10_05_seed42_alpha05_xdo_rppo (3).csv       → entropy, seed 42
  10_05_seed123_alpha05_xdo_rppo (2).csv      → entropy, seed 123
  10_05_seed42_alpha05_xdo_rppo (4).csv       → BC accuracy, seed 42
  10_05_seed123_alpha05_xdo_rppo (3).csv      → BC accuracy, seed 123
  10_05_seed42_alpha05_xdo_rppo (5).csv       → swap regret, seed 42
  10_05_seed123_alpha05_xdo_rppo (4).csv      → swap regret, seed 123

Usage:
  python scripts/plot_training_diagnostics.py
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "text.usetex":      False,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DL = os.path.expanduser("~/Downloads")

# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------

def load_csv(path):
    """Return (steps, values) arrays from a TensorBoard CSV export."""
    steps, vals = [], []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            steps.append(int(parts[1]))
            vals.append(float(parts[2]))
    return np.array(steps), np.array(vals)

def try_load(path):
    """Load CSV, return (None, None) if file is missing."""
    if not os.path.exists(path):
        print(f"  [MISSING] {os.path.basename(path)}")
        return None, None
    return load_csv(path)

# Mean joint score (stochastic training episodes)
s42_score_x,   s42_score   = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo 2.csv")
s123_score_x,  s123_score  = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo 2.csv")

# Greedy rollout score (evaluation)
s42_roll_x,    s42_roll    = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo.csv")
s123_roll_x,   s123_roll   = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo.csv")

# L1 divergence — α=0.5 coupled
s42_l1_x,     s42_l1      = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo (1).csv")
s123_l1_x,    s123_l1     = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo (1).csv")

# L1 divergence — α=0 uncoupled
s42_l1_a0_x,  s42_l1_a0   = try_load(f"{DL}/10_05_seed42_alpha0_xdo_rppo.csv")
s123_l1_a0_x, s123_l1_a0  = try_load(f"{DL}/10_05_seed123_alpha0_xdo_rppo.csv")

# Entropy
s42_ent_x,    s42_ent     = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo (3).csv")
s123_ent_x,   s123_ent    = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo (2).csv")

# BC accuracy
s42_bc_x,     s42_bc      = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo (4).csv")
s123_bc_x,    s123_bc     = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo (3).csv")

# Swap regret
s42_sr_x,     s42_sr      = try_load(f"{DL}/10_05_seed42_alpha05_xdo_rppo (5).csv")
s123_sr_x,    s123_sr     = try_load(f"{DL}/10_05_seed123_alpha05_xdo_rppo (4).csv")

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
C42     = "#2166ac"   # seed 42  — blue
C123    = "#d6604d"   # seed 123 — red
C42_a0  = "#92c5de"   # seed 42,  α=0 — light blue
C123_a0 = "#f4a582"   # seed 123, α=0 — light red/orange

def add_smoothed(ax, x, y, color, label, alpha_raw=0.25, window=5, linestyle="-"):
    """Plot raw as faint line + smoothed line."""
    ax.plot(x, y, color=color, alpha=alpha_raw, linewidth=1.0, linestyle=linestyle)
    if len(y) >= window:
        kernel = np.ones(window) / window
        y_sm   = np.convolve(y, kernel, mode="valid")
        x_sm   = x[window // 2: window // 2 + len(y_sm)]
        ax.plot(x_sm, y_sm, color=color, linewidth=2.0, label=label, linestyle=linestyle)
    else:
        ax.plot(x, y, color=color, linewidth=2.0, label=label, linestyle=linestyle)

def finish(fig, ax, path_stem, xlabel="XDO iteration"):
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.legend(loc="best", framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"{path_stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
    print(f"Saved: {path_stem}.pdf / .png")

# ---------------------------------------------------------------------------
# Figure 1 — Mean joint score (stochastic training episodes)
# ---------------------------------------------------------------------------
if s42_score is not None and s123_score is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_score_x,  s42_score,  C42,  "Seed 42")
    add_smoothed(ax, s123_score_x, s123_score, C123, "Seed 123")

    ax.set_ylabel("Mean joint episode score")
    ax.set_title("Within-Run Learning Curve\n(mean joint score per XDO iteration, 5-step rolling mean)")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(1.5, 4.0)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "joint_score")
else:
    print("Skipping joint_score (missing CSVs)")

# ---------------------------------------------------------------------------
# Figure 2 — Greedy rollout score (evaluation)
# ---------------------------------------------------------------------------
if s42_roll is not None and s123_roll is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_roll_x,  s42_roll,  C42,  "Seed 42")
    add_smoothed(ax, s123_roll_x, s123_roll, C123, "Seed 123")

    ax.set_ylabel("Mean joint episode score")
    ax.set_title("Greedy Evaluation Rollout Score\n(per XDO iteration, 5-step rolling mean)")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(1.5, 4.0)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "rollout_score")
else:
    print("Skipping rollout_score (missing CSVs)")

# ---------------------------------------------------------------------------
# Figure 2 — L1 divergence (coupled runs, α=0.5)
# ---------------------------------------------------------------------------
if s42_l1 is not None and s123_l1 is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_l1_x,  s42_l1,  C42,  "Seed 42")
    add_smoothed(ax, s123_l1_x, s123_l1, C123, "Seed 123")

    ax.axhline(0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axhline(2, color="#b2182b", linestyle=":", linewidth=1.0, alpha=0.6,
               label="Max divergence (2.0)")

    ax.set_ylabel(r"L1 divergence $\ell_1^{(t)}$")
    ax.set_title(r"Meta-Strategy L1 Divergence ($\alpha=0.5$)")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(-0.05, 2.1)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "l1_divergence_coupled")
else:
    print("Skipping l1_divergence_coupled (missing CSVs)")

# ---------------------------------------------------------------------------
# Figure 3 — L1 divergence comparison: α=0 vs α=0.5
# ---------------------------------------------------------------------------
has_a05 = s42_l1 is not None and s123_l1 is not None
has_a0  = s42_l1_a0 is not None and s123_l1_a0 is not None

if has_a05 or has_a0:
    fig, ax = plt.subplots(figsize=(7, 4))

    if has_a05:
        add_smoothed(ax, s42_l1_x,     s42_l1,    C42,     r"Seed 42, $\alpha=0.5$",  linestyle="-")
        add_smoothed(ax, s123_l1_x,    s123_l1,   C123,    r"Seed 123, $\alpha=0.5$", linestyle="-")

    if has_a0:
        add_smoothed(ax, s42_l1_a0_x,  s42_l1_a0,  C42_a0,  r"Seed 42, $\alpha=0$",   linestyle="--")
        add_smoothed(ax, s123_l1_a0_x, s123_l1_a0, C123_a0, r"Seed 123, $\alpha=0$",  linestyle="--")

    ax.axhline(1.7, color="#b2182b", linestyle=":", linewidth=1.0, alpha=0.65,
               label="Saturation (~1.70)")
    ax.axhline(0,   color="#888888", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_ylabel(r"L1 divergence $\ell_1^{(t)}$")
    ax.set_title(r"Meta-Strategy L1 Divergence: $\alpha=0$ vs $\alpha=0.5$")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(-0.05, 2.1)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "l1_comparison")
else:
    print("Skipping l1_comparison (missing all L1 CSVs)")

# ---------------------------------------------------------------------------
# Figure 4 — Meta-strategy entropy
# ---------------------------------------------------------------------------
if s42_ent is not None and s123_ent is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_ent_x,  s42_ent,  C42,  "Seed 42")
    add_smoothed(ax, s123_ent_x, s123_ent, C123, "Seed 123")

    for k, style in [(2, ":"), (4, "--"), (8, "-.")]:
        ax.axhline(np.log(k), color="#888888", linestyle=style, linewidth=0.9,
                   alpha=0.55, label=f"Uniform over {k} profiles (H={np.log(k):.2f})")

    ax.set_ylabel(r"Meta-strategy entropy $H(\mathbf{w})$")
    ax.set_title(r"Meta-Strategy Entropy ($\alpha=0.5$)")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(0, 2.2)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "entropy")
else:
    print("Skipping entropy (missing CSVs)")

# ---------------------------------------------------------------------------
# Figure 5 — BC accuracy
# ---------------------------------------------------------------------------
if s42_bc is not None and s123_bc is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_bc_x,  s42_bc,  C42,  "Seed 42")
    add_smoothed(ax, s123_bc_x, s123_bc, C123, "Seed 123")

    ax.set_ylabel("BC profile accuracy")
    ax.set_title("Behaviour Cloning Profile Accuracy per XDO Iteration")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(0.80, 1.00)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "bc_accuracy")
else:
    print("Skipping bc_accuracy (missing CSVs)")

# ---------------------------------------------------------------------------
# Figure 6 — Swap regret
# ---------------------------------------------------------------------------
if s42_sr is not None and s123_sr is not None:
    fig, ax = plt.subplots(figsize=(7, 4))

    add_smoothed(ax, s42_sr_x,  s42_sr,  C42,  "Seed 42")
    add_smoothed(ax, s123_sr_x, s123_sr, C123, "Seed 123")

    ax.set_ylabel("Swap regret")
    ax.set_title("Swap Regret per XDO Iteration")
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(0, 2.2)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", left=True)

    finish(fig, ax, "swap_regret")
else:
    print("Skipping swap_regret (missing CSVs)")

plt.show()
print("\nDone.")
