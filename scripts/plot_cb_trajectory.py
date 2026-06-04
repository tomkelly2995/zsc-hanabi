"""
Two figures:
  1. figures/cb_score_trajectory.pdf — 50-game rolling window trajectories
  2. figures/cb_method_comparison.pdf — bar chart comparing all test-time methods
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

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

# ===========================================================================
# DATA
# ===========================================================================

# ── Trajectory data (50-game windows) ──────────────────────────────────────
# β=0.3 run (500 games → 10 windows)
cb_beta03_traj = [2.680, 3.360, 2.860, 3.220, 3.080, 2.840, 2.940, 3.200, 2.740, 3.260]

# β=0.2 run (650 games → 13 windows — paste from your run output)
cb_beta02_traj = [2.980, 3.380, 2.920, 3.260, 3.200, 2.760, 2.980, 3.100, 2.940, 3.420]

# ── Overall mean scores per method (from run output summary tables) ─────────
SCORES = {
    # α=0.5 agents
    "Generic":          (2.646, 1.776),
    "Fixed BR (LL)":    (2.982, 1.671),
    "CB b=0.2 (best)":  (float(np.mean(cb_beta02_traj)), None),  # computed from windows
    "CB β=0.3":         (3.018, None),
    "Last-layer FT":    (2.626, None),
    "Adapter FT":       (2.514, None),
    "Self-play ceiling":(3.430, 1.353),
}

# ── Replace None stds if you have them from run output ─────────────────────
SCORES["CB b=0.2 (best)"] = (float(np.mean(cb_beta02_traj)), None)
SCORES["CB β=0.3"]   = (3.018, None)

# ===========================================================================
# FIGURE 1 — Score Trajectory
# ===========================================================================

WINDOW_SIZE = 50

def window_mids(n_windows):
    return [WINDOW_SIZE * i + WINDOW_SIZE // 2 for i in range(n_windows)]

COLORS = {
    "generic":   "#888888",
    "fixed_br":  "#2166ac",
    "cb_02":     "#d6604d",
    "cb_03":     "#f4a582",
    "selfplay":  "#4dac26",
}

fig1, ax1 = plt.subplots(figsize=(8, 4.5))

# Baselines
ax1.axhline(SCORES["Generic"][0],    color=COLORS["generic"],  linestyle="--",
            linewidth=1.4, label=f"Generic baseline: {SCORES['Generic'][0]:.3f}", zorder=2)
ax1.axhline(SCORES["Fixed BR (LL)"][0], color=COLORS["fixed_br"], linestyle="-.",
            linewidth=1.4, label=f"Fixed BR (β=0): {SCORES['Fixed BR (LL)'][0]:.3f}", zorder=2)
ax1.axhline(SCORES["Self-play ceiling"][0], color=COLORS["selfplay"], linestyle=":",
            linewidth=1.4, label=f"Self-play ceiling: {SCORES['Self-play ceiling'][0]:.3f}", zorder=2)

# β=0.3 trajectory
mids03 = window_mids(len(cb_beta03_traj))
n_eval03 = len(cb_beta03_traj) * WINDOW_SIZE
ax1.plot(mids03, cb_beta03_traj,
         color=COLORS["cb_03"], marker="o", markersize=5,
         linewidth=1.6, linestyle="-", alpha=0.8,
         label=f"CB β=0.3 (mean={SCORES['CB β=0.3'][0]:.3f})", zorder=3)

# β=0.2 trajectory
mids02 = window_mids(len(cb_beta02_traj))
n_eval02 = len(cb_beta02_traj) * WINDOW_SIZE
ax1.plot(mids02, cb_beta02_traj,
         color=COLORS["cb_02"], marker="s", markersize=5,
         linewidth=2.0, linestyle="-",
         label=f"CB b=0.2 (best) (mean={SCORES['CB b=0.2 (best)'][0]:.3f})", zorder=4)

n_eval_max = max(n_eval02, n_eval03)
ax1.set_xlabel("Evaluation game")
ax1.set_ylabel("Mean score (50-game window)")
ax1.set_title("Convention-Conditioned Belief Blending: Score Trajectory\n"
              "(A uses LL-selected BR + BC blend; B fixed generic)")
ax1.set_xlim(0, n_eval_max + WINDOW_SIZE // 2)
ax1.set_ylim(1.8, 4.0)
ax1.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax1.tick_params(axis="y", which="minor", left=True)
ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")
fig1.tight_layout()

p = os.path.join(OUT_DIR, "cb_score_trajectory")
fig1.savefig(p + ".pdf", bbox_inches="tight")
fig1.savefig(p + ".png", bbox_inches="tight", dpi=200)
print(f"Saved: {p}.pdf / .png")

# ===========================================================================
# FIGURE 2 — Method Comparison Bar Chart
# ===========================================================================

# Ordered from worst to best for readability
METHOD_ORDER = [
    "Adapter FT",
    "Last-layer FT",
    "Generic",
    "Fixed BR (LL)",
    "CB β=0.3",
    "CB b=0.2 (best)",
    "Self-play ceiling",
]

# Colour by category
BAR_COLORS = {
    "Adapter FT":        "#b2182b",
    "Last-layer FT":     "#ef8a62",
    "Generic":           "#888888",
    "Fixed BR (LL)":     "#2166ac",
    "CB β=0.3":          "#f4a582",
    "CB b=0.2 (best)":        "#d6604d",
    "Self-play ceiling": "#4dac26",
}

means = [SCORES[m][0] for m in METHOD_ORDER]
stds  = [SCORES[m][1] if SCORES[m][1] is not None else 0.0 for m in METHOD_ORDER]
colors = [BAR_COLORS[m] for m in METHOD_ORDER]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(METHOD_ORDER))
bars = ax2.bar(x, means, color=colors, edgecolor="white", linewidth=0.8,
               width=0.6, zorder=3)

# Error bars only where we have std
for i, (mean, std) in enumerate(zip(means, stds)):
    if std > 0:
        ax2.errorbar(x[i], mean, yerr=std, fmt="none",
                     color="black", capsize=4, linewidth=1.2, zorder=4)

# Score labels on top of each bar
for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             mean + 0.04,
             f"{mean:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="normal")

# Generic baseline reference line
ax2.axhline(SCORES["Generic"][0], color="#888888", linestyle="--",
            linewidth=1.2, alpha=0.6, zorder=2, label="Generic baseline")

ax2.set_xticks(x)
ax2.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
ax2.set_ylabel("Mean score (500 games)")
ax2.set_title("Test-Time Adaptation: Method Comparison\n"
              "(α=0.5 coupled agents, A uses LL inference; B fixed generic)")
ax2.set_ylim(1.8, 4.0)
ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax2.tick_params(axis="y", which="minor", left=True)
ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)

# Legend patches for category grouping
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color="#888888",  label="Baseline"),
    mpatches.Patch(color="#2166ac",  label="LL inference (fixed)"),
    mpatches.Patch(color="#d6604d",  label="CB blending (test-time)"),
    mpatches.Patch(color="#ef8a62",  label="Gradient fine-tuning"),
    mpatches.Patch(color="#4dac26",  label="Self-play ceiling"),
]
ax2.legend(handles=legend_elements, loc="upper left",
           framealpha=0.92, edgecolor="#cccccc")

fig2.tight_layout()

p = os.path.join(OUT_DIR, "cb_method_comparison")
fig2.savefig(p + ".pdf", bbox_inches="tight")
fig2.savefig(p + ".png", bbox_inches="tight", dpi=200)
print(f"Saved: {p}.pdf / .png")

plt.show()
