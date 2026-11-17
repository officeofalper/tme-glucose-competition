"""
Starved in the Tumor: a simplified model of glucose competition
between tumor cells and effector T cells in the tumor microenvironment (TME).

IMPORTANT: This is a simplified, illustrative model. All parameter values
are chosen to show qualitative behaviour. They are NOT measured values,
and the outputs are NOT experimental data or new scientific findings.

Model (all quantities in arbitrary units):
    G  = glucose concentration in the TME
    N  = number of tumor cells
    E  = effector activity of T cells (0 = none, 1 = maximal)
    T  = number of effector T cells (kept constant)

    Glucose:  dG/dt = supply - washout*G - tumor uptake - T-cell uptake
    Tumor:    dN/dt = glucose-dependent growth - killing by active T cells
    Effector activity depends on how much glucose the T cells can get.

The equations are solved with the simple Euler method:
    new value = old value + (rate of change) * dt
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Default parameters (arbitrary units, chosen for illustration)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Glucose
    "supply": 1.0,          # glucose entering the TME from blood per time unit
    "washout": 0.1,         # fraction of glucose leaving the TME per time unit
    # Tumor cells
    "tumor_vmax": 0.05,     # maximal glucose uptake per tumor cell
    "tumor_km": 1.0,        # glucose level at which tumor uptake is half-maximal
    "tumor_growth": 0.3,    # maximal tumor growth rate
    "tumor_capacity": 200.0,  # maximal tumor size the tissue can hold
    # T cells
    "n_tcells": 10.0,       # number of effector T cells (constant)
    "tcell_vmax": 0.02,     # maximal glucose uptake per T cell
    "tcell_km": 3.0,        # glucose level at which T-cell uptake is half-maximal
    "kill_rate": 0.04,      # tumor cells killed per active T cell per time unit
    # Interventions (0 = no effect, 1 = complete block)
    "tumor_inhibition": 0.0,  # block of glucose uptake in tumor cells only
    "tcell_inhibition": 0.0,  # block of glycolysis in T cells (off-target effect)
    # Simulation settings
    "initial_glucose": 10.0,
    "initial_tumor": 5.0,
    "dt": 0.05,
    "t_end": 100.0,
}


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------
def saturating(glucose, km):
    """Michaelis-Menten-like curve: 0 when glucose is 0, approaches 1 when
    glucose is much larger than km."""
    return glucose / (km + glucose)


def simulate(params):
    """Run one simulation and return the time course of G, N and E."""
    p = params
    n_steps = int(p["t_end"] / p["dt"])
    time = np.arange(n_steps) * p["dt"]

    # Empty arrays that will store the value at every time step
    glucose = np.zeros(n_steps)
    tumor = np.zeros(n_steps)
    effector = np.zeros(n_steps)

    glucose[0] = p["initial_glucose"]
    tumor[0] = p["initial_tumor"]

    # How strongly each cell type can still use glucose after an intervention
    tumor_capacity_left = 1.0 - p["tumor_inhibition"]
    tcell_capacity_left = 1.0 - p["tcell_inhibition"]

    for i in range(n_steps):
        G = glucose[i]
        N = tumor[i]

        # Fraction of maximal glucose use for each cell type
        tumor_fraction = tumor_capacity_left * saturating(G, p["tumor_km"])
        tcell_fraction = tcell_capacity_left * saturating(G, p["tcell_km"])

        # T-cell effector activity follows their glucose use (glycolysis)
        effector[i] = tcell_fraction

        if i == n_steps - 1:
            break  # last step: values stored, nothing left to update

        # Glucose uptake by each population
        tumor_uptake = p["tumor_vmax"] * tumor_fraction * N
        tcell_uptake = p["tcell_vmax"] * tcell_fraction * p["n_tcells"]

        # Rates of change
        dG = p["supply"] - p["washout"] * G - tumor_uptake - tcell_uptake
        growth = p["tumor_growth"] * tumor_fraction * N * (1 - N / p["tumor_capacity"])
        killing = p["kill_rate"] * effector[i] * p["n_tcells"] * N
        dN = growth - killing

        # Euler update (max(..., 0) keeps values from going negative)
        glucose[i + 1] = max(G + dG * p["dt"], 0.0)
        tumor[i + 1] = max(N + dN * p["dt"], 0.0)

    return {"time": time, "glucose": glucose, "tumor": tumor, "effector": effector}


def with_changes(**changes):
    """Copy the default parameters and change only the ones given."""
    params = dict(DEFAULT_PARAMS)
    params.update(changes)
    return params


# ---------------------------------------------------------------------------
# 3. Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "No tumor": with_changes(initial_tumor=0.0),
    "Small tumor": with_changes(initial_tumor=5.0),
    "Large tumor": with_changes(initial_tumor=60.0),
    "Large tumor + tumor-selective inhibitor": with_changes(
        initial_tumor=60.0, tumor_inhibition=0.5),
    "Large tumor + non-selective inhibitor": with_changes(
        initial_tumor=60.0, tumor_inhibition=0.5, tcell_inhibition=0.5),
}

COLORS = {
    "No tumor": "#4d4d4d",
    "Small tumor": "#2a9d8f",
    "Large tumor": "#d62828",
    "Large tumor + tumor-selective inhibitor": "#1d3557",
    "Large tumor + non-selective inhibitor": "#e9a23b",
}


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def plot_scenarios(results, filename):
    """Figure 1: time courses of glucose, tumor size and T-cell activity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("glucose", "TME glucose (a.u.)", "A. Glucose in the TME"),
        ("tumor", "Tumor cells (a.u.)", "B. Tumor size"),
        ("effector", "Effector activity (0-1)", "C. T-cell effector activity"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name, res in results.items():
            ax.plot(res["time"], res[key], label=name, color=COLORS[name], lw=2,
                    ls=":" if name == "No tumor" else "-")  # dotted: tells it apart without color
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, frameon=False)
    fig.suptitle("Illustrative model only - not experimental data", fontsize=9,
                 color="gray", x=0.99, ha="right")
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_threshold(filename):
    """Figure 2: final tumor size for different starting tumor sizes,
    with and without a tumor-selective inhibitor."""
    starting_sizes = np.linspace(0, 100, 51)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for inhibition, name in [(0.0, "Large tumor"),
                             (0.5, "Large tumor + tumor-selective inhibitor")]:
        final_sizes = []
        for n0 in starting_sizes:
            res = simulate(with_changes(initial_tumor=n0,
                                        tumor_inhibition=inhibition))
            final_sizes.append(res["tumor"][-1])
        label = "No treatment" if inhibition == 0 else "Tumor-selective inhibitor (50%)"
        ax.plot(starting_sizes, final_sizes, "o-", ms=4, lw=2,
                color=COLORS[name], label=label)
    ax.set_xlabel("Starting tumor size (a.u.)")
    ax.set_ylabel("Tumor size at end of simulation (a.u.)")
    ax.set_title("Escape threshold: when does the tumor outcompete T cells?",
                 loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_selectivity(filename):
    """Figure 3: how selective must an inhibitor be? Final tumor size as a
    function of how much the drug also hits T cells."""
    off_target = np.linspace(0, 1, 41)
    final_sizes = []
    final_activity = []
    for f in off_target:
        res = simulate(with_changes(initial_tumor=60.0, tumor_inhibition=0.5,
                                    tcell_inhibition=0.5 * f))
        final_sizes.append(res["tumor"][-1])
        final_activity.append(res["effector"][-1])

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(off_target * 100, final_sizes, lw=2, color="#d62828",
             label="Final tumor size")
    ax1.set_xlabel("Off-target effect on T cells (% of effect on tumor)")
    ax1.set_ylabel("Final tumor size (a.u.)", color="#d62828")
    ax2 = ax1.twinx()
    ax2.plot(off_target * 100, final_activity, lw=2, ls="--", color="#1d3557",
             label="Final T-cell activity")
    ax2.set_ylabel("Final T-cell effector activity (0-1)", color="#1d3557")
    ax2.set_ylim(0, 1)
    ax1.set_title("Selectivity matters: inhibitor hitting T cells as well",
                  loc="left", fontweight="bold")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = {name: simulate(params) for name, params in SCENARIOS.items()}

    print("Final values at t = {:.0f} (arbitrary units)".format(DEFAULT_PARAMS["t_end"]))
    print("{:<42} {:>8} {:>8} {:>9}".format("Scenario", "Glucose", "Tumor", "Effector"))
    for name, res in results.items():
        print("{:<42} {:>8.2f} {:>8.1f} {:>9.2f}".format(
            name, res["glucose"][-1], res["tumor"][-1], res["effector"][-1]))

    plot_scenarios(results, "figures/fig1_scenarios.png")
    plot_threshold("figures/fig2_escape_threshold.png")
    plot_selectivity("figures/fig3_selectivity.png")
    print("Figures saved in the 'figures' folder.")
