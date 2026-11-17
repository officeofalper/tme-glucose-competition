"""
Starved in the Tumor, version 2: a spatial model of nutrient competition
in the tumor microenvironment (TME).

IMPORTANT: This is a simplified, illustrative model. All parameter values
are chosen to show qualitative behaviour. They are NOT measured values,
and the outputs are NOT experimental data or new scientific findings.

What is new compared with tme_glucose.py (version 1):
  1. Lactate:   tumor cells turn much of the glucose they take up into
                lactate, and lactate suppresses T-cell effector function.
  2. Glutamine: a second nutrient that tumor cells and T cells compete for.
  3. Effector <-> memory-like switch: a starved effector T cell does not
                die; it switches to a quiet, oxidative (OXPHOS/FAO-like)
                state that uses little fuel and does not kill. If nutrients
                return, it can switch back.
  4. Space:     the tissue is a 2D grid. Nutrients enter from a blood vessel
                on the left edge and diffuse into the tissue, so cells far
                from the vessel see less glucose and glutamine.

Model pieces (all in arbitrary units):
  Fields on the grid:   G (glucose), Q (glutamine), L (lactate), N (tumor cells)
  Agents:               individual T cells with a position and a state
                        ("effector" or "memory")

Each field changes by diffusion + uptake/production, solved with the
simple Euler method on the grid:
    new value = old value + (rate of change) * dt
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Default parameters (arbitrary units, chosen for illustration)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Grid and time
    "nx": 60, "ny": 40,        # grid size (cells across, cells down)
    "dt": 0.1, "t_end": 300.0,
    # Blood vessel (left edge): concentrations in blood
    "blood_glucose": 10.0,
    "blood_glutamine": 5.0,
    # Diffusion coefficients (grid units^2 per time unit)
    "D_glucose": 1.0, "D_glutamine": 1.0, "D_lactate": 1.0, "D_tumor": 0.02,
    # Normal tissue background use of nutrients, and lactate breakdown
    "tissue_use": 0.002,
    "lactate_decay": 0.01,
    # Tumor cells (values per tumor cell)
    "tumor_glc_vmax": 0.05, "tumor_glc_km": 1.0,
    "tumor_gln_vmax": 0.02, "tumor_gln_km": 0.5,
    "lactate_yield": 1.5,      # lactate made per unit glucose taken up (Warburg)
    "tumor_growth": 0.04,
    "tumor_capacity": 10.0,    # maximum tumor cells per grid square
    # T cells (values per T cell)
    "n_tcells": 40,
    "tcell_glc_vmax": 0.1, "tcell_glc_km": 1.5,
    "tcell_gln_vmax": 0.05, "tcell_gln_km": 1.0,
    "lactate_ki": 3.0,          # lactate level that halves T-cell function
    "kill_rate": 2.0,          # tumor cells killed per fully active effector per time
    "effector_speed": 1.5,     # grid squares per time unit
    "memory_speed": 0.15,
    "memory_fuel_use": 0.2,    # memory-like cells use 20% of effector fuel
    "switch_down": 0.15,       # effector -> memory below this fitness
    "switch_up": 0.35,         # memory -> effector above this fitness
    "switch_delay": 5.0,       # time a cell must spend below/above threshold
    # Interventions (0 = no effect, 1 = complete block)
    "tumor_inhibition": 0.0,   # drug blocking tumor glucose uptake
    "tcell_inhibition": 0.0,   # the same drug's side effect on T cells
    "tumor_gln_inhibition": 0.0,  # second drug: blocks tumor glutamine uptake
    "drug_start": 0.0,         # time at which the drug is given
    # Starting tumor
    "tumor_radius": 7.0,       # grid squares
    "tumor_center": (38, 20),
    "seed": 1,
}


# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------
def saturating(c, km):
    """Michaelis-Menten-like curve: 0 when c = 0, close to 1 when c >> km."""
    return c / (km + c)


def laplacian(field):
    """Diffusion term: how much a square differs from its 4 neighbours.
    Edges copy their neighbour (no flow in or out through the edge)."""
    p = np.pad(field, 1, mode="edge")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * field


def drug(p, t):
    """Drug blocks at time t (all 0 before the drugs start): tumor glucose
    uptake, T-cell glucose uptake (side effect), tumor glutamine uptake."""
    on = t >= p["drug_start"]
    if not on:
        return 0.0, 0.0, 0.0
    return p["tumor_inhibition"], p["tcell_inhibition"], p["tumor_gln_inhibition"]


def tcell_fitness(G, Q, L, p, tcell_block=0.0):
    """How well a T cell can work at a given spot (0 to 1): it needs glucose
    AND glutamine, and lactate pushes its function down."""
    fuel = (1 - tcell_block) * saturating(G, p["tcell_glc_km"]) \
        * saturating(Q, p["tcell_gln_km"])
    return fuel / (1 + L / p["lactate_ki"])


# ---------------------------------------------------------------------------
# 3. Setting up the tissue
# ---------------------------------------------------------------------------
def initial_state(p):
    ny, nx = p["ny"], p["nx"]
    rng = np.random.default_rng(p["seed"])
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Start nutrients at their healthy-tissue levels by letting them settle
    G = np.full((ny, nx), p["blood_glucose"])
    Q = np.full((ny, nx), p["blood_glutamine"])
    L = np.zeros((ny, nx))

    # Tumor: a disc of tumor cells
    cx, cy = p["tumor_center"]
    dist = np.hypot(xx - cx, yy - cy)
    N = np.where(dist < p["tumor_radius"], 0.8 * p["tumor_capacity"], 0.0)

    # T cells: start next to the vessel, all as effectors
    n = p["n_tcells"]
    tcells = {
        "x": rng.uniform(1, 4, n),
        "y": rng.uniform(1, ny - 2, n),
        "effector": np.ones(n, dtype=bool),
        "timer": np.zeros(n),
    }
    return {"t": 0.0, "G": G, "Q": Q, "L": L, "N": N, "tcells": tcells, "rng": rng}


# ---------------------------------------------------------------------------
# 4. One time step
# ---------------------------------------------------------------------------
def step(s, p):
    dt = p["dt"]
    G, Q, L, N, tc = s["G"], s["Q"], s["L"], s["N"], s["tcells"]

    tumor_block, tcell_block, gln_block = drug(p, s["t"])

    # --- Tumor uptake, lactate production and growth (per grid square) ---
    tumor_glc_frac = (1 - tumor_block) * saturating(G, p["tumor_glc_km"])
    tumor_gln_frac = (1 - gln_block) * saturating(Q, p["tumor_gln_km"])
    tumor_glc_use = p["tumor_glc_vmax"] * tumor_glc_frac * N
    tumor_gln_use = p["tumor_gln_vmax"] * tumor_gln_frac * N
    lactate_made = p["lactate_yield"] * tumor_glc_use
    growth = p["tumor_growth"] * tumor_glc_frac * tumor_gln_frac * N \
        * (1 - N / p["tumor_capacity"])

    # --- T cells: where they are, how fit they are, what they use ---
    ix = np.clip(tc["x"].astype(int), 0, p["nx"] - 1)
    iy = np.clip(tc["y"].astype(int), 0, p["ny"] - 1)
    fit = tcell_fitness(G[iy, ix], Q[iy, ix], L[iy, ix], p, tcell_block)
    use_factor = np.where(tc["effector"], 1.0, p["memory_fuel_use"])
    gfrac = (1 - tcell_block) * saturating(G[iy, ix], p["tcell_glc_km"])
    qfrac = saturating(Q[iy, ix], p["tcell_gln_km"])

    tcell_glc_use = np.zeros_like(G)
    tcell_gln_use = np.zeros_like(Q)
    killing = np.zeros_like(N)
    np.add.at(tcell_glc_use, (iy, ix), p["tcell_glc_vmax"] * gfrac * use_factor)
    np.add.at(tcell_gln_use, (iy, ix), p["tcell_gln_vmax"] * qfrac * use_factor)
    np.add.at(killing, (iy, ix), p["kill_rate"] * fit * tc["effector"])
    killing = np.minimum(killing, N / dt)          # cannot kill more than exist

    # --- Update the fields: diffusion + sources - uses ---
    G_new = G + dt * (p["D_glucose"] * laplacian(G) - p["tissue_use"] * G
                      - tumor_glc_use - tcell_glc_use)
    Q_new = Q + dt * (p["D_glutamine"] * laplacian(Q) - p["tissue_use"] * Q
                      - tumor_gln_use - tcell_gln_use)
    L_new = L + dt * (p["D_lactate"] * laplacian(L) - p["lactate_decay"] * L
                      + lactate_made)
    N_new = N + dt * (p["D_tumor"] * laplacian(N) + growth - killing)

    # The vessel (left column) keeps blood levels and washes lactate away
    G_new[:, 0] = p["blood_glucose"]
    Q_new[:, 0] = p["blood_glutamine"]
    L_new[:, 0] = 0.0
    N_new[:, 0] = 0.0

    s["G"] = np.maximum(G_new, 0)
    s["Q"] = np.maximum(Q_new, 0)
    s["L"] = np.maximum(L_new, 0)
    s["N"] = np.clip(N_new, 0, p["tumor_capacity"])

    # --- T-cell state switch (with a delay, so cells do not flicker) ---
    eff = tc["effector"]
    wants_to_switch = np.where(eff, fit < p["switch_down"], fit > p["switch_up"])
    tc["timer"] = np.where(wants_to_switch, tc["timer"] + dt, 0.0)
    flip = tc["timer"] >= p["switch_delay"]
    tc["effector"] = np.where(flip, ~eff, eff)
    tc["timer"][flip] = 0.0

    # --- T-cell movement: effectors climb toward tumor, all wander a bit ---
    move_t_cells(s, p, ix, iy)
    s["t"] += dt


def move_t_cells(s, p, ix, iy):
    tc, rng, N = s["tcells"], s["rng"], s["N"]
    n = len(tc["x"])
    # Direction of increasing tumor (a simple stand-in for chemokine signals)
    p_pad = np.pad(N, 1, mode="edge")
    grad_x = (p_pad[1:-1, 2:] - p_pad[1:-1, :-2])[iy, ix]
    grad_y = (p_pad[2:, 1:-1] - p_pad[:-2, 1:-1])[iy, ix]
    norm = np.hypot(grad_x, grad_y) + 1e-9
    # Far from the tumor, effectors drift toward its centre instead
    cx, cy = p["tumor_center"]
    far = norm < 1e-6
    dx_c, dy_c = cx - tc["x"], cy - tc["y"]
    dc = np.hypot(dx_c, dy_c) + 1e-9
    dir_x = np.where(far, dx_c / dc, grad_x / norm)
    dir_y = np.where(far, dy_c / dc, grad_y / norm)
    if N.sum() < 0.5:                       # no tumor left: just wander
        dir_x[:] = 0
        dir_y[:] = 0

    speed = np.where(tc["effector"], p["effector_speed"], p["memory_speed"])
    bias = np.where(tc["effector"], 0.5, 0.0)
    angle = rng.uniform(0, 2 * np.pi, n)
    vx = speed * (bias * dir_x + (1 - bias) * np.cos(angle))
    vy = speed * (bias * dir_y + (1 - bias) * np.sin(angle))
    tc["x"] = np.clip(tc["x"] + vx * p["dt"], 1, p["nx"] - 1.01)
    tc["y"] = np.clip(tc["y"] + vy * p["dt"], 0, p["ny"] - 0.01)


# ---------------------------------------------------------------------------
# 5. Running a scenario
# ---------------------------------------------------------------------------
def with_changes(**changes):
    params = dict(DEFAULT_PARAMS)
    params.update(changes)
    return params


def settle_nutrients(s, p, time=200.0):
    """Let glucose and glutamine reach their healthy-tissue gradient
    before the tumor starts using them."""
    ref = with_changes(**{**p, "tumor_radius": 0.0, "n_tcells": 0})
    tmp = initial_state(ref)
    for _ in range(int(time / p["dt"])):
        step(tmp, ref)
    s["G"], s["Q"] = tmp["G"].copy(), tmp["Q"].copy()


def run(params, snapshot_times=()):
    p = params
    s = initial_state(p)
    settle_nutrients(s, p)
    hist = {k: [] for k in ["t", "tumor", "glucose_in_tumor", "lactate_max",
                            "n_effector", "n_memory"]}
    snaps = {}
    n_steps = int(round(p["t_end"] / p["dt"]))
    for i in range(n_steps + 1):
        t = round(s["t"], 6)
        if i % 10 == 0:
            in_tumor = s["N"] > 1
            hist["t"].append(t)
            hist["tumor"].append(s["N"].sum())
            hist["glucose_in_tumor"].append(s["G"][in_tumor].mean() if in_tumor.any() else np.nan)
            hist["lactate_max"].append(s["L"].max())
            hist["n_effector"].append(int(s["tcells"]["effector"].sum()))
            hist["n_memory"].append(int((~s["tcells"]["effector"]).sum()))
        for ts in snapshot_times:
            if abs(t - ts) < p["dt"] / 2:
                tc = s["tcells"]
                snaps[ts] = {k: s[k].copy() for k in "GQLN"}
                snaps[ts]["tc"] = {k: v.copy() for k, v in tc.items()}
        if i < n_steps:
            step(s, p)
    return {k: np.array(v) for k, v in hist.items()}, snaps


SCENARIOS = {
    "Small tumor": with_changes(tumor_radius=3.0),
    "Large tumor": with_changes(tumor_radius=7.0),
    "Large tumor + glucose drug (80%)": with_changes(tumor_radius=7.0, tumor_inhibition=0.8),
    "Large tumor + glucose drug that also hits T cells": with_changes(
        tumor_radius=7.0, tumor_inhibition=0.8, tcell_inhibition=0.8),
    "Large tumor + glucose and glutamine drugs (50% each)": with_changes(
        tumor_radius=7.0, tumor_inhibition=0.5, tumor_gln_inhibition=0.5),
}

COLORS = {
    "Small tumor": "#2a9d8f",
    "Large tumor": "#d62828",
    "Large tumor + glucose drug (80%)": "#1d3557",
    "Large tumor + glucose drug that also hits T cells": "#e9a23b",
    "Large tumor + glucose and glutamine drugs (50% each)": "#7b2cbf",
}
STYLES = {"Large tumor + glucose drug (80%)": (0, (5, 3))}   # dashed, so overlaps stay visible


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def plot_maps(snaps, filename, title):
    """Figure 4: maps of glucose, lactate and tumor + T cells at several times."""
    times = sorted(snaps)
    fig, axes = plt.subplots(4, len(times), figsize=(4 * len(times), 10.6))
    rows = [("G", "Glucose", "YlOrBr", 0, 10),
            ("Q", "Glutamine", "Greens", 0, 5),
            ("L", "Lactate", "PuRd", None, None),
            ("N", "Tumor + T cells", "Purples", 0, 10)]
    lmax = max(sn["L"].max() for sn in snaps.values()) or 1
    for c, ts in enumerate(times):
        sn = snaps[ts]
        for r, (key, label, cmap, vmin, vmax) in enumerate(rows):
            ax = axes[r, c]
            vmax_ = lmax if key == "L" else vmax
            im = ax.imshow(sn[key], cmap=cmap, vmin=0, vmax=vmax_, origin="upper")
            ax.axvline(-0.5, color="#c1121f", lw=6)
            if key == "N":
                tc = sn["tc"]
                e = tc["effector"]
                # different shapes as well as colors, so the states are clear for color-blind readers
                ax.scatter(tc["x"][e], tc["y"][e], s=22, marker="o", c="#3d8bff", edgecolor="white", lw=.7, label="Effector T cell")
                ax.scatter(tc["x"][~e], tc["y"][~e], s=20, marker="s", c="white", edgecolor="black", lw=.8, label="Memory-like T cell")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {ts:.0f}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
            if c == len(times) - 1:
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    axes[3, 0].legend(loc="lower left", fontsize=8, framealpha=.9)
    fig.suptitle(title + "  (blood vessel = red line on the left; illustrative model only)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def plot_scenarios(results, filename):
    """Figure 5: tumor size, glucose inside the tumor, and T-cell states."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for name, h in results.items():
        ls = STYLES.get(name, "-")
        axes[0].plot(h["t"], h["tumor"], color=COLORS[name], lw=2, ls=ls, label=name)
        axes[1].plot(h["t"], h["glucose_in_tumor"], color=COLORS[name], lw=2, ls=ls)
        frac = h["n_effector"] / (h["n_effector"] + h["n_memory"])
        axes[2].plot(h["t"], 100 * frac, color=COLORS[name], lw=2, ls=ls)
    titles = ["A. Total tumor cells", "B. Mean glucose inside the tumor",
              "C. T cells in effector state (%)"]
    ylabels = ["Tumor cells (a.u.)", "Glucose (a.u.)", "% of T cells"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, loc="left", fontweight="bold")
        ax.set_xlabel("Time (a.u.)"); ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0, 105)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Illustrative model only - not experimental data", fontsize=9,
                 color="gray", x=0.99, ha="right")
    fig.tight_layout(rect=[0, 0.17, 1, 0.97])
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Run everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    for name, params in SCENARIOS.items():
        snap_times = (0, 50, 150, 300) if name == "Large tumor" else ()
        hist, snaps = run(params, snap_times)
        results[name] = hist
        if snaps:
            plot_maps(snaps, "figures/fig4_spatial_maps.png", "Large tumor, no treatment")
        print("{:<54} tumor {:7.1f}   effector T cells {:2d}/{:2d}   max lactate {:5.1f}".format(
            name, hist["tumor"][-1], hist["n_effector"][-1],
            hist["n_effector"][-1] + hist["n_memory"][-1], hist["lactate_max"][-1]))
    plot_scenarios(results, "figures/fig5_spatial_scenarios.png")
    print("Figures saved in the 'figures' folder.")
