"""Does the tumour escape depend on competition, or on *unequal* competition?

The headline behaviour of the ODE model is an escape threshold: small tumours
are cleared, large ones escape. The model gives the tumour a higher glucose
affinity than the T cells (half-saturation 1.0 vs 3.0) - a value I chose, not
one I measured. This script asks how much of the result that single choice
carries.

For each affinity setting we sweep the initial tumour size and ask whether the
tumour is larger at the end of the run than it started. Nothing here is a new
scientific finding; it is a sensitivity check on my own model.

Run:  python sensitivity_affinity.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tme_glucose as M

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SIZES = range(1, 151)          # initial tumour size, arbitrary units
CAPACITY = M.DEFAULT_PARAMS["tumor_capacity"]   # 200

SETTINGS = [
    ("default: tumour 1.0 / T cell 3.0", 1.0, 3.0),
    ("equal, both 1.0 (tumour's value)", 1.0, 1.0),
    ("equal, both 3.0 (T cell's value)", 3.0, 3.0),
    ("reversed: tumour 3.0 / T cell 1.0", 3.0, 1.0),
]


def final_size(n0, tumor_km, tcell_km):
    p = M.with_changes(initial_tumor=n0, tumor_km=tumor_km, tcell_km=tcell_km)
    return M.simulate(p)["tumor"][-1]


def main():
    rows, curves = [], {}
    for label, t_km, e_km in SETTINGS:
        finals = np.array([final_size(n, t_km, e_km) for n in SIZES])
        grew = [n for n, f in zip(SIZES, finals) if f > n]
        rows.append({"setting": label, "tumor_km": t_km, "tcell_km": e_km,
                     "first_escape": grew[0] if grew else None,
                     "n_escaping": len(grew),
                     "max_final": round(float(finals.max()), 2)})
        curves[label] = finals
        first = grew[0] if grew else None
        print(f"{label:38} escapes from n0 = {str(first):>4}   "
              f"({len(grew):3d} of {len(list(SIZES))} sizes)   "
              f"largest final size {finals.max():6.1f}")

    json.dump(rows, open(os.path.join(FIG, "sensitivity_affinity.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.array(list(SIZES))
    ax.plot(x, x, color="0.6", lw=1, ls="--", label="no change (final = initial)")
    # the three non-default curves sit on top of each other at zero, so vary
    # the width and dash pattern to keep all four readable
    styles = [(2.4, "-"), (4.0, "-"), (2.4, (0, (6, 3))), (1.3, (0, (1.5, 2)))]
    for (label, _, _), (lw, ls) in zip(SETTINGS, styles):
        ax.plot(x, curves[label], lw=lw, ls=ls, label=label)
    ax.axhline(CAPACITY, color="0.8", lw=1)
    ax.annotate("tissue carrying capacity", (4, CAPACITY - 12), fontsize=8, color="0.45")
    ax.set_xlabel("initial tumour size (arbitrary units)")
    ax.set_ylabel("tumour size at t = 100")
    ax.set_title("Escape needs unequal affinity, not competition as such",
                 loc="left", fontweight="bold", pad=18)
    ax.text(0.0, 1.015, "the three non-default curves lie on top of one another at zero",
            transform=ax.transAxes, fontsize=8, color="0.45")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_affinity_sensitivity.png"), dpi=150)
    print("\nwrote figures/fig6_affinity_sensitivity.png")


if __name__ == "__main__":
    main()
