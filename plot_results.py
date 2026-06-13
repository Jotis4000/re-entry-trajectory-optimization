"""
plot_results.py
===============

Author: Panagiotis Sachinis
Year: 2026

Render the solution produced by main.py.

"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Report-quality look & feel
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,        # set True if you have a LaTeX install
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.axisbelow": True,       # grid sits behind the data
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.color": "#b8b8b8",
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

# Coordinated palette (ColorBrewer RdBu pairing -> reads well in print & greyscale)
COL_DATA  = "#2166ac"   # primary trajectory traces
COL_LIMIT = "#b2182b"   # constraint limits
COL_PT    = "#1a1a1a"   # control-point markers
COL_PEAK  = "#762a83"   # peak markers / annotations


def _polish(ax):
    """Common per-axis tidy-up: drop the top/right spines for a clean look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)


def _mark_peak(ax, x, y, label):
    """Mark the maximum of a curve with a dot and a small annotation."""
    i = int(np.nanargmax(y))
    ax.plot(x[i], y[i], "o", color=COL_PEAK, ms=5, zorder=6)
    ax.annotate(label.format(y[i]), (x[i], y[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=9, color=COL_PEAK, fontweight="bold")


def _shade_limit(ax, limit):
    """Draw a constraint line and lightly shade the forbidden region above it."""
    ax.axhline(limit, ls="--", lw=1.6, color=COL_LIMIT, label="limit", zorder=4)
    y0, y1 = ax.get_ylim()
    if y1 > limit:
        ax.axhspan(limit, y1, color=COL_LIMIT, alpha=0.07, lw=0, zorder=0)
        ax.set_ylim(y0, y1)


def load(fname):
    try:
        d = np.load(fname, allow_pickle=True)
    except FileNotFoundError:
        raise SystemExit(f"'{fname}' not found. Run 'python main.py' first.")
    cols = list(d["columns"])
    H = d["hist"]
    data = {name: H[:, i] for i, name in enumerate(cols)}
    data["v_nodes"] = d["v_nodes"]
    data["alpha_nodes_deg"] = np.degrees(d["alpha_nodes_rad"])
    data["qdot_max"] = float(d["qdot_max"])
    data["n_max"] = float(d["n_max"])
    # scalars for the printed summary
    for k in ("status", "t_final", "range_final", "v_final",
              "gamma_final_deg", "heat_load", "qdot_peak", "n_peak"):
        data[k] = d[k]
    return data


def print_summary(d):
    print("=" * 60)
    print("RE-ENTRY SOLUTION SUMMARY")
    print("=" * 60)
    print(f"  status            : {d['status']}")
    print(f"  time of flight    : {float(d['t_final']):8.1f} s")
    print(f"  downrange         : {float(d['range_final'])/1000:8.1f} km")
    print(f"  terminal velocity : {float(d['v_final']):8.1f} m/s")
    print(f"  integrated heat   : {float(d['heat_load'])/1e6:8.2f} MJ/m^2")
    print(f"  peak heat flux    : {float(d['qdot_peak'])/1e3:8.1f} kW/m^2 "
          f"(limit {d['qdot_max']/1e3:.0f})")
    print(f"  peak deceleration : {float(d['n_peak']):8.2f} g     "
          f"(limit {d['n_max']:.0f})")
    print("=" * 60)


def make_figure(d):
    t, v, h = d["t"], d["v"], d["h"]
    gamma, qdot, n_aero, alpha = d["gamma_deg"], d["qdot"], d["n_aero"], d["alpha_deg"]

    fig, ax = plt.subplots(2, 4, figsize=(18.5, 9))
    fig.suptitle("Optimised Apollo Re-entry Trajectory",
                 fontsize=15, fontweight="bold")

    # --- (0,0) altitude vs velocity ---
    ax[0, 0].plot(v, h / 1000, color=COL_DATA)
    ax[0, 0].set(xlabel="velocity [m/s]", ylabel="altitude [km]",
                 title="Altitude vs Velocity")
    ax[0, 0].invert_xaxis()

    # --- (0,1) altitude vs time ---
    ax[0, 1].plot(t, h / 1000, color=COL_DATA)
    ax[0, 1].set(xlabel="time [s]", ylabel="altitude [km]", title="Altitude vs Time")

    # --- (0,2) stagnation heat flux (constrained) ---
    ax[0, 2].plot(t, qdot / 1e3, color=COL_DATA)
    ax[0, 2].set(xlabel="time [s]", ylabel="heat flux [kW/m$^2$]",
                 title="Stagnation Heat Flux")
    _mark_peak(ax[0, 2], t, qdot / 1e3, "{:.0f} kW/m$^2$")
    _shade_limit(ax[0, 2], d["qdot_max"] / 1e3)
    ax[0, 2].legend(loc="upper right")

    # --- (0,3) lift-to-drag ratio ---
    if "L_over_D" in d:
        ax[0, 3].plot(t, d["L_over_D"], color=COL_DATA)
        ax[0, 3].axhline(0.0, color="#999999", lw=0.8, zorder=1)
        ax[0, 3].set(xlabel="time [s]", ylabel="L/D [-]",
                     title="Lift-to-Drag Ratio")
    else:
        ax[0, 3].axis("off")

    # --- (1,0) aerodynamic deceleration (constrained) ---
    ax[1, 0].plot(t, n_aero, color=COL_DATA)
    ax[1, 0].set(xlabel="time [s]", ylabel="load factor [g]",
                 title="Aerodynamic Deceleration")
    _mark_peak(ax[1, 0], t, n_aero, "{:.2f} g")
    _shade_limit(ax[1, 0], d["n_max"])
    ax[1, 0].legend(loc="lower left")

    # --- (1,1) flight-path angle ---
    ax[1, 1].plot(t, gamma, color=COL_DATA)
    ax[1, 1].axhline(0.0, color="#999999", lw=0.8, zorder=1)
    ax[1, 1].set(xlabel="time [s]", ylabel=r"$\gamma$ [deg]",
                 title="Flight-Path Angle")

    # --- (1,2) angle-of-attack schedule ---
    ax[1, 2].plot(v, alpha, color=COL_DATA, label=r"flown $\alpha(V)$")
    ax[1, 2].plot(d["v_nodes"], d["alpha_nodes_deg"], "o", color=COL_PT,
                  ms=6, mec="white", mew=0.8, label="control points", zorder=5)
    ax[1, 2].set(xlabel="velocity [m/s]", ylabel=r"$\alpha$ [deg]",
                 title="Angle-of-Attack Schedule")
    ax[1, 2].invert_xaxis()
    ax[1, 2].legend(loc="best")

    # --- (1,3) velocity vs time  (NEW) ---
    ax[1, 3].plot(t, v, color=COL_DATA)
    ax[1, 3].set(xlabel="time [s]", ylabel="velocity [m/s]", title="Velocity vs Time")

    for a in ax.ravel():
        _polish(a)

    fig.text(0.995, 0.006, "P. Sachinis, 2026", ha="right", va="bottom",
             fontsize=8, color="#888888", style="italic")
    fig.tight_layout(rect=[0, 0.015, 1, 0.96])
    return fig

def main():
    results_file = "data/reentry_results.npz"
    figure_file = "figs/reentry_trajectory.png"
    d = load(results_file)
    print_summary(d)
    fig = make_figure(d)
    fig.savefig(figure_file)
    print(f"\nSaved figure -> {figure_file}")
    plt.show()

if __name__ == "__main__":
    main()