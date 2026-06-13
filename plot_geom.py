"""
plot_geom.py
============

Author: Panagiotis Sachinis
Year: 2026

Visualise the Apollo capsule geometry written by main.py.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.axisbelow": True,
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

COL_DATA = "#2166ac"
COL_EDGE = "#0b3a66"
COL_REF  = "#999999"

def load(fname="data/surface_coordinates.npz"):
    try:
        d = np.load(fname)
    except FileNotFoundError:
        raise SystemExit(f"'{fname}' not found. Run 'python main.py' first.")
    return d["X"], d["Y"], d["Z"]


def make_profile_figure(X, Y, Z):
    """Figure 1: the 2-D meridian profile recovered from the surface grid."""
    x_prof = X[:, 0]
    r_prof = Y[:, 0]

    fig = plt.figure(figsize=(7.2, 6.6))
    fig.suptitle("Apollo Command Module \u2014 Meridian Profile",
                 fontsize=14, fontweight="bold")
    ax = fig.add_subplot(111)

    ax.fill_between(x_prof, r_prof, -r_prof, color=COL_DATA, alpha=0.13, zorder=1)
    ax.plot(x_prof, r_prof, color=COL_EDGE, lw=2.2, zorder=3)
    ax.plot(x_prof, -r_prof, color=COL_EDGE, lw=2.2, zorder=3)
    ax.axhline(0.0, color=COL_REF, lw=0.9, ls="--", zorder=2)

    ax.set(xlabel="axial station  $x$ [m]", ylabel="radius  $r$ [m]")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0.08)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def make_surface_figure(X, Y, Z):
    """Figure 2: the shaded 3-D surface render with true proportions."""
    fig = plt.figure(figsize=(8.0, 7.0))
    fig.suptitle("Apollo Command Module \u2014 3-D Surface",
                 fontsize=14, fontweight="bold")
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, color=COL_DATA, shade=True,
                    rstride=1, cstride=2, linewidth=0,
                    antialiased=True, alpha=1.0)

    ax.set_box_aspect((X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()))
    ax.view_init(elev=20, azim=-65)

    ax.set(xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"].update(color="#dddddd", linestyle="--", linewidth=0.6)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def main():
    X, Y, Z = load()

    fig_profile = make_profile_figure(X, Y, Z)
    fig_profile.savefig("figs/capsule_meridian.png")

    fig_surface = make_surface_figure(X, Y, Z)
    fig_surface.savefig("figs/capsule_surface.png")

    print("Saved figures -> capsule_meridian.png, capsule_surface.png")
    plt.show()

if __name__ == "__main__":
    main()