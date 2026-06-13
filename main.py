"""
main.py
=======

Author: Panagiotis Sachinis
Year: 2026

A first-estimate re-entry trajectory optimizer for an Apollo-shaped capsule based on
minimizing integrated stagnation heat load during re-entry, as well as peak stagnation
heating and G-load constraints.

Model uses a point-mass (3-DOF) physics engine, with angle of attack scheduling at
discrete nodes (linearly interpolated).

Optimizer uses PyGMO Self-Adaptive Differential Evolution (SADE) to evaluate trajectories
from the entry interface down to a terminal altitude.

Physics (based on concepts from Anderson, "Hypersonic and High-Temperature Gas Dynamics"):
  * Aerodynamics : (modified) Newtonian theory integrated over the Apollo geometry
                   (spherical heat shield + conical afterbody).  Cp = Cp_max*sin^2(th)
                   on windward panels, 0 on shadowed ones; Cp_max is the modified-
                   Newtonian normal-shock value (function of Mach).
  * Heating      : Sutton-Graves stagnation point, q = k*sqrt(rho/Rn)*V^3.
  * Atmosphere   : 1976 US Standard Atmosphere

Run:  python main.py            -> Writes reentry_results.npz
      python plot_results.py    -> Plots trajectory output (separate, PyGMO-free process)
      python plot_geom.py       -> Plots the capsule geometry

      Processes were separated to avoid C++ segfault as a result of 
      PyGMO and matplotlib library clashes.
"""

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import atmosphere as atm

#######################################################
# SIMULATION PARAMETERS
#######################################################

### Basic Constants
G0          = 9.80665           # standard gravity             [m/s^2]
RE          = 6371000.0         # Earth radius for the EOM     [m]

MASS        = 4532.0            # capsule mass                 [kg]
SREF        = 12.02             # reference area               [m^2]
RN          = 4.694             # effective nose radius        [m]
CD0         = 0.0               # parasitic drag coefficient   [-]
GAMMA_AIR   = 1.4               # ratio of specific heats      [-]

K_SG        = 1.7415e-4         # Sutton-Graves constant       [kg^0.5/m]

### Entry Interface
H0          = 120000.0          # entry altitude               [m]
V0          = 7800.0            # entry speed                  [m/s]
GAMMA0      = np.radians(-2.0)  # entry flight-path angle      [rad]
HF          = 25_000.0          # terminal altitude            [m]

### Simulation Constraints
QDOT_MAX    = 5.0e5             # peak heat-flux limit         [W/m^2]
N_MAX       = 3                 # peak deceleration limit      [g]

N_NODES     = 12                                # No. alpha control points
V_LO        = 500.0                             # Lowest alpha scheduling velocity
ALPHA_MIN   = np.radians(0.0)                   # Min alpha
ALPHA_MAX   = np.radians(45.0)                  # Max alpha
V_NODES     = np.linspace(V_LO, V0, N_NODES)    # velocities the alpha nodes are tied to

DT_OPT      = 2.0               # integration time step        [s]
DT_FINE     = 0.5               # reconstruction time step     [s]
T_MAX       = 3500.0            # integration time cap         [s]

POP_SIZE    = 32                # SADE population
GEN         = 250               # SADE generations
SEED        = 21                # Sim. seed

#######################################################
# ATMOSPHERIC CALCULATIONS
#######################################################
# Tabulates atmospheric data in advance and interpolates to avoid expensive external calls to atmosphere.py
H = np.linspace(0.0, H0, 4001)
RHO, _, _, A, _, _ = atm.getAtmosphere(H)     # returns [rho, T, P, a, nu, mu]
LOGRHO = np.log(RHO)

def getDensity(h):
    return float(np.exp(np.interp(min(max(h, 0.0), H0), H, LOGRHO)))

def getSpeedOfSound(h):
    return float(np.interp(min(max(h, 0.0), H0), H, A))


#######################################################
# AERODYNAMICS
#######################################################
R_SHIELD = RN                    # heat-shield spherical radius     [m]
R_MAX    = 1.956                 # max body radius (D = 3.912 m)    [m]
CONE_ANG = np.radians(32.5)      # angle of cone                    [deg]
R_TOP    = 0.20                  # aft truncation radius            [m]
A_REF    = np.pi * R_MAX**2      # coefficient reference area       [m^2]

def genMesh(n_shield=50, n_cone=25, n_az=120):
    '''Panelise the Apollo body of revolution -> centroids, outward normals, areas.'''
    th = np.linspace(0.0, np.arcsin(R_MAX / R_SHIELD), n_shield)
    xs, rs = R_SHIELD * (1 - np.cos(th)), R_SHIELD * np.sin(th)       # heat shield
    L = (rs[-1] - R_TOP) / np.tan(CONE_ANG)
    xc = np.linspace(xs[-1], xs[-1] + L, n_cone)
    rc = rs[-1] - (xc - xs[-1]) * np.tan(CONE_ANG)                    # afterbody cone
    x = np.concatenate([xs, xc[1:]])
    r = np.concatenate([rs, rc[1:]])

    phi = np.linspace(0.0, 2 * np.pi, n_az + 1)
    X = x[:, None] * np.ones_like(phi)[None, :]
    Y = r[:, None] * np.cos(phi)[None, :]
    Z = r[:, None] * np.sin(phi)[None, :]

    np.savez('data/surface_coordinates.npz', X=X, Y=Y, Z=Z)
    print("Saved surface coordinates to 'surface_coordinates.npz'")

    P00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], -1)
    P10 = np.stack([X[1:, :-1],  Y[1:, :-1],  Z[1:, :-1]],  -1)
    P11 = np.stack([X[1:, 1:],   Y[1:, 1:],   Z[1:, 1:]],   -1)
    P01 = np.stack([X[:-1, 1:],  Y[:-1, 1:],  Z[:-1, 1:]],  -1)
    c = 0.25 * (P00 + P10 + P11 + P01)
    av = 0.5 * np.cross(P11 - P00, P01 - P10)        # area-weighted normal
    area = np.linalg.norm(av, axis=-1)
    n = av / (area[..., None] + 1e-30)
    c, n, area = c.reshape(-1, 3), n.reshape(-1, 3), area.reshape(-1)
    ref = c.mean(0)                                  # interior point (body is convex)
    n *= np.sign(np.einsum('ij,ij->i', n, c - ref))[:, None]   # orient outward
    return c, n, area

PAN_CENTER, PAN_NORMAL, PAN_AREA = genMesh()

def tabAeroCoeffs(alpha):
    '''Generates the aero coefficients for the Newton method independent of Cp_max.
    Makes life a bit easier computationally down the road as we don't need to recalculate
    these every time, just multiply by whatever Cp_max is.'''
    V = np.array([np.cos(alpha), 0.0, np.sin(alpha)])        # freestream direction in 3D
    imp = -(PAN_NORMAL * V).sum(axis=1)                      # sin(local inclination)
    Cp = np.where(imp > 0.0, imp * imp, 0.0)                 # Newtonian, Cp_max = 1
    F = -(Cp[:, None] * PAN_NORMAL * PAN_AREA[:, None]).sum(0) / A_REF # This was an actual headache to make
    lift = np.array([np.sin(alpha), 0.0, -np.cos(alpha)])    # +lift, perp. to V
    return float((F * lift).sum()), float((F * V).sum())

# Tabulate the lift and drag coeffs based on Newton in advance, removes some headache.
ALPHA_RANGE = np.radians(np.linspace(np.degrees(ALPHA_MIN)-2, np.degrees(ALPHA_MAX)+2, 241)) # Range of alphas to tabulate (with some wiggle room)
CL_TAB = np.array([tabAeroCoeffs(a)[0] for a in ALPHA_RANGE])
CD_TAB = np.array([tabAeroCoeffs(a)[1] for a in ALPHA_RANGE])

def getCPmax(M):
    """Modified-Newtonian Cp_max from the normal-shock stagnation pressure."""
    g = GAMMA_AIR
    return (2/(GAMMA_AIR*M**2))*(((GAMMA_AIR+1)**2*M**2/(4*GAMMA_AIR*M**2-2*(GAMMA_AIR-1)))**(GAMMA_AIR/(GAMMA_AIR-1))*((1-GAMMA_AIR+2*GAMMA_AIR*M**2)/(GAMMA_AIR+1))-1)

def aero(alpha, M):
    """Calculates CL and CD"""
    cp = getCPmax(M)
    a = min(max(alpha, ALPHA_RANGE[0]), ALPHA_RANGE[-1])
    CL = cp * np.interp(a, ALPHA_RANGE, CL_TAB)
    CD = cp * np.interp(a, ALPHA_RANGE, CD_TAB) + CD0
    return CL, CD

#######################################################
# PHYSICS AND TRAJECTORY PROPOGATION
#######################################################
def applyEOM(y, alpha):
    """d/dt [v, gamma, h, s] for the planar re-entry EOM."""
    v, gamma, h, s = y
    rho = getDensity(h)
    M = v / getSpeedOfSound(h)
    CL, CD = aero(alpha, M)
    r = RE + h
    g = G0 * (RE / r)**2
    q = 0.5 * rho * v * v
    L = q * SREF * CL
    D = q * SREF * CD
    dv = -D / MASS - g * np.sin(gamma)
    dgamma = L / (MASS * v) - (g / v) * np.cos(gamma) + (v / r) * np.cos(gamma)
    dh = v * np.sin(gamma)
    ds = v * np.cos(gamma) * RE / r
    return np.array([dv, dgamma, dh, ds])
 
def propagate(alpha_nodes, dt=DT_OPT, record=False):
    """Integrate one trajectory. Returns a dict of metrics (+ history if `record`). Standard Runge-Kutta integration (RK4)."""
    y = np.array([V0, GAMMA0, H0, 0.0])
    t = qdot_peak = n_peak = heat_load = 0.0
    hist = []
    while True:
        v, gamma, h, s = y
        alpha = float(np.interp(v, V_NODES, alpha_nodes))
        rho = getDensity(h)
        CL, CD = aero(alpha, v / getSpeedOfSound(h))
        q = 0.5 * rho * v * v
        qdot = K_SG * np.sqrt(rho / RN) * v**3
        n = q * SREF * np.sqrt(CL * CL + CD * CD) / (MASS * G0)
        qdot_peak = max(qdot_peak, qdot)
        n_peak = max(n_peak, n)
        heat_load += qdot * dt
        if record:
            hist.append((t, v, np.degrees(gamma), h, s, qdot, n, np.degrees(alpha), q,
                         CL / CD if CD != 0 else 0.0))
 
        if h <= HF:
            status = "reached"; break
        if t > T_MAX or h > H0 or v < 150.0:
            status = "failed"; break
 
        k1 = applyEOM(y, alpha)
        k2 = applyEOM(y + 0.5 * dt * k1, alpha)
        k3 = applyEOM(y + 0.5 * dt * k2, alpha)
        k4 = applyEOM(y + dt * k3, alpha)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
 
    return dict(status=status, reached=(status == "reached"), t_final=t,
                heat_load=heat_load, qdot_peak=qdot_peak, n_peak=n_peak,
                v_final=y[0], gamma_final_deg=np.degrees(y[1]),
                h_final=y[2], range_final=y[3],
                hist=(np.array(hist) if record else None))


#######################################################
# OPTIMIZATION UDP FOR PYGMO
#######################################################
class ReentryProblem:
    """Minimal PyGMO problem. x = alpha at the N velocity nodes [rad]."""

    def get_bounds(self):
        return ([ALPHA_MIN] * N_NODES, [ALPHA_MAX] * N_NODES)

    def get_name(self):
        return "Apollo Re-Entry Trajectory (Newtonian Aero, Heat-Flux Constrained)"

    def fitness(self, x):
        try:
            r = propagate(x, DT_OPT)
        except Exception:
            return [1e9] # Penalize heavily if something goes kaput
        if not r["reached"]:                       
            return [1e5 + max(0.0, r["h_final"] - HF) / 1000.0] # beun to push toward the ground
        Q = r["heat_load"] / 1e6                    # MJ/m^2  (objective)
        penalty = 1e6 * max(0.0, r["qdot_peak"] / QDOT_MAX - 1.0) + 1e6 * max(0.0, r["n_peak"] / N_MAX - 1.0) # 1e3?
        return [Q + penalty]


def main():
    ### Initialize PyGMO UDP
    prob = pg.problem(ReentryProblem())
    pop = pg.population(prob, size=POP_SIZE, seed=SEED)
    pop.set_x(0, np.linspace(ALPHA_MIN + np.radians(15), ALPHA_MAX - np.radians(2), N_NODES)) # Give a reasonable first guess
 
    algo = pg.algorithm(pg.sade(gen=GEN, seed=SEED))
    algo.set_verbosity(20)
    print("Optimising (min heat load; peak-q and g-load as penalties)...")
    pop = algo.evolve(pop)
 
    x = pop.champion_x
    r = propagate(x, DT_FINE, record=True)
 
    ### Output stuff
    print("\n" + "=" * 60)
    print("OPTIMISED APOLLO RE-ENTRY")
    print("=" * 60)
    print(f"  status            : {r['status']}")
    print(f"  time of flight    : {r['t_final']:8.1f} s")
    print(f"  downrange         : {r['range_final']/1000:8.1f} km")
    print(f"  terminal velocity : {r['v_final']:8.1f} m/s")
    print(f"  integrated heat   : {r['heat_load']/1e6:8.2f} MJ/m^2  (objective)")
    print(f"  peak heat flux    : {r['qdot_peak']/1e3:8.1f} kW/m^2 (limit {QDOT_MAX/1e3:.0f})")
    print(f"  peak deceleration : {r['n_peak']:8.2f} g      (limit {N_MAX:.0f})")
    lod = r["hist"][:, 9]
    print(f"  L/D range         : {lod.min():8.3f} to {lod.max():.3f}")
    print("  alpha schedule [deg] vs velocity [m/s]:")
    for vn, an in zip(V_NODES, np.degrees(x)):
        print(f"      V = {vn:7.0f}  ->  alpha = {an:5.2f}")
    print("=" * 60)
 
    np.savez("data/reentry_results.npz",
             hist=r["hist"],
             columns=np.array(["t", "v", "gamma_deg", "h", "s",
                               "qdot", "n_aero", "alpha_deg", "qdyn", "L_over_D"]),
             v_nodes=V_NODES, alpha_nodes_rad=np.asarray(x, dtype=float),
             qdot_max=QDOT_MAX, n_max=N_MAX,
             status=r["status"], t_final=r["t_final"], range_final=r["range_final"],
             v_final=r["v_final"], gamma_final_deg=r["gamma_final_deg"],
             heat_load=r["heat_load"], qdot_peak=r["qdot_peak"], n_peak=r["n_peak"])
    print("\nResults saved to reentry_results.npz")

if __name__ == "__main__":
    main()