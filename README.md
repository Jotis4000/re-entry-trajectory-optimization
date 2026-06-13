# re-entry-optimization-hypersonic

TEST FOR README COMPATIBILITY

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyGMO](https://img.shields.io/badge/PyGMO-Optimization-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-EOM-lightgrey.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Viz-green.svg)

> **Executive Summary:** A 3-Degree-of-Freedom (3-DOF) flight dynamics simulation and trajectory optimization tool for an Apollo-style re-entry capsule. Built using the **European Space Agency's PyGMO framework**, this project utilizes Self-Adaptive Differential Evolution (SADE) to find the optimal Angle of Attack ($\alpha$) schedule that minimizes integrated stagnation heat load while strictly adhering to peak heating and aerodynamic G-load constraints.

---

## Physics & Modeling Methodology

This simulation avoids reliance on empirical wind-tunnel data by deriving aerodynamic and environmental properties computationally:

* **Aerodynamics (Modified Newtonian Theory):** The Apollo capsule geometry (spherical heat shield + conical afterbody) is dynamically meshed into 3D panels. Outward normals are computed, and local pressure coefficients are integrated over the surface using $C_p = C_{p,max} \sin^2(\theta)$ on windward panels. $C_{p,max}$ is dynamically derived using normal-shock relations.
* **Thermal Heating:** Stagnation point convective heating is modeled using the Sutton-Graves relation: $\dot{q} = k \sqrt{\frac{\rho}{R_N}} V^3$.
* **Atmosphere:** A vectorized implementation of the 1976 US Standard Atmosphere model provides high-speed density and speed-of-sound lookups.
* **Optimization Strategy:** The problem is formulated as a non-linear control optimization. The trajectory is integrated using an RK4 solver across a 12-node discrete Angle of Attack schedule, penalized heavily for violating the 5.0e5 $W/m^2$ heat flux limit or the 3G deceleration limit.

---

## Repository Structure

To ensure system stability, the numerical optimization and the data visualization pipelines are strictly decoupled into separate scripts. This prevents known C++ segmentation faults caused by threading clashes between PyGMO and Matplotlib.

* `main.py` — The core 3-DOF physics engine and PyGMO optimization routine. Generates the 3D surface mesh, runs the evolutionary algorithm, and exports results.
* `atmosphere.py` — The standalone 1976 US Standard Atmosphere calculator.
* `plot_results.py` — Ingests the optimization output and generates report-quality trajectory subplots (Altitude, Velocity, Heating, G-Load, and Control History).
* `plot_geom.py` — Visualizes the generated 3D surface mesh and 2D meridian profile of the Apollo capsule.

---

## Installation & Usage

### 1. Dependencies
Because PyGMO is a wrapper around heavily optimized C++ libraries, it is highly recommended to install it via `conda` rather than standard pip to avoid build-chain errors.

```bash
conda install -c conda-forge pygmo
pip install numpy scipy matplotlib