import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

"""
atmosphere.py
=======

Author: Panagiotis Sachinis
Year: 2026

Calculates atmospheric properties using the 1976 US Standard Atmosphere model.
"""

G0 = 9.80665 # Hardcode these
R = 287
GAMMA_AIR = 1.4

def getAtmosphere(h):
    """
    Extends isothermal properties for LEO reentry simulations.
    
    Inputs:
    h       : Geometric altitude [m] (can be a scalar or NUmpy array)
    G0      : Standard gravity [m/s^2]
    R       : Specific gas constant for air [J/(kg*K)]
    GAMMA_AIR  : Ratio of specific heats (gamma)
    
    Outputs:
    T, P, RHO, A, MU, NU (All as NUmpy arrays or scalars depending on input)
    """
    # Ensure input is a NUmpy array for vectorized operations
    h_arr = np.atleast_1d(h) # headaaaaches
    
    # Constants - realistically I could have imported all of these from main but this is old code so no reason to change it
    RE = 6356766.0          # Earth radius
    BETA = 1.458e-6         # Sutherland's constant 1 [kg/(m*s*K^0.5)]
    S = 110.4               # Sutherland's constant 2 [K]
    
    H = (RE * h_arr) / (RE + h_arr) # Convert to geopotential altitude
    
    # Define ISA Layers (Base Altitude, Base Temp, Base Press, Lapse Rate)
    # Layers: Troposphere, Tropopause, Stratosphere 1, Stratosphere 2, Stratopause, Mesosphere 1, Mesosphere 2, Mesopause/Thermosphere
    Hb = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
    Tb = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    Pb = np.array([101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.9564, 0.3734])
    Lb = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
    
    # Boolean masks to identify which layer each altitude falls into
    conds = [
        (H >= Hb[0]) & (H < Hb[1]),
        (H >= Hb[1]) & (H < Hb[2]),
        (H >= Hb[2]) & (H < Hb[3]),
        (H >= Hb[3]) & (H < Hb[4]),
        (H >= Hb[4]) & (H < Hb[5]),
        (H >= Hb[5]) & (H < Hb[6]),
        (H >= Hb[6]) & (H < Hb[7]),
        (H >= Hb[7])
    ]
    
    # Calculate Temperature (T) and Pressure (P)
    T = np.zeros_like(H)
    P = np.zeros_like(H)
    
    for i in range(len(Hb)):
        mask = conds[i]
        if not np.any(mask):
            continue
            
        if Lb[i] == 0.0: # Isothermal Layer
            T[mask] = Tb[i]
            P[mask] = Pb[i] * np.exp(-G0 * (H[mask] - Hb[i]) / (R * Tb[i]))
        else:            # Gradient Layer
            T[mask] = Tb[i] + Lb[i] * (H[mask] - Hb[i])
            P[mask] = Pb[i] * (T[mask] / Tb[i]) ** (-G0 / (Lb[i] * R))
            
    # Calculate the rest of the properties
    RHO = P / (R * T)                     # Density [kg/m^3]
    A = np.sqrt(GAMMA_AIR * R * T)        # Speed of Sound [m/s]
    MU = (BETA * T**1.5) / (T + S)        # Dynamic Viscosity [kg/(m*s)]
    NU = MU / RHO                         # Kinematic Viscosity [m^2/s]
    
    # If a scalar was passed in, return scalars. Otherwise, return arrays. Headaches..
    if np.isscalar(h):
        return [RHO[0], T[0], P[0], A[0], NU[0], MU[0]]
    
    return [RHO, T, P, A, NU, MU]