"""
cushion_model.py — Axisymmetric 2D cushion + annular seal model.

Provides:
- Cushion pressure for a target load (axisymmetric disk).
- Two leakage models through the annular seal:
  (A) viscous lubrication; (B) orifice/inertial with discharge coefficient Cd.
- Functions to compute Q (m^3/s), ideal power (W), p(r), u_bar(r).

Author: You
License: MIT
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    R: float = 1.0        # disk radius [m]
    h_c: float = 0.20     # central clearance [m]
    w: float = 0.05       # seal width [m]
    h_eff: float = 0.03   # effective seal gap [m]
    m_load: float = 40.0  # useful load [kg]
    rho: float = 1.20     # air density [kg/m^3]
    mu: float = 1.8e-5    # air dynamic viscosity [Pa*s]
    g: float = 9.81       # gravity [m/s^2]
    Cd: float = 0.65      # discharge coefficient for orifice model

def cushion_pressure(p: Params) -> float:
    """Required cushion pressure to support the load."""
    A = np.pi * p.R**2
    return (p.m_load * p.g) / A

def leakage_viscous(p: Params, p_c: float) -> dict:
    """Lubrication approximation in annulus [R-w, R]."""
    ln_factor = np.log(p.R / (p.R - p.w))
    Q = (np.pi * p.h_eff**3 * p_c) / (6.0 * p.mu * ln_factor)
    P = p_c * Q
    r = np.linspace(p.R - p.w, p.R, 400)
    # pressure drop (logarithmic)
    p_r = p_c - (6.0 * p.mu * Q) / (np.pi * p.h_eff**3) * np.log(r / (p.R - p.w))
    # mean radial velocity
    u_bar = Q / (2.0 * np.pi * r * p.h_eff)
    Re = p.rho * u_bar[-1] * p.h_eff / p.mu
    return dict(Q=Q, P=P, r=r, p_r=p_r, u_bar=u_bar, Re=Re)

def leakage_orifice(p: Params, p_c: float) -> dict:
    """Orifice-like annular gap with discharge coefficient Cd."""
    A_eff = 2.0 * np.pi * p.R * p.h_eff
    V = np.sqrt(2.0 * p_c / p.rho)
    Q = p.Cd * A_eff * V
    P = p_c * Q
    r = np.linspace(p.R - p.w, p.R, 400)
    # velocity profile assuming constant Q in the seal
    u_bar = Q / (2.0 * np.pi * r * p.h_eff)
    # piecewise pressure: flat p_c then sharp drop at r=R
    p_r = np.piecewise(r, [r < p.R, r >= p.R], [p_c, 0.0])
    Re = p.rho * V * p.h_eff / p.mu
    return dict(Q=Q, P=P, r=r, p_r=p_r, u_bar=u_bar, Re=Re)
