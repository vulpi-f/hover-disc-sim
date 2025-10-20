#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc — Core Solver + Shooting + Mass/Power (aligned with LaTeX)
======================================================================

What’s new vs. previous version:
1) Rim pressure profile now decays with height:  Φ(ζ) = (1-ζ)^m  (m ≥ 1).
2) Shooting loop adjusts curtain intensity (via Δp_edge ↔ U_out) to enforce
   the target mean cushion pressure  p̄ = p_c = W/(π R_tot^2).
3) Leakage model (film vs orifice) to estimate ṁ_loss, plus flows of
   the outer curtain and inner make-up, and pneumatic power figures.
4) Parameters/naming aligned with the TeX (lam→m; add β, C_d, thresholds).
5) Same non-dimensional plots saved to ../figs for LaTeX inclusion.

PDE model (core domain: r∈[0,R_minus], z∈[0,h]):
    (1/r) ∂r ( r ρ k_r ∂r p ) + ∂z ( ρ k_z ∂z p ) = 0,  with  ρ = p/(Rg T).
BCs:
    ∂p/∂r = 0 at r=0 ;  ∂p/∂z = 0 at z=0,h ;  p(R_minus,z) = p_edge(z).
Rim pressure (curtain-imposed):
    p_edge(z) = p0 + Δp_edge * Φ(z/h),   Φ(ζ) = (1-ζ)^m.
Curtain link:
    Δp_edge = C_t * (ρ_j U_out^2 b) / h_eff   ⇔   U_out = sqrt( Δp_edge h_eff / (C_t ρ_j b) ).

Non-dimensionalization for plots (as in paper):
    p̂ = (p - p0)/p_c,  with  p_c = W/(π R_tot^2);
    û = -∂_{r̂} p̂,  ŵ = -∂_{ẑ} p̂,  S = (α_z/α_r) (R_tot/h).

Outputs (printed):
    - Shooting summary and error
    - Regime for leakage (film/orifice)
    - ṁ_loss, ṁ_out, ṁ_in, U_out (final), Δp_edge (final)
    - P_out, P_in (pneumatic power)
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict


# ---------------------------- Parameters ----------------------------
@dataclass
class Params:
    # Geometry
    R_tot: float = 0.50       # [m] total radius
    w: float = 0.05           # [m] leakage ring width (radial gap outside core)
    h: float = 0.20           # [m] hover height
    # Payload / ambient
    W: float = 400.0          # [N] payload
    p0: float = 101325.0      # [Pa] ambient pressure
    T_inf: float = 293.0      # [K] (uniform here)
    # Fluid properties
    mu: float = 1.85e-5       # [Pa s]
    Rg: float = 287.0         # [J/(kg K)]
    # Curtain (turned jet → rim pressure)
    b: float = 0.003          # [m] slot thickness
    h_eff: float = 0.010      # [m] effective sealing height
    Ct: float = 1.0           # [-] curtain factor
    m: float = 1.2            # [-] profile exponent in Φ(ζ)=(1-ζ)^m  (was lam)
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_out: float = 40.0       # [m/s] initial guess for jet speed
    # Stokes–Darcy closure
    alpha_r: float = 0.12     # [-] k_r = α_r h^2
    alpha_z: float = 0.05     # [-] k_z = α_z h^2
    # Leakage / power model
    beta: float = 0.15        # [-] fraction of curtain recirculated into cushion
    C_d: float = 0.62         # [-] discharge coefficient (orifice regime)
    Re_h_thr: float = 120.0   # [-] threshold for switching film→orifice
    eta_in: float = 1.0       # [-] blower efficiency (inner) if converting to shaft power
    eta_out: float = 1.0      # [-] blower efficiency (outer)
    # Numerical grid
    Nr: int = 180
    Nz: int = 90
    # Solver (elliptic)
    max_iter: int = 6000
    tol_rel: float = 1e-4     # relative to p_c (for elliptic fix-point)
    omega: float = 1.6        # SOR relaxation
    # Shooting
    shoot_max_iter: int = 25
    shoot_tol: float = 5e-3   # relative on p_c (|p̄ - p_c| <= shoot_tol * p_c)
    shoot_gain: float = 0.6   # proportional gain updating Δp_edge
    # IO
    SAVEFIG: int = 1
    FIGDIR: str = "../figs"   # keep linkage with LaTeX


# ---------------------------- Helpers ----------------------------
def ensure_figdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def phi_down(z: np.ndarray, h: float, m: float) -> np.ndarray:
    """Φ(z/h) = (1 - z/h)^m clipped in [0,1]. Decays with z; larger near the floor."""
    zhat = np.clip(z / max(h, 1e-12), 0.0, 1.0)
    return (1.0 - zhat)**m


def rho_of(P: np.ndarray, Rg: float, T: float) -> np.ndarray:
    """Ideal gas with uniform T (consistent with TeX simplification)."""
    return P / (Rg * T)


def average_cushion_pressure(P: np.ndarray, p0: float,
                             r: np.ndarray, z: np.ndarray,
                             R_tot: float) -> float:
    """
    Compute areal average of overpressure over the *whole disc area* πR_tot^2,
    using the z-averaged pressure of the core solution up to r=R_minus, and
    assuming the (thin) rim contributes with its boundary value. Here we use the
    resolved core only (r ≤ R_minus) and renormalize by π R_tot^2 as in the TeX.
    """
    # z-average inside the core
    p_bar_z = np.mean(P, axis=0)  # shape (Nr,)
    # radial integral over core (0..R_minus)
    integrand = (p_bar_z - p0) * 2.0 * np.pi * r
    core_area = np.pi * (r[-1]**2)
    avg_over_core = np.trapezoid(integrand, r) / max(core_area, 1e-16)
    # Renormalize to total disc area (π R_tot^2)
    return avg_over_core * (core_area / (np.pi * R_tot**2))


# ---------------------------- Solver ----------------------------
def solve_core_once(pars: Params, Delta_p_edge: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray, float, float,
                               Dict[str, float]]:
    """Single elliptic solve with given Δp_edge; returns fields and diagnostics."""
    R_minus = pars.R_tot - pars.w

    # References
    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Grid (dimensional)
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0

    # Rim pressure profile p_edge(z) = p0 + Δp_edge * Φ(z/h)
    p_edge = pars.p0 + Delta_p_edge * phi_down(z, pars.h, pars.m)  # (Nz,)

    # Initial guess: radial interpolation from center to rim (quadratic in r)
    P = np.zeros((pars.Nz, pars.Nr))
    r_safe = max(R_minus, 1e-12)
    rr = (r / r_safe)**2
    for i in range(pars.Nz):
        P[i, :] = p_center - (p_center - p_edge[i]) * rr

    def apply_bc(Pf):
        # r = R_minus: Dirichlet
        Pf[:, -1] = p_edge[:]
        # r = 0: Neumann (mirror)
        Pf[:, 0] = Pf[:, 1]
        # z = 0, z = h: Neumann (mirror)
        Pf[0, :]  = Pf[1, :]
        Pf[-1, :] = Pf[-2, :]
        return Pf

    P = apply_bc(P)

    T = pars.T_inf
    tol_abs = pars.tol_rel * p_c

    for _ in range(pars.max_iter):
        P_old = P.copy()
        rho = rho_of(P, pars.Rg, T)

        # SOR Gauss–Seidel
        for i in range(1, pars.Nz-1):
            for j in range(1, pars.Nr-1):
                rj = r[j] if r[j] > 1e-12 else 0.5*dr
                # ρ at faces (arithmetic mean)
                rho_jp = 0.5*(rho[i, j] + rho[i, j+1])
                rho_jm = 0.5*(rho[i, j] + rho[i, j-1])
                rho_ip = 0.5*(rho[i+1, j] + rho[i, j])
                rho_im = 0.5*(rho[i-1, j] + rho[i, j])

                Ar_p = (rj + 0.5*dr) * rho_jp * kr / (dr**2)
                Ar_m = (rj - 0.5*dr) * rho_jm * kr / (dr**2)
                Az_p = rho_ip * kz / (dz**2)
                Az_m = rho_im * kz / (dz**2)

                denom = (Ar_p + Ar_m) / rj + (Az_p + Az_m)
                rhs = (Ar_p * P[i, j+1] + Ar_m * P[i, j-1]) / rj                     + (Az_p * P[i+1, j] + Az_m * P[i-1, j])

                P_new = rhs / (denom + 1e-30)
                P[i, j] = (1 - pars.omega) * P[i, j] + pars.omega * P_new

        P = apply_bc(P)
        err = np.max(np.abs(P - P_old))
        if err < tol_abs:
            break

    # Derivatives
    dPdr = np.zeros_like(P)
    dPdz = np.zeros_like(P)
    dPdr[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2*dr)
    dPdr[:, 0]    = (P[:, 1] - P[:, 0]) / dr
    dPdr[:, -1]   = (P[:, -1] - P[:, -2]) / dr
    dPdz[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2*dz)
    dPdz[0, :]    = (P[1, :] - P[0, :]) / dz
    dPdz[-1, :]   = (P[-1, :] - P[-2, :]) / dz

    # Non-dimensional fields
    r_hat = r / pars.R_tot
    z_hat = z / pars.h
    R_hat, Z_hat = np.meshgrid(r_hat, z_hat, indexing='xy')

    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_hat = (P - pars.p0) / p_c
    dp_hat_drhat = (pars.R_tot / p_c) * dPdr
    dp_hat_dzhat = (pars.h / p_c) * dPdz
    u_hat = - dp_hat_drhat
    w_hat = - dp_hat_dzhat
    S = (pars.alpha_z / pars.alpha_r) * (pars.R_tot / pars.h)

    # Diagnostics needed for shooting and leakage
    pbar = average_cushion_pressure(P, pars.p0, r, z, pars.R_tot)
    # Δp_leak estimated at rim from the solution (z-average on last column)
    Delta_p_leak = float(np.mean(P[:, -1] - pars.p0))

    diag = {
        "pbar": float(pbar),
        "Delta_p_leak": float(Delta_p_leak),
        "R_minus": float(R_minus),
        "p_c": float(p_c),
    }

    return (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, R_minus/pars.R_tot, diag)


def solve_core_with_shooting(pars: Params):
    """Outer loop to adjust curtain intensity until p̄ ≈ p_c."""
    p_c = pars.W / (np.pi * pars.R_tot**2)
    # Start from U_out guess → Δp_edge
    Delta_p_edge = pars.Ct * (pars.rho_j * pars.U_out**2 * pars.b) / pars.h_eff

    history = []
    final_fields = None
    for it in range(1, pars.shoot_max_iter+1):
        results = solve_core_once(pars, Delta_p_edge)
        (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag) = results
        pbar = diag["pbar"]
        err = pbar - p_c
        rel_err = err / max(p_c, 1e-16)
        history.append((it, Delta_p_edge, pbar, rel_err))

        # Stop?
        if abs(rel_err) <= pars.shoot_tol:
            final_fields = results
            break

        # Proportional update on Δp_edge (bounded positive)
        Delta_p_edge = max(1e-1, Delta_p_edge * (1.0 - pars.shoot_gain * rel_err))

        final_fields = results  # keep latest

    # Map back to U_out from final Δp_edge
    Delta_p_edge_final = history[-1][1]
    U_out_final = np.sqrt(max(Delta_p_edge_final, 0.0) * pars.h_eff / (pars.Ct * pars.rho_j * pars.b))

    return final_fields, {
        "history": history,
        "Delta_p_edge": float(Delta_p_edge_final),
        "U_out_final": float(U_out_final),
    }


# ---------------------------- Leakage / Flows / Power ----------------------------
def leakage_and_power(pars: Params, diag: Dict[str, float]) -> Dict[str, float]:
    """
    Estimate leakage mass flow and related power numbers.
    Uses Δp_leak from the solution (z-avg at rim). Selects regime by Re_h.
    """
    Delta_p_leak = diag["Delta_p_leak"]
    R_minus = diag["R_minus"]
    # Gas density in cushion (use mean pressure p0 + p_c at T_inf)
    rho_c = (pars.p0 + diag["p_c"]) / (pars.Rg * pars.T_inf)

    # Geometric areas
    A_leak = 2.0 * np.pi * R_minus * pars.h  # side annulus opening
    # ---- Film (laminar slot) estimate ----
    # volumetric flow per unit length along circumference q' ≈ h^3/(12 μ w) Δp
    qprime = (pars.h**3 / (12.0 * pars.mu * max(pars.w, 1e-12))) * Delta_p_leak  # [m^2/s per m] ≡ [m^3/(s·m)]
    Q_film = qprime * (2.0 * np.pi * R_minus)   # [m^3/s]
    mdot_film = rho_c * Q_film

    # Provisional velocity and Re_h
    U_h = Q_film / max(A_leak, 1e-16)
    Re_h = rho_c * U_h * pars.h / pars.mu

    # ---- Orifice (inertial) estimate ----
    Q_orif = pars.C_d * A_leak * np.sqrt(max(2.0 * Delta_p_leak / max(rho_c, 1e-16), 0.0))
    mdot_orif = rho_c * Q_orif

    # Select regime
    if Re_h >= pars.Re_h_thr:
        regime = "orifice"
        mdot_loss = mdot_orif
    else:
        regime = "film"
        mdot_loss = mdot_film

    return {
        "rho_c": float(rho_c),
        "A_leak": float(A_leak),
        "Q_film": float(Q_film),
        "Q_orif": float(Q_orif),
        "mdot_film": float(mdot_film),
        "mdot_orif": float(mdot_orif),
        "mdot_loss": float(mdot_loss),
        "Re_h": float(Re_h),
        "regime": regime,
    }


def curtain_and_powers(pars: Params, ctrl: Dict[str, float], leak: Dict[str, float]) -> Dict[str, float]:
    """Curtain outflow, make-up inflow, and pneumatic power figures."""
    U_out = ctrl["U_out_final"]
    Delta_p_edge = ctrl["Delta_p_edge"]
    # Curtain mass flow
    mdot_out = pars.rho_j * U_out * (2.0 * np.pi * pars.R_tot * pars.b)
    # Recirculation fraction
    mdot_in = leak["mdot_loss"] - pars.beta * mdot_out

    # Pneumatic powers (volumetric flow × Δp); convert from mdot by dividing by density
    P_out = (mdot_out / max(pars.rho_j, 1e-16)) * Delta_p_edge
    P_in = (mdot_in / max(leak["rho_c"], 1e-16)) * ( (pars.W / (np.pi * pars.R_tot**2)) )

    # Shaft power (if η<1)
    P_out_shaft = P_out / max(pars.eta_out, 1e-16)
    P_in_shaft  = P_in  / max(pars.eta_in,  1e-16)

    return {
        "mdot_out": float(mdot_out),
        "mdot_in": float(mdot_in),
        "P_out": float(P_out),
        "P_in": float(P_in),
        "P_out_shaft": float(P_out_shaft),
        "P_in_shaft": float(P_in_shaft),
        "U_out_final": float(U_out),
        "Delta_p_edge": float(Delta_p_edge),
    }


# ---------------------------- Plots ----------------------------
def make_plots(pars: Params, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat):
    ensure_figdir(pars.FIGDIR)
    # Quiver downsample
    Nr, Nz = R_hat.shape[1], Z_hat.shape[0]
    step_r = max(1, Nr // 30)
    step_z = max(1, Nz // 15)
    Rq = R_hat[::step_z, ::step_r]
    Zq = Z_hat[::step_z, ::step_r]
    uq = u_hat[::step_z, ::step_r]
    wq = w_hat[::step_z, ::step_r]

    # Isotropic magnitude for cmap_speed.png
    Viso = np.sqrt(u_hat**2 + (S * w_hat)**2)

    # Utility to draw vertical line at r̂ = Rminus_hat
    def add_Rminus_line():
        plt.plot([Rminus_hat, Rminus_hat], [0.0, 1.0])

    # 1) Quiver of (û, S ŵ)
    plt.figure(figsize=(7, 4.5))
    plt.quiver(Rq, Zq, uq, S*wq, angles='xy', scale_units='xy', scale=None)
    add_Rminus_line()
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title('Velocity field (quiver) — non-dimensional')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'quiver_velocity.png'), dpi=180)

    # 2) |V̂_iso|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(R_hat, Z_hat, Viso, shading='auto')
    add_Rminus_line()
    plt.colorbar(label=r'$|\hat V_{\mathrm{iso}}|$')
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title(r'Speed magnitude $|\hat V_{\mathrm{iso}}|$ (non-dimensional)')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'cmap_speed.png'), dpi=180)

    # 3) |û|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(R_hat, Z_hat, np.abs(u_hat), shading='auto')
    add_Rminus_line()
    plt.colorbar(label=r'$|\hat u|$')
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title(r'Radial component $|\hat u|$ (non-dimensional)')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'cmap_ur.png'), dpi=180)

    # 4) |ŵ|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(R_hat, Z_hat, np.abs(w_hat), shading='auto')
    add_Rminus_line()
    plt.colorbar(label=r'$|\hat w|$')
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title(r'Axial component $|\hat w|$ (non-dimensional)')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'cmap_uz.png'), dpi=180)

    # 5) p̂
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(R_hat, Z_hat, p_hat, shading='auto')
    add_Rminus_line()
    plt.colorbar(label=r'$\hat p$')
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title(r'Pressure $\hat p$ (non-dimensional)')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'cmap_pressure.png'), dpi=180)

    plt.show()


# ---------------------------- Main ----------------------------
def print_report(pars: Params, shoot_ctrl: dict, leak: dict, flows: dict):
    print(" Shooting (p̄ → p_c) ===")
    hist = shoot_ctrl["history"]
    for it, Dp, pbar, rel in hist:
        print(f"  it {it:2d}: Δp_edge={Dp:10.3f} Pa | p̄={pbar:9.3f} Pa | rel_err={rel:+8.4f}")
    print(f"  Final U_out = {shoot_ctrl['U_out_final']:.3f} m/s")
    print(f"  Final Δp_edge = {shoot_ctrl['Delta_p_edge']:.3f} Pa")

    print("=== Leakage Regime ===")
    print(f"  regime = {leak['regime']} | Re_h = {leak['Re_h']:.1f}")
    print(f"  ṁ_loss = {leak['mdot_loss']:.6f} kg/s  (film: {leak['mdot_film']:.6f}, orif: {leak['mdot_orif']:.6f})")

    print("=== Flows & Power ===")
    print(f"  ṁ_out = {flows['mdot_out']:.6f} kg/s")
    print(f"  ṁ_in  = {flows['mdot_in']:.6f} kg/s  (β={pars.beta})")
    print(f"  P_out (pneum.)  = {flows['P_out']:.2f} W  | shaft ≈ {flows['P_out_shaft']:.2f} W (η_out={pars.eta_out})")
    print(f"  P_in  (pneum.)  = {flows['P_in']:.2f} W  | shaft ≈ {flows['P_in_shaft']:.2f} W (η_in={pars.eta_in})")


if __name__ == '__main__':
    pars = Params()

    # Elliptic core with shooting on Δp_edge via U_out
    (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag), shoot_ctrl = solve_core_with_shooting(pars)

    # Leakage model and power/flows
    leak = leakage_and_power(pars, diag)
    flows = curtain_and_powers(pars, shoot_ctrl, leak)

    # Report + plots
    print_report(pars, shoot_ctrl, leak, flows)
    make_plots(pars, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat)
