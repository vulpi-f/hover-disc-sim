#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc — Core + Outer-Jet (plots core & seal together)
==========================================================

Modifiche principali per aderire alla procedura descritta nel file TeX e per
visualizzare nello stesso plot sia il core sia l'outer jet di sigillo:

1) Shooting sul core: come da TeX, il target è la sovrapressione media sul solo core.
2) Rim pressure conforme alla forma composita (termine statico + sigillo a momento limitato).
3) Figure uniche: ogni figura mostra tutto il dominio — core (0→R^-) e annulus (R^-→R_tot).
   - Quiver: frecce del campo di velocità nel core e frecce del getto nel sigillo (verticali);
     uso di due scale indipendenti (una per il core, una per il getto) con due quiverkey.
   - Colormap: mappa del modulo di velocità nel core (adimensionale) e mappa di U_j(z) (m/s)
     nel sigillo; due colorbar separate.
4) Solo quiver o colormap: niente iso-contour/streamplot; soltanto le due tipologie richieste.
"""
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------------------- Parameters ----------------------------
@dataclass
class Params:
    # Geometry
    R_tot: float = 0.50       # [m] total radius
    w: float = 0.05           # [m] leakage ring width; R^- = R_tot - w
    h: float = 0.20           # [m] hover height
    # Payload / ambient
    W: float = 400.0          # [N] payload
    p0: float = 101325.0      # [Pa] ambient
    T_inf: float = 293.0      # [K]
    # Fluid properties
    mu: float = 1.85e-5       # [Pa s]
    Rg: float = 287.0         # [J/(kg K)]
    # Curtain / rim pressure parameters
    b: float = 0.003          # [m] slot thickness at injection b0
    h_eff: float = None       # [m] effective curtain height H; if None, set to h (paper)
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_out: float = 40.0       # [m/s] initial guess for U0
    # Rim pressure model coefficients (from tex)
    m: float = 1.2            # [-] exponent for sealing weight ζ^m
    n_exp: float = 2.0        # [-] exponent for static build-up (1-ζ)^n
    C_p: float = 0.15         # [-] static coefficient (O(1e-1)); multiplies ρ_j U0^2
    Dm_min: float = 6.0       # [-] minimum deflection modulus for sealing
    s_spread: float = 0.07    # [-] jet spreading parameter s (≈0.06–0.09)
    DeltaP_cap: float = 5e5   # [Pa] cap for momentum term (set large to disable)
    # Stokes–Darcy closure
    alpha_r: float = 0.12     # [-] κ_r = α_r h^2
    alpha_z: float = 0.05     # [-] κ_z = α_z h^2
    # Leakage / power model
    beta: float = 0.15        # [-] fraction of curtain recirculated into cushion
    C_d: float = 0.62         # [-] discharge coefficient (orifice regime)
    Re_h_thr: float = 120.0   # [-] film→orifice threshold
    eta_in: float = 1.0       # [-] blower efficiency (inner)
    eta_out: float = 1.0      # [-] blower efficiency (outer)
    # Numerical grid
    Nr: int = 180
    Nz: int = 90
    # Elliptic solver
    max_iter: int = 6000
    tol_rel: float = 1e-4     # relative to p_c
    omega: float = 1.6        # SOR relaxation
    # Shooting on U_out
    shoot_max_iter: int = 25
    shoot_tol: float = 5e-3   # |p̄ - p_c_core| <= shoot_tol * p_c_core
    shoot_gain: float = 0.6   # proportional gain on U_out
    # IO
    SAVEFIG: int = 1
    FIGDIR: str = "../figs"

    def __post_init__(self):
        if self.h_eff is None or self.h_eff <= 0.0:
            self.h_eff = self.h  # H = h per paper

# ---------------------------- Helpers ----------------------------
def ensure_figdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def rho_of(P: np.ndarray, Rg: float, T: float) -> np.ndarray:
    return P / (Rg * T)

def phi_static(z: np.ndarray, H: float, n_exp: float) -> np.ndarray:
    zhat = np.clip(z / max(H, 1e-12), 0.0, 1.0)
    return (1.0 - zhat) ** n_exp

def phi_seal(z: np.ndarray, H: float, m: float) -> np.ndarray:
    zhat = np.clip(z / max(H, 1e-12), 0.0, 1.0)
    return zhat ** m

def jet_profile(U0: float, b0: float, z: np.ndarray, H: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple spreading: b(z)=b0*(1+s*ζ), U(z)=U0*b0/b(z)."""
    zhat = np.clip(z / max(H, 1e-12), 0.0, 1.0)
    b_z = b0 * (1.0 + s * zhat)
    U_z = U0 * (b0 / b_z)
    return U_z, b_z

def average_cushion_overpressure_core(P: np.ndarray, p0: float,
                                      r: np.ndarray) -> float:
    """
    Area-weighted mean of (P - p0) over the *core* (r ∈ [0, r_max]).
    r is 1D array over the core; P has shape (Nz, Nr_core).
    """
    p_bar_z = np.mean(P, axis=0)  # (Nr,)
    integrand = (p_bar_z - p0) * 2.0 * np.pi * r
    core_area = np.pi * (r[-1]**2)
    return np.trapezoid(integrand, r) / max(core_area, 1e-16)

# ---------------------------- Solver ----------------------------
def solve_core_once(pars: Params, U0: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray, float, float,
                               Dict[str, float], np.ndarray, np.ndarray]:
    """
    Single elliptic solve with given U0; returns fields and diagnostics.
    Now also returns the outer-jet profile U_z(z) and p_edge(z) for plotting.
    """
    R_minus = pars.R_tot - pars.w

    # References
    p_c = pars.W / (np.pi * pars.R_tot**2)  # cushion overpressure over total area
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Grid (core only)
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0

    # Rim pressure p_edge(z) per paper.tex
    H = pars.h_eff  # equals h by __post_init__
    U_z, b_z = jet_profile(U0, pars.b, z, H, pars.s_spread)

    # Static part: C_p * rho_j * U0^2
    Delta_p_static = pars.C_p * pars.rho_j * (U0**2)

    # Sealing momentum-limited part: min(ΔP_cap, ρ_j U(z)^2 b(z) / (H Dm_min))
    momentum_term = (pars.rho_j * (U_z**2) * b_z) / max(H * max(pars.Dm_min, 1e-12), 1e-12)
    Delta_p_seal_z = np.minimum(pars.DeltaP_cap, momentum_term)

    p_edge = (pars.p0
              + Delta_p_static * phi_static(z, H, pars.n_exp)
              + Delta_p_seal_z * phi_seal(z, H, pars.m))

    # Initial guess: quadratic radial interpolation towards rim
    P = np.zeros((pars.Nz, pars.Nr))
    r_safe = max(R_minus, 1e-12)
    rr = (r / r_safe)**2
    for i in range(pars.Nz):
        P[i, :] = p_center - (p_center - p_edge[i]) * rr

    def apply_bc(Pf):
        Pf[:, -1] = p_edge[:]         # r = R^-
        Pf[:, 0]  = Pf[:, 1]          # r = 0 symmetry
        Pf[0, :]  = Pf[1, :]          # z = 0 Neumann
        Pf[-1, :] = Pf[-2, :]         # z = h Neumann
        return Pf

    P = apply_bc(P)

    T = pars.T_inf
    tol_abs = pars.tol_rel * p_c  # relative to p_c (same scale)

    for _ in range(pars.max_iter):
        P_old = P.copy()
        rho = rho_of(P, pars.Rg, T)

        # SOR-Gauss–Seidel for variable-coefficient elliptic equation
        for i in range(1, pars.Nz-1):
            for j in range(1, pars.Nr-1):
                rj = r[j] if r[j] > 1e-12 else 0.5*dr
                rho_jp = 0.5*(rho[i, j] + rho[i, j+1])
                rho_jm = 0.5*(rho[i, j] + rho[i, j-1])
                rho_ip = 0.5*(rho[i+1, j] + rho[i, j])
                rho_im = 0.5*(rho[i-1, j] + rho[i, j])

                Ar_p = (rj + 0.5*dr) * rho_jp * kr / (dr**2)
                Ar_m = (rj - 0.5*dr) * rho_jm * kr / (dr**2)
                Az_p = rho_ip * kz / (dz**2)
                Az_m = rho_im * kz / (dz**2)

                denom = (Ar_p + Ar_m) / rj + (Az_p + Az_m)
                rhs = (Ar_p * P[i, j+1] + Ar_m * P[i, j-1]) / rj + (Az_p * P[i+1, j] + Az_m * P[i-1, j])

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

    # Non-dimensional fields (core)
    r_hat = (np.linspace(0.0, R_minus, pars.Nr)) / pars.R_tot
    z_hat = (np.linspace(0.0, pars.h, pars.Nz)) / pars.h
    R_hat, Z_hat = np.meshgrid(r_hat, z_hat, indexing='xy')

    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_hat = (P - pars.p0) / p_c
    dp_hat_drhat = (pars.R_tot / p_c) * dPdr
    dp_hat_dzhat = (pars.h / p_c) * dPdz
    u_hat = - dp_hat_drhat
    w_hat = - dp_hat_dzhat
    S = (pars.alpha_z / pars.alpha_r) * (pars.R_tot / pars.h)

    # Diagnostics for shooting & leakage
    pbar_core = average_cushion_overpressure_core(P, pars.p0, np.linspace(0.0, R_minus, pars.Nr))
    Delta_p_leak = float(np.mean(P[:, -1] - pars.p0))  # z-avg at rim

    diag = {
        "pbar_core": float(pbar_core),
        "Delta_p_leak": float(Delta_p_leak),
        "R_minus": float(R_minus),
        "p_c": float(p_c),
        "U_profile_mean": float(np.mean(U_z)),
    }

    # Return also U_z(z) [m/s] and p_edge(z) [Pa] for the annulus plots
    return (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, R_minus/pars.R_tot, diag, U_z, p_edge)

def solve_core_with_shooting(pars: Params):
    """Adjust U_out so that mean overpressure over the core equals p_c_core."""
    p_c = pars.W / (np.pi * pars.R_tot**2)
    R_minus = pars.R_tot - pars.w
    p_c_core = p_c * (pars.R_tot / max(R_minus, 1e-12))**2  # target over core area
    U0 = pars.U_out
    history = []
    final_fields = None

    for it in range(1, pars.shoot_max_iter+1):
        results = solve_core_once(pars, U0)
        (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag, U_z, p_edge) = results
        pbar_core = diag["pbar_core"]
        err = pbar_core - p_c_core
        rel_err = err / max(p_c_core, 1e-16)
        history.append((it, U0, pbar_core, rel_err))

        if abs(rel_err) <= pars.shoot_tol:
            final_fields = results
            break

        # Proportional gain on U0 (keep positive)
        U0 = max(0.5, U0 * (1.0 - pars.shoot_gain * rel_err))
        final_fields = results

    # Final diagnostics based on U0
    U_out_final = float(history[-1][1])
    ctrl = {
        "history": history,
        "U_out_final": U_out_final,
        "p_c": p_c,
        "p_c_core": p_c_core,
    }
    return final_fields, ctrl

# ---------------------------- Leakage / Flows / Power ----------------------------
def leakage_and_power(pars: Params, diag: Dict[str, float]) -> Dict[str, float]:
    Delta_p_leak = diag["Delta_p_leak"]
    R_minus = diag["R_minus"]
    rho_c = (pars.p0 + diag["p_c"]) / (pars.Rg * pars.T_inf)

    A_leak = 2.0 * np.pi * R_minus * pars.h
    qprime = (pars.h**3 / (12.0 * pars.mu * max(pars.w, 1e-12))) * Delta_p_leak
    Q_film = qprime * (2.0 * np.pi * R_minus)
    mdot_film = rho_c * Q_film

    U_h = Q_film / max(A_leak, 1e-16)
    Re_h = rho_c * U_h * pars.h / pars.mu

    Q_orif = pars.C_d * A_leak * np.sqrt(max(2.0 * Delta_p_leak / max(rho_c, 1e-16), 0.0))
    mdot_orif = rho_c * Q_orif

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
    U_out = ctrl["U_out_final"]
    # Curtain mass flow (annular slot): ṁ_out = ρ_j U_out (2π R_tot b0)
    mdot_out = pars.rho_j * U_out * (2.0 * np.pi * pars.R_tot * pars.b)
    # Recirculated fraction β reduces make-up flow demand
    mdot_in = leak["mdot_loss"] - pars.beta * mdot_out
    mdot_in = float(mdot_in)

    # Pneumatic powers
    # For outer jet, estimate aerodynamic power via momentum flux per unit circumference ~ ρ U^3 b * 2πR
    P_out = (pars.rho_j * (U_out**3) * pars.b) * (2.0 * np.pi * pars.R_tot)
    rho_c = leak["rho_c"]
    p_c = ctrl["p_c"]
    P_in = ( (mdot_in / max(rho_c, 1e-16)) * p_c )

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
    }

# ---------------------------- Plot Utils ----------------------------
def _style_axes(ax, Rminus_hat, title=""):
    ax.axvline(Rminus_hat, linestyle='-', linewidth=1.2)
    # shade the outer annulus up to r̂=1
    ax.axvspan(Rminus_hat, 1.0, alpha=0.15, hatch='//')
    ax.set_xlabel(r'$\hat r$'); ax.set_ylabel(r'$\hat z$')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_title(title)

def _outer_grid(Nz, Nr_outer, Rminus_hat):
    r_hat_outer = np.linspace(Rminus_hat, 1.0, max(Nr_outer, 2))
    z_hat = np.linspace(0.0, 1.0, Nz)
    R_out, Z_out = np.meshgrid(r_hat_outer, z_hat, indexing='xy')
    return R_out, Z_out

# ---------------------------- Plots (core + outer jet together) ----------------------------
def make_plots(pars: Params,
               r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat,
               U_z, p_edge):
    """
    Figure 1: Quiver con core + outer jet (due scale).
    Figure 2: Colormap con |V̂_iso| nel core + U_j(z) nel sigillo (due colorbar).
    Figure 3: Colormap della pressione adimensionale nel core + p_edge(z) nel sigillo.
    """
    ensure_figdir(pars.FIGDIR)

    # ---------- Common grids ----------
    Nr_core, Nz = R_hat.shape[1], Z_hat.shape[0]
    step_r = max(1, Nr_core // 30)
    step_z = max(1, Nz // 15)
    Rq = R_hat[::step_z, ::step_r]
    Zq = Z_hat[::step_z, ::step_r]
    uq = u_hat[::step_z, ::step_r]
    wq = w_hat[::step_z, ::step_r]

    # Outer annulus grid
    Nr_outer = max(2, Nr_core // 6)
    R_out, Z_out = _outer_grid(Nz, Nr_outer, Rminus_hat)

    # ---------- QU I V E R ----------
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    # core quiver (adimensionale)
    Q1 = ax.quiver(Rq, Zq, uq, S*wq, angles='xy', scale_units='xy', scale=None)
    # outer jet quiver (verticale, dimensionale in m/s, scala distinta)
    Vjet = np.tile(-U_z.reshape(-1, 1), (1, R_out.shape[1]))  # negativo: verso il basso
    jet_scale = np.nanmax(np.abs(U_z)) if np.nanmax(np.abs(U_z)) > 0 else 1.0
    Q2 = ax.quiver(R_out, Z_out, np.zeros_like(Vjet), Vjet/jet_scale, angles='xy',
                   scale_units='xy', scale=None, alpha=0.85)
    _style_axes(ax, Rminus_hat, title='Quiver: core (adim.) + outer jet (m/s, scala separata)')
    # Quiver keys (legenda delle scale)
    ax.quiverkey(Q1, 0.15, -0.04, 1.0, r'core: $(\hat u, S\hat w)$ = 1', labelpos='E')
    ax.quiverkey(Q2, 0.55, -0.04, 1.0, r'outer jet: $U_j$ = {:.1f} m/s'.format(jet_scale), labelpos='E')
    plt.tight_layout(rect=[0,0.05,1,1])
    if pars.SAVEFIG:
        fig.savefig(os.path.join(pars.FIGDIR, 'quiver_velocity.png'), dpi=200)

    # ---------- C O L O R M A P  (velocità) ----------
    Viso = np.sqrt(u_hat**2 + (S * w_hat)**2)  # adimensionale nel core
    fig2, ax2 = plt.subplots(figsize=(8.4, 4.8))
    # core colormap
    core_cmap = plt.get_cmap('viridis')
    core_norm = Normalize(vmin=np.nanmin(Viso), vmax=np.nanmax(Viso))
    m1 = ax2.pcolormesh(R_hat, Z_hat, Viso, shading='auto', cmap=core_cmap, norm=core_norm)
    # outer colormap: U_j(z) [m/s]
    Ujet_field = np.tile(U_z.reshape(-1,1), (1, R_out.shape[1]))
    jet_cmap = plt.get_cmap('plasma')
    jet_norm = Normalize(vmin=np.nanmin(U_z), vmax=np.nanmax(U_z))
    m2 = ax2.pcolormesh(R_out, Z_out, Ujet_field, shading='auto', cmap=jet_cmap, norm=jet_norm, alpha=0.9)

    _style_axes(ax2, Rminus_hat, title=r'Colormap: $|\hat V_{\mathrm{iso}}|$ (core) + $U_j$ (outer jet)')
    # due colorbar separate
    cbar1 = fig2.colorbar(m1, ax=ax2, pad=0.02)
    cbar1.set_label(r'$|\hat V_{\mathrm{iso}}|$ (core)')
    cbar2 = fig2.colorbar(m2, ax=ax2, pad=0.10)
    cbar2.set_label(r'$U_j$ [m/s] (outer)')
    plt.tight_layout()
    if pars.SAVEFIG:
        fig2.savefig(os.path.join(pars.FIGDIR, 'cmap_speed.png'), dpi=200)

    # ---------- C O L O R M A P  (pressione) ----------
    fig3, ax3 = plt.subplots(figsize=(8.4, 4.8))
    # core: p_hat
    m3 = ax3.pcolormesh(R_hat, Z_hat, p_hat, shading='auto', cmap='viridis')
    # outer: p_edge(z) (Pa), replicata radialmente
    pedge_field = np.tile(p_edge.reshape(-1,1), (1, R_out.shape[1]))
    m4 = ax3.pcolormesh(R_out, Z_out, pedge_field, shading='auto', cmap='magma')
    _style_axes(ax3, Rminus_hat, title=r'Colormap: $\hat p$ (core) + $p_\mathrm{edge}$ (outer)')
    cbar3 = fig3.colorbar(m3, ax=ax3, pad=0.02); cbar3.set_label(r'$\hat p$ (core)')
    cbar4 = fig3.colorbar(m4, ax=ax3, pad=0.10); cbar4.set_label(r'$p_\mathrm{edge}$ [Pa] (outer)')
    plt.tight_layout()
    if pars.SAVEFIG:
        fig3.savefig(os.path.join(pars.FIGDIR, 'cmap_pressure.png'), dpi=200)

    plt.show()

# ---------------------------- Main ----------------------------
def print_report(pars: Params, shoot_ctrl: dict, leak: dict, flows: dict):
    print(" Shooting (p̄_core → p_c_core) ===")
    hist = shoot_ctrl["history"]
    for it, U0, pbar_core, rel in hist:
        print(f"  it {it:2d}: U_out={U0:8.3f} m/s | p̄_core={pbar_core:9.3f} Pa | rel_err={rel:+8.4f}")
    print(f"  Final U_out = {shoot_ctrl['U_out_final']:.3f} m/s")
    print(f"  Targets: p_c = {shoot_ctrl['p_c']:.3f} Pa,  p_c_core = {shoot_ctrl['p_c_core']:.3f} Pa")

    print("=== Leakage Regime ===")
    print(f"  regime = {leak['regime']} | Re_h = {leak['Re_h']:.1f}")
    print(f"  ṁ_loss = {leak['mdot_loss']:.6f} kg/s  (film: {leak['mdot_film']:.6f}, orif: {leak['mdot_orif']:.6f})")

    print("=== Flows & Power ===")
    print(f"  ṁ_out = {flows['mdot_out']:.6f} kg/s")
    print(f"  ṁ_in  = {flows['mdot_in']:.6f} kg/s  (β={pars.beta})")
    print(f"  P_out (jet power)  = {flows['P_out']:.2f} W  | shaft ≈ {flows['P_out_shaft']:.2f} W (η_out={pars.eta_out})")
    print(f"  P_in  (pneum.)     = {flows['P_in']:.2f} W  | shaft ≈ {flows['P_in_shaft']:.2f} W (η_in={pars.eta_in})")

if __name__ == '__main__':
    pars = Params()

    # Solve core with shooting on U_out
    (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag, U_z, p_edge), shoot_ctrl = solve_core_with_shooting(pars)

    # Leakage model and power/flows
    leak = leakage_and_power(pars, diag)
    flows = curtain_and_powers(pars, shoot_ctrl, leak)

    # Report + plots
    print_report(pars, shoot_ctrl, leak, flows)
    make_plots(pars, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, U_z, p_edge)
