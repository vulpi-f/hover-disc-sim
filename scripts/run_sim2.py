#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc — Momentum-Based Jet-Induced Sealing (per PDF model)
================================================================

Implements the steady, incompressible, axisymmetric model structured in three
coupled regions:
  1) Core thin film (lubrication) on 0 ≤ r ≤ R−
  2) Vertical annular jet and turning pedestal near r ≈ r_t
  3) Radial wall-jet after impact (integral equations)

Unknowns are solved so that: (i) edge pressure consistency is satisfied via the
turning pedestal, (ii) global mass holds, and (iii) lift balances W.

This script reproduces the same figure set as the user's reference (all non-dimensional):
  1) Quiver: core (û, S ŵ) + outer jet (−Û_j)
  2) Colormap: |V̂_iso| (core) + Û_j (outer)
  3) Colormap: p̂ (core) + p̂_edge (outer)  [p_c scale]
  4) Colormap: û (core) + û_outer≈0 (outer)
  5) Colormap: ŵ (core) + ŵ_outer = −Û_j (outer)

Notes:
- Core is solved from the lubrication equation (thin-film) with Neumann at r=0
  and Dirichlet at r=R− equal to p_edge, which comes from the jet turning
  (pedestal) momentum balance.
- The wall-jet is advanced with integral balances for volume and momentum with a
  friction law Cf(Re_δ) and a simple growth law for δ(r). Outside the narrow
  turning region, dp/dr≈0 in the wall-jet ODEs.
- The jet speed Uj is adjusted so that the lift ∫(p_c−p0)2πrdr matches W.

This is a practical, robust implementation meant for design iteration.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# ---------------------------- Parameters ----------------------------
@dataclass
class Params:
    # Geometry
    R_tot: float = 0.50     # [m] total radius (disc)
    b0: float = 0.05        # [m] outer annulus width; R_minus = R_tot - b0
    h: float = 0.20         # [m] hover gap

    # Ambient / load
    W: float = 400.0        # [N] external load to support
    p0: float = 101325.0    # [Pa] ambient pressure

    # Fluid
    rho: float = 1.20       # [kg/m^3] (incompressible here)
    mu: float = 1.85e-5     # [Pa s]
    nu: float = 1.85e-5/1.20  # [m^2/s]

    # Jet & turning
    Uj_init: float = 40.0   # [m/s] initial jet speed guess
    rt_offset: float = 0.5  # [-] rt = R_minus - rt_offset*b0
    K_turn: float = 1.2     # [-] denominator (1+K_turn) for jet power & losses
    sigma_fac: float = 0.6  # [-] pedestal width σ = sigma_fac * b0
    loss_frac: float = 0.0  # [-] ΔM_z / (ṁ_j U_j); set 0 to enforce pure momentum→pedestal

    # Wall-jet closure
    E: float = 0.0          # [-] entrainment (0 = none)
    Cf0: float = 0.030      # [-] coefficient in Cf = Cf0 / Re_δ^a (clamped)
    Cf_exp: float = 1/7     # [-] exponent a
    Cf_min: float = 3e-3    # [-] floor for Cf
    Cf_max: float = 3e-2    # [-] cap for Cf
    k_delta: float = 0.12   # [-] growth law δ' = k_delta (δ/r)

    # Numerical grids
    Nr: int = 240           # core radial nodes
    refine_edge: float = 6.0  # edge clustering strength (≥1)
    Nr_outer: int = 40      # outer annulus plot grid (for overlays)

    # Nonlinear control (shooting on Uj)
    max_iter: int = 25
    tol_rel_L: float = 5e-3   # |L-W|/W ≤ tol
    relax: float = 0.6

    # Plot / IO
    SAVEFIG: int = 1
    FIGDIR: str = "../figs"

    # Small numbers
    eps: float = 1e-12

    def __post_init__(self):
        self.R_minus = self.R_tot - self.b0

# ---------------------------- Helpers ----------------------------
def ensure_figdir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def core_grid(pars: Params) -> np.ndarray:
    """Edge-refined radial grid on [0, R_minus]."""
    xi = np.linspace(0.0, 1.0, pars.Nr)
    beta = pars.refine_edge
    s = xi**beta
    r = pars.R_minus * s
    return r

def assemble_core_matrix(pars: Params, r: np.ndarray, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-volume discretization of (1/r) d/dr[ - r K dp/dr ] = 0 with K = rho h^3/(12 mu).
    Neumann at r=0 (dp/dr=0), Dirichlet at r=R_minus: p=p_edge.
    """
    N = len(r)
    dr = np.diff(r)
    K = pars.rho * pars.h**3 / (12.0 * pars.mu)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # center (Neumann: dp/dr = 0) -> mirrored ghost gives p1 = p0
    A[0, 0] = 1.0
    A[0, 1] = -1.0
    b[0] = 0.0

    # interior control volumes
    for i in range(1, N-1):
        r_w = 0.5*(r[i-1] + r[i])
        r_e = 0.5*(r[i] + r[i+1])
        dw = r[i] - r_w
        de = r_e - r[i]

        Aw = (r_w * K) / max(dw, pars.eps)
        Ae = (r_e * K) / max(de, pars.eps)

        A[i, i-1] =  Aw / r[i]
        A[i, i]   = -(Aw + Ae) / r[i]
        A[i, i+1] =  Ae / r[i]

    # edge (Dirichlet)
    A[-1, -1] = 1.0
    b[-1] = p_edge
    return A, b

def solve_core_pressure(pars: Params, p_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    r = core_grid(pars)
    A, b = assemble_core_matrix(pars, r, p_edge)
    p = np.linalg.solve(A, b)
    return r, p

def lift_from_pressure(r: np.ndarray, p: np.ndarray, p0: float) -> float:
    integrand = (p - p0) * 2.0 * np.pi * r
    return float(np.trapz(integrand, r))

def leakage_from_edge(pars: Params, r: np.ndarray, p: np.ndarray) -> float:
    # From ṁleak = - π rho h^3 /(6 mu) * R_minus * (dp/dr)|_{R-}
    dpdr = np.gradient(p, r, edge_order=2)
    dpdr_edge = float(dpdr[-1])
    mdot_leak = - np.pi * pars.rho * pars.h**3 / (6.0 * pars.mu) * pars.R_minus * dpdr_edge
    return float(max(mdot_leak, 0.0))

# ---------------------------- Wall-jet (integral) ----------------------------

def cf_from_Re_delta(pars: Params, Re_delta: float) -> float:
    Cf = pars.Cf0 / max(Re_delta, 1.0)**(pars.Cf_exp)
    return float(np.clip(Cf, pars.Cf_min, pars.Cf_max))


def walljet_integrate(pars: Params, Uj: float, rt: float, r_end: float, Nr: int = 600):
    """Advance q(r), m(r), δ(r) for the wall-jet with dq/dr = 2π r E U_w and
    dm/dr = - π r Cf U_w^2 / (cl), where U_w = m/(rho q) and δ = rho q^2 / m.
    Pressure gradient term is neglected outside the narrow turning region.
    """
    rho, nu = pars.rho, pars.nu

    r = np.linspace(rt, r_end, Nr)
    dr = r[1]-r[0] if Nr > 1 else 1.0

    m_j = rho * Uj * pars.b0         # per unit circumference (kg/(s·m))
    q0 = m_j / (2.0 * np.pi * rt * rho)
    m0 = m_j * Uj / (2.0 * np.pi * rt)

    q = np.zeros_like(r)
    m = np.zeros_like(r)
    delta = np.zeros_like(r)

    q[0], m[0] = q0, m0
    delta[0] = rho * (q0*q0) / max(m0, pars.eps)

    for i in range(1, Nr):
        Uw = m[i-1] / max(rho * q[i-1], pars.eps)
        Re_delta = q[i-1] / max(nu, pars.eps)
        Cf = cf_from_Re_delta(pars, Re_delta)

        # dq/dr = 2π r E Uw
        dq = 2.0 * np.pi * r[i-1] * pars.E * Uw * dr
        q[i] = max(q[i-1] + dq, pars.eps)

        # dm/dr = - τ_w 2π r / rho = - (0.5 rho Cf Uw^2) 2π r / rho
        dm = - np.pi * r[i-1] * Cf * Uw*Uw * dr
        m[i] = max(m[i-1] + dm, pars.eps)

        delta[i] = rho * (q[i]*q[i]) / max(m[i], pars.eps)

    return r, q, m, delta

# ---------------------------- Pedestal / edge pressure ----------------------------

def pedestal_profile(pars: Params, Uj: float, r: np.ndarray, rt: float) -> np.ndarray:
    """Gaussian pedestal p_ped(r) whose line integral equals (1 - loss_frac) ṁ_j U_j.
    p_edge = p0 + p_ped(R_minus). Width σ = sigma_fac b0.
    """
    rho = pars.rho
    m_j = rho * Uj * pars.b0
    Mz_avail = (1.0 - pars.loss_frac) * (m_j * Uj)  # units: N per unit circumference

    sigma = max(pars.sigma_fac * pars.b0, 1e-6)
    # Normalize Gaussian so that ∫_{-∞}^{∞} A exp(-(r-rt)^2/(2σ^2)) dr = Mz_avail
    A = Mz_avail / (np.sqrt(2.0*np.pi) * sigma)
    p_ped = A * np.exp(-0.5 * ((r - rt)/sigma)**2)
    return p_ped

# ---------------------------- Solver (outer loop on Uj) ----------------------------

def solve_once(pars: Params, Uj: float) -> Dict[str, object]:
    # Turning radius and pedestal
    rt = max(pars.R_minus - pars.rt_offset*pars.b0, 0.7*pars.R_minus)
    rline = core_grid(pars)
    p_ped = pedestal_profile(pars, Uj, rline, rt)

    # Edge pressure from pedestal value at R_minus
    p_edge = pars.p0 + float(np.interp(pars.R_minus, rline, p_ped))

    # Core solve
    r_core, p_core = solve_core_pressure(pars, p_edge)
    L = lift_from_pressure(r_core, p_core, pars.p0)
    mdot_leak = leakage_from_edge(pars, r_core, p_core)

    # Wall-jet (for diagnostics and figures); integrate from rt to R_tot
    r_wj, q_wj, m_wj, delta_wj = walljet_integrate(pars, Uj, rt, pars.R_tot)

    # Pack
    out = {
        "Uj": Uj,
        "rt": rt,
        "p_edge": p_edge,
        "r_core": r_core,
        "p_core": p_core,
        "L": L,
        "mdot_leak": mdot_leak,
        "r_wj": r_wj,
        "q_wj": q_wj,
        "m_wj": m_wj,
        "delta_wj": delta_wj,
    }
    return out


def solve_with_shooting(pars: Params) -> Tuple[Dict[str, object], Dict[str, object]]:
    Uj = max(0.5, pars.Uj_init)
    history = []
    final = None

    for it in range(1, pars.max_iter+1):
        res = solve_once(pars, Uj)
        L = res["L"]
        err = L - pars.W
        rel = err / max(pars.W, 1e-12)
        history.append((it, Uj, L, rel))

        # Convergence
        if abs(rel) <= pars.tol_rel_L:
            final = res
            break

        # Simple proportional update on Uj using sign of error
        Uj = max(0.5, Uj * (1.0 - pars.relax * rel))
        final = res

    ctrl = {"history": history, "Uj_final": final["Uj"]}
    return final, ctrl

# ---------------------------- Diagnostic numbers ----------------------------

def nondimensional_fields(pars: Params, sol: Dict[str, object]):
    r = sol["r_core"]
    p = sol["p_core"]
    p_c = pars.W / (np.pi * pars.R_minus**2)

    # Thin-film averaged radial velocity ū(r)
    dpdr = np.gradient(p, r, edge_order=2)
    ubar = - pars.h**2 / (12.0 * pars.mu) * dpdr

    # Build pseudo 2D fields by assuming weak z-variation (for visualization)
    Nr = 180
    Nz = 90
    r_hat = np.linspace(0.0, pars.R_minus/pars.R_tot, Nr)
    z_hat = np.linspace(0.0, 1.0, Nz)
    R_hat, Z_hat = np.meshgrid(r_hat, z_hat, indexing='xy')

    # Interpolate ū to plotting grid and shape along z with a parabolic profile
    ubar_grid = np.interp(R_hat*pars.R_tot, r, ubar)
    shape_z = 6.0 * Z_hat * (1.0 - Z_hat)  # Poiseuille-like shape for display
    u_field = (ubar_grid * shape_z)

    # w from continuity (scaled for display): 1/r d(r h ū)/dr = ρ h ∂w/∂z → ŵ ∼ integral of source
    term = np.gradient(pars.rho * pars.h * ubar_grid * R_hat, r_hat, axis=1, edge_order=2)
    w_field = - (term / max(pars.rho*pars.h, 1e-12))

    p_grid = np.interp(R_hat*pars.R_tot, r, p)

    # Non-dimensionalization
    p_hat = (p_grid - pars.p0) / max(p_c, 1e-30)
    # use Uj_final for scaling
    u_hat = u_field / max(sol["Uj"], 1e-12)
    w_hat = w_field / max(sol["Uj"], 1e-12)

    # isotropic scaling factor like in the reference script
    S = (pars.R_tot / max(pars.h, 1e-12))

    # Vorticity surrogate ωθ ≈ ∂ŵ/∂r̂ − ∂û/∂ẑ
    drh = r_hat[1]-r_hat[0]
    dzh = z_hat[1]-z_hat[0]
    dwhat_drhat = np.zeros_like(w_hat)
    dwhat_drhat[:, 1:-1] = (w_hat[:, 2:] - w_hat[:, :-2]) / (2.0 * drh)
    dwhat_drhat[:, 0]    = (w_hat[:, 1] - w_hat[:, 0]) / drh
    dwhat_drhat[:, -1]   = (w_hat[:, -1] - w_hat[:, -2]) / drh

    duhat_dzhat = np.zeros_like(u_hat)
    duhat_dzhat[1:-1, :] = (u_hat[2:, :] - u_hat[:-2, :]) / (2.0 * dzh)
    duhat_dzhat[0, :]    = (u_hat[1, :] - u_hat[0, :]) / dzh
    duhat_dzhat[-1, :]   = (u_hat[-1, :] - u_hat[-2, :]) / dzh

    omega_hat = dwhat_drhat - duhat_dzhat

    # Outer overlays: jet magnitude along z in annulus (non-dimensional)
    z_plot = np.linspace(0.0, 1.0, Nz)
    Uj_profile = sol["Uj"] * z_plot**2  # simple accelerating vertical profile (for plot only)
    Uhat_j = Uj_profile / max(sol["Uj"], 1e-12)

    return (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, omega_hat, S,
            Uhat_j)

# ---------------------------- Plotting ----------------------------

def _style_axes(ax, Rminus_hat: float, title: str = ""):
    ax.axvline(Rminus_hat, linestyle='-', linewidth=1.2)
    ax.axvspan(Rminus_hat, 1.0, alpha=0.15, hatch='//')
    ax.set_xlabel(r'$\hat r$')
    ax.set_ylabel(r'$\hat z$')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_title(title)


def _outer_grid(Nz, Nr_outer, Rminus_hat):
    r_hat_outer = np.linspace(Rminus_hat, 1.0, max(Nr_outer, 2))
    z_hat = np.linspace(0.0, 1.0, Nz)
    R_out, Z_out = np.meshgrid(r_hat_outer, z_hat, indexing='xy')
    return R_out, Z_out


def make_plots(pars: Params, sol: Dict[str, object]):
    ensure_figdir(pars.FIGDIR)

    (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, omega_hat, S, Uhat_j) = nondimensional_fields(pars, sol)

    Rminus_hat = pars.R_minus / pars.R_tot
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

    # 1) Quiver: core (û, S ŵ) + outer (−Û_j)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    Q1 = ax.quiver(Rq, Zq, uq, S*wq, angles='xy', scale_units='xy', scale=None)
    Vjet_hat = np.tile(-Uhat_j.reshape(-1, 1), (1, R_out.shape[1]))
    Q2 = ax.quiver(R_out, Z_out, np.zeros_like(Vjet_hat), Vjet_hat, angles='xy',
                   scale_units='xy', scale=None, alpha=0.85)
    _style_axes(ax, Rminus_hat, title='Quiver: core $(\hat u, S\hat w)$ + outer jet $-\hat U_j$')
    ax.quiverkey(Q1, 0.15, -0.04, 1.0, r'core: $(\hat u, S\hat w)=1$', labelpos='E')
    ax.quiverkey(Q2, 0.55, -0.04, 1.0, r'outer: $\hat U_j=1$', labelpos='E')
    plt.tight_layout(rect=[0,0.05,1,1])
    if pars.SAVEFIG:
        fig.savefig(os.path.join(pars.FIGDIR, 'quiver_velocity.png'), dpi=200)

    # 2) Colormap: |V̂_iso|
    Viso = np.sqrt(u_hat**2 + (S * w_hat)**2)
    fig2, ax2 = plt.subplots(figsize=(8.4, 4.8))
    m1 = ax2.pcolormesh(R_hat, Z_hat, Viso, shading='auto')
    Uhat_field = np.tile(Uhat_j.reshape(-1,1), (1, R_out.shape[1]))
    m2 = ax2.pcolormesh(R_out, Z_out, Uhat_field, shading='auto', alpha=0.9)
    _style_axes(ax2, Rminus_hat, title=r'Colormap: $|\hat V_{\mathrm{iso}}|$ (core) + $\hat U_j$ (outer)')
    cbar1 = fig2.colorbar(m1, ax=ax2, pad=0.02); cbar1.set_label(r'$|\hat V_{\mathrm{iso}}|$ (core)')
    cbar2 = fig2.colorbar(m2, ax=ax2, pad=0.10); cbar2.set_label(r'$\hat U_j$ (outer)')
    plt.tight_layout()
    if pars.SAVEFIG:
        fig2.savefig(os.path.join(pars.FIGDIR, 'cmap_speed.png'), dpi=200)

    # 3) Colormap: p̂ (core) + p̂_edge (outer)
    fig3, ax3 = plt.subplots(figsize=(8.4, 4.8))
    p_hat_edge = np.full((Z_out.shape[0], R_out.shape[1]), (sol['p_edge']-pars.p0)/(pars.W/(np.pi*pars.R_minus**2)))

    vmin = min(p_hat.min(), p_hat_edge.min())
    vmax = max(p_hat.max(), p_hat_edge.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    m3 = ax3.pcolormesh(R_hat, Z_hat, p_hat, shading='auto', cmap=cmap, norm=norm)
    m4 = ax3.pcolormesh(R_out, Z_out, p_hat_edge, shading='auto', cmap=cmap, norm=norm, alpha=0.9)

    _style_axes(ax3, Rminus_hat, title=r'Colormap: $\hat p$ (core) + $\hat p_{\mathrm{edge}}$ (outer, $p_c$)')
    cbar3 = fig3.colorbar(m3, ax=ax3, pad=0.02)
    cbar3.set_label(r'$\hat p$ (core, $p_c$)')
    plt.tight_layout()
    if pars.SAVEFIG:
        fig3.savefig(os.path.join(pars.FIGDIR, 'cmap_pressure.png'), dpi=200)

    # 4) Colormap: û with sign (core) + û_outer≈0 (outer)
    figu, axu = plt.subplots(figsize=(8.4, 4.8))
    umax = float(np.nanmax(np.abs(u_hat))) or 1.0
    umax = umax if umax > 0 else 1.0
    m_uc = axu.pcolormesh(R_hat, Z_hat, u_hat, shading='auto',
                          cmap='coolwarm', norm=TwoSlopeNorm(vmin=-umax, vcenter=0.0, vmax=umax))
    U_r_outer_hat = np.zeros((Z_out.shape[0], R_out.shape[1]))
    m_uo = axu.pcolormesh(R_out, Z_out, U_r_outer_hat, shading='auto',
                          cmap='coolwarm', norm=TwoSlopeNorm(vmin=-1e-9, vcenter=0.0, vmax=1e-9), alpha=0.9)
    _style_axes(axu, Rminus_hat, title=r'Colormap: $\hat u$ (core) + $\hat u_{\mathrm{outer}}\!\approx 0$ (annulus)')
    cbar_uc = figu.colorbar(m_uc, ax=axu, pad=0.02); cbar_uc.set_label(r'$\hat u$ (core)')
    cbar_uo = figu.colorbar(m_uo, ax=axu, pad=0.10); cbar_uo.set_label(r'$\hat u$ (outer, $\approx 0$)')
    plt.tight_layout()
    if pars.SAVEFIG:
        figu.savefig(os.path.join(pars.FIGDIR, 'cmap_ur.png'), dpi=200)

    # 5) Colormap: ŵ with sign (core) + ŵ_outer = −Û_j
    figw, axw = plt.subplots(figsize=(8.4, 4.8))
    wmax = float(np.nanmax(np.abs(w_hat))) or 1.0
    wmax = wmax if wmax > 0 else 1.0
    m_wc = axw.pcolormesh(R_hat, Z_hat, w_hat, shading='auto',
                          cmap='coolwarm', norm=TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax))
    Wjet_hat_field = np.tile((-Uhat_j).reshape(-1, 1), (1, R_out.shape[1]))
    wj_max = float(np.nanmax(np.abs(Wjet_hat_field))) or 1.0
    wj_max = wj_max if wj_max > 0 else 1.0
    m_wo = axw.pcolormesh(R_out, Z_out, Wjet_hat_field, shading='auto',
                          cmap='coolwarm', norm=TwoSlopeNorm(vmin=-wj_max, vcenter=0.0, vmax=wj_max), alpha=0.9)
    _style_axes(axw, Rminus_hat, title=r'Colormap: $\hat w$ (core) + $\hat w_{\mathrm{outer}}=-\hat U_j$ (annulus)')
    cbar_wc = figw.colorbar(m_wc, ax=axw, pad=0.02); cbar_wc.set_label(r'$\hat w$ (core)')
    cbar_wo = figw.colorbar(m_wo, ax=axw, pad=0.10); cbar_wo.set_label(r'$\hat w$ (outer, $-\hat U_j$)')
    plt.tight_layout()
    if pars.SAVEFIG:
        figw.savefig(os.path.join(pars.FIGDIR, 'cmap_uz.png'), dpi=200)

    # 6) Vorticity-like field
    fig6, ax6 = plt.subplots(figsize=(8.4, 4.8))
    om_max = float(np.nanmax(np.abs(omega_hat))) or 1.0
    m6 = ax6.pcolormesh(R_hat, Z_hat, omega_hat, shading='auto',
                        cmap='coolwarm', norm=TwoSlopeNorm(vmin=-om_max, vcenter=0.0, vmax=om_max))
    _style_axes(ax6, Rminus_hat, title=r'Colormap: $\hat\omega_\theta$ (core)')
    cbar6 = fig6.colorbar(m6, ax=ax6, pad=0.02); cbar6.set_label(r'$\hat\omega_\theta$')
    if pars.SAVEFIG:
        fig6.savefig(os.path.join(pars.FIGDIR, 'cmap_vorticity.png'), dpi=200)

    plt.show()

# ---------------------------- Metrics & report ----------------------------

def seal_leak_indices(pars: Params, sol: Dict[str, object]) -> Dict[str, float]:
    Uj = sol["Uj"]
    p_c = pars.W / (np.pi * pars.R_minus**2)
    Pi_seal = pars.rho * Uj*Uj * pars.b0 / max(pars.h * p_c, 1e-30)
    Pi_leak = sol["mdot_leak"] / max(pars.rho * Uj * 2.0*np.pi*pars.R_minus*pars.b0, 1e-30)
    return {"Pi_seal": float(Pi_seal), "Pi_leak": float(Pi_leak)}


def flows_and_power(pars: Params, sol: Dict[str, object]) -> Dict[str, float]:
    Uj = sol["Uj"]
    mdot_out = pars.rho * Uj * (2.0 * np.pi * pars.R_tot * pars.b0)
    mdot_in = mdot_out + sol["mdot_leak"]
    P_jet = 0.5 * mdot_out * Uj*Uj / (1.0 + pars.K_turn)
    return {"mdot_out": float(mdot_out), "mdot_in": float(mdot_in), "P_jet": float(P_jet)}


def print_report(pars: Params, sol: Dict[str, object], ctrl: Dict[str, object]):
    print(" Shooting (L → W) ===")
    for it, Uj, L, rel in ctrl["history"]:
        print(f"  it {it:2d}: Uj={Uj:8.3f} m/s | L={L:9.3f} N | rel_err={rel:+8.4f}")
    print(f"  Final Uj = {ctrl['Uj_final']:.3f} m/s")
    p_c = pars.W / (np.pi * pars.R_minus**2)
    print(f"  Targets: W = {pars.W:.3f} N, p_c (core-avg) = {p_c:.3f} Pa")

    mdot_leak = sol["mdot_leak"]
    print("=== Leakage ===")
    print(f"  ṁ_leak = {mdot_leak:.6f} kg/s")

    flows = flows_and_power(pars, sol)
    print("=== Flows & Power ===")
    print(f"  ṁ_out = {flows['mdot_out']:.6f} kg/s")
    print(f"  ṁ_in  = {flows['mdot_in']:.6f} kg/s")
    print(f"  Ẇ_jet ≈ {flows['P_jet']:.2f} W")

    inds = seal_leak_indices(pars, sol)
    print("=== Indici non-dimensionali ===")
    print(f"  Π_seal = {inds['Pi_seal']:.4f}")
    print(f"  Π_leak = {inds['Pi_leak']:.6f}")

# ---------------------------- Main ----------------------------
if __name__ == '__main__':
    pars = Params()

    # Nonlinear solve (shoot Uj so that lift matches W)
    sol, ctrl = solve_with_shooting(pars)

    # Report + plots
    print_report(pars, sol, ctrl)
    make_plots(pars, sol)
