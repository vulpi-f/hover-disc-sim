#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc — Non-dimensional Core Solver and Plots (aligned with LaTeX)
=======================================================================

- Core-only domain: r ∈ [0, R_minus], z ∈ [0, h].
- Low-Mach compressible, Stokes–Darcy closure:
      u = -(k_r/μ) ∂p/∂r,   w = -(k_z/μ) ∂p/∂z
  with k_r = α_r h^2, k_z = α_z h^2.
- Elliptic pressure solve:
      (1/r) ∂r ( r ρ k_r ∂r p ) + ∂z ( ρ k_z ∂z p ) = 0,  ρ = p/(Rg T)
  BCs: ∂p/∂r=0 at r=0; ∂p/∂z=0 at z=0,h; p(R_minus,z)=p_edge(z).
- Rim (curtain) pressure (vertical jet at z=h projected at rim):
      p_edge(z) = p0 + Δp * ψ(z),   ψ(z) = 1 + λ z/h  (increasing → w < 0).

Non-dimensionalization (as in the LaTeX):
    r̂=r/R_tot, ẑ=z/h, p̂=(p-p0)/p_c,   with p_c = W/(π R_tot^2).
    û = -∂_{r̂} p̂,  ŵ = -∂_{ẑ} p̂.
    S = U_z^0 / U_r^0 = (α_z/α_r) * (R_tot/h).
    For isotropic quiver magnitude, use (û, S ŵ).

Plots saved to ../figs/ with the same filenames used in the paper:
    quiver_velocity.png  (quiver of (û, S ŵ))
    cmap_speed.png       (|V̂_iso| = sqrt(û^2 + (S ŵ)^2))
    cmap_ur.png          (|û|)
    cmap_uz.png          (|ŵ|)   [note: not multiplied by S to show pure component]
    cmap_pressure.png    (p̂)

Set SAVEFIG=1 to save PNGs; figures are always shown for interactive checks.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------- Parameters ----------------------------
@dataclass
class Params:
    # Geometry
    R_tot: float = 0.50       # [m] total radius
    w: float = 0.05           # [m] leakage ring width
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
    lam: float = 0.4          # [-] ψ(z)=1+lam*z/h
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_out: float = 40.0    # [m/s] jet speed
    # Stokes–Darcy closure
    alpha_r: float = 0.12     # [-] k_r = α_r h^2
    alpha_z: float = 0.05     # [-] k_z = α_z h^2
    # Numerical grid
    Nr: int = 180
    Nz: int = 90
    # Solver
    max_iter: int = 6000
    tol_rel: float = 1e-4     # relative to p_c
    omega: float = 1.6        # SOR relaxation
    # IO
    SAVEFIG: int = 1
    FIGDIR: str = "../figs"   # keep linkage with LaTeX


def psi_increasing(z, h, lam):
    """ψ(z) = 1 + lam * z/h (increasing with z ⇒ ∂p/∂z > 0 ⇒ w < 0)."""
    return 1.0 + lam * np.clip(z / h, 0.0, 1.0)


def ensure_figdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def solve_core(pars: Params):
    R_minus = pars.R_tot - pars.w

    # Reference cushion pressure and center pressure
    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Rim pressure increment from curtain momentum
    Delta_p_edge = pars.Ct * (pars.rho_j * pars.U_out**2 * pars.b) / pars.h_eff

    # Grid (dimensional)
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0
    Rg, Zg = np.meshgrid(r, z, indexing='xy')

    # Rim pressure profile p_edge(z) = p0 + Δp * ψ(z)
    p_edge = pars.p0 + Delta_p_edge * psi_increasing(z, pars.h, pars.lam)  # (Nz,)

    # Initial guess: radial interpolation from center to rim
    P = np.zeros((pars.Nz, pars.Nr))
    for i in range(pars.Nz):
        P[i, :] = p_center - (p_center - p_edge[i]) * (r / max(R_minus, 1e-12))**2

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

    def rho_of(Pf, Tval):
        return Pf / (pars.Rg * Tval)

    T = pars.T_inf
    tol_abs = pars.tol_rel * p_c

    for it in range(pars.max_iter):
        P_old = P.copy()
        rho = rho_of(P, T)

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
                rhs = (Ar_p * P[i, j+1] + Ar_m * P[i, j-1]) / rj \
                    + (Az_p * P[i+1, j] + Az_m * P[i-1, j])

                P_new = rhs / (denom + 1e-30)
                P[i, j] = (1 - pars.omega) * P[i, j] + pars.omega * P_new

        P = apply_bc(P)
        err = np.max(np.abs(P - P_old))
        if err < tol_abs:
            break

    # Derivatives (dimensional)
    dPdr = np.zeros_like(P)
    dPdz = np.zeros_like(P)
    dPdr[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2*dr)
    dPdr[:, 0]    = (P[:, 1] - P[:, 0]) / dr
    dPdr[:, -1]   = (P[:, -1] - P[:, -2]) / dr
    dPdz[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2*dz)
    dPdz[0, :]    = (P[1, :] - P[0, :]) / dz
    dPdz[-1, :]   = (P[-1, :] - P[-2, :]) / dz

    # Dimensional velocities (not plotted, but could be kept for reference)
    # u = -(kr / pars.mu) * dPdr
    # w = -(kz / pars.mu) * dPdz

    # ------------------ Non-dimensional fields ------------------
    # Coordinates
    r_hat = r / pars.R_tot
    z_hat = z / pars.h
    R_hat, Z_hat = np.meshgrid(r_hat, z_hat, indexing='xy')

    # Pressure hat
    p_hat = (P - pars.p0) / p_c

    # Derivatives of p̂ wrt r̂ and ẑ:
    # ∂p̂/∂r̂ = (R_tot/ p_c) ∂p/∂r ;   ∂p̂/∂ẑ = (h/ p_c) ∂p/∂z
    dp_hat_drhat = (pars.R_tot / p_c) * dPdr
    dp_hat_dzhat = (pars.h / p_c) * dPdz

    # Velocity hats
    u_hat = - dp_hat_drhat
    w_hat = - dp_hat_dzhat

    # Anisotropy scaling for isotropic quiver
    S = (pars.alpha_z / pars.alpha_r) * (pars.R_tot / pars.h)

    return (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S,
            R_minus/pars.R_tot)


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


if __name__ == '__main__':
    pars = Params()
    r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat = solve_core(pars)
    # Sanity: mean ŵ < 0 (downwards)
    # print('mean w_hat =', float(np.mean(w_hat)))
    make_plots(pars, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat)
