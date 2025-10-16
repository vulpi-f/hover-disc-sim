#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc 2D Axisymmetric Core Solver — consistent with LaTeX and sign conventions
===================================================================================
Domain: r ∈ [0, R_minus], z ∈ [0, h], with ground at z=0 and disc underside at z=h.
Both jets (inner and corona) are purely vertical **downward** at z = h (towards negative z).
In the core-only model (Option A in the LaTeX), we do not explicitly impose the inlet
velocity; instead, the turned outer jet is projected as a pressure boundary at r = R^-.
To obtain a **downward** vertical component inside the core (w < 0), the model must
produce ∂p/∂z > 0 (pressure increasing with z), because w = -(kz/μ) ∂p/∂z.

We therefore set the sealing pressure at the rim to INCREASE with z:
    p_edge(z) = p0 + Δp * ψ(z),       ψ(z) = 1 + λ z/h   (λ > 0).
This is the minimal adjustment that makes w < 0 while remaining consistent with the
physical situation of a vertical jet at z = h. If you strictly prefer the sealing
load to peak near the floor for other reasons, you can switch to φ(z)=(1-z/h)^m and
accept that the core model may yield w ≥ 0 unless additional momentum terms are added.

Governing core solve (Stokes–Darcy closure, compressible low-Mach with uniform T):
    (1/r) ∂r ( r ρ k_r ∂r p ) + ∂z ( ρ k_z ∂z p ) = 0,   ρ = p/(Rg T)
    u_r = -(k_r/μ) ∂p/∂r,   u_z = -(k_z/μ) ∂p/∂z.
BCs (core-only):
    r=0:       ∂p/∂r = 0
    r=R_minus: p = p_edge(z)
    z=0, z=h:  ∂p/∂z = 0
Plots: quiver (velocity), |v|, |u_r|, |u_z|, and p; each includes a vertical line at r=R^-.
Figures are SAVED only if SAVEFIG == 1 (and always SHOWN).

Author: updated per latest LaTeX + discussion.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


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
    T_inf: float = 293.0      # [K] uniform temperature (core-only solve)
    # Fluid properties
    mu: float = 1.85e-5       # [Pa s]
    Rg: float = 287.0         # [J/(kg K)]
    # Curtain (turned jet projected as rim pressure)
    b: float = 0.003          # [m] slot thickness
    h_eff: float = 0.010      # [m] effective sealing height
    Ct: float = 1.0           # [-] curtain factor
    lam: float = 0.4          # [-] slope for ψ(z) = 1 + lam*z/h (controls ∂p/∂z > 0)
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_corona: float = 40.0    # [m/s] jet exit speed
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
    # Saving
    SAVEFIG: bool = 1          # set to 1 to save figures
    savepath = "../figs/"


def psi_increasing(z, h, lam):
    """ψ(z) = 1 + lam * z/h  (increasing with z -> ∂p/∂z > 0 -> w < 0)."""
    return 1.0 + lam * np.clip(z / h, 0.0, 1.0)


def solve_core(pars: Params):
    R_minus = pars.R_tot - pars.w
    # Reference cushion pressure
    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Rim pressure increment from curtain momentum
    Delta_p_edge = pars.Ct * (pars.rho_j * pars.U_corona**2 * pars.b) / pars.h_eff

    # Grid
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0
    Rg, Zg = np.meshgrid(r, z, indexing='xy')

    # Rim pressure profile p_edge(z) = p0 + Δp * ψ(z) with ψ increasing
    p_edge = pars.p0 + Delta_p_edge * psi_increasing(z, pars.h, pars.lam)  # (Nz,)

    # Initial guess: smooth radial interpolation from center to rim
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

        # Interior sweep (SOR)
        for i in range(1, pars.Nz-1):
            for j in range(1, pars.Nr-1):
                rj = r[j] if r[j] > 1e-12 else 0.5*dr

                # Densities at faces (arithmetic mean)
                rho_jp = 0.5*(rho[i, j] + rho[i, j+1])
                rho_jm = 0.5*(rho[i, j] + rho[i, j-1])
                rho_ip = 0.5*(rho[i+1, j] + rho[i, j])
                rho_im = 0.5*(rho[i-1, j] + rho[i, j])

                # Flux coefficients
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

    # Velocities (Stokes–Darcy)
    dPdr = np.zeros_like(P)
    dPdz = np.zeros_like(P)
    dPdr[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2*dr)
    dPdr[:, 0]    = (P[:, 1] - P[:, 0]) / dr
    dPdr[:, -1]   = (P[:, -1] - P[:, -2]) / dr
    dPdz[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2*dz)
    dPdz[0, :]    = (P[1, :] - P[0, :]) / dz
    dPdz[-1, :]   = (P[-1, :] - P[-2, :]) / dz

    ur = -(kr / pars.mu) * dPdr
    uz = -(kz / pars.mu) * dPdz

    return r, z, Rg, Zg, P, ur, uz, p_c, R_minus


def make_plots(pars: Params, r, z, Rg, Zg, P, ur, uz, R_minus):
    speed = np.sqrt(ur**2 + uz**2)
    ur_abs = np.abs(ur)
    uz_abs = np.abs(uz)

    # Quiver downsample
    Nr, Nz = Rg.shape[1], Zg.shape[0]
    step_r = max(1, Nr // 30)
    step_z = max(1, Nz // 15)
    Rq = Rg[::step_z, ::step_r]
    Zq = Zg[::step_z, ::step_r]
    urq = ur[::step_z, ::step_r]
    uzq = uz[::step_z, ::step_r]

    def add_Rminus_line():
        plt.plot([R_minus, R_minus], [0.0, pars.h])

    # 1) Quiver
    plt.figure(figsize=(7, 4.5))
    plt.quiver(Rq, Zq, urq, uzq, angles='xy', scale_units='xy', scale=None)
    add_Rminus_line()
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Velocity field (quiver)')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(pars.savepath+'quiver_velocity.png', dpi=180)

    # 2) |v|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, speed, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|v| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Speed magnitude |v|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(pars.savepath+'cmap_speed.png', dpi=180)

    # 3) |u_r|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, ur_abs, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|u_r| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Radial component |u_r|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(pars.savepath+'cmap_ur.png', dpi=180)

    # 4) |u_z|
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, uz_abs, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|u_z| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Axial component |u_z|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(pars.savepath+'cmap_uz.png', dpi=180)

    # 5) Pressure
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, P, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='p [Pa]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Pressure field')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(pars.savepath+'cmap_pressure.png', dpi=180)

    plt.show()


if __name__ == '__main__':
    pars = Params()
    r, z, Rg, Zg, P, ur, uz, p_c, R_minus = solve_core(pars)
    # Sanity check for direction: expect mean uz < 0 (downward)
    # print('Mean uz:', np.mean(uz))
    make_plots(pars, r, z, Rg, Zg, P, ur, uz, R_minus)
