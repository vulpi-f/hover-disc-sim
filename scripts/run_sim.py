#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc 2D Axisymmetric Core Solver (updated to match LaTeX assumptions)
============================================================================

- Core region (0 <= r <= R_minus, 0 <= z <= h) with axisymmetry, no swirl.
- Compressible, low-Mach: ρ = p / (Rg T). (Here T is taken uniform; extend with energy loop if needed.)
- Stokes–Darcy closure for mean velocities in the core:
      u_r = -(k_r/μ) ∂p/∂r,   u_z = -(k_z/μ) ∂p/∂z,
  with k_r = α_r h^2, k_z = α_z h^2.
- Elliptic equation for p(r,z):
      (1/r) ∂r ( r ρ k_r ∂r p ) + ∂z ( ρ k_z ∂z p ) = 0
  in (r,z) ∈ [0,R_minus]×[0,h].
- Boundary conditions (core option, as in LaTeX Option A):
    r = 0      : symmetry (∂p/∂r = 0)
    r = R_minus: Dirichlet sealing pressure from the turned (vertical) corona jet:
                 p_edge(z) = p0 + Ct * (ρ_j U_corona^2 b / h_eff) * φ(z),
                 with φ(z) = (1 - z/h)^m (max near ground, ≈0 near top).
    z = 0, h   : no normal flow for the Darcy closure (∂p/∂z = 0).
- Velocities recovered via Stokes–Darcy.
- Plots: quiver of velocity, |v| colormap, |u_r| colormap, |u_z| colormap, pressure colormap.
  Each plot draws a vertical line at r = R_minus. Plots are saved only if SAVEFIG=1.

You can tweak parameters below. To save figures, set SAVEFIG = 1.
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
    T_inf: float = 293.0      # [K] ambient temperature (uniform here)
    # Fluid properties
    mu: float = 1.85e-5       # [Pa s]
    Rg: float = 287.0         # [J/(kg K)]
    # Curtain / leakage
    b: float = 0.003          # [m] slot thickness
    h_eff: float = 0.010      # [m] effective sealing height
    Ct: float = 1.0           # [-] curtain factor
    m_phi: float = 1.5        # [-] exponent in φ(z) = (1 - z/h)^m
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_corona: float = 40.0    # [m/s] jet exit speed
    # Stokes–Darcy closure
    alpha_r: float = 0.12     # [-] k_r = α_r * h^2
    alpha_z: float = 0.05     # [-] k_z = α_z * h^2
    # Numerical grid
    Nr: int = 180
    Nz: int = 90
    # Solver
    max_iter: int = 5000
    tol_rel: float = 1e-4     # relative to p_c
    omega: float = 1.6        # SOR relaxation
    # Saving
    SAVEFIG: int = 0          # set to 1 to save figures


def phi_z(z, h, m):
    """Sealing vertical distribution: φ(z) = (1 - z/h)^m, max at ground, ~0 near top."""
    zeta = np.clip(z / h, 0.0, 1.0)
    return (1.0 - zeta)**m


def solve_core(pars: Params):
    # Derived geometry
    R_minus = pars.R_tot - pars.w
    # Cushion pressure (for reference)
    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Curtain pressure increment at rim
    Delta_p_edge = pars.Ct * (pars.rho_j * pars.U_corona**2 * pars.b) / pars.h_eff  # [Pa]

    # Grid
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0
    Rg, Zg = np.meshgrid(r, z, indexing='xy')

    # Edge pressure (Dirichlet at r = R_minus), varying with z
    p_edge = pars.p0 + Delta_p_edge * phi_z(z, pars.h, pars.m_phi)  # shape (Nz,)

    # Initial guess for p(r,z): smooth interpolation between center and edge
    P = np.zeros((pars.Nz, pars.Nr))
    for i in range(pars.Nz):
        P[i, :] = p_center - (p_center - p_edge[i]) * (r / max(R_minus, 1e-12))**2

    # Helper: apply boundary conditions to P in-place
    def apply_bc(Pf):
        # r = R_minus: Dirichlet from curtain
        Pf[:, -1] = p_edge[:]
        # r = 0: Neumann ∂p/∂r = 0 -> mirror
        Pf[:, 0] = Pf[:, 1]
        # z = 0 and z = h: Neumann ∂p/∂z = 0 -> mirror
        Pf[0, :]  = Pf[1, :]
        Pf[-1, :] = Pf[-2, :]
        return Pf

    P = apply_bc(P)

    # Density function
    def rho_of(Pf, Tval):
        return Pf / (pars.Rg * Tval)

    # Iterative SOR solve for (1/r)∂r(r ρ kr ∂r P) + ∂z(ρ kz ∂z P) = 0
    T = pars.T_inf  # use uniform T; extend with energy loop if needed
    rho = rho_of(P, T)
    tol_abs = pars.tol_rel * p_c

    for it in range(pars.max_iter):
        P_old = P.copy()
        rho = rho_of(P, T)

        # Sweep interior points
        for i in range(1, pars.Nz-1):
            for j in range(1, pars.Nr-1):
                rj = r[j]
                # Avoid division by zero near the axis
                if rj < 1e-12:
                    rj = dr * 0.5

                # ρ at faces (arithmetic mean)
                rho_jp = 0.5*(rho[i, j] + rho[i, j+1])
                rho_jm = 0.5*(rho[i, j] + rho[i, j-1])
                rho_ip = 0.5*(rho[i+1, j] + rho[i, j])
                rho_im = 0.5*(rho[i-1, j] + rho[i, j])

                # Flux coefficients
                # Radial term: (1/r) * ∂r [ r * rho * kr * ∂r P ]
                Ar_p = (rj + 0.5*dr) * rho_jp * kr / (dr**2)
                Ar_m = (rj - 0.5*dr) * rho_jm * kr / (dr**2)
                # Axial term: ∂z [ rho * kz * ∂z P ]
                Az_p = rho_ip * kz / (dz**2)
                Az_m = rho_im * kz / (dz**2)

                denom = (Ar_p + Ar_m) / rj + (Az_p + Az_m)
                rhs = (Ar_p * P[i, j+1] + Ar_m * P[i, j-1]) / rj \
                    + (Az_p * P[i+1, j] + Az_m * P[i-1, j])

                P_new = rhs / (denom + 1e-30)
                P[i, j] = (1 - pars.omega) * P[i, j] + pars.omega * P_new

        # Re-apply BCs
        P = apply_bc(P)

        err = np.max(np.abs(P - P_old))
        if err < tol_abs:
            # print(f"Converged at iter {it}, err={err:.3e}")
            break

    # Velocities from Darcy
    dPdr = np.zeros_like(P)
    dPdz = np.zeros_like(P)
    dPdr[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2*dr)
    dPdr[:, 0]    = (P[:, 1] - P[:, 0]) / dr
    dPdr[:, -1]   = (P[:, -1] - P[:, -2]) / dr
    dPdz[1:-1, :] = (P[2:, :] - P[:-2, :]) / (2*dz)
    dPdz[0, :]    = (P[1, :] - P[0, :]) / dz
    dPdz[-1, :]   = (P[-1, :] - P[-2, :]) / dz

    u_r = -(kr / pars.mu) * dPdr
    u_z = -(kz / pars.mu) * dPdz

    return r, z, Rg, Zg, P, u_r, u_z, p_c, R_minus


def make_plots(pars: Params, r, z, Rg, Zg, P, ur, uz, R_minus):
    speed = np.sqrt(ur**2 + uz**2)
    ur_abs = np.abs(ur)
    uz_abs = np.abs(uz)

    # Downsample vectors for quiver clarity
    Nr, Nz = Rg.shape[1], Zg.shape[0]
    step_r = max(1, Nr // 30)
    step_z = max(1, Nz // 15)
    Rq = Rg[::step_z, ::step_r]
    Zq = Zg[::step_z, ::step_r]
    urq = ur[::step_z, ::step_r]
    uzq = uz[::step_z, ::step_r]

    # Helper to add the vertical line at r = R_minus
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
        plt.savefig('quiver_velocity.png', dpi=180)

    # 2) |v| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, speed, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|v| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Speed magnitude |v|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig('cmap_speed.png', dpi=180)

    # 3) |u_r| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, ur_abs, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|u_r| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Radial component |u_r|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig('cmap_ur.png', dpi=180)

    # 4) |u_z| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, uz_abs, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='|u_z| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Axial component |u_z|')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig('cmap_uz.png', dpi=180)

    # 5) Pressure colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, P, shading='auto')
    add_Rminus_line()
    plt.colorbar(label='p [Pa]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Pressure field')
    plt.xlim(0, R_minus); plt.ylim(0, pars.h)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig('cmap_pressure.png', dpi=180)

    # Show figures (always)
    plt.show()


# ---------------------------- Main ----------------------------
if __name__ == '__main__':
    pars = Params()
    r, z, Rg, Zg, P, ur, uz, p_c, R_minus = solve_core(pars)
    make_plots(pars, r, z, Rg, Zg, P, ur, uz, R_minus)
    # To save figures as PNG, set pars.SAVEFIG = 1
