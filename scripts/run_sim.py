#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc 2D Axisymmetric Solver (low‑Mach, compressible core with Stokes–Darcy closure)
========================================================================================

This script implements the algorithm described in the LaTeX:
- Core region (0 <= r <= R_minus, 0 <= z <= h) with axisymmetry, no swirl.
- Pressure p(r,z) solves: (1/r)∂r(r * ρ * k_r ∂r p) + ∂z(ρ * k_z ∂z p) = 0,
  where ρ = p/(R_g T). We use anisotropic permeabilities k_r = α_r h^2, k_z = α_z h^2.
- Boundary conditions:
    r = 0        : symmetry (∂p/∂r = 0)
    r = R_minus  : curtain sealing pressure p_edge(z) = p0 + Δp * fz(z)
    z = 0, z = h : no normal flow (∂p/∂z = 0)
- Velocities from Stokes–Darcy:
    u_r = -(k_r/μ) ∂p/∂r,    u_z = -(k_z/μ) ∂p/∂z
- Curtain profile fz(z) is chosen to make vertical flow *downward* in the core,
  consistent with design intent. To that end we set p_edge larger near the TOP
  (z ~ h) than near the ground, i.e. fz(z) = 1 + λ z/h with λ > 0, so ∂p/∂z > 0
  and therefore u_z = -(k_z/μ) ∂p/∂z < 0 (downwards).

Outputs (saved to PNG files):
  - quiver_velocity.png        : quiver plot of (u_r,u_z)
  - cmap_speed.png             : |v| colormap
  - cmap_ur.png                : |u_r| colormap
  - cmap_uz.png                : |u_z| colormap
  - cmap_pressure.png          : pressure colormap

You can tweak parameters in the 'Parameters' section.
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
    T_inf: float = 293.0      # [K] ambient temperature
    # Fluid properties
    mu: float = 1.85e-5       # [Pa s]
    Rg: float = 287.0         # [J/(kg K)]
    k_air: float = 0.026      # [W/(m K)] (not used in this core-only solve)
    # Curtain / leakage
    b: float = 0.003          # [m] slot thickness
    h_eff: float = 0.010      # [m] effective sealing height
    Ct: float = 1.0           # [-] curtain factor
    lam: float = 0.4          # [-] vertical distribution strength (fz = 1 + lam*z/h)
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_corona: float = 40.0    # [m/s] jet exit speed
    # Stokes–Darcy closure
    alpha_r: float = 0.12     # [-] kr = alpha_r * h^2
    alpha_z: float = 0.05     # [-] kz = alpha_z * h^2
    # Numerical grid
    Nr: int = 180
    Nz: int = 90
    # Solver
    max_iter: int = 5000
    tol_rel: float = 1e-4     # relative to p_c
    omega: float = 1.6        # SOR relaxation


def fz_linear(z, h, lam):
    """Vertical loading shape for the curtain: fz(z) = 1 + lam * z/h.
       With lam > 0 this produces p_edge larger near the TOP, yielding ∂p/∂z > 0
       and u_z = -(kz/μ) ∂p/∂z < 0 (downwards) inside the core."""
    return 1.0 + lam * (z / h)


def solve_core(p: Params):
    # Derived geometry
    R_minus = p.R_tot - p.w
    # Cushion pressure (for reference)
    p_c = p.W / (np.pi * p.R_tot**2)
    p_center = p.p0 + p_c

    # Permeabilities
    kr = p.alpha_r * p.h**2
    kz = p.alpha_z * p.h**2

    # Curtain pressure increment at rim
    Delta_p_edge = p.Ct * (p.rho_j * p.U_corona**2 * p.b) / p.h_eff  # [Pa]

    # Grid
    r = np.linspace(0.0, R_minus, p.Nr)
    z = np.linspace(0.0, p.h, p.Nz)
    dr = r[1]-r[0] if p.Nr > 1 else 1.0
    dz = z[1]-z[0] if p.Nz > 1 else 1.0
    Rg, Zg = np.meshgrid(r, z, indexing='xy')

    # Edge pressure (Dirichlet at r = R_minus), varying with z
    p_edge = p.p0 + Delta_p_edge * fz_linear(z, p.h, p.lam)  # shape (Nz,)

    # Initial guess for p(r,z): smooth interpolation between center and edge
    P = np.zeros((p.Nz, p.Nr))
    for i in range(p.Nz):
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
        return Pf / (p.Rg * Tval)

    # Iterative SOR solve for (1/r)∂r(r ρ kr ∂r P) + ∂z(ρ kz ∂z P) = 0
    T = p.T_inf  # use uniform T; extend with energy solve if needed
    rho = rho_of(P, T)
    tol_abs = p.tol_rel * p_c

    for it in range(p.max_iter):
        P_old = P.copy()
        rho = rho_of(P, T)

        # Sweep interior points
        for i in range(1, p.Nz-1):
            for j in range(1, p.Nr-1):
                rj = r[j]
                # Avoid division by zero near the axis
                if rj < 1e-12:
                    rj = dr * 0.5

                # Coeffs at half steps (arithmetic mean for ρ at faces)
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
                P[i, j] = (1 - p.omega) * P[i, j] + p.omega * P_new

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

    u_r = -(kr / p.mu) * dPdr
    u_z = -(kz / p.mu) * dPdz

    return r, z, Rg, Zg, P, u_r, u_z, p_c, R_minus


# ---------------------------- Plot helpers ----------------------------
def make_plots(r, z, Rg, Zg, P, ur, uz, R_minus, h):
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

    # 1) Quiver
    plt.figure(figsize=(7, 4.5))
    plt.quiver(Rq, Zq, urq, uzq, angles='xy', scale_units='xy', scale=None)
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Velocity field (quiver)')
    plt.xlim(0, R_minus); plt.ylim(0, h)
    plt.tight_layout()
    plt.savefig('quiver_velocity.png', dpi=180)

    # 2) |v| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, speed, shading='auto')
    plt.colorbar(label='|v| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Speed magnitude |v|')
    plt.xlim(0, R_minus); plt.ylim(0, h)
    plt.tight_layout()
    plt.savefig('cmap_speed.png', dpi=180)

    # 3) |u_r| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, ur_abs, shading='auto')
    plt.colorbar(label='|u_r| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Radial component |u_r|')
    plt.xlim(0, R_minus); plt.ylim(0, h)
    plt.tight_layout()
    plt.savefig('cmap_ur.png', dpi=180)

    # 4) |u_z| colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, uz_abs, shading='auto')
    plt.colorbar(label='|u_z| [arb. units]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Axial component |u_z|')
    plt.xlim(0, R_minus); plt.ylim(0, h)
    plt.tight_layout()
    plt.savefig('cmap_uz.png', dpi=180)

    # 5) Pressure colormap
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(Rg, Zg, P, shading='auto')
    plt.colorbar(label='p [Pa]')
    plt.xlabel('r [m]'); plt.ylabel('z [m]')
    plt.title('Pressure field')
    plt.xlim(0, R_minus); plt.ylim(0, h)
    plt.tight_layout()
    plt.savefig('cmap_pressure.png', dpi=180)


# ---------------------------- Main ----------------------------
if __name__ == '__main__':
    pars = Params()
    r, z, Rg, Zg, P, ur, uz, p_c, R_minus = solve_core(pars)
    # Direction sanity check: expect downward vertical velocities in the core (uz < 0)
    # Uncomment to print simple diagnostics:
    # print('Mean uz sign (should be negative):', np.sign(np.mean(uz)))
    make_plots(r, z, Rg, Zg, P, ur, uz, R_minus, pars.h)
    print('Saved: quiver_velocity.png, cmap_speed.png, cmap_ur.png, cmap_uz.png, cmap_pressure.png')
