
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hover Disc — Core Solver + Shooting + Curtain Model (aligned with paper.tex)
=============================================================================

Model highlights aligned with LaTeX:
1) Core PDE (low-Mach, axisymmetric, Stokes–Darcy closure):
   (1/r) ∂r ( r ρ κ_r ∂r p ) + ∂z ( ρ κ_z ∂z p ) = 0,   ρ = p/(R_g T).
   κ_r = α_r h^2, κ_z = α_z h^2.  BCs: ∂r p|_{r=0}=0, ∂z p|_{z=0,h}=0, p|_{r=R^-}=p_edge(z).

2) Rim pressure (composite static + sealing momentum, Eq. p_edge_momentum in paper.tex):
   Let ζ = z/H with H ≡ h_eff (effective curtain height).
   b(z) = b0 * (1 + s ζ)        # jet thickens with ζ (spreading parameter s ~ 0.06–0.09)
   U(z) = U0 * b0 / b(z)        # mass conservation per unit circumference (2D slot jet)
   p_edge(z) = p0
              + Δp_static * (1 - ζ)^n
              + min(ΔP_cap, ρ_j U(z)^2 b(z) / (H * Dm_min)) * ζ^m

   - Δp_static = C_p * p_c encodes static build-up from turning curtain near floor.
   - The momentum/sealing term is capped by ΔP_cap (optional), as in the text.

3) Shooting loop adjusts U0 (=U_out) to enforce target mean cushion pressure p̄ ≈ p_c = W/(π R_tot^2).

4) Leakage model (film vs. orifice) and pneumatic powers identical in spirit to the paper.

Outputs:
  - Shooting history and final U_out
  - Leakage regime and flows
  - Curtain outflow, make-up inflow, pneumatic powers
  - Non-dimensional plots saved under FIGDIR for LaTeX
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
    h_eff: float = 0.010      # [m] effective curtain height H
    rho_j: float = 1.20       # [kg/m^3] jet density
    U_out: float = 40.0       # [m/s] initial guess for U0
    # Rim pressure model coefficients (from tex)
    m: float = 1.2            # [-] exponent for sealing weight ζ^m
    n_exp: float = 2.0        # [-] exponent for static build-up (1-ζ)^n
    C_p: float = 0.15         # [-] static-pressure coefficient (O(1e-1))
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
    shoot_tol: float = 5e-3   # |p̄ - p_c| <= shoot_tol * p_c
    shoot_gain: float = 0.6   # proportional gain on U_out
    # IO
    SAVEFIG: int = 1
    FIGDIR: str = "../figs"

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

def average_cushion_pressure(P: np.ndarray, p0: float,
                             r: np.ndarray, z: np.ndarray,
                             R_tot: float) -> float:
    p_bar_z = np.mean(P, axis=0)  # (Nr,)
    integrand = (p_bar_z - p0) * 2.0 * np.pi * r
    core_area = np.pi * (r[-1]**2)
    avg_over_core = np.trapezoid(integrand, r) / max(core_area, 1e-16)
    return avg_over_core * (core_area / (np.pi * R_tot**2))

# ---------------------------- Solver ----------------------------
def solve_core_once(pars: Params, U0: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray, float, float,
                               Dict[str, float]]:
    """Single elliptic solve with given U0; returns fields and diagnostics."""
    R_minus = pars.R_tot - pars.w

    # References
    p_c = pars.W / (np.pi * pars.R_tot**2)
    p_center = pars.p0 + p_c

    # Permeabilities
    kr = pars.alpha_r * pars.h**2
    kz = pars.alpha_z * pars.h**2

    # Grid
    r = np.linspace(0.0, R_minus, pars.Nr)
    z = np.linspace(0.0, pars.h, pars.Nz)
    dr = r[1]-r[0] if pars.Nr > 1 else 1.0
    dz = z[1]-z[0] if pars.Nz > 1 else 1.0

    # Rim pressure p_edge(z) per paper.tex
    U_z, b_z = jet_profile(U0, pars.b, z, pars.h_eff, pars.s_spread)
    zhat_H = np.clip(z / max(pars.h_eff, 1e-12), 0.0, 1.0)
    # static part
    Delta_p_static = pars.C_p * p_c
    # sealing momentum-limited part
    momentum_term = (pars.rho_j * (U_z**2) * b_z) / max(pars.h_eff * max(pars.Dm_min, 1e-12), 1e-12)
    Delta_p_seal_z = np.minimum(pars.DeltaP_cap, momentum_term)
    p_edge = (pars.p0
              + Delta_p_static * phi_static(z, pars.h_eff, pars.n_exp)
              + Delta_p_seal_z * phi_seal(z, pars.h_eff, pars.m))

    # Initial guess: quadratic radial interpolation
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
    tol_abs = pars.tol_rel * p_c

    for _ in range(pars.max_iter):
        P_old = P.copy()
        rho = rho_of(P, pars.Rg, T)

        # SOR-Gauss–Seidel
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

    # Non-dimensional fields
    r_hat = r / pars.R_tot
    z_hat = z / pars.h
    R_hat, Z_hat = np.meshgrid(r_hat, z_hat, indexing='xy')

    p_hat = (P - pars.p0) / p_c
    dp_hat_drhat = (pars.R_tot / p_c) * dPdr
    dp_hat_dzhat = (pars.h / p_c) * dPdz
    u_hat = - dp_hat_drhat
    w_hat = - dp_hat_dzhat
    S = (pars.alpha_z / pars.alpha_r) * (pars.R_tot / pars.h)

    # Diagnostics for shooting & leakage
    pbar = average_cushion_pressure(P, pars.p0, r, z, pars.R_tot)
    Delta_p_leak = float(np.mean(P[:, -1] - pars.p0))  # z-avg at rim

    diag = {
        "pbar": float(pbar),
        "Delta_p_leak": float(Delta_p_leak),
        "R_minus": float(R_minus),
        "p_c": float(p_c),
        "U_profile_mean": float(np.mean(U_z)),
    }

    return (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, R_minus/pars.R_tot, diag)

def solve_core_with_shooting(pars: Params):
    """Adjust U_out so that p̄ ≈ p_c."""
    p_c = pars.W / (np.pi * pars.R_tot**2)
    U0 = pars.U_out
    history = []
    final_fields = None

    for it in range(1, pars.shoot_max_iter+1):
        results = solve_core_once(pars, U0)
        (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag) = results
        pbar = diag["pbar"]
        err = pbar - p_c
        rel_err = err / max(p_c, 1e-16)
        history.append((it, U0, pbar, rel_err))

        if abs(rel_err) <= pars.shoot_tol:
            final_fields = results
            break

        # Proportional gain on U0 (keep positive)
        U0 = max(0.5, U0 * (1.0 - pars.shoot_gain * rel_err))
        final_fields = results

    # Final diagnostics based on U0
    U_out_final = float(history[-1][1])
    return final_fields, {
        "history": history,
        "U_out_final": U_out_final,
    }

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

    # Pneumatic powers: P = Q * Δp, with Q = ṁ/ρ
    # Use Δp ≈ p_c for make-up and an effective Δp at the rim for the outer jet.
    P_out = (mdot_out / max(pars.rho_j, 1e-16)) * leak["mdot_loss"] * 0.0  # placeholder in case needed
    # For outer jet power, better estimate via momentum flux per unit circumf. ~ ρ U^3 b * 2πR
    P_out = (pars.rho_j * (U_out**3) * pars.b) * (2.0 * np.pi * pars.R_tot)  # aerodynamic power in jet
    rho_c = leak["rho_c"]
    p_c = ctrl.get("p_c", (pars.W / (np.pi * pars.R_tot**2)))
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

# ---------------------------- Plots ----------------------------
def make_plots(pars: Params, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat):
    ensure_figdir(pars.FIGDIR)
    Nr, Nz = R_hat.shape[1], Z_hat.shape[0]
    step_r = max(1, Nr // 30)
    step_z = max(1, Nz // 15)
    Rq = R_hat[::step_z, ::step_r]
    Zq = Z_hat[::step_z, ::step_r]
    uq = u_hat[::step_z, ::step_r]
    wq = w_hat[::step_z, ::step_r]

    Viso = np.sqrt(u_hat**2 + (S * w_hat)**2)

    def add_Rminus_line():
        import matplotlib.pyplot as plt
        plt.plot([Rminus_hat, Rminus_hat], [0.0, 1.0])

    # Quiver (û, S ŵ)
    plt.figure(figsize=(7, 4.5))
    plt.quiver(Rq, Zq, uq, S*wq, angles='xy', scale_units='xy', scale=None)
    add_Rminus_line()
    plt.xlabel(r'$\hat r$'); plt.ylabel(r'$\hat z$')
    plt.title('Velocity field (quiver) — non-dimensional')
    plt.xlim(0, Rminus_hat); plt.ylim(0, 1.0)
    plt.tight_layout()
    if pars.SAVEFIG == 1:
        plt.savefig(os.path.join(pars.FIGDIR, 'quiver_velocity.png'), dpi=180)

    # |V̂_iso|
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

    # |û|
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

    # |ŵ|
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

    # p̂
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
    for it, U0, pbar, rel in hist:
        print(f"  it {it:2d}: U_out={U0:8.3f} m/s | p̄={pbar:9.3f} Pa | rel_err={rel:+8.4f}")
    print(f"  Final U_out = {shoot_ctrl['U_out_final']:.3f} m/s")

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
    (r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat, diag), shoot_ctrl = solve_core_with_shooting(pars)

    # Leakage model and power/flows
    leak = leakage_and_power(pars, diag)
    flows = curtain_and_powers(pars, shoot_ctrl, leak)

    # Report + plots
    print_report(pars, shoot_ctrl, leak, flows)
    make_plots(pars, r_hat, z_hat, R_hat, Z_hat, p_hat, u_hat, w_hat, S, Rminus_hat)
