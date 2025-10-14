#!/usr/bin/env python3
"""
run_sim.py — Hover disc 2D (assi-simmetrico): geometria + campi 2D
-------------------------------------------------------------------
• Parametri settati qui (niente argparse) per esecuzione da VS Code.
• Figura 1: geometria (sezione r–z, suolo a z=0).
• Figura 2: tre subplot con colormap 2D di:
   (a) pressione p(r,z) [Pa]
   (b) velocità radiale u_r(r,z) [m/s]
   (c) velocità assiale u_z(r,z) [m/s]

Modello (stazionario, incomprimibile, assi-simmetrico):
- Interno (0 ≤ r ≤ R-w): gap ~ h_c, p ~ p_c costante, u ~ 0.
- Sigillo (R-w ≤ r ≤ R, 0 ≤ z ≤ h_eff): flusso tra lastre (lubrificazione).
  u_r(r,z) = 6·ū(r)·φ·(1−φ),  φ = z/h_eff,  ū(r) = Q / (2π r h_eff).
  p(r) = p_c − (6 μ Q)/(π h_eff^3) · ln(r/(R−w)).
- Esterno (r > R): p = 0, u = 0.
- In questo modello, dalla continuità con ū∝1/r segue u_z ≈ 0 nel sigillo (e no-penetration a z=0,h_eff).

NOTA: se vuoi rappresentare un u_z non nullo (alimentazione/risucchio), possiamo estendere il modello
con una sorgente volumetrica o con uno strato d’ingresso nel sigillo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parametri (modificabili) ----------------
R      = 1.00   # m, raggio disco (Ø=2R)
h_c    = 0.20   # m, luce centrale
w      = 0.05   # m, larghezza del tratto di sigillo
h_eff  = 0.03   # m, gap efficace nel sigillo
m_load = 40.0   # kg, carico utile sostenuto
rho    = 1.20   # kg/m^3, densità aria (unused qui)
mu     = 1.8e-5 # Pa*s, viscosità dinamica aria
g      = 9.81   # m/s^2

# Griglia 2D (r–z). r ∈ [0, R + w], z ∈ [0, h_c]
Nr, Nz = 400, 240
r = np.linspace(0.0, R + w, Nr)
z = np.linspace(0.0, h_c, Nz)
Rmesh, Zmesh = np.meshgrid(r, z)  # (Nz, Nr)

# ---------------- Grandezze derivate ----------------
A_disk = np.pi * R**2
p_c = (m_load * g) / A_disk  # Pa
rin  = R - w                 # raggio interno del sigillo
ln_factor = np.log(R / rin)

# Portata nel sigillo (lubrificazione): Q = π h_eff^3 p_c / (6 μ ln(R/rin))
Q = (np.pi * h_eff**3 * p_c) / (6.0 * mu * ln_factor)  # m^3/s
P_ideal = p_c * Q  # W

def ubar(rval: np.ndarray) -> np.ndarray:
    """Velocità media radiale nel sigillo (ū = Q/(2π r h_eff))."""
    return Q / (2.0 * np.pi * np.clip(rval, 1e-6, None) * h_eff)

def u_r_profile(rval: np.ndarray, zval: np.ndarray) -> np.ndarray:
    """Profilo parabolico tra piastre: u_r = 6·ū(r)·φ·(1−φ), φ=z/h_eff, nel solo sigillo."""
    phi = np.clip(zval / h_eff, 0.0, 1.0)
    return 6.0 * ubar(rval) * phi * (1.0 - phi)

# ---------------- Campi 2D ----------------
P  = np.zeros_like(Rmesh)  # pressione [Pa]
Ur = np.zeros_like(Rmesh)  # velocità radiale [m/s]
Uz = np.zeros_like(Rmesh)  # velocità assiale [m/s] (≈0 nel modello)

mask_interior = (Rmesh <= rin) & (Zmesh <= h_c)
mask_seal     = (Rmesh >  rin) & (Rmesh <= R) & (Zmesh <= h_eff)
mask_outside  = (Rmesh >  R)

# Pressione
P[mask_interior] = p_c
coef = (6.0 * mu * Q) / (np.pi * h_eff**3)
r_sig = np.clip(Rmesh, rin, R)
P[mask_seal] = p_c - coef * np.log(r_sig[mask_seal] / rin)
P[mask_outside] = 0.0

# Velocità
Ur[mask_seal] = u_r_profile(Rmesh[mask_seal], Zmesh[mask_seal])
# Uz rimane 0 con questo modello (continuità + no-penetration)

# ---------------- Report ----------------
print("=== Hover Disc 2D — VS Code ===")
print(f"R                : {R:.2f} m (Ø = {2*R:.2f} m)")
print(f"h_c (centro)     : {h_c:.3f} m")
print(f"w (sigillo)      : {w:.3f} m")
print(f"h_eff (sigillo)  : {h_eff:.3f} m")
print(f"Carico           : {m_load:.1f} kg")
print(f"p_c              : {p_c:.2f} Pa")
print(f"Q (lubr.)        : {Q:.4f} m^3/s")
print(f"P_ideal          : {P_ideal/1000:.3f} kW")

# ---------------- Plot 1: Geometria ----------------
os.makedirs("figs", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
# Suolo
ax1.plot([0, R + w], [0, 0], lw=2)
# Sottosuperficie disco (interno)
ax1.plot([0, rin], [h_c, h_c], lw=3)
# Labbro verticale del bordo
ax1.plot([R, R], [h_c, h_eff], lw=3)
# Regione sigillo (riempimento)
ax1.fill_betweenx([0, h_eff], rin, R, alpha=0.12, label="Sigillo (h_eff)")
# Quote
ax1.annotate("", xy=(rin, h_c*1.02), xytext=(R, h_c*1.02),
             arrowprops=dict(arrowstyle="<->"))
ax1.text(rin + 0.5*(R-rin), h_c*1.04, "w", ha="center", va="bottom")
ax1.text(0.02*R, h_c+0.01, "Interno disco", va="bottom")
ax1.text(R+0.01, (h_c+h_eff)/2, "Labbro", va="center", rotation=90)
ax1.set_xlim(0, R + w)
ax1.set_ylim(-0.01, h_c + 0.08)
ax1.set_xlabel("r [m]")
ax1.set_ylabel("z [m]")
ax1.set_title("Geometria (sezione r–z)")
ax1.legend(loc="upper right")
fig1.tight_layout()
fig1.savefig("figs/geometry.png", dpi=160)

# ---------------- Plot 2: Colormap con 3 subplot ----------------
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

# (a) Pressione
im0 = axes[0].imshow(P, origin="lower",
                     extent=[r.min(), r.max(), z.min(), z.max()],
                     aspect="auto")
axes[0].set_title("p(r,z) [Pa]")
axes[0].set_xlabel("r [m]"); axes[0].set_ylabel("z [m]")
fig2.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# (b) Velocità radiale u_r
im1 = axes[1].imshow(Ur, origin="lower",
                     extent=[r.min(), r.max(), z.min(), z.max()],
                     aspect="auto")
axes[1].set_title("u_r(r,z) [m/s]")
axes[1].set_xlabel("r [m]"); axes[1].set_ylabel("z [m]")
fig2.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# (c) Velocità assiale u_z (≈0 nel modello)
im2 = axes[2].imshow(Uz, origin="lower",
                     extent=[r.min(), r.max(), z.min(), z.max()],
                     aspect="auto")
axes[2].set_title("u_z(r,z) [m/s]")
axes[2].set_xlabel("r [m]"); axes[2].set_ylabel("z [m]")
fig2.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig2.savefig("figs/fields_tripanel.png", dpi=160)

# Mostra figure (utile in VS Code)
plt.show()
