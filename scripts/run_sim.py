
#!/usr/bin/env python3
"""
run_sim_vscode.py — Hover disc 2D (assi-simmetrico) visual & fields
---------------------------------------------------------------
• Parametri definiti direttamente qui (niente argparse): comodo da lanciare in VS Code.
• Figura 1: sezione 2D della geometria (r-y).
• Figura 2: due subplot con mappe di colore 2D per pressione p(r,y) e velocità |u|(r,y).
Assunzioni essenziali
• Assi-simmetria: r è radiale, y è verticale (suolo a y=0).
• Cuscino interno (0 ≤ r ≤ R-w): gap ≈ h_c, p≈p_c, u≈0.
• Sigillo anulare (R-w ≤ r ≤ R): gap = h_eff, flusso radiale viscoso (modello lubrificazione).
• Fuori dal bordo (r>R): p=0, u=0 (ambiente).
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
rho    = 1.20   # kg/m^3, densità aria
mu     = 1.8e-5 # Pa*s, viscosità dinamica aria
g      = 9.81   # m/s^2

# Griglia del dominio per i campi 2D (sezione r–y)
# Dominio: r in [0, R + w] per includere un po' di esterno; y in [0, h_c]
Nr, Ny = 400, 200
r = np.linspace(0.0, R + w, Nr)
y = np.linspace(0.0, h_c, Ny)
Rmesh, Ymesh = np.meshgrid(r, y)  # shape (Ny, Nr)

# ---------------- Grandezze derivate ----------------
A_disk = np.pi * R**2
p_c = (m_load * g) / A_disk  # Pa, pressione richiesta per il carico
rin  = R - w                 # raggio interno del sigillo
ln_factor = np.log(R / rin)

# Portata attraverso il sigillo (modello lubrificazione, stazionario)
Q = (np.pi * h_eff**3 * p_c) / (6.0 * mu * ln_factor)  # m^3/s
P_ideal = p_c * Q  # W

# Velocità radiale media nel sigillo
def ubar(rval):
    return Q / (2.0 * np.pi * np.clip(rval, 1e-6, None) * h_eff)

# Profilo parabolico tra piastre: u(r,y) = 6 u_bar(r) * (y/h) * (1 - y/h)
def u_parabolic(rval, yval):
    phi = np.clip(yval / h_eff, 0.0, 1.0)
    return 6.0 * ubar(rval) * phi * (1.0 - phi)

# ---------------- Costruzione dei campi 2D ----------------
P = np.zeros_like(Rmesh)     # pressione [Pa]
U = np.zeros_like(Rmesh)     # modulo velocità [m/s]

mask_interior = (Rmesh <= rin) & (Ymesh <= h_c)
mask_seal     = (Rmesh >  rin) & (Rmesh <= R) & (Ymesh <= h_eff)
mask_outside  = (Rmesh >  R)

# Pressione: interno ~ p_c
P[mask_interior] = p_c

# Pressione nel sigillo (drop logaritmico dal modello lubrificazione)
coef = (6.0 * mu * Q) / (np.pi * h_eff**3)
r_sig = np.clip(Rmesh, rin, R)  # evita log(0)
P[mask_seal] = p_c - coef * np.log(r_sig[mask_seal] / rin)

# Fuori: P=0
P[mask_outside] = 0.0

# Velocità: interno ~ 0, sigillo parabolico
U[mask_seal] = u_parabolic(Rmesh[mask_seal], Ymesh[mask_seal])

# ---------------- Output testuale ----------------
print("=== Hover Disc 2D (assi-simmetrico) — VS Code run ===")
print(f"R                : {R:.2f} m (Ø = {2*R:.2f} m)")
print(f"h_c (centro)     : {h_c:.3f} m")
print(f"w (sigillo)      : {w:.3f} m")
print(f"h_eff (sigillo)  : {h_eff:.3f} m")
print(f"Carico           : {m_load:.1f} kg")
print(f"p_c              : {p_c:.2f} Pa")
print(f"Q (lubr.)        : {Q:.4f} m^3/s")
print(f"P_ideal          : {P_ideal/1000:.3f} kW")

# ---------------- Plot 1: Geometria (sezione r–y) ----------------
os.makedirs("figs", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(9, 4.5))
# Suolo
ax1.plot([0, R + w], [0, 0], lw=2, color="black")
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
ax1.set_ylabel("y [m]")
ax1.set_title("Geometria (sezione 2D r–y)")
ax1.legend(loc="upper right")
fig1.tight_layout()
fig1.savefig("figs/geometry.png", dpi=160)

# ---------------- Plot 2: Mappe 2D p(r,y) e |u(r,y)| ----------------
fig2, (axp, axu) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

pc = axp.imshow(P, origin="lower", extent=[r.min(), r.max(), y.min(), y.max()],
                aspect="auto")
axp.set_title("Pressione p(r,y) [Pa]")
axp.set_xlabel("r [m]")
axp.set_ylabel("y [m]")
fig2.colorbar(pc, ax=axp, fraction=0.046, pad=0.04)

uc = axu.imshow(U, origin="lower", extent=[r.min(), r.max(), y.min(), y.max()],
                aspect="auto")
axu.set_title("Velocità |u(r,y)| [m/s]")
axu.set_xlabel("r [m]")
axu.set_ylabel("y [m]")
fig2.colorbar(uc, ax=axu, fraction=0.046, pad=0.04)

fig2.savefig("figs/fields.png", dpi=160)

plt.show()
