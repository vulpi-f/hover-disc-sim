#!/usr/bin/env python3
# Plot geometrico (assi-simmetria completa, r<0…r>0) — SOLO DISPLAY, NESSUN SALVATAGGIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ---------------- Parametri (modificabili) ----------------
R_tot  = 1.00   # m, raggio totale del disco (R_tot)
w      = 0.08   # m, larghezza out (getto di bordo)
h_c    = 0.20   # m, distanza dal suolo della faccia inferiore
R_i    = R_tot - w

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 6))

# Suolo (z = 0)
ax.plot([-R_tot - 0.15, R_tot + 0.15], [0, 0], lw=2, color="black")

# Asse di simmetria a r=0 (linea punto-tratto, verticale)
ax.plot([0, 0], [-0.02, h_c + 0.07], linestyle="-.", color="black", linewidth=1.5, alpha=0.9)

ax.plot([R_i, R_i], [0, h_c], linestyle="--", color="black", linewidth=1.5, alpha=0.9)
ax.plot([-R_i, -R_i], [0, h_c], linestyle="--", color="black", linewidth=1.5, alpha=0.9)

# Faccia inferiore del disco a z = h_c
# Parte centrale (|r| <= R_i) — colore 1
ax.plot([-R_i, R_i], [h_c, h_c], lw=3, color="black", solid_capstyle="round")
# Parti outli (R_i -> R_tot su entrambi i lati) — colore 2 per evidenziare w
ax.plot([-R_tot, -R_i], [h_c, h_c], lw=3, color="#ff0000")  # sinistra
ax.plot([ R_i,  R_tot], [h_c, h_c], lw=3, color="#ff0000")  # destra

# Quota h (altezza da terra)
h_y_dist = 0.08
ax.annotate("", xy=(h_y_dist, 0.0), xytext=(h_y_dist, h_c),
            arrowprops=dict(arrowstyle="<->", lw=1.8))
ax.text(h_y_dist+0.02, 0.5*h_c, r"$h$", va="center", fontsize=12)

# Indicazione della larghezza w a destra
v_dist = 0.02
ax.annotate("", xy=(R_i, h_c + v_dist), xytext=(R_tot, h_c + v_dist),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color="#FF0000"))
ax.text(R_tot+0.01, h_c+v_dist, r"$b_0$", ha="left", va="center", color="#FF0000", fontsize=11)

# Portata massica entrante nella zona centrale (da sopra verso il cuscino)
arr_L = h_c-0.04
y_mass_in = h_c + 0.04
arr_width = 0.05
ax.add_patch(FancyArrow(0.0, y_mass_in, 0.0, -arr_L, width=arr_width, head_width=arr_width*2, length_includes_head=True, head_length=0.5*arr_L, color="#fb00ff"))
ax.text(0.08, y_mass_in - 0.03, r"$\dot{m}_{in}$", ha="center", va="bottom", color="#fb00ff", fontsize=12)

# Portata massica della out (getto radiale nella fascia w), su entrambi i lati
y_cor = h_c * 0.9
arr_Lc = arr_L
y_mass_c_in = y_mass_in
arr_width2 = 0.03
ax.add_patch(FancyArrow(-R_i-w/2, y_mass_c_in, 0.0, -arr_Lc, width=arr_width2, head_width=arr_width2*2, length_includes_head=True, head_length=0.5*arr_Lc, color="#1f77b4"))
ax.text(-(R_i + 0.6*w), y_mass_c_in + 0.012, r"$\dot{m}_{out}$", ha="center", va="bottom", fontsize=11, color="#1f77b4")
ax.add_patch(FancyArrow(R_i+w/2, y_mass_c_in, 0.0, -arr_Lc, width=arr_width2, head_width=arr_width2*2, length_includes_head=True, head_length=0.5*arr_Lc, color="#1f77b4"))
ax.text((R_i + 0.6*w), y_mass_c_in + 0.012, r"$\dot{m}_{out}$", ha="center", va="bottom", fontsize=11, color="#1f77b4")

# Portata massica dispersa lateralmente oltre R_tot (perdite)
y_loss = h_c * 0.15
x_dist = 0.04
y_dist =0.005
loss_L = w+0.10
ax.add_patch(FancyArrow(-R_i, y_loss, -loss_L, 0.0, width=0.003, length_includes_head=True, color="tab:red"))
ax.text(-R_i - x_dist, y_loss+y_dist, r"$\dot{m}_{loss}$", ha="right", va="bottom", fontsize=11, color="tab:red")
ax.add_patch(FancyArrow(R_i, y_loss,  loss_L, 0.0, width=0.003, length_includes_head=True, color="tab:red"))
ax.text(R_i + x_dist, y_loss+y_dist, r"$\dot{m}_{loss}$", ha="left", va="bottom", fontsize=11, color="tab:red")

# Limiti e stile
ax.set_xlim(-R_tot - 0.15, R_tot + 0.15)
ax.set_ylim(-0.03, h_c + 0.08)
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
#ax.set_title("Schema geometrico: asse di simmetria, out (w), h, $\dot{m}_{in}$, $\dot{m}_{out}$, $\dot{m}_{loss}$")
ax.grid(alpha=0.15)

plt.show()

fig.tight_layout()
fig.savefig("../figs/schema_geometry.png", dpi=220)
print("Salvato: figs/schema_geometry.png")