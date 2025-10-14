
#!/usr/bin/env python3
# Geometry-only plot per user specs
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

R_tot  = 1.00
w      = 0.05
h_c    = 0.20
R_i    = R_tot - w

os.makedirs("figs", exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6))

# Ground
ax.plot([-R_tot - 0.15, R_tot + 0.15], [0, 0], lw=2, color="black")

# Axis of symmetry at r=0 (dash-dot)
ax.plot([0, 0], [-0.02, h_c + 0.12], linestyle="-.", color="black", linewidth=1.5, alpha=0.9)
ax.text(0.01, h_c + 0.11, "asse di simmetria (r=0)", va="bottom", fontsize=10)

# Underside of disc at z = h_c
ax.plot([-R_i, R_i], [h_c, h_c], lw=3, color="black", solid_capstyle="round")
ax.plot([-R_tot, -R_i], [h_c, h_c], lw=3, color="#1f77b4")
ax.plot([ R_i,  R_tot], [h_c, h_c], lw=3, color="#1f77b4")

# Relation annotation
ax.text(0.0, h_c + 0.035, r"$R_i + w = R_{tot}$", ha="center", va="bottom", fontsize=12)

# Height h
ax.annotate("", xy=(R_tot + 0.06, 0.0), xytext=(R_tot + 0.06, h_c),
            arrowprops=dict(arrowstyle="<->", lw=1.8))
ax.text(R_tot + 0.08, 0.5*h_c, r"$h$", va="center", fontsize=12)

# Width w (right side)
ax.annotate("", xy=(R_i, h_c + 0.02), xytext=(R_tot, h_c + 0.02),
            arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1f77b4"))
ax.text(0.5*(R_i + R_tot), h_c + 0.025, r"$w$", ha="center", va="bottom", color="#1f77b4", fontsize=11)

# Mass flow in (central)
ax.add_patch(FancyArrow(0.0, h_c + 0.09, 0.0, -0.06, width=0.002, length_includes_head=True))
ax.text(0.0, h_c + 0.10, r"$\dot{m}_{in}$", ha="center", va="bottom", fontsize=12)

# Corona mass flow (both sides), shallow horizontal arrows under the disc
y_cor = h_c * 0.9
ax.add_patch(FancyArrow(-R_i, y_cor, -(w - 0.02), 0.0, width=0.004, length_includes_head=True, color="#1f77b4"))
ax.text(-(R_i + 0.6*w), y_cor + 0.012, r"$\dot{m}_{corona}$", ha="center", va="bottom", fontsize=11, color="#1f77b4")
ax.add_patch(FancyArrow(R_i, y_cor,  (w - 0.02), 0.0, width=0.004, length_includes_head=True, color="#1f77b4"))
ax.text((R_i + 0.6*w), y_cor + 0.012, r"$\dot{m}_{corona}$", ha="center", va="bottom", fontsize=11, color="#1f77b4")

# Lateral dispersion beyond R_tot (loss)
y_loss = h_c * 0.15
ax.add_patch(FancyArrow(-R_tot, y_loss, -0.10, 0.0, width=0.003, length_includes_head=True, color="tab:red"))
ax.text(-R_tot - 0.11, y_loss + 0.01, r"$\dot{m}_{loss}$", ha="right", va="bottom", fontsize=11, color="tab:red")
ax.add_patch(FancyArrow(R_tot, y_loss,  0.10, 0.0, width=0.003, length_includes_head=True, color="tab:red"))
ax.text(R_tot + 0.11, y_loss + 0.01, r"$\dot{m}_{loss}$", ha="left", va="bottom", fontsize=11, color="tab:red")

ax.set_xlim(-R_tot - 0.15, R_tot + 0.15)
ax.set_ylim(-0.03, h_c + 0.14)
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.set_title("Schema geometrico: asse di simmetria, corona (w), h, \\dot{m}_{in}, \\dot{m}_{corona}, \\dot{m}_{loss}")
ax.grid(alpha=0.15)

fig.tight_layout()
fig.savefig("figs/schema_geometry.png", dpi=220)
print("Salvato: figs/schema_geometry.png")
