"""
plots.py — plotting utilities for cushion model.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_pressure(r, p_r, p_c, R, w, out_png):
    plt.figure(figsize=(7,4))
    plt.plot([0, R-w], [p_c, p_c], lw=2, label="Interior ~ p_c")
    plt.plot(r, p_r, lw=2, label="Seal region")
    plt.xlabel("Radius r [m]")
    plt.ylabel("Pressure p(r) [Pa]")
    plt.title("Pressure distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_velocity(r, u_bar, out_png):
    plt.figure(figsize=(7,4))
    plt.plot(r, u_bar, lw=2)
    plt.xlabel("Radius r [m]")
    plt.ylabel("Mean radial velocity ū(r) [m/s]")
    plt.title("Annular seal: mean radial velocity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def save_results_json(path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
