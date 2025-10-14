#!/usr/bin/env python3
"""
CLI runner for the hover-disc cushion model.
"""
import argparse, os
from src.cushion_model import Params, cushion_pressure, leakage_viscous, leakage_orifice
from src.plots import plot_pressure, plot_velocity, save_results_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--h_c", type=float, default=0.20)
    ap.add_argument("--w", type=float, default=0.05)
    ap.add_argument("--h_eff", type=float, default=0.03)
    ap.add_argument("--m_load", type=float, default=40.0)
    ap.add_argument("--rho", type=float, default=1.20)
    ap.add_argument("--mu", type=float, default=1.8e-5)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--model", choices=["auto","viscous","orifice"], default="auto")
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    p = Params(R=args.R, h_c=args.h_c, w=args.w, h_eff=args.h_eff,
               m_load=args.m_load, rho=args.rho, mu=args.mu, Cd=args.cd)

    p_c = cushion_pressure(p)
    res_visc = leakage_viscous(p, p_c)
    res_orif = leakage_orifice(p, p_c)

    # choose model
    if args.model == "auto":
        # pick by Reynolds number (gap-based) threshold ~2000
        chosen = "orifice" if res_orif["Re"] > 2000 else "viscous"
    else:
        chosen = args.model

    res = res_orif if chosen == "orifice" else res_visc

    # plots
    p_png = os.path.join(args.outdir, "pressure.png")
    u_png = os.path.join(args.outdir, "velocity.png")
    plot_pressure(res["r"], res["p_r"], p_c, p.R, p.w, p_png)
    plot_velocity(res["r"], res["u_bar"], u_png)

    # report
    print("=== Hover Disc Cushion (2D axisymmetric) ===")
    print(f"Chosen model     : {chosen}")
    print(f"R                : {p.R} m (Ø={2*p.R} m)")
    print(f"h_c (center)     : {p.h_c} m")
    print(f"w (seal width)   : {p.w} m")
    print(f"h_eff (seal gap) : {p.h_eff} m")
    print(f"Load             : {p.m_load} kg")
    print(f"Cushion pressure : {p_c:.2f} Pa")
    print(f"Leakage Q        : {res['Q']:.3f} m^3/s")
    print(f"Ideal power      : {res['P']/1000:.3f} kW")
    print(f"Re (diagnostic)  : {res['Re']:.0f}")
    print(f"Saved plots to   : {args.outdir}/")

    save_results_json(os.path.join(args.outdir, "last_results.json"), {
        "model": chosen,
        "R": p.R, "h_c": p.h_c, "w": p.w, "h_eff": p.h_eff,
        "m_load": p.m_load, "rho": p.rho, "mu": p.mu, "Cd": p.Cd,
        "p_c": p_c, "Q": res["Q"], "P": res["P"], "Re": float(res["Re"]),
        "pressure_png": p_png, "velocity_png": u_png
    })

if __name__ == "__main__":
    main()
