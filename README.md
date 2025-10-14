# Hover Disc Cushion — 2D Axisymmetric Model

This repo contains a minimal **Python** model and a **LaTeX** write-up for a disc that levitates at ~20 cm using a central cushion and an annular aerodynamic seal.

## Features
- Axisymmetric 2D model with two leakage regimes: **viscous (lubrication)** and **orifice/inertial**.
- Pressure and velocity profiles in the annular seal, cushion pressure for a target load.
- Plot generation to `figs/`.
- LaTeX document with governing equations and figure includes.

## Quickstart

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a simulation (defaults: R=1.0 m, h_c=0.20 m, w=0.05 m, h_eff=0.03 m, m_load=40 kg)
python scripts/run_sim.py --R 1.0 --h_c 0.20 --w 0.05 --h_eff 0.03 --m_load 40 --cd 0.65

# Outputs:
# - Printed report in terminal
# - Plots saved under figs/
# - JSON results under figs/last_results.json
```

### Create a new GitHub repository

Using **git** only:
```bash
git init
git add .
git commit -m "Initial commit: 2D cushion model + LaTeX"
git branch -M main
# Create an empty repo on GitHub named hover-disc-sim, then:
git remote add origin https://github.com/<your-username>/hover-disc-sim.git
git push -u origin main
```

Using the **GitHub CLI** (`gh`):
```bash
gh repo create hover-disc-sim --public --source=. --remote=origin --push
```

## Repository layout
```
src/        # Python modules (model + plotting)
scripts/    # CLI runner
figs/       # Output plots and result JSON
tex/        # LaTeX write-up
```

## License
MIT
