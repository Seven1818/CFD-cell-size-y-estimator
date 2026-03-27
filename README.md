# CFD y⁺ Cell Size Estimator

A lightweight Python tool to estimate the **first-cell height (Δy)** required 
to achieve a target y⁺ value in CFD wall-bounded flow simulations,
particularly useful when setting up snappyHexMesh configurations in OpenFOAM.


![CFD y+ Estimator GUI](/Images/GUI_y_2.png)
---

## Installation

No dependencies beyond the Python standard library.
```bash
git clone https://github.com/Seven1818/yplus-estimator.git
cd yplus-estimator
python yplus_estimator.py
```

Tested on Python 3.9+.

---

## Why y⁺ matters

The y⁺ value determines whether your near-wall mesh resolution is appropriate 
for your chosen turbulence modelling approach:

| y⁺ range | Wall treatment |
|---|---|
| y⁺ ≈ 1 | Resolved boundary layer (low-Re models, e.g. k-ω SST) |
| 30 < y⁺ < 300 | Wall functions (high-Re models) |

Getting this wrong leads to inaccurate wall shear stress, heat transfer 
predictions, and drag estimates.

---

## Physics pipeline

The tool implements the following sequence [1]:

1. **Reynolds number:** Re = U · L · ρ/μ  
2. **Skin-friction coefficient** (Schlichting correlation, valid for Re < 10⁹):  
   Cf = (2 · log₁₀(Re) − 0.65)^(−2.3)  
3. **Wall shear stress:** τ_w = 0.5 · ρ · U² · Cf  
4. **Friction velocity:** u_τ = √(τ_w / ρ)  
5. **First-cell height:** Δy = y⁺ · μ / u_τ ·ρ 

> **Note:** This uses a flat-plate boundary layer approximation. 
> Results are a good starting estimate; always verify with your 
> actual simulation once running.

---


## Usage

### GUI mode
Run the script directly to launch the graphical interface:
```bash
python yplus_estimator.py
```

Enter your flow parameters and press **Calculate**.

### Programmatic use
The physics functions are fully decoupled from the GUI and can be 
imported directly:
```python
from yplus_estimator import estimate_cell_size

result = estimate_cell_size(
    velocity=30.0,          # m/s
    density=1.225,          # kg/m³
    dynamic_viscosity=1.81e-5,  # Pa·s
    char_length=1.5,        # m
    y_plus=1.0,
)

print(f"First-cell height: {result['delta_y']:.4e} m")
print(f"Reynolds number:   {result['Re']:.3e}")
```


## Inputs

| Parameter | Symbol | Unit | Default |
|---|---|---|---|
| Free-stream velocity | U | m/s | 10.0 |
| Fluid density | ρ | kg/m³ | 1.225 |
| Dynamic viscosity | μ | Pa·s | 1.81e-5 |
| Characteristic length | L | m | 1.0 |
| Target y⁺ | y⁺ | — | 1.0 |

---

## Limitations

- Assumes flat-plate turbulent boundary layer (Schlichting correlation)
- Not valid for separated flows, strong pressure gradients, or 
  Re > 10⁹
- Use as an **initial estimate** ! validate against your simulation

---

## Author

**Massimiliano Toffoli**  
MSc Mechanical Engineering — TU Delft (Fluid Dynamics & CFD)  
[LinkedIn](https://www.linkedin.com/in/massimiliano-toffoli/) · 
[GitHub](https://github.com/Seven1818)

## References
[1] CFD-Online Community. Y Plus Wall Distance Estimation. *url: https://www.cfd-online.com/Wiki/Y_plus_wall_distance_estimation* (visited on 03/12/2026)

---

## Licence

MIT
