"""
CFD y+ Cell Size Estimator
--------------------------
Estimates the minimum first-cell height required to achieve a target y+ value
for use in OpenFOAM snappyHexMesh configurations.

Physics pipeline:
  1. Compute Reynolds number:          Re   = U * L / nu
  2. Schlichting skin-friction:        Cf   = (2*log10(Re) - 0.65)^(-2.3)
  3. Wall shear stress:                tau_w = 0.5 * rho * U^2 * Cf
  4. Friction velocity:                u_tau = sqrt(tau_w / rho)
  5. First-cell height:               delta_y = y_plus * nu / u_tau* rho
"""

import math
import tkinter as tk
from tkinter import messagebox


# ---------------------------------------------------------------------------
# Core physics functions (no GUI dependency – easy to unit-test)
# ---------------------------------------------------------------------------

def compute_reynolds(velocity: float, char_length: float, kinematic_viscosity: float) -> float:
    """Return the Reynolds number Re = U * L / nu."""
    if kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be positive.")
    return velocity * char_length / kinematic_viscosity


def compute_skin_friction(re: float) -> float:
    """
    Return the skin-friction coefficient using the Schlichting empirical
    correlation (valid for Re<10^9)

        Cf = (2 * log10(Re) - 0.65) ** (-2.3)
    """
    if re <= 0:
        raise ValueError("Reynolds number must be positive.")
    return (2.0 * math.log10(re) - 0.65) ** (-2.3)


def compute_wall_shear_stress(density: float, velocity: float, cf: float) -> float:
    """Return the wall shear stress tau_w = 0.5 * rho * U^2 * Cf."""
    return 0.5 * density * velocity ** 2 * cf


def compute_friction_velocity(tau_w: float, density: float) -> float:
    """Return the friction velocity u_tau = sqrt(tau_w / rho)."""
    if density <= 0:
        raise ValueError("Density must be positive.")
    return math.sqrt(tau_w / density)


def compute_cell_size(y_plus: float, kinematic_viscosity: float, u_tau: float, density:float) -> float:
    """Return the first-cell height delta_y = y+ * nu / u_tau*rho."""
    if u_tau <= 0:
        raise ValueError("Friction velocity must be positive.")
    return y_plus * kinematic_viscosity / (u_tau * density)


def estimate_cell_size(
    velocity: float,
    density: float,
    dynamic_viscosity: float,
    char_length: float,
    y_plus: float,
) -> dict:
    """
    Run the full y+ estimation pipeline.

    Parameters
    ----------
    velocity          : free-stream velocity [m/s]
    density           : fluid density        [kg/m³]
    dynamic_viscosity : dynamic viscosity    [Pa·s]
    char_length       : characteristic length [m]
    y_plus            : target y+ value

    Returns
    -------
    dict with keys: nu, Re, Cf, tau_w, u_tau, delta_y
    """
    if velocity <= 0:
        raise ValueError("Velocity must be positive.")
    if density <= 0:
        raise ValueError("Density must be positive.")
    if dynamic_viscosity <= 0:
        raise ValueError("Dynamic viscosity must be positive.")
    if char_length <= 0:
        raise ValueError("Characteristic length must be positive.")
    if y_plus <= 0:
        raise ValueError("y+ must be positive.")

    nu = dynamic_viscosity / density
    re = compute_reynolds(velocity, char_length, nu)
    cf = compute_skin_friction(re)
    tau_w = compute_wall_shear_stress(density, velocity, cf)
    u_tau = compute_friction_velocity(tau_w, density)
    delta_y = compute_cell_size(y_plus, nu, u_tau,density)

    return {
        "nu": nu,
        "Re": re,
        "Cf": cf,
        "tau_w": tau_w,
        "u_tau": u_tau,
        "delta_y": delta_y,
    }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class YPlusApp(tk.Tk):
    """Main application window for the CFD y+ Cell Size Estimator."""

    # Colour palette
    BG = "#1e2330"
    PANEL = "#252c3d"
    ACCENT = "#4a9eff"
    TEXT = "#dce3f0"
    LABEL = "#8a95aa"
    RESULT_BG = "#2a3348"
    SUCCESS = "#43c78a"
    ERROR = "#ff5f5f"

    def __init__(self):
        super().__init__()
        self.title("CFD y⁺ Cell Size Estimator")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 18, "pady": 6}

        # ── Title bar ──────────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=self.ACCENT)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="CFD  y⁺  Cell Size Estimator",
            font=("Helvetica", 16, "bold"),
            bg=self.ACCENT,
            fg="white",
            pady=10,
        ).pack()

        tk.Label(
            title_frame,
            text="for OpenFOAM  snappyHexMesh",
            font=("Helvetica", 9),
            bg=self.ACCENT,
            fg="#cce5ff",
            pady=2,
        ).pack()

        # ── Input panel ────────────────────────────────────────────────
        input_frame = tk.LabelFrame(
            self,
            text="  Flow Parameters  ",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL,
            fg=self.TEXT,
            bd=0,
            labelanchor="nw",
        )
        input_frame.pack(fill="x", padx=14, pady=(14, 6))

        self._fields = {}
        fields_def = [
            ("velocity",          "Free-stream velocity  U",    "m/s",    "10.0"),
            ("density",           "Fluid density  ρ",           "kg/m³",  "1.225"),
            ("dynamic_viscosity", "Dynamic viscosity  μ",       "Pa·s",   "1.81e-5"),
            ("char_length",       "Characteristic length  L",   "m",      "1.0"),
            ("y_plus",            "Target  y⁺",                 "–",      "1.0"),
        ]

        for row_idx, (key, label, unit, default) in enumerate(fields_def):
            tk.Label(
                input_frame,
                text=label,
                font=("Helvetica", 10),
                bg=self.PANEL,
                fg=self.TEXT,
                anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", padx=14, pady=5)

            entry = tk.Entry(
                input_frame,
                font=("Courier", 10),
                width=14,
                bg=self.BG,
                fg=self.TEXT,
                insertbackground=self.TEXT,
                relief="flat",
                highlightthickness=1,
                highlightcolor=self.ACCENT,
                highlightbackground="#3a4258",
            )
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, padx=10, pady=5)

            tk.Label(
                input_frame,
                text=unit,
                font=("Helvetica", 9),
                bg=self.PANEL,
                fg=self.LABEL,
            ).grid(row=row_idx, column=2, padx=(0, 14), pady=5, sticky="w")

            self._fields[key] = entry

        # ── Calculate button ───────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=self.BG)
        btn_frame.pack(pady=8)

        self._calc_btn = tk.Button(
            btn_frame,
            text="  Calculate  ",
            font=("Helvetica", 11, "bold"),
            bg=self.ACCENT,
            fg="white",
            activebackground="#3388ee",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=8,
            command=self._on_calculate,
        )
        self._calc_btn.pack()

        # ── Results panel ──────────────────────────────────────────────
        result_frame = tk.LabelFrame(
            self,
            text="  Results  ",
            font=("Helvetica", 10, "bold"),
            bg=self.PANEL,
            fg=self.TEXT,
            bd=0,
            labelanchor="nw",
        )
        result_frame.pack(fill="x", padx=14, pady=(6, 14))

        result_rows = [
            ("re_var",      "Reynolds number  Re"),
            ("cf_var",      "Skin-friction coeff.  Cƒ"),
            ("tauw_var",    "Wall shear stress  τ_w"),
            ("utau_var",    "Friction velocity  u_τ"),
            ("delta_var",   "▶  First-cell height  Δy"),
        ]

        self._result_vars = {}
        for row_idx, (var_key, label) in enumerate(result_rows):
            is_main = var_key == "delta_var"
            row_bg = self.RESULT_BG if is_main else self.PANEL

            tk.Label(
                result_frame,
                text=label,
                font=("Helvetica", 10, "bold" if is_main else "normal"),
                bg=row_bg,
                fg=self.ACCENT if is_main else self.TEXT,
                anchor="w",
            ).grid(row=row_idx, column=0, sticky="ew", padx=14, pady=(5 if is_main else 3))

            var = tk.StringVar(value="—")
            tk.Label(
                result_frame,
                textvariable=var,
                font=("Courier", 10, "bold" if is_main else "normal"),
                bg=row_bg,
                fg=self.SUCCESS if is_main else self.TEXT,
                anchor="e",
                width=22,
            ).grid(row=row_idx, column=1, sticky="e", padx=(0, 14), pady=(5 if is_main else 3))

            self._result_vars[var_key] = var

        result_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(1, weight=0)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Enter parameters and press Calculate.")
        tk.Label(
            self,
            textvariable=self._status_var,
            font=("Helvetica", 9, "italic"),
            bg=self.BG,
            fg=self.LABEL,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_calculate(self):
        """Read inputs, run estimation, display results."""
        try:
            inputs = {k: float(self._fields[k].get()) for k in self._fields}
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "All fields must contain valid numbers.\nPlease check your inputs.",
            )
            self._status_var.set("⚠  Invalid input — all fields must be numeric.")
            return

        try:
            result = estimate_cell_size(**inputs)
        except ValueError as exc:
            messagebox.showerror("Calculation Error", str(exc))
            self._status_var.set(f"⚠  {exc}")
            return

        self._result_vars["re_var"].set(f"{result['Re']:.4e}")
        self._result_vars["cf_var"].set(f"{result['Cf']:.4e}")
        self._result_vars["tauw_var"].set(f"{result['tau_w']:.4e}  Pa")
        self._result_vars["utau_var"].set(f"{result['u_tau']:.4e}  m/s")
        self._result_vars["delta_var"].set(f"{result['delta_y']:.4e}  m")

        self._status_var.set(
            f"✔  Calculated for Re = {result['Re']:.3e}  |  "
            f"Δy = {result['delta_y']:.4e} m"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = YPlusApp()
    app.mainloop()
