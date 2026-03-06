"""
Unit tests for the physics calculation functions in yplus_estimator.py. to Test units (just debugging purposes)
"""

import math
import pytest

from yplus_estimator import (
    compute_reynolds,
    compute_skin_friction,
    compute_wall_shear_stress,
    compute_friction_velocity,
    compute_cell_size,
    estimate_cell_size,
)


# ---------------------------------------------------------------------------
# compute_reynolds
# ---------------------------------------------------------------------------

class TestComputeReynolds:
    def test_basic(self):
        re = compute_reynolds(velocity=10.0, char_length=1.0, kinematic_viscosity=1e-5)
        assert re == pytest.approx(1_000_000.0)

    def test_proportional_to_velocity(self):
        re1 = compute_reynolds(velocity=5.0, char_length=1.0, kinematic_viscosity=1e-5)
        re2 = compute_reynolds(velocity=10.0, char_length=1.0, kinematic_viscosity=1e-5)
        assert re2 == pytest.approx(2 * re1)

    def test_zero_viscosity_raises(self):
        with pytest.raises(ValueError):
            compute_reynolds(velocity=10.0, char_length=1.0, kinematic_viscosity=0.0)

    def test_negative_viscosity_raises(self):
        with pytest.raises(ValueError):
            compute_reynolds(velocity=10.0, char_length=1.0, kinematic_viscosity=-1e-5)


# ---------------------------------------------------------------------------
# compute_skin_friction
# ---------------------------------------------------------------------------

class TestComputeSkinFriction:
    def test_known_value(self):
        # For Re = 1e6, Cf = (2*log10(1e6) - 0.65)**(-2.3)
        re = 1e6
        expected = (2.0 * math.log10(re) - 0.65) ** (-2.3)
        assert compute_skin_friction(re) == pytest.approx(expected, rel=1e-9)

    def test_decreases_with_re(self):
        """Higher Re → lower Cf (turbulent flat-plate behaviour)."""
        cf_low = compute_skin_friction(1e5)
        cf_high = compute_skin_friction(1e7)
        assert cf_low > cf_high

    def test_zero_re_raises(self):
        with pytest.raises(ValueError):
            compute_skin_friction(0.0)

    def test_negative_re_raises(self):
        with pytest.raises(ValueError):
            compute_skin_friction(-1e6)


# ---------------------------------------------------------------------------
# compute_wall_shear_stress
# ---------------------------------------------------------------------------

class TestComputeWallShearStress:
    def test_basic(self):
        # tau_w = 0.5 * 1.225 * 10^2 * 0.003
        tau_w = compute_wall_shear_stress(density=1.225, velocity=10.0, cf=0.003)
        assert tau_w == pytest.approx(0.5 * 1.225 * 100 * 0.003)

    def test_scales_with_velocity_squared(self):
        tau1 = compute_wall_shear_stress(density=1.0, velocity=10.0, cf=0.003)
        tau2 = compute_wall_shear_stress(density=1.0, velocity=20.0, cf=0.003)
        assert tau2 == pytest.approx(4 * tau1)


# ---------------------------------------------------------------------------
# compute_friction_velocity
# ---------------------------------------------------------------------------

class TestComputeFrictionVelocity:
    def test_basic(self):
        # u_tau = sqrt(1.0 / 1.0) = 1.0
        assert compute_friction_velocity(tau_w=1.0, density=1.0) == pytest.approx(1.0)

    def test_sqrt_relation(self):
        u1 = compute_friction_velocity(tau_w=4.0, density=1.0)
        u2 = compute_friction_velocity(tau_w=16.0, density=1.0)
        assert u2 == pytest.approx(2 * u1)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError):
            compute_friction_velocity(tau_w=1.0, density=0.0)


# ---------------------------------------------------------------------------
# compute_cell_size
# ---------------------------------------------------------------------------

class TestComputeCellSize:
    def test_basic(self):
        # delta_y = 1.0 * 1e-5 / 1.0 = 1e-5
        assert compute_cell_size(y_plus=1.0, kinematic_viscosity=1e-5, u_tau=1.0) == pytest.approx(1e-5)

    def test_proportional_to_y_plus(self):
        d1 = compute_cell_size(y_plus=1.0, kinematic_viscosity=1e-5, u_tau=0.5)
        d2 = compute_cell_size(y_plus=5.0, kinematic_viscosity=1e-5, u_tau=0.5)
        assert d2 == pytest.approx(5 * d1)

    def test_zero_u_tau_raises(self):
        with pytest.raises(ValueError):
            compute_cell_size(y_plus=1.0, kinematic_viscosity=1e-5, u_tau=0.0)


# ---------------------------------------------------------------------------
# estimate_cell_size (integration test)
# ---------------------------------------------------------------------------

class TestEstimateCellSize:
    """End-to-end pipeline tests using air at standard conditions."""

    STANDARD_INPUTS = dict(
        velocity=10.0,
        density=1.225,
        dynamic_viscosity=1.81e-5,
        char_length=1.0,
        y_plus=1.0,
    )

    def test_returns_all_keys(self):
        result = estimate_cell_size(**self.STANDARD_INPUTS)
        assert set(result.keys()) == {"nu", "Re", "Cf", "tau_w", "u_tau", "delta_y"}

    def test_nu_value(self):
        result = estimate_cell_size(**self.STANDARD_INPUTS)
        expected_nu = 1.81e-5 / 1.225
        assert result["nu"] == pytest.approx(expected_nu, rel=1e-6)

    def test_reynolds_positive(self):
        result = estimate_cell_size(**self.STANDARD_INPUTS)
        assert result["Re"] > 0

    def test_cell_size_positive(self):
        result = estimate_cell_size(**self.STANDARD_INPUTS)
        assert result["delta_y"] > 0

    def test_cell_size_decreases_with_velocity(self):
        """Higher velocity → higher Re → larger tau_w → larger u_tau → smaller Δy."""
        result_slow = estimate_cell_size(**{**self.STANDARD_INPUTS, "velocity": 5.0})
        result_fast = estimate_cell_size(**{**self.STANDARD_INPUTS, "velocity": 50.0})
        assert result_fast["delta_y"] < result_slow["delta_y"]

    def test_cell_size_scales_with_y_plus(self):
        result1 = estimate_cell_size(**{**self.STANDARD_INPUTS, "y_plus": 1.0})
        result5 = estimate_cell_size(**{**self.STANDARD_INPUTS, "y_plus": 5.0})
        assert result5["delta_y"] == pytest.approx(5 * result1["delta_y"], rel=1e-6)

    def test_invalid_velocity_raises(self):
        with pytest.raises(ValueError):
            estimate_cell_size(**{**self.STANDARD_INPUTS, "velocity": -1.0})

    def test_invalid_density_raises(self):
        with pytest.raises(ValueError):
            estimate_cell_size(**{**self.STANDARD_INPUTS, "density": 0.0})

    def test_invalid_viscosity_raises(self):
        with pytest.raises(ValueError):
            estimate_cell_size(**{**self.STANDARD_INPUTS, "dynamic_viscosity": -1e-5})

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            estimate_cell_size(**{**self.STANDARD_INPUTS, "char_length": 0.0})

    def test_invalid_y_plus_raises(self):
        with pytest.raises(ValueError):
            estimate_cell_size(**{**self.STANDARD_INPUTS, "y_plus": 0.0})

    def test_water_conditions(self):
        """Smoke-test with water properties."""
        result = estimate_cell_size(
            velocity=2.0,
            density=998.0,
            dynamic_viscosity=1.002e-3,
            char_length=0.1,
            y_plus=1.0,
        )
        assert result["delta_y"] > 0
        assert result["Re"] > 0
