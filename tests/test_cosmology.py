import numpy as np

from physics.cosmology import LambdaCDM
from utils.constants import G, OMEGA_B, OMEGA_DM, RHO_CRIT, SECONDS_PER_YEAR, c
from utils.cosmology_utils import calculate_scale_factor_at_time


def test_expansion_acceleration_uses_requested_time_not_current_scale_factor():
    cosmology = LambdaCDM()
    time_years = 1.0e9
    time_seconds = time_years * SECONDS_PER_YEAR

    # Deliberately set a different current state. The method should use the
    # requested time, not this mutable instance value.
    cosmology.scale_factor = 1.0

    a_at_time = calculate_scale_factor_at_time(time_years)
    rho_m = (OMEGA_DM + OMEGA_B) * RHO_CRIT / (a_at_time**3)
    rho_lambda = cosmology.dark_energy_density()
    p_lambda = cosmology.dark_energy_pressure()
    expected = -4 * np.pi * G / 3 * (rho_m + rho_lambda + 3 * p_lambda / (c**2))

    np.testing.assert_allclose(
        cosmology.expansion_acceleration(time_seconds),
        expected,
        rtol=1e-12,
    )
