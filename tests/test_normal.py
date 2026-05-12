"""
Test for normal scenario (universe expansion only, no collapse)

Checks values at current universe age (~13.8 billion years):
- Particle horizon
- Event horizon  
- FLRW c/H helper
- de Sitter horizon
- Masses inside horizons
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from physics.cosmology import LambdaCDM
from physics.mass_calculator import MassCalculator
from physics.matter_simulation import MatterSimulation
from physics.objects import Universe
from utils.constants import (
    G, c, RHO_CRIT, OMEGA_B, OMEGA_DM, OMEGA_LAMBDA,
    NARIAI_BLACK_HOLE_MASS_KG
)
from utils.cosmology_utils import calculate_scale_factor_at_time
import config

# Constants
BILLION_YEARS_IN_SECONDS = 3.154e16
CURRENT_UNIVERSE_AGE_YEARS = 13.8e9
CURRENT_UNIVERSE_AGE_SECONDS = 13.8 * BILLION_YEARS_IN_SECONDS
LIGHT_YEAR_IN_METERS = 9.461e15
BILLION_LY = 1e9 * LIGHT_YEAR_IN_METERS

# Expected values at t = 13.8 billion years
EXPECTED_PARTICLE_HORIZON_GLY = 46.5
EXPECTED_FLRW_HUBBLE_HORIZON_GLY = 14.4
EXPECTED_EVENT_HORIZON_GLY = 16.5
EXPECTED_DE_SITTER_HORIZON_GLY = 17.3

TOLERANCE = 0.20


def test_normal_scenario():
    """Test normal scenario at t = 13.8 billion years."""
    print("=" * 70)
    print("TEST: normal scenario")
    print("=" * 70)
    print(f"Target time: {CURRENT_UNIVERSE_AGE_YEARS/1e9:.1f} billion years")
    print()
    
    cosmology = LambdaCDM()
    universe = Universe()
    universe.time = CURRENT_UNIVERSE_AGE_SECONDS
    
    time_years = universe.time / BILLION_YEARS_IN_SECONDS * 1e9
    scale_factor = calculate_scale_factor_at_time(time_years)
    print(f"Scale factor a(t): {scale_factor:.4f}")
    
    # Calculate horizons
    r_particle = cosmology.particle_horizon(universe.time)
    # LambdaCDM.hubble_horizon remains the homogeneous FLRW c/H helper.
    # The scenario LTB Hubble radius lives in MassCalculator.r_hubble_horizon_m.
    r_hubble_flrw = cosmology.hubble_horizon(universe.time, 0)
    r_event = cosmology.cosmological_event_horizon(universe.time, 0)
    r_de_sitter = cosmology.de_sitter_horizon(0)
    
    # Convert to billion light years
    particle_gly = r_particle / BILLION_LY
    hubble_flrw_gly = r_hubble_flrw / BILLION_LY
    event_gly = r_event / BILLION_LY
    de_sitter_gly = r_de_sitter / BILLION_LY
    
    print()
    print("HORIZONS:")
    print("-" * 50)
    print(f"  Particle horizon:   {particle_gly:8.2f} Gly (expected ~{EXPECTED_PARTICLE_HORIZON_GLY})")
    print(f"  FLRW c/H radius:    {hubble_flrw_gly:8.2f} Gly (expected ~{EXPECTED_FLRW_HUBBLE_HORIZON_GLY})")
    print(f"  Event horizon:      {event_gly:8.2f} Gly (expected ~{EXPECTED_EVENT_HORIZON_GLY})")
    print(f"  de Sitter horizon:  {de_sitter_gly:8.2f} Gly (expected ~{EXPECTED_DE_SITTER_HORIZON_GLY})")
    print()
    
    results = []
    
    def check(name, actual, expected):
        if abs(actual - expected) / expected < TOLERANCE:
            results.append((name, "OK", actual, expected))
        else:
            results.append((name, "FAIL", actual, expected))
    
    check("Particle horizon", particle_gly, EXPECTED_PARTICLE_HORIZON_GLY)
    check("FLRW c/H radius", hubble_flrw_gly, EXPECTED_FLRW_HUBBLE_HORIZON_GLY)
    check("Event horizon", event_gly, EXPECTED_EVENT_HORIZON_GLY)
    check("de Sitter horizon", de_sitter_gly, EXPECTED_DE_SITTER_HORIZON_GLY)
    
    # Calculate masses
    rho_matter = (OMEGA_DM + OMEGA_B) * RHO_CRIT / (scale_factor**3)
    M_particle = (4/3) * np.pi * r_particle**3 * rho_matter
    M_hubble_flrw = (4/3) * np.pi * r_hubble_flrw**3 * rho_matter
    
    print("MASSES:")
    print("-" * 50)
    print(f"  Mass in particle horizon: {M_particle:.2e} kg")
    print(f"  Mass in FLRW c/H sphere:  {M_hubble_flrw:.2e} kg")
    print()
    
    if 1e52 < M_particle < 1e55:
        results.append(("Mass (order)", "OK", M_particle, "10^52 - 10^55"))
    else:
        results.append(("Mass (order)", "FAIL", M_particle, "10^52 - 10^55"))
    
    # Check Hubble constant
    H_0 = cosmology.hubble_parameter(universe.time)
    H_0_km_s_Mpc = H_0 * 3.086e19
    EXPECTED_H0 = 67.4
    
    print("HUBBLE CONSTANT:")
    print("-" * 50)
    print(f"  H_0 = {H_0_km_s_Mpc:.1f} km/s/Mpc (expected ~{EXPECTED_H0})")
    print()
    
    check("Hubble constant", H_0_km_s_Mpc, EXPECTED_H0)
    
    # Results
    print("=" * 70)
    print("TEST RESULTS:")
    print("=" * 70)
    
    passed = sum(1 for r in results if r[1] == "OK")
    for name, status, actual, expected in results:
        symbol = "[PASS]" if status == "OK" else "[FAIL]"
        if isinstance(expected, str):
            print(f"  {symbol} {name}: {actual:.2e} (expected {expected})")
        else:
            print(f"  {symbol} {name}: {actual:.2f} (expected {expected})")
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    print("=" * 70)
    
    return passed == len(results)


if __name__ == "__main__":
    success = test_normal_scenario()
    sys.exit(0 if success else 1)
