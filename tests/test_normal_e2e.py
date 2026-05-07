"""
E2E test for normal scenario.

Runs the ACTUAL simulation loop (without GUI) from INITIAL_TIME_YEARS to 13.8 billion years
and verifies the results at that time.

This test simulates the core physics loop without rendering.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time as time_module
from physics.cosmology import LambdaCDM
from physics.mass_calculator import MassCalculator
from physics.matter_simulation import MatterSimulation
from physics.objects import Universe
from utils.constants import RHO_CRIT, OMEGA_B, OMEGA_DM
import config

# Constants
BILLION_YEARS_IN_SECONDS = 3.154e16
TARGET_UNIVERSE_AGE_YEARS = 13.8e9
TARGET_UNIVERSE_AGE_SECONDS = 13.8 * BILLION_YEARS_IN_SECONDS
BILLION_LY = 1e9 * 9.461e15

# Expected values
EXPECTED_PARTICLE_HORIZON_GLY = 46.5
EXPECTED_HUBBLE_HORIZON_GLY = 14.4
EXPECTED_H0 = 67.4
TOLERANCE = 0.25


def run_e2e_test():
    """E2E test: runs simulation loop to t = 13.8 billion years."""
    print("=" * 70)
    print("E2E TEST: normal scenario")
    print("=" * 70)
    
    # Сценарий «расширение без коллапса» в пределах прогона (только config, без профилей)
    config.LASER_START_TIME_YEARS = 1e100
    config.INITIAL_TIME_YEARS = 10e6

    # Initialize
    cosmology = LambdaCDM()
    universe = Universe()
    matter_sim = MatterSimulation()
    mass_calc = MassCalculator()
    
    # Get config values
    initial_time_years = config.INITIAL_TIME_YEARS
    dt_years = config.DT_YEARS
    dt_seconds = dt_years * 365.25 * 24 * 3600
    
    universe.time = initial_time_years * 365.25 * 24 * 3600
    
    print(f"Initial: {initial_time_years/1e9:.1f} Gyr")
    print(f"Target: {TARGET_UNIVERSE_AGE_YEARS/1e9:.1f} Gyr")
    print(f"Step: {dt_years/1e6:.0f} Myr")
    print()
    
    # Calculate number of steps
    total_steps = int((TARGET_UNIVERSE_AGE_SECONDS - universe.time) / dt_seconds)
    print(f"Total steps needed: {total_steps}")
    print()
    
    # Run simulation
    print("Running simulation...")
    start_time = time_module.perf_counter()
    
    step = 0
    last_gyr = 0
    
    while universe.time < TARGET_UNIVERSE_AGE_SECONDS:
        cosmology.update_scale_factor(dt_seconds)
        universe.time += dt_seconds
        step += 1
        
        # Progress every 2 Gyr
        current_gyr = int(universe.time / BILLION_YEARS_IN_SECONDS)
        if current_gyr >= last_gyr + 2:
            last_gyr = current_gyr
            print(f"  t = {current_gyr} Gyr ({step}/{total_steps})")
    
    elapsed = time_module.perf_counter() - start_time
    print(f"\nSimulation done: {step} steps in {elapsed:.2f}s")
    print()
    
    # Get results
    r_particle = cosmology.particle_horizon(universe.time)
    r_hubble = cosmology.hubble_horizon(universe.time, 0)
    H0 = cosmology.hubble_parameter(universe.time) * 3.086e19  # to km/s/Mpc
    
    particle_gly = r_particle / BILLION_LY
    hubble_gly = r_hubble / BILLION_LY
    
    print("RESULTS:")
    print("-" * 40)
    print(f"  Scale factor: {cosmology.scale_factor:.4f}")
    print(f"  Particle horizon: {particle_gly:.2f} Gly (expected ~{EXPECTED_PARTICLE_HORIZON_GLY})")
    print(f"  Hubble horizon: {hubble_gly:.2f} Gly (expected ~{EXPECTED_HUBBLE_HORIZON_GLY})")
    print(f"  H_0: {H0:.1f} km/s/Mpc (expected ~{EXPECTED_H0})")
    print()
    
    # Check results
    results = []
    
    def check(name, actual, expected):
        diff = abs(actual - expected) / expected
        status = "PASS" if diff < TOLERANCE else "FAIL"
        results.append((name, status, actual, expected, diff * 100))
    
    check("Particle horizon", particle_gly, EXPECTED_PARTICLE_HORIZON_GLY)
    check("Hubble horizon", hubble_gly, EXPECTED_HUBBLE_HORIZON_GLY)
    check("Hubble constant", H0, EXPECTED_H0)
    
    print("=" * 70)
    print("TEST RESULTS:")
    print("=" * 70)
    
    passed = 0
    for name, status, actual, expected, diff in results:
        symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {symbol} {name}: {actual:.2f} (expected {expected}, diff {diff:.1f}%)")
        if status == "PASS":
            passed += 1
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Time: {elapsed:.2f}s")
    print("=" * 70)
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
