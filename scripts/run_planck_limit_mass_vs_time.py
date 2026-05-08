"""
Скрипт для расчета максимальной массы черной дыры при суммарной мощности эмиттеров = мощность Планка.

Запускает симуляцию в headless-режиме для разных моментов времени старта лазера.

Примеры запуска:
    Продолжить расчет (загружает кэш и пропускает посчитанные эпохи):
    python scripts/run_planck_limit_mass_vs_time.py

    Сбросить кэш и начать расчет заново (перезапишет старые данные):
    python scripts/run_planck_limit_mass_vs_time.py --force
"""

import os
import sys
import json
import argparse

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

import config
from physics.objects import Universe
from physics.cosmology import LambdaCDM
from physics.matter_simulation import MatterSimulation
from physics.mass_calculator import MassCalculator
from utils.constants import SECONDS_PER_YEAR, NARIAI_BLACK_HOLE_MASS_KG
from utils.cosmology_utils import calculate_scale_factor_at_time

# =============================================================================
# Настройки
# =============================================================================
SWEEP_START_TIME_YEARS = 0.1e9
SWEEP_END_TIME_YEARS = 15.0e9
SWEEP_STEP_YEARS = 0.1e9
MAX_SIMULATION_WAIT_YEARS = 100.0e9

def calculate_masses_wrapper(calc, universe, cosmology, sim):
    def get_physical_points_wrapper(particle_horizon_physical):
        return sim.get_physical_points_and_distances(universe, cosmology, particle_horizon_physical)

    def initialize_matter_points_wrapper():
        sim.initialize_matter_points(universe, cosmology)

    def add_matter_points_wrapper(num_new_points, radius_physical):
        sim.add_matter_points(universe, cosmology, num_new_points, radius_physical)

    return calc.calculate_masses(
        universe,
        cosmology,
        sim.matter_points,
        False,  # paused
        get_physical_points_wrapper,
        initialize_matter_points_wrapper,
        add_matter_points_wrapper,
    )

def run_planck_simulation(laser_start_years: float, shared_cosmology: LambdaCDM) -> tuple[float, float]:
    """
    Запускает симуляцию без UI, возвращает (duration_years, max_bh_mass)
    """
    config.DEBUG = False
    
    # Включаем режим Планка
    config.LIMIT_TOTAL_POWER_TO_PLANCK = True
    config.MATTER_THRUST_POWER_PER_KG_W = 0.0
    
    universe = Universe()
    
    laser_start_seconds = laser_start_years * SECONDS_PER_YEAR
    universe.time = laser_start_seconds
    config.LASER_START_TIME_YEARS = laser_start_years
    
    max_simulation_wait_seconds = laser_start_seconds + MAX_SIMULATION_WAIT_YEARS * SECONDS_PER_YEAR
    
    shared_cosmology.scale_factor = shared_cosmology._get_scale_factor_for_time(universe.time)
    shared_cosmology.time_history.clear()
    shared_cosmology.scale_factor_history.clear()
    
    sim = MatterSimulation(mode='uniform')
    
    dt_step_years = 1.0e7
    dt = dt_step_years * SECONDS_PER_YEAR
    max_steps = int((max_simulation_wait_seconds - laser_start_seconds) / dt)
    
    cosmology = LambdaCDM()
    cosmology.scale_factor = calculate_scale_factor_at_time(laser_start_years)
    cosmology.time_history = [universe.time]
    cosmology.scale_factor_history = [cosmology.scale_factor]
    
    calc = MassCalculator()
    
    M_black_hole_kg = 0.0
    r_bh = 0.0
    
    for step_count in range(max_steps):
        masses = calculate_masses_wrapper(calc, universe, cosmology, sim)
        M_black_hole_kg = masses.get('M_black_hole_kg', 0.0)
        r_bh = masses.get('r_black_hole_schwarzschild_m', 0.0)
        
        if M_black_hole_kg >= NARIAI_BLACK_HOLE_MASS_KG:
            return universe.time / SECONDS_PER_YEAR, M_black_hole_kg
            
        if universe.time >= max_simulation_wait_seconds:
            return universe.time / SECONDS_PER_YEAR, M_black_hole_kg
        
        universe.time += dt
        cosmology.update_scale_factor(dt)
        cosmology.time_history.append(universe.time)
        cosmology.scale_factor_history.append(cosmology.scale_factor)
        
        sim.update_collapse(universe, cosmology, paused=False, r_black_hole=r_bh, dt_step_signed=dt)
        
        mp = sim.matter_points
        if step_count % 10 == 0 and sim.matter_points_initialized:
            if universe.time > laser_start_seconds + 10 * dt:
                power_ret = mp.total_laser_emitters_power_w(cosmology.scale_factor, r_bh)
                power_total = float(power_ret) if power_ret is not None else 0.0
                
                r_event = cosmology.cosmological_event_horizon(universe.time)
                
                if mp._laser_photon_chi is not None and len(mp._laser_photon_chi) > 0:
                    photon_distances = mp._laser_photon_chi * cosmology.scale_factor
                    num_viable_photons = np.sum(photon_distances <= r_event)
                else:
                    num_viable_photons = 0
                    
                if num_viable_photons == 0 and power_total <= 0.0:
                    return universe.time / SECONDS_PER_YEAR, M_black_hole_kg
                    
    masses = calculate_masses_wrapper(calc, universe, cosmology, sim)
    return universe.time / SECONDS_PER_YEAR, masses.get('M_black_hole_kg', 0.0)

def main():
    parser = argparse.ArgumentParser(description="Sweep laser start time and measure max BH mass with Planck power limit.")
    parser.add_argument('--force', action='store_true', help="Сбросить кэш и пересчитать всё заново")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    
    num_steps = int(round((SWEEP_END_TIME_YEARS - SWEEP_START_TIME_YEARS) / SWEEP_STEP_YEARS)) + 1
    times_to_test = [SWEEP_START_TIME_YEARS + i * SWEEP_STEP_YEARS for i in range(num_steps)]
    
    saved_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
        except json.JSONDecodeError:
            print("Ошибка чтения JSON, файл будет перезаписан.")
            
    if not args.force and "planck_limit_runs" in saved_data:
        planck_times = saved_data["planck_limit_runs"].get("times_billion_years", [])
        planck_max_mass_pcts = saved_data["planck_limit_runs"].get("max_mass_pct_of_nariai", [])
        if planck_times:
            print(f"Загружено из кэша: {len(planck_times)} расчётов в режиме Planck limit.")
        processed_times = set([round(t, 2) for t in planck_times])
    else:
        saved_data["planck_limit_runs"] = {"times_billion_years": [], "max_mass_pct_of_nariai": []}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
        planck_times = []
        planck_max_mass_pcts = []
        processed_times = set()

    shared_cosmology = LambdaCDM()
    
    for t_years in times_to_test:
        t_b_years = t_years / 1e9
        if round(t_b_years, 2) in processed_times:
            print(f"\n--- Пропускаем: {t_b_years:.2f} млрд лет (уже посчитано) ---")
            continue
            
        print(f"\n--- Testing Laser Start Time: {t_b_years:.2f} Billion Years (Planck Limit) ---")
        
        t_final, m_final = run_planck_simulation(t_years, shared_cosmology)
        
        pct = 100.0 * m_final / NARIAI_BLACK_HOLE_MASS_KG
        print(f"  -> Max BH mass achieved: {m_final:.4e} kg ({pct:.4f}% of Nariai)")
        planck_times.append(t_b_years)
        planck_max_mass_pcts.append(pct)
            
        saved_data["planck_limit_runs"] = {
            "times_billion_years": planck_times,
            "max_mass_pct_of_nariai": planck_max_mass_pcts
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
            
    print("\n=== SWEEP COMPLETED ===")
    print(f"\nИтоговые данные успешно сохранены в файл {json_path}")
    
    if planck_times:
        try:
            sorted_indices = np.argsort(planck_times)
            sorted_times = np.array(planck_times)[sorted_indices]
            sorted_pcts = np.array(planck_max_mass_pcts)[sorted_indices]
            
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_times, sorted_pcts, marker='o', linestyle='-', color='purple')
            
            plt.title('Max Achievable BH Mass vs Laser Start Time (Total Power = Planck)')
            plt.xlabel('Laser Start Time (Billion Years)')
            plt.ylabel('Max Black Hole Mass (% of Nariai Limit)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            img_path = os.path.join(data_dir, 'nariai_planck_limit_mass_vs_time.png')
            plt.savefig(img_path)
            print(f"График успешно сохранен в файл {img_path}")
        except Exception as e:
            print(f"\nОшибка при построении графика: {e}")
    else:
        print("\nNo data points found to plot.")

if __name__ == "__main__":
    main()
