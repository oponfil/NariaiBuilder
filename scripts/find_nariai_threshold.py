"""
Скрипт для поиска порога энергии, необходимой для создания черной дыры Нариаи.

Запускает симуляцию в headless-режиме (без графического интерфейса) и использует 
бинарный поиск для нахождения минимальной удельной мощности лазера (Вт/кг), 
при которой масса центральной черной дыры достигнет предела Нариаи.
"""

import os
import sys
import json

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import matplotlib.pyplot as plt

import config
import numpy as np
from physics.objects import Universe
from physics.cosmology import LambdaCDM
from physics.matter_simulation import MatterSimulation
from physics.mass_calculator import MassCalculator
from utils.config_utils import get_initial_time_seconds, get_collapse_start_time_seconds, get_dt
from utils.constants import SECONDS_PER_YEAR, NARIAI_BLACK_HOLE_MASS_KG
from utils.cosmology_utils import calculate_scale_factor_at_time

# =============================================================================
# Настройки поиска и симуляции
# =============================================================================
# Начальное время сканирования по времени (в годах)
SWEEP_START_TIME_YEARS = 1.0e9
# Максимальное время сканирования по времени (в годах)
SWEEP_END_TIME_YEARS = 15.0e9
# Шаг сканирования по времени (в годах)
SWEEP_STEP_YEARS = 0.1e9
# Максимальное время ожидания симуляции, пока фотоны долетят до центра (в годах)
MAX_SIMULATION_WAIT_YEARS = 100.0e9
# Максимально допустимая мощность, чтобы избежать бесконечного цикла (Вт/кг)
SEARCH_MAX_POWER_W = 1.0e6
# Требуемая точность бинарного поиска (Вт/кг)
SEARCH_POWER_TOLERANCE_W = 0.1


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

def run_simulation_headless(power_w_per_kg: float, laser_start_years: float, shared_cosmology: LambdaCDM) -> tuple[bool, float, float]:
    """
    Запускает симуляцию без UI, возвращает (success, duration_years, max_bh_mass)
    laser_start_years - космологическое время включения лазера (в годах от Большого взрыва).
    """
    # Отключаем вывод профилирования
    config.DEBUG = False
    
    # Настраиваем параметры для конкретного прогона
    config.MATTER_THRUST_POWER_PER_KG_W = power_w_per_kg
    
    universe = Universe()
    
    laser_start_seconds = laser_start_years * SECONDS_PER_YEAR
    universe.time = laser_start_seconds
    
    # Записываем в конфиг, чтобы MatterPoints и MassCalculator знали, когда начался коллапс
    config.LASER_START_TIME_YEARS = laser_start_years
    
    # Максимальное время ожидания симуляции (пока все фотоны не долетят)
    max_simulation_wait_seconds = laser_start_seconds + MAX_SIMULATION_WAIT_YEARS * SECONDS_PER_YEAR
    
    # Сбрасываем переданный объект космологии
    shared_cosmology.scale_factor = shared_cosmology._get_scale_factor_for_time(universe.time)
    shared_cosmology.time_history.clear()
    shared_cosmology.scale_factor_history.clear()
    cosmology = shared_cosmology
    
    sim = MatterSimulation(mode='uniform')
    
    # Вычисляем максимальное количество шагов
    dt_step_years = 1.0e7
    dt = dt_step_years * SECONDS_PER_YEAR
    max_steps = int((max_simulation_wait_seconds - laser_start_seconds) / dt)
    
    # Создаем ЧИСТУЮ космологию для каждой симуляции, чтобы scale_factor не перетекал из прошлого прогона
    cosmology = LambdaCDM()
    # Устанавливаем правильный начальный scale_factor для времени старта
    cosmology.scale_factor = calculate_scale_factor_at_time(laser_start_years)
    cosmology.time_history = [universe.time]
    cosmology.scale_factor_history = [cosmology.scale_factor]
    
    calc = MassCalculator()
    
    M_black_hole_kg = 0.0
    r_bh = 0.0
    
    for step_count in range(max_steps):
        # 1. Считаем текущую массу ЦЧД
        masses = calculate_masses_wrapper(calc, universe, cosmology, sim)
        M_black_hole_kg = masses.get('M_black_hole_kg', 0.0)
        r_bh = masses.get('r_black_hole_schwarzschild_m', 0.0)
        
        # Проверяем успех
        if M_black_hole_kg >= NARIAI_BLACK_HOLE_MASS_KG:
            return True, universe.time / SECONDS_PER_YEAR, M_black_hole_kg
            
        if universe.time >= max_simulation_wait_seconds:
            # Достигли предела ожидания 100 млрд лет
            return False, universe.time / SECONDS_PER_YEAR, M_black_hole_kg
        
        # 2. Обновляем время и космологию
        universe.time += dt
        cosmology.update_scale_factor(dt)
        cosmology.time_history.append(universe.time)
        cosmology.scale_factor_history.append(cosmology.scale_factor)
        
        # 3. Шаг физики
        sim.update_collapse(universe, cosmology, paused=False, r_black_hole=r_bh, dt_step_signed=dt)
        
        mp = sim.matter_points
        
        # 4. Проверка ранней остановки (если лазеры отработали и жизнеспособных фотонов нет)
        if step_count % 10 == 0 and sim.matter_points_initialized:
            if universe.time > laser_start_seconds + 10 * dt:
                power_ret = mp.total_laser_emitters_power_w(cosmology.scale_factor, r_bh)
                power_total = float(power_ret) if power_ret is not None else 0.0
                
                r_event = cosmology.cosmological_event_horizon(universe.time)
                
                if mp._laser_photon_chi is not None and len(mp._laser_photon_chi) > 0:
                    photon_distances = mp._laser_photon_chi * cosmology.scale_factor
                    # Считаем только те фотоны, которые внутри горизонта событий
                    # Фотоны снаружи никогда не долетят до центра
                    num_viable_photons = np.sum(photon_distances <= r_event)
                else:
                    num_viable_photons = 0
                    
                if num_viable_photons == 0 and power_total <= 0.0:
                    return False, universe.time / SECONDS_PER_YEAR, M_black_hole_kg
                    
    masses = calculate_masses_wrapper(calc, universe, cosmology, sim)
    return False, universe.time / SECONDS_PER_YEAR, masses.get('M_black_hole_kg', 0.0)


def find_threshold_for_time(target_time_years: float, shared_cosmology: LambdaCDM) -> tuple[float, float]:
    power_low = 0.0
    power_high = SEARCH_MAX_POWER_W
    best_failed_mass = 0.0
    
    current_power = SEARCH_MAX_POWER_W
    
    # Определяем количество знаков для округления на основе погрешности
    decimals = max(1, int(math.ceil(-math.log10(SEARCH_POWER_TOLERANCE_W))))
    
    # Phase 1A: Вначале спускаемся по 3 порядка (делим на 1000) до провала
    phase_1_done = False
    while True:
        success, t_final, m_final = run_simulation_headless(current_power, target_time_years, shared_cosmology)
        print(f"  [Bound Search 1A] Power: {current_power:.{decimals}f} W/kg -> {'SUCCESS' if success else 'FAILED'} (Max BH: {m_final/NARIAI_BLACK_HOLE_MASS_KG*100:.4f}% Nariai)")
        
        if not success:
            if m_final > best_failed_mass:
                best_failed_mass = m_final
            if current_power == SEARCH_MAX_POWER_W:
                # Если даже максимальная мощность не помогла
                return float('inf'), best_failed_mass
            break
            
        if current_power <= SEARCH_POWER_TOLERANCE_W:
            # Даже минимальная погрешность дает УСПЕХ (порог стремится к нулю)
            power_low = 0.0
            power_high = current_power
            phase_1_done = True
            break
            
        current_power = max(current_power / 1000.0, SEARCH_POWER_TOLERANCE_W)
        
    if not phase_1_done:
        # Phase 1B: Поднимаемся на порядок (умножаем на 10) до первого успеха
        current_power *= 10.0
        while True:
            success, t_final, m_final = run_simulation_headless(current_power, target_time_years, shared_cosmology)
            print(f"  [Bound Search 1B] Power: {current_power:.{decimals}f} W/kg -> {'SUCCESS' if success else 'FAILED'} (Max BH: {m_final/NARIAI_BLACK_HOLE_MASS_KG*100:.4f}% Nariai)")
            
            if success:
                break
                
            if m_final > best_failed_mass:
                best_failed_mass = m_final
                
            current_power *= 10.0
            
        # Phase 1C: Спускаемся по двойке (делим на 2) до первого провала
        power_high = current_power
        current_power /= 2.0
        while True:
            success, t_final, m_final = run_simulation_headless(current_power, target_time_years, shared_cosmology)
            print(f"  [Bound Search 1C] Power: {current_power:.{decimals}f} W/kg -> {'SUCCESS' if success else 'FAILED'} (Max BH: {m_final/NARIAI_BLACK_HOLE_MASS_KG*100:.4f}% Nariai)")
            
            if not success:
                if m_final > best_failed_mass:
                    best_failed_mass = m_final
                power_low = current_power
                break
                
            power_high = current_power
            current_power /= 2.0
                
    # Phase 2: Binary search
    while power_high - power_low > SEARCH_POWER_TOLERANCE_W:
        power_mid = round((power_low + power_high) / 2.0, decimals)
        success, t_final, m_final = run_simulation_headless(power_mid, target_time_years, shared_cosmology)
        print(f"  [Binary Search] Power: {power_mid:.{decimals}f} W/kg -> {'SUCCESS' if success else 'FAILED'} (Max BH: {m_final/NARIAI_BLACK_HOLE_MASS_KG*100:.4f}% Nariai)")
        if success:
            power_high = power_mid
        else:
            power_low = power_mid
            
    # КРИТИЧЕСКИ ВАЖНО: всегда округляем ВВЕРХ, чтобы гарантировать успех.
    # Обычный round(1.25, 1) в Python округляет к ближайшему четному (то есть к 1.2),
    # что приводит к выдаче "провального" порога!
    factor = 10 ** decimals
    power_high = math.ceil(power_high * factor) / factor
    return power_high, best_failed_mass


def sweep_time_limits():
    print("=== Nariai Black Hole Threshold vs Laser Start Time (Cosmological Epoch) ===")
    
    # Начинаем с начального времени, заданного в конфигурации
    start_time = SWEEP_START_TIME_YEARS
    
    # Генерируем список временных отсечек
    times_to_test = np.arange(start_time, SWEEP_END_TIME_YEARS + SWEEP_STEP_YEARS, SWEEP_STEP_YEARS)
    
    successful_times = []
    threshold_powers = []
    failed_times = []
    failed_max_mass_pcts = []
    
    # Создаем космологию один раз, чтобы не загружать JSON на каждой итерации
    shared_cosmology = LambdaCDM()
    
    for t_years in times_to_test:
        print(f"\n--- Testing Laser Start Time: {t_years/1e9:.2f} Billion Years ---")
        
        # Получаем порог мощности для данного времени старта
        required_power, best_mass = find_threshold_for_time(t_years, shared_cosmology)
        
        # Вычисляем decimals снова для корректного форматирования вывода
        decimals = max(1, int(math.ceil(-math.log10(SEARCH_POWER_TOLERANCE_W))))
        
        if required_power != float('inf'):
            print(f"  -> Threshold Power: {required_power:.{decimals}f} W/kg")
            successful_times.append(t_years / 1e9)
            threshold_powers.append(required_power)
        else:
            pct = 100.0 * best_mass / NARIAI_BLACK_HOLE_MASS_KG
            print(f"  -> Failed. Max BH mass achieved: {best_mass:.4e} kg ({pct:.4f}% of Nariai)")
            failed_times.append(t_years / 1e9)
            failed_max_mass_pcts.append(pct)
            
    print("\n=== SWEEP COMPLETED ===")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Сохраняем сырые данные (и успешные, и провальные) в единый JSON
    json_path = os.path.join(data_dir, 'nariai_threshold_sweep.json')
    results_data = {
        "successful_runs": {
            "times_billion_years": successful_times,
            "threshold_powers_w_per_kg": threshold_powers
        },
        "failed_runs": {
            "times_billion_years": failed_times,
            "max_mass_pct_of_nariai": failed_max_mass_pcts
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
    print(f"\nДанные успешно сохранены в файл {json_path}")
    
    if successful_times:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(successful_times, threshold_powers, marker='o', linestyle='-', color='r')
            
            # Логарифмическая шкала Y для наглядности (мощности могут сильно отличаться)
            plt.yscale('log')
            
            plt.title('Minimum Power Required vs Laser Start Time')
            plt.xlabel('Laser Start Time (Billion Years)')
            plt.ylabel('Minimum Specific Power (W/kg)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            # Сохраняем в файл в папку data
            img_path = os.path.join(data_dir, 'nariai_power_vs_time_sweep.png')
            plt.savefig(img_path)
            print(f"График успешно сохранен в файл {img_path}")
            # plt.show() # Покажем все графики в конце
        except Exception as e:
            print(f"\nОшибка при построении графика успешных прогонов: {e}")
    else:
        print("\nNo successful data points found to plot.")

    if failed_times:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(failed_times, failed_max_mass_pcts, marker='o', linestyle='-', color='b')
            
            plt.title('Max Achievable BH Mass vs Laser Start Time (When Power is Infinite)')
            plt.xlabel('Laser Start Time (Billion Years)')
            plt.ylabel('Max Black Hole Mass (% of Nariai Limit)')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            img_path_failed = os.path.join(data_dir, 'nariai_failed_mass_vs_time.png')
            plt.savefig(img_path_failed)
            print(f"График провальных запусков сохранен в файл {img_path_failed}")
        except Exception as e:
            print(f"\nОшибка при построении графика провальных прогонов: {e}")
            
    # Если мы построили хотя бы один график
    if successful_times or failed_times:
        try:
            plt.show()
        except Exception:
            pass

if __name__ == "__main__":
    sweep_time_limits()
