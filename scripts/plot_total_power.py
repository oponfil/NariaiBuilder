"""
Скрипт для построения графика зависимости максимальной суммарной мощности 
(необходимой для создания черной дыры Нариаи) от времени старта лазера.

Берет данные об удельной мощности из кэша (nariai_simulation_data.json),
вычисляет общую массу эмиттеров для каждой эпохи и перемножает их.

Примеры запуска:
    Обычный запуск (использует кэш, если есть):
    python scripts/plot_total_power.py

    Принудительный пересчет (игнорирует кэш и перезаписывает его):
    python scripts/plot_total_power.py --force
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from physics.cosmology import LambdaCDM
from physics.objects import Universe
from physics.matter_simulation import MatterSimulation
from utils.constants import SECONDS_PER_YEAR
from utils.cosmology_utils import calculate_scale_factor_at_time

def get_emitters_mass(target_time_years: float, shared_cosmology: LambdaCDM) -> float:
    # Имитируем начало симуляции для заданного времени,
    # чтобы узнать, сколько массы попадает в горизонт событий и становится эмиттером
    config.LASER_START_TIME_YEARS = target_time_years
    
    universe = Universe()
    universe.time = target_time_years * SECONDS_PER_YEAR
    # Устанавливаем правильный начальный scale_factor для времени старта
    shared_cosmology.scale_factor = calculate_scale_factor_at_time(target_time_years)
    shared_cosmology.time_history = [universe.time]
    shared_cosmology.scale_factor_history = [shared_cosmology.scale_factor]
    
    sim = MatterSimulation()
    sim.initialize_matter_points(universe, shared_cosmology)
    
    # Делаем один шаг, чтобы сработал авто-выбор эмиттеров
    sim.update_collapse(universe, shared_cosmology, paused=False, r_black_hole=0.0, dt_step_signed=1.0)
    
    mask = sim.matter_points.laser_emitter_mask
    if mask is None:
        return 0.0
    
    m_rest = sim.matter_points.masses_per_point
    if m_rest is None:
        return 0.0
        
    return float(np.sum(m_rest[mask]))

def main():
    parser = argparse.ArgumentParser(description="Plot Total Maximum Power vs Time.")
    parser.add_argument('--force', action='store_true', help='Force recalculation and overwrite total power in JSON')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    
    if not os.path.exists(json_path):
        print(f"Файл {json_path} не найден! Сначала запустите find_nariai_threshold.py")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        
    successful_runs = saved_data.get("successful_runs", {})
    times_billion_years = successful_runs.get("times_billion_years", [])
    threshold_powers = successful_runs.get("threshold_powers_w_per_kg", [])
    
    if not times_billion_years or not threshold_powers:
        print("В кэше нет успешных точек для построения графика.")
        sys.exit(1)
        
    # Сортируем по времени, чтобы график не скакал
    times = np.array(times_billion_years)
    powers_spec = np.array(threshold_powers)
    
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    powers_spec = powers_spec[sort_idx]
    
    total_power_data = saved_data.get("total_power", {})
    cached_times = total_power_data.get("times_billion_years", [])
    cached_total_powers = total_power_data.get("total_powers_w", [])
    
    # Сравниваем списки времен, если они совпадают, то данные можно взять из кэша
    can_use_cache = False
    if len(cached_times) == len(times):
        if np.allclose(np.array(cached_times), times):
            can_use_cache = True
            
    if not args.force and can_use_cache:
        print(f"Загружены кэшированные данные суммарной мощности ({len(cached_times)} точек).")
        total_powers = np.array(cached_total_powers)
    else:
        total_powers = []
        shared_cosmology = LambdaCDM()
        
        print("Вычисляем суммарную массу эмиттеров для каждой эпохи...")
        for i, t_b_years in enumerate(times):
            t_years = t_b_years * 1e9
            mass_kg = get_emitters_mass(t_years, shared_cosmology)
            
            # Полная мощность = удельная мощность * массу эмиттеров
            p_total = powers_spec[i] * mass_kg
            total_powers.append(p_total)
            
            print(f"Time: {t_b_years:5.2f} Gyr | Emitters Mass: {mass_kg:.2e} kg | Spec Power: {powers_spec[i]:.2e} W/kg -> Total Power: {p_total:.2e} W")
            
        total_powers = np.array(total_powers)
        
        # Сохраняем в кэш
        saved_data["total_power"] = {
            "times_billion_years": times.tolist(),
            "total_powers_w": total_powers.tolist()
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
        print("Суммарная мощность успешно кэширована в JSON.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_powers, marker='o', linestyle='-', color='purple', label='Total Max Power')
    plt.yscale('log')
    plt.xlabel('Laser Start Time (Billion Years)')
    plt.ylabel('Total Maximum Power (W)')
    plt.title('Total Required Power to Form Nariai Black Hole vs Time')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    img_path = os.path.join(data_dir, 'nariai_total_power_vs_time.png')
    plt.savefig(img_path)
    print(f"\nГрафик успешно сохранен в файл: {img_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
