"""
Скрипт для построения графика зависимости итоговой массы черной дыры от мощности лазера
для одной фиксированной космологической эпохи (времени старта).

Скрипт перебирает логарифмически распределенные значения мощности и строит график
процента достижения предела Нариаи.

Примеры запуска:
    С настройками по умолчанию (13.8 млрд лет, 30 точек, от 10^-1 до 10^9 Вт/кг):
    python scripts/plot_mass_vs_power.py

    С указанием другой космологической эпохи (например, 8.5 млрд лет):
    python scripts/plot_mass_vs_power.py --time 8.5

    Принудительный пересчет данных (игнорируя кэш) для текущей эпохи:
    python scripts/plot_mass_vs_power.py --time 13.8 --force
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- НАСТРОЙКИ ПО УМОЛЧАНИЮ ---
DEFAULT_START_TIME_GYR = 13.8
DEFAULT_NUM_POINTS = 30
DEFAULT_MIN_POWER_LOG10 = -1.0
DEFAULT_MAX_POWER_LOG10 = 9.0

# Добавляем корневую директорию в путь для импорта модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics.cosmology import LambdaCDM
from utils.constants import NARIAI_BLACK_HOLE_MASS_KG
# Импортируем готовую headless-симуляцию из основного скрипта поиска
from scripts.find_nariai_threshold import run_simulation_headless

def main():
    parser = argparse.ArgumentParser(description="Plot Black Hole Mass vs Laser Power for a specific cosmological epoch.")
    parser.add_argument('--time', type=float, default=DEFAULT_START_TIME_GYR, help=f'Laser start time in billion years (default: {DEFAULT_START_TIME_GYR})')
    parser.add_argument('--force', action='store_true', help='Force recalculation and overwrite this epoch in JSON')
    args = parser.parse_args()
    
    target_time = args.time * 1e9
    
    print(f"=== Black Hole Mass vs Power ===")
    print(f"Target Time: {args.time:.2f} Billion Years")
    print(f"Power range: 10^{DEFAULT_MIN_POWER_LOG10} to 10^{DEFAULT_MAX_POWER_LOG10} W/kg ({DEFAULT_NUM_POINTS} points)")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    
    time_str = f"{args.time:.2f}"
    
    # Пытаемся загрузить данные из кэша
    saved_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
        except Exception:
            pass
            
    # Проверяем, есть ли уже посчитанные данные для этой эпохи
    mass_vs_power_data = saved_data.get("mass_vs_power", {})
    
    if not args.force and time_str in mass_vs_power_data:
        print(f"Загружены кэшированные данные для {time_str} млрд лет.")
        epoch_data = mass_vs_power_data[time_str]
        powers = np.array(epoch_data["powers_w_per_kg"])
        masses_pct = np.array(epoch_data["max_mass_pct_of_nariai"])
    else:
        # Считаем заново
        powers = np.logspace(DEFAULT_MIN_POWER_LOG10, DEFAULT_MAX_POWER_LOG10, num=DEFAULT_NUM_POINTS)
        masses_pct = []
        shared_cosmology = LambdaCDM()
        
        for p in powers:
            print(f"Testing power: {p:.3e} W/kg... ", end="", flush=True)
            success, t_final, m_final = run_simulation_headless(p, target_time, shared_cosmology)
            pct = (m_final / NARIAI_BLACK_HOLE_MASS_KG) * 100.0
            masses_pct.append(pct)
            print(f"[{'SUCCESS' if success else 'FAILED'}] Max Mass: {pct:.2f}% of Nariai")
            
        # Сохраняем в общий JSON
        if "mass_vs_power" not in saved_data:
            saved_data["mass_vs_power"] = {}
        saved_data["mass_vs_power"][time_str] = {
            "powers_w_per_kg": powers.tolist(),
            "max_mass_pct_of_nariai": masses_pct
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
        print(f"Данные сохранены в {json_path}")
        
    # --- Построение графика ---
    plt.figure(figsize=(10, 6))
    
    # Строим основную линию (синим цветом)
    plt.plot(masses_pct, powers, marker='o', linestyle='-', color='b', label=f'BH Mass at {args.time} Gyr')
    
    # Добавляем вертикальную линию для предела Нариаи (100%)
    plt.axvline(x=100.0, color='r', linestyle='--', linewidth=2, label='Nariai Limit (100%)')
    
    # Настраиваем оси
    plt.yscale('log')
    plt.ylabel('Laser Specific Power (W/kg)')
    plt.xlabel('Max Black Hole Mass (% of Nariai Limit)')
    plt.title(f'Laser Power vs Black Hole Mass at {args.time:.2f} Billion Years')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Сохраняем график в директорию data
    img_path = os.path.join(data_dir, f'mass_vs_power_{args.time:.2f}Gyr.png')
    
    plt.savefig(img_path)
    print(f"\nГрафик успешно сохранен в файл: {img_path}")
    
    # Показываем график пользователю
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
