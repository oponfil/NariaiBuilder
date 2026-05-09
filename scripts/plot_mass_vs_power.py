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
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import _bootstrap  # noqa: F401  -- python scripts/<name>.py
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  -- from scripts.<name> import ...

from utils.constants import NARIAI_BLACK_HOLE_MASS_KG
from utils.headless_sim import SimConfig, run_headless_simulation
from utils.json_cache import JsonCache
from utils.plotting import DEFAULT_FIGSIZE, show_open_figures

# --- НАСТРОЙКИ ПО УМОЛЧАНИЮ ---
_DEFAULT_START_TIME_GYR = 13.8
_DEFAULT_NUM_POINTS = 30
_DEFAULT_MIN_POWER_LOG10 = -1.0
_DEFAULT_MAX_POWER_LOG10 = 9.0

_CACHE_KEY = "mass_vs_power"


def _resolve_data_paths() -> tuple[str, str]:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    return data_dir, json_path


def _compute_mass_curve(target_time_years: float, powers: np.ndarray) -> list[float]:
    masses_pct: list[float] = []
    for p in powers:
        print(f"Testing power: {p:.3e} W/kg... ", end="", flush=True)
        result = run_headless_simulation(SimConfig(
            laser_start_years=target_time_years,
            power_w_per_kg=float(p),
        ))
        pct = (result.max_bh_mass_kg / NARIAI_BLACK_HOLE_MASS_KG) * 100.0
        masses_pct.append(pct)
        print(f"[{'SUCCESS' if result.success else 'FAILED'}] Max Mass: {pct:.2f}% of Nariai")
    return masses_pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Black Hole Mass vs Laser Power for a specific cosmological epoch.")
    parser.add_argument('--time', type=float, default=_DEFAULT_START_TIME_GYR,
                        help=f'Laser start time in billion years (default: {_DEFAULT_START_TIME_GYR})')
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation and overwrite this epoch in JSON')
    args = parser.parse_args()

    target_time = args.time * 1e9
    time_str = f"{args.time:.2f}"

    print("=== Black Hole Mass vs Power ===")
    print(f"Target Time: {args.time:.2f} Billion Years")
    print(f"Power range: 10^{_DEFAULT_MIN_POWER_LOG10} to 10^{_DEFAULT_MAX_POWER_LOG10} W/kg ({_DEFAULT_NUM_POINTS} points)")

    data_dir, json_path = _resolve_data_paths()
    cache = JsonCache(json_path)

    cached_curves = cache.get_section(_CACHE_KEY, default={})

    if not args.force and time_str in cached_curves:
        print(f"Загружены кэшированные данные для {time_str} млрд лет.")
        epoch_data = cached_curves[time_str]
        powers = np.array(epoch_data["powers_w_per_kg"])
        masses_pct = np.array(epoch_data["max_mass_pct_of_nariai"])
    else:
        powers = np.logspace(_DEFAULT_MIN_POWER_LOG10, _DEFAULT_MAX_POWER_LOG10, num=_DEFAULT_NUM_POINTS)
        masses_pct_list = _compute_mass_curve(target_time, powers)

        cached_curves[time_str] = {
            "powers_w_per_kg": powers.tolist(),
            "max_mass_pct_of_nariai": masses_pct_list,
        }
        cache.set_section(_CACHE_KEY, cached_curves)
        print(f"Данные сохранены в {json_path}")
        masses_pct = np.array(masses_pct_list)

    # Здесь оси нестандартные (X — масса в %, Y — мощность), поэтому общий plot_vs_time не подходит.
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(masses_pct, powers, marker='o', linestyle='-', color='b', label=f'BH Mass at {args.time} Gyr')
    ax.axvline(x=100.0, color='r', linestyle='--', linewidth=2, label='Nariai Limit (100%)')
    ax.set_yscale('log')
    ax.set_ylabel('Laser Specific Power (W/kg)')
    ax.set_xlabel('Max Black Hole Mass (% of Nariai Limit)')
    ax.set_title(f'Laser Power vs Black Hole Mass at {args.time:.2f} Billion Years')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()

    img_path = os.path.join(data_dir, f'mass_vs_power_{args.time:.2f}Gyr.png')
    fig.savefig(img_path)
    print(f"\nГрафик успешно сохранен в файл: {img_path}")

    show_open_figures()


if __name__ == "__main__":
    main()
