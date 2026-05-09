"""
Скрипт для расчета максимальной массы черной дыры при суммарной мощности эмиттеров = мощность Планка.

Запускает симуляцию в headless-режиме для разных моментов времени старта лазера.

Примеры запуска:
    Продолжить расчет (загружает кэш и пропускает посчитанные эпохи):
    python scripts/run_planck_limit_mass_vs_time.py

    Сбросить кэш и начать расчет заново (перезапишет старые данные):
    python scripts/run_planck_limit_mass_vs_time.py --force
"""
import argparse
import os

try:
    import _bootstrap  # noqa: F401  -- python scripts/<name>.py
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  -- from scripts.<name> import ...

from utils.constants import NARIAI_BLACK_HOLE_MASS_KG
from utils.headless_sim import SimConfig, run_headless_simulation
from utils.json_cache import JsonCache
from utils.plotting import plot_vs_time, show_open_figures
from utils.sweep import (
    collect_processed_times_billion_years,
    generate_sweep_times_years,
    is_time_processed,
)

# =============================================================================
# Настройки свипа по эпохам и симуляции
# =============================================================================
# Начальное время сканирования по времени (в годах)
_SWEEP_START_TIME_YEARS = 0.1e9
# Максимальное время сканирования по времени (в годах)
_SWEEP_END_TIME_YEARS = 15.0e9
# Шаг сканирования по времени (в годах)
_SWEEP_STEP_YEARS = 0.1e9
# Максимальное время ожидания симуляции, пока фотоны долетят до центра (в годах)
_MAX_SIMULATION_WAIT_YEARS = 100.0e9

_CACHE_KEY = "planck_limit_runs"
_EMPTY_SECTION = {"times_billion_years": [], "max_mass_pct_of_nariai": []}


def _resolve_data_paths() -> tuple[str, str]:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    return data_dir, json_path


def _load_cached_state(cache: JsonCache, force: bool) -> tuple[list, list, set]:
    if force or not cache.has_section(_CACHE_KEY):
        cache.reset_section(_CACHE_KEY, dict(_EMPTY_SECTION))
        return [], [], set()

    section = cache.get_section(_CACHE_KEY, default=dict(_EMPTY_SECTION))
    planck_times = list(section.get("times_billion_years", []))
    planck_max_mass_pcts = list(section.get("max_mass_pct_of_nariai", []))
    if planck_times:
        print(f"Загружено из кэша: {len(planck_times)} расчётов в режиме Planck limit.")
    processed = collect_processed_times_billion_years(planck_times)
    return planck_times, planck_max_mass_pcts, processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep laser start time and measure max BH mass with Planck power limit.")
    parser.add_argument('--force', action='store_true', help='Force recalculation and clear cache')
    args = parser.parse_args()

    data_dir, json_path = _resolve_data_paths()
    cache = JsonCache(json_path)

    planck_times, planck_max_mass_pcts, processed = _load_cached_state(cache, args.force)
    times_to_test = generate_sweep_times_years(
        _SWEEP_START_TIME_YEARS, _SWEEP_END_TIME_YEARS, _SWEEP_STEP_YEARS,
    )

    for t_years in times_to_test:
        t_b_years = t_years / 1e9
        if is_time_processed(t_b_years, processed):
            print(f"\n--- Пропускаем: {t_b_years:.2f} млрд лет (уже посчитано) ---")
            continue

        print(f"\n--- Testing Laser Start Time: {t_b_years:.2f} Billion Years (Planck Limit) ---")
        result = run_headless_simulation(SimConfig(
            laser_start_years=t_years,
            use_planck_limit=True,
            max_wait_years=_MAX_SIMULATION_WAIT_YEARS,
        ))

        pct = 100.0 * result.max_bh_mass_kg / NARIAI_BLACK_HOLE_MASS_KG
        print(f"  -> Max BH mass achieved: {result.max_bh_mass_kg:.4e} kg ({pct:.4f}% of Nariai)")
        planck_times.append(t_b_years)
        planck_max_mass_pcts.append(pct)

        cache.set_section(_CACHE_KEY, {
            "times_billion_years": planck_times,
            "max_mass_pct_of_nariai": planck_max_mass_pcts,
        })

    print("\n=== SWEEP COMPLETED ===")
    print(f"\nИтоговые данные успешно сохранены в файл {json_path}")

    if planck_times:
        try:
            plot_vs_time(
                planck_times, planck_max_mass_pcts,
                title='Max Achievable BH Mass vs Laser Start Time (Total Power = Planck)',
                ylabel='Max Black Hole Mass (% of Nariai Limit)',
                color='purple',
                out_path=os.path.join(data_dir, 'nariai_planck_limit_mass_vs_time.png'),
            )
            show_open_figures()
        except Exception as e:
            print(f"\nОшибка при построении графика: {e}")
    else:
        print("\nNo data points found to plot.")


if __name__ == "__main__":
    main()
