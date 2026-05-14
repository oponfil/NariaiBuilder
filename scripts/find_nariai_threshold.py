"""
Скрипт для поиска порога энергии, необходимой для создания черной дыры Нариаи.

Запускает симуляцию в headless-режиме (без графического интерфейса) и использует 
бинарный поиск для нахождения минимальной удельной мощности лазера (Вт/кг), 
при которой масса центральной черной дыры достигнет предела Нариаи.

Примеры запуска:
    Продолжить расчет (загружает кэш и пропускает посчитанные эпохи):
    python scripts/find_nariai_threshold.py

    Сбросить кэш и начать расчет заново (перезапишет старые данные):
    python scripts/find_nariai_threshold.py --force
"""
import argparse
import logging
import math
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import _bootstrap  # noqa: F401  -- python scripts/<name>.py
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  -- from scripts.<name> import ...

import config
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
_SWEEP_START_TIME_YEARS = 0.01e9
# Максимальное время сканирования по времени (в годах)
_SWEEP_END_TIME_YEARS = 1.5e9
# Шаг сканирования по времени (в годах)
_SWEEP_STEP_YEARS = 0.01e9
# Максимальное время ожидания симуляции, пока фотоны долетят до центра (в годах)
_MAX_SIMULATION_WAIT_YEARS = float(config.MAX_TIME_YEARS)

# =============================================================================
# Настройки бинарного поиска
# =============================================================================
# Максимально допустимая мощность, чтобы избежать бесконечного цикла (Вт/кг)
_SEARCH_MAX_POWER_W = 1.0e6
# Требуемая точность бинарного поиска (Вт/кг)
_SEARCH_POWER_TOLERANCE_W = 0.01

# Ключи в общем JSON-кэше
_CACHE_KEY_SUCCESS = "successful_runs"
_CACHE_KEY_FAILED = "failed_runs"

# Шаблоны пустых секций (используются и при чтении, и при сбросе через --force)
_EMPTY_SUCCESS = {"times_billion_years": [], "threshold_powers_w_per_kg": []}
_EMPTY_FAILED = {"times_billion_years": [], "max_mass_pct_of_nariai": []}


def _power_decimals() -> int:
    """Сколько знаков после запятой нужно для отображения мощности."""
    return max(1, int(math.ceil(-math.log10(_SEARCH_POWER_TOLERANCE_W))))


def _run_at_power(power_w_per_kg: float, target_time_years: float):
    """Прогон headless-симуляции при заданной удельной мощности лазера."""
    return run_headless_simulation(SimConfig(
        laser_start_years=target_time_years,
        power_w_per_kg=power_w_per_kg,
        max_wait_years=_MAX_SIMULATION_WAIT_YEARS,
    ))


def find_threshold_for_time(target_time_years: float) -> tuple[float, float]:
    """Найти минимальную удельную мощность, дающую успех на данной эпохе.

    Возвращает `(power_w_per_kg, best_failed_mass_kg)`. Если порог не найден
    даже при `_SEARCH_MAX_POWER_W`, возвращает `(inf, best_failed_mass)`.
    """
    power_low = 0.0
    power_high = _SEARCH_MAX_POWER_W
    best_failed_mass = 0.0
    current_power = _SEARCH_MAX_POWER_W

    decimals = _power_decimals()

    # Кэш для предотвращения повторных симуляций с одной и той же мощностью
    power_cache: dict[float, tuple[bool, float, float]] = {}

    def run_sim_cached(power: float, phase_name: str) -> tuple[bool, float, float]:
        if power in power_cache:
            # Не выводим в консоль повторные вычисления, чтобы не засорять лог
            return power_cache[power]

        result = _run_at_power(power, target_time_years)
        triple = (result.success, result.duration_years, result.max_bh_mass_kg)
        power_cache[power] = triple
        verdict = 'SUCCESS' if result.success else 'FAILED'
        pct = result.max_bh_mass_kg / NARIAI_BLACK_HOLE_MASS_KG * 100
        logger.info(f"  [{phase_name}] Power: {power:.6g} W/kg -> {verdict} (Max BH: {pct:.4f}% Nariai)")
        return triple

    # Phase 1A: спускаемся по 3 порядка (делим на 1000) до провала
    phase_1_done = False
    while True:
        success, _, m_final = run_sim_cached(current_power, "Bound Search 1A")

        if not success:
            best_failed_mass = max(best_failed_mass, m_final)
            if current_power == _SEARCH_MAX_POWER_W:
                # Даже максимальная мощность не помогла
                return float('inf'), best_failed_mass
            break

        if current_power <= _SEARCH_POWER_TOLERANCE_W:
            # Даже минимальная погрешность даёт УСПЕХ — порог стремится к нулю
            power_low = 0.0
            power_high = current_power
            phase_1_done = True
            break

        current_power = max(current_power / 1000.0, _SEARCH_POWER_TOLERANCE_W)

    if not phase_1_done:
        # Phase 1B: поднимаемся на порядок (умножаем на 10) до первого успеха
        current_power *= 10.0
        while True:
            success, _, m_final = run_sim_cached(current_power, "Bound Search 1B")
            if success:
                break
            best_failed_mass = max(best_failed_mass, m_final)
            current_power *= 10.0

        # Phase 1C: спускаемся по двойке (делим на 2) до первого провала
        power_high = current_power
        current_power /= 2.0
        while True:
            success, _, m_final = run_sim_cached(current_power, "Bound Search 1C")
            if not success:
                best_failed_mass = max(best_failed_mass, m_final)
                power_low = current_power
                break
            power_high = current_power
            current_power /= 2.0

    # Phase 2: бинарный поиск
    # Используем высокую точность float для поиска порога
    while power_high - power_low > max(_SEARCH_POWER_TOLERANCE_W, power_high * 1e-12):
        power_mid = (power_low + power_high) / 2.0
        # Защита от бесконечного цикла из-за проблем с точностью float
        if power_mid <= power_low or power_mid >= power_high:
            break

        success, _, _ = run_sim_cached(power_mid, "Binary Search")
        if success:
            power_high = power_mid
        else:
            power_low = power_mid

    return power_high, best_failed_mass


def _resolve_data_paths() -> tuple[str, str]:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    return data_dir, json_path


def _load_cached_state(cache: JsonCache, force: bool) -> tuple[list, list, list, list, set]:
    if force:
        cache.reset_section(_CACHE_KEY_SUCCESS, dict(_EMPTY_SUCCESS))
        cache.reset_section(_CACHE_KEY_FAILED, dict(_EMPTY_FAILED))
        return [], [], [], [], set()

    successes = cache.get_section(_CACHE_KEY_SUCCESS, default=dict(_EMPTY_SUCCESS))
    failures = cache.get_section(_CACHE_KEY_FAILED, default=dict(_EMPTY_FAILED))
    successful_times = list(successes.get("times_billion_years", []))
    threshold_powers = list(successes.get("threshold_powers_w_per_kg", []))
    failed_times = list(failures.get("times_billion_years", []))
    failed_max_mass_pcts = list(failures.get("max_mass_pct_of_nariai", []))

    if successful_times or failed_times:
        logger.info(f"Загружено из кэша: {len(successful_times)} успешных и {len(failed_times)} провальных точек.")

    processed = collect_processed_times_billion_years(successful_times, failed_times)
    return successful_times, threshold_powers, failed_times, failed_max_mass_pcts, processed


def _persist_state(cache: JsonCache,
                   successful_times: list, threshold_powers: list,
                   failed_times: list, failed_max_mass_pcts: list) -> None:
    cache.set_section(_CACHE_KEY_SUCCESS, {
        "times_billion_years": successful_times,
        "threshold_powers_w_per_kg": threshold_powers,
    })
    cache.set_section(_CACHE_KEY_FAILED, {
        "times_billion_years": failed_times,
        "max_mass_pct_of_nariai": failed_max_mass_pcts,
    })


def sweep_time_limits(force: bool = False) -> None:
    logger.info("=== Nariai Black Hole Threshold vs Laser Start Time (Cosmological Epoch) ===")

    data_dir, json_path = _resolve_data_paths()
    cache = JsonCache(json_path)

    successful_times, threshold_powers, failed_times, failed_max_mass_pcts, processed = \
        _load_cached_state(cache, force)

    decimals = _power_decimals()
    times_to_test = generate_sweep_times_years(
        _SWEEP_START_TIME_YEARS, _SWEEP_END_TIME_YEARS, _SWEEP_STEP_YEARS,
    )

    for t_years in times_to_test:
        t_b_years = t_years / 1e9
        if is_time_processed(t_b_years, processed):
            logger.info(f"\n--- Пропускаем: {t_b_years:.2f} млрд лет (уже посчитано) ---")
            continue

        logger.info(f"\n--- Testing Laser Start Time: {t_b_years:.2f} Billion Years ---")
        required_power, best_mass = find_threshold_for_time(t_years)

        if required_power != float('inf'):
            logger.info(f"  -> Threshold Power: {required_power:.6g} W/kg")
            successful_times.append(t_b_years)
            threshold_powers.append(required_power)
        else:
            pct = 100.0 * best_mass / NARIAI_BLACK_HOLE_MASS_KG
            logger.warning(f"  -> Failed. Max BH mass achieved: {best_mass:.4e} kg ({pct:.4f}% of Nariai)")
            failed_times.append(t_b_years)
            failed_max_mass_pcts.append(pct)

        _persist_state(cache, successful_times, threshold_powers, failed_times, failed_max_mass_pcts)

    logger.info("\n=== SWEEP COMPLETED ===")
    logger.info(f"\nИтоговые данные успешно сохранены в файл {json_path}")

    if successful_times:
        try:
            plot_vs_time(
                successful_times, threshold_powers,
                title='Minimum Power Required vs Laser Start Time',
                ylabel='Minimum Specific Power (W/kg)',
                color='r',
                out_path=os.path.join(data_dir, 'nariai_power_vs_time_sweep.png'),
            )
        except Exception as e:
            logger.error(f"\nОшибка при построении графика успешных прогонов: {e}")
    else:
        logger.info("\nNo successful data points found to plot.")

    if failed_times:
        try:
            plot_vs_time(
                failed_times, failed_max_mass_pcts,
                title='Max Achievable BH Mass vs Laser Start Time (When Power is Infinite)',
                ylabel='Max Black Hole Mass (% of Nariai Limit)',
                color='b',
                out_path=os.path.join(data_dir, 'nariai_failed_mass_vs_time.png'),
            )
        except Exception as e:
            logger.error(f"\nОшибка при построении графика провальных прогонов: {e}")

    if successful_times or failed_times:
        show_open_figures()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Nariai threshold over time.")
    parser.add_argument('--force', action='store_true', help='Force recalculation and clear cache')
    args = parser.parse_args()

    sweep_time_limits(force=args.force)
