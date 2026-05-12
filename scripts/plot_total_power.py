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
import argparse
import logging
import os
import sys

import numpy as np

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
from physics.cosmology import LambdaCDM
from physics.matter_simulation import MatterSimulation
from physics.objects import Universe
from utils.constants import SECONDS_PER_YEAR
from utils.cosmology_utils import calculate_scale_factor_at_time
from utils.json_cache import JsonCache
from utils.plotting import plot_vs_time, show_open_figures

_CACHE_KEY_THRESHOLDS = "successful_runs"
_CACHE_KEY_TOTAL = "total_power"


def get_emitters_mass(target_time_years: float, shared_cosmology: LambdaCDM) -> float:
    """Сколько массы попадает в горизонт событий и становится лазерным эмиттером."""
    config.LASER_START_TIME_YEARS = target_time_years

    universe = Universe()
    universe.time = target_time_years * SECONDS_PER_YEAR
    shared_cosmology.scale_factor = calculate_scale_factor_at_time(target_time_years)
    shared_cosmology.time_history = [universe.time]
    shared_cosmology.scale_factor_history = [shared_cosmology.scale_factor]

    sim = MatterSimulation()
    sim.initialize_matter_points(universe, shared_cosmology)
    # Один шаг, чтобы сработал авто-выбор эмиттеров
    sim.update_collapse(universe, shared_cosmology, paused=False, r_black_hole=0.0, dt_step_signed=1.0)

    mask = sim.matter_points.laser_emitter_mask
    if mask is None:
        return 0.0

    m_rest = sim.matter_points.masses_per_point
    if m_rest is None:
        return 0.0

    return float(np.sum(m_rest[mask]))


def _resolve_data_paths() -> tuple[str, str]:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    json_path = os.path.join(data_dir, 'nariai_simulation_data.json')
    return data_dir, json_path


def _load_thresholds(cache: JsonCache) -> tuple[np.ndarray, np.ndarray] | None:
    section = cache.get_section(_CACHE_KEY_THRESHOLDS, default={})
    times = section.get("times_billion_years", [])
    powers = section.get("threshold_powers_w_per_kg", [])
    if not times or not powers:
        return None
    times_arr = np.array(times)
    powers_arr = np.array(powers)
    sort_idx = np.argsort(times_arr)
    return times_arr[sort_idx], powers_arr[sort_idx]


def _try_use_total_power_cache(cache: JsonCache, times: np.ndarray) -> np.ndarray | None:
    section = cache.get_section(_CACHE_KEY_TOTAL, default={})
    cached_times = section.get("times_billion_years", [])
    cached_total = section.get("total_powers_w", [])
    if len(cached_times) != len(times):
        return None
    if not np.allclose(np.array(cached_times), times):
        return None
    logger.info(f"Загружены кэшированные данные суммарной мощности ({len(cached_times)} точек).")
    return np.array(cached_total)


def _compute_total_powers(times: np.ndarray, specific_powers: np.ndarray) -> np.ndarray:
    shared_cosmology = LambdaCDM()
    total_powers: list[float] = []
    logger.info("Вычисляем суммарную массу эмиттеров для каждой эпохи...")
    for i, t_b_years in enumerate(times):
        mass_kg = get_emitters_mass(float(t_b_years) * 1e9, shared_cosmology)
        p_total = specific_powers[i] * mass_kg
        total_powers.append(p_total)
        logger.info(f"Time: {t_b_years:5.2f} Gyr | Emitters Mass: {mass_kg:.2e} kg | "
              f"Spec Power: {specific_powers[i]:.2e} W/kg -> Total Power: {p_total:.2e} W")
    return np.array(total_powers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Total Maximum Power vs Time.")
    parser.add_argument('--force', action='store_true',
                        help='Force recalculation and overwrite total power in JSON')
    args = parser.parse_args()

    data_dir, json_path = _resolve_data_paths()
    if not os.path.exists(json_path):
        logger.error(f"Файл {json_path} не найден! Сначала запустите find_nariai_threshold.py")
        sys.exit(1)

    cache = JsonCache(json_path)
    thresholds = _load_thresholds(cache)
    if thresholds is None:
        logger.error("В кэше нет успешных точек для построения графика.")
        sys.exit(1)
    times, powers_spec = thresholds

    cached_total = None if args.force else _try_use_total_power_cache(cache, times)
    if cached_total is not None:
        total_powers = cached_total
    else:
        total_powers = _compute_total_powers(times, powers_spec)
        cache.set_section(_CACHE_KEY_TOTAL, {
            "times_billion_years": times.tolist(),
            "total_powers_w": total_powers.tolist(),
        })
        logger.info("Суммарная мощность успешно кэширована в JSON.")

    plot_vs_time(
        times, total_powers,
        title='Total Required Power to Form Nariai Black Hole vs Time',
        ylabel='Total Maximum Power (W)',
        color='purple',
        out_path=os.path.join(data_dir, 'nariai_total_power_vs_time.png'),
        label='Total Max Power',
    )
    show_open_figures()


if __name__ == "__main__":
    main()
