"""
Унифицированный headless-прогон симуляции коллапса для CLI-скриптов.

Раньше идентичный цикл (с минимальными отличиями в установке режима лазера)
жил отдельно в `scripts/find_nariai_threshold.py` и
`scripts/run_planck_limit_mass_vs_time.py`. Теперь обе симуляции вызывают
`run_headless_simulation` с разными `SimConfig`.
"""
from dataclasses import dataclass

import numpy as np

import config
from physics.cosmology import LambdaCDM
from physics.mass_calculator import MassCalculator
from physics.matter_simulation import MatterSimulation
from physics.objects import Universe
from utils.constants import NARIAI_BLACK_HOLE_MASS_KG, SECONDS_PER_YEAR
from utils.cosmology_utils import calculate_scale_factor_at_time


# По умолчанию мы ждём, пока ЦЧД дорастёт до предела Нариаи, не более 100 млрд лет.
DEFAULT_MAX_WAIT_YEARS = 100.0e9
# Шаг физики headless-симуляции (10 млн лет).
DEFAULT_DT_YEARS = 1.0e6
# Как часто (в шагах) проверять условие ранней остановки симуляции.
DEFAULT_EARLY_STOP_INTERVAL_STEPS = 100
# Сколько шагов dt должно пройти после старта лазера перед первой проверкой
# ранней остановки. Защищает от мгновенного выхода до того, как фотоны
# успели вылететь.
EARLY_STOP_WARMUP_STEPS = 100


@dataclass
class SimConfig:
    """Параметры одного headless-прогона.

    Только один из `power_w_per_kg` / `use_planck_limit` должен задавать режим.
    Если `use_planck_limit=True`, удельная мощность игнорируется,
    `config.LIMIT_TOTAL_POWER_TO_PLANCK` принудительно становится True.
    """
    laser_start_years: float
    power_w_per_kg: float | None = None
    use_planck_limit: bool = False
    max_wait_years: float = DEFAULT_MAX_WAIT_YEARS
    dt_years: float = DEFAULT_DT_YEARS
    early_stop_check_interval: int = DEFAULT_EARLY_STOP_INTERVAL_STEPS
    matter_distribution: str = 'uniform'


@dataclass
class SimResult:
    """Результат одного прогона.

    `success=True` означает, что масса ЦЧД дошла до `NARIAI_BLACK_HOLE_MASS_KG`
    в пределах `max_wait_years`.
    """
    success: bool
    duration_years: float
    max_bh_mass_kg: float


def _calculate_masses(calc: MassCalculator, universe: Universe,
                      cosmology: LambdaCDM, sim: MatterSimulation) -> dict:
    """Тонкий адаптер вокруг `MassCalculator.calculate_masses`.

    `MassCalculator` ожидает один колбэк — ленивую инициализацию точек
    материи; привязываем его к текущим объектам.
    """
    return calc.calculate_masses(
        universe,
        cosmology,
        sim.matter_points,
        False,  # paused
        lambda: sim.initialize_matter_points(universe, cosmology),
    )


def _viable_photons_remaining(sim: MatterSimulation, cosmology: LambdaCDM,
                              universe_time: float) -> int:
    """Сколько лазерных фотонов ещё могут долететь до центра.

    Фотон жизнеспособен, если его текущее физическое расстояние меньше
    космологического горизонта событий — иначе расширение унесёт его быстрее.
    """
    mp = sim.matter_points
    if mp._laser_photon_chi is None or len(mp._laser_photon_chi) == 0:
        return 0
    photon_distances = mp._laser_photon_chi * cosmology.scale_factor
    universe = Universe()
    universe.time = universe_time
    r_boundary = sim._emission_boundary_radius(universe, cosmology)
    return int(np.sum(photon_distances <= r_boundary))


def _should_early_stop(sim: MatterSimulation, cosmology: LambdaCDM,
                       universe_time: float, r_bh: float) -> bool:
    """Лазеры выгорели и в пути нет фотонов, способных дойти, — ждать нечего."""
    mp = sim.matter_points
    power_ret = mp.total_laser_emitters_power_w(cosmology.scale_factor, r_bh)
    power_total = float(power_ret) if power_ret is not None else 0.0
    if power_total > 0.0:
        return False
    return _viable_photons_remaining(sim, cosmology, universe_time) == 0


def run_headless_simulation(cfg: SimConfig) -> SimResult:
    """Прогнать симуляцию без UI и вернуть итоговую массу ЦЧД.

    Имеет два режима:
      * фиксированная удельная мощность лазера (`cfg.power_w_per_kg`);
      * жёсткое ограничение суммарной мощности значением Планка
        (`cfg.use_planck_limit`).
    """
    config.DEBUG = False
    config.LASER_START_TIME_YEARS = cfg.laser_start_years

    if cfg.use_planck_limit:
        config.LIMIT_TOTAL_POWER_TO_PLANCK = True
        config.MATTER_THRUST_POWER_PER_KG_W = 0.0
    else:
        config.LIMIT_TOTAL_POWER_TO_PLANCK = False
        if cfg.power_w_per_kg is None:
            raise ValueError("SimConfig.power_w_per_kg is required when use_planck_limit=False")
        config.MATTER_THRUST_POWER_PER_KG_W = cfg.power_w_per_kg

    laser_start_seconds = cfg.laser_start_years * SECONDS_PER_YEAR
    max_simulation_wait_seconds = laser_start_seconds + cfg.max_wait_years * SECONDS_PER_YEAR
    dt = cfg.dt_years * SECONDS_PER_YEAR
    max_steps = int((max_simulation_wait_seconds - laser_start_seconds) / dt)

    universe = Universe()
    universe.time = laser_start_seconds

    cosmology = LambdaCDM()
    cosmology.scale_factor = calculate_scale_factor_at_time(cfg.laser_start_years)
    cosmology.time_history = [universe.time]
    cosmology.scale_factor_history = [cosmology.scale_factor]

    sim = MatterSimulation(mode=cfg.matter_distribution)
    calc = MassCalculator()

    M_black_hole_kg = 0.0
    r_bh = 0.0

    for step_count in range(max_steps):
        masses = _calculate_masses(calc, universe, cosmology, sim)
        M_black_hole_kg = masses.get('M_black_hole_kg', 0.0)
        r_bh = masses.get('r_black_hole_schwarzschild_m', 0.0)

        if M_black_hole_kg >= NARIAI_BLACK_HOLE_MASS_KG:
            return SimResult(True, universe.time / SECONDS_PER_YEAR, M_black_hole_kg)

        if universe.time >= max_simulation_wait_seconds:
            return SimResult(False, universe.time / SECONDS_PER_YEAR, M_black_hole_kg)

        universe.time += dt
        cosmology.update_scale_factor(dt)
        cosmology.time_history.append(universe.time)
        cosmology.scale_factor_history.append(cosmology.scale_factor)

        sim.update_collapse(universe, cosmology, paused=False, r_black_hole=r_bh, dt_step_signed=dt)

        if (step_count % cfg.early_stop_check_interval == 0
                and sim.matter_points_initialized
                and universe.time > laser_start_seconds + EARLY_STOP_WARMUP_STEPS * dt
                and _should_early_stop(sim, cosmology, universe.time, r_bh)):
            return SimResult(False, universe.time / SECONDS_PER_YEAR, M_black_hole_kg)

    masses = _calculate_masses(calc, universe, cosmology, sim)
    return SimResult(False, universe.time / SECONDS_PER_YEAR, masses.get('M_black_hole_kg', 0.0))
