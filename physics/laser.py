"""
Расчёты лазерной эмиссии и переноса энергии фотонов.

Модуль намеренно не хранит состояние симуляции: массивы точек и фотонных
пакетов остаются в MatterPoints, а здесь лежат только физические формулы.
"""

from __future__ import annotations

import numpy as np

from utils.constants import c
import config


def laser_mass_floor(start_mass_kg, efficiency: float):
    """
    Несжигаемый остаток массы для заданного КПД преобразования в фотоны.

    NaN в start_mass означает, что лазер для этой точки ещё не включался.
    """
    eff = float(np.clip(efficiency, 0.0, 1.0))
    start_mass = np.asarray(start_mass_kg, dtype=np.float64)
    return np.where(np.isnan(start_mass), 0.0, start_mass * (1.0 - eff))


def burnable_mass_kg(current_mass_kg, mass_floor_kg):
    """Масса, доступная для преобразования в лазерное излучение."""
    burnable = np.asarray(current_mass_kg, dtype=np.float64) - np.asarray(mass_floor_kg, dtype=np.float64)
    # Отсечка: из-за экспоненциального закона масса затухает асимптотически.
    # Если оставшаяся сжигаемая масса ничтожно мала (меньше 10^-5 от массы остатка),
    # принудительно обнуляем её, чтобы остановить генерацию бесконечного числа микро-фотонов.
    # Порог: приравниваем к массе остатка, как просил пользователь
    threshold = np.asarray(mass_floor_kg, dtype=np.float64)
    return np.where(burnable > threshold, burnable, 0.0)


def emitted_rest_mass_for_step(
    thrust_power_per_kg_w: float,
    rest_mass_kg,
    dt_seconds: float,
    gamma,
    sqrt_lapse,
    burnable_mass,
):
    """
    Потеря массы покоя за лабораторный шаг dt.

    Внутри одного шага gamma и sqrt(f) считаются постоянными, а доступная
    масса выгорает непрерывно:

        dm = burnable * (1 - exp(-k * dt)),
        k = s * sqrt(f) / (gamma * c^2).

    Возвращается также effective_fraction = dm / dm_linear. Она нужна старой
    одношаговой динамике как усреднённая доля тяги за dt.
    """
    rest_mass = np.asarray(rest_mass_kg, dtype=np.float64)
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    sqrt_f = np.asarray(sqrt_lapse, dtype=np.float64)
    burnable = np.asarray(burnable_mass, dtype=np.float64)

    dm_linear = (
        float(thrust_power_per_kg_w) * rest_mass * float(dt_seconds)
        * sqrt_f / (gamma_arr * c * c)
    )
    exponent = (
        float(thrust_power_per_kg_w) * float(dt_seconds)
        * sqrt_f / (gamma_arr * c * c)
    )
    emitted_dm = burnable * (-np.expm1(-exponent))
    emitted_dm = np.minimum(burnable, emitted_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        active_fraction = np.where(dm_linear > 0.0, emitted_dm / dm_linear, 0.0)
    return emitted_dm, active_fraction


def thrust_delta_gamma_v_per_second(thrust_power_per_kg_w: float, active_fraction):
    """
    Вклад лазерной тяги в d(gamma*v)/dt.

    active_fraction учитывает частичный шаг, если сжигаемая масса закончилась.
    """
    return (float(thrust_power_per_kg_w) / c) * np.asarray(active_fraction, dtype=np.float64)


def emitted_photon_mass_coord_kg(emitted_rest_mass_kg, gamma, beta_radial, sqrt_lapse):
    """
    Координатная масса-эквивалент фотонного пакета E_inf / c^2.

    E_phot_inf/c^2 = gamma * (1 - beta_radial) * dm_rest * sqrt(f_emit).
    """
    mass_emit = (
        np.asarray(gamma, dtype=np.float64)
        * (1.0 - np.asarray(beta_radial, dtype=np.float64))
        * np.asarray(emitted_rest_mass_kg, dtype=np.float64)
        * np.asarray(sqrt_lapse, dtype=np.float64)
    )
    return np.maximum(mass_emit, 0.0)


def cosmological_redshifted_photon_mass_kg(mass_emit_kg, scale_factor_emit, scale_factor_now: float):
    """Масса-эквивалент фотона после космологического redshift: m_emit * a_emit / a_now."""
    return (
        np.asarray(mass_emit_kg, dtype=np.float64)
        * np.asarray(scale_factor_emit, dtype=np.float64)
        / float(scale_factor_now)
    )
