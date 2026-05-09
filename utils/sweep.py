"""
Хелперы для скриптов, делающих свип по моментам старта лазера.

Константы свипа (диапазон, шаг) намеренно живут в самих скриптах: каждый
эксперимент может крутить их независимо. Этот модуль — только функции.
"""
from typing import Iterable

import numpy as np


def generate_sweep_times_years(start_years: float, end_years: float, step_years: float) -> np.ndarray:
    """Список моментов старта лазера для свипа (в годах)."""
    return np.arange(start_years, end_years + step_years, step_years)


def collect_processed_times_billion_years(
    *lists_billion_years: Iterable[float],
    decimals: int = 2,
) -> set[float]:
    """Слить переданные списки времён (в млрд лет) в множество для проверки `in`.

    Округление до `decimals` знаков нивелирует ошибки float, появляющиеся
    при `np.arange` с дробным шагом.
    """
    processed: set[float] = set()
    for lst in lists_billion_years:
        for t in lst:
            processed.add(round(float(t), decimals))
    return processed


def is_time_processed(t_billion_years: float, processed: set[float], decimals: int = 2) -> bool:
    return round(float(t_billion_years), decimals) in processed
