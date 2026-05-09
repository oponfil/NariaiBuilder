"""
Скрипт для предвычисления космологических горизонтов
Вычисляет горизонты один раз и сохраняет в файл для быстрого доступа
"""
import json
import os

import numpy as np
from scipy import integrate

try:
    import _bootstrap  # noqa: F401  -- python scripts/<name>.py
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  -- from scripts.<name> import ...

from utils.constants import (
    H0_s,
    OMEGA_B,
    OMEGA_DM,
    OMEGA_LAMBDA,
    SECONDS_PER_YEAR,
    c,
)
from utils.cosmology_utils import calculate_scale_factor_at_time

# =============================================================================
# Предрасчёт горизонтов (годы)
# =============================================================================
PRECOMPUTE_DT_YEARS = 1e6  # 1 миллион лет
PRECOMPUTE_TIME_START_YEARS = PRECOMPUTE_DT_YEARS
PRECOMPUTE_TIME_END_YEARS = 150e9

_TIME_START_YEARS = PRECOMPUTE_TIME_START_YEARS
_TIME_END_YEARS = PRECOMPUTE_TIME_END_YEARS

# Вычисляем количество точек для сетки предвычисления горизонтов.
# Количество точек рассчитывается из желаемого шага симуляции (PRECOMPUTE_DT_YEARS).
_NUM_POINTS = int((_TIME_END_YEARS - _TIME_START_YEARS) / PRECOMPUTE_DT_YEARS)

# Константы для численного интегрирования горизонтов
_MIN_TIME_YEARS = 1e6
_MIN_SCALE_FACTOR = 1e-10
_MIN_TIME_FRACTION = 1e-6

# Параметры решателя ОДУ для масштабного фактора
_ODE_PAST_RTOL = 1e-4
_ODE_PAST_ATOL = 1e-7
_ODE_FUTURE_RTOL = 1e-3
_ODE_FUTURE_ATOL = 1e-6

# Параметры квадратурного интегрирования (горизонты)
_QUAD_LIMIT = 500
_QUAD_EPSABS = 1e-6
_QUAD_EPSREL = 1e-4

# Время в будущем для интегрирования горизонта событий (1000 млрд лет)
_EVENT_HORIZON_FUTURE_INTEGRATION_TIME_SECONDS = 1000.0 * 1e9 * SECONDS_PER_YEAR

def compute_particle_horizon(time_years, scale_factor, previous_integral=None, previous_time=None):
    """
    Вычислить горизонт частиц для заданного времени
    
    Args:
        time_years: Время в годах
        scale_factor: Масштабный фактор в это время
        previous_integral: Предыдущее значение интеграла (для накопления)
        previous_time: Предыдущее время (для накопления)
    
    Returns:
        (particle_horizon_physical, integral_result) - физический горизонт и значение интеграла
    """
    if time_years <= 0:
        return (0.0, 0.0)
    
    # Для очень малых времен возвращаем 0
    if time_years < _MIN_TIME_YEARS:
        return (0.0, 0.0)
    
    time_seconds = time_years * SECONDS_PER_YEAR
    omega_m = OMEGA_DM + OMEGA_B
    
    def da_dt_past(t, a_val):
        if a_val <= 0:
            a_val = _MIN_SCALE_FACTOR
        h = H0_s * np.sqrt(omega_m / (a_val**3) + OMEGA_LAMBDA)
        return a_val * h
    
    # ВАЖНО: Решаем ОДУ от очень малого времени до текущего времени
    # Начинаем с времени, когда Вселенная уже начала расширяться
    t_start = max(_MIN_TIME_YEARS * SECONDS_PER_YEAR, time_seconds * _MIN_TIME_FRACTION)
    
    # Вычисляем начальное значение a при t_start используя приближение для ранней Вселенной
    # При доминировании материи: a(t) ∝ t^(2/3)
    a_start = scale_factor * (t_start / time_seconds) ** (2/3)
    if a_start <= 0:
        a_start = _MIN_SCALE_FACTOR
    
    sol_past = None
    try:
        # Увеличиваем точность для более точного результата
        result_past = integrate.solve_ivp(da_dt_past, [t_start, time_seconds], [a_start],
                                          method='RK45', dense_output=True,
                                          rtol=_ODE_PAST_RTOL, atol=_ODE_PAST_ATOL)
        if result_past.success:
            sol_past = result_past.sol
    except:
        pass
    
    def integrand(t_prime):
        """Возвращает 1/a(t') для интегрирования"""
        if t_prime <= 0:
            return 1.0 / _MIN_SCALE_FACTOR
        if sol_past is not None:
            try:
                a_at_t = float(sol_past(t_prime)[0])
                if a_at_t <= 0:
                    return 1.0 / _MIN_SCALE_FACTOR
                return 1.0 / a_at_t
            except:
                pass
        return 1.0 / _MIN_SCALE_FACTOR
    
    try:
        # ВАЖНО: Используем накопление, если доступно предыдущее значение
        if previous_integral is not None and previous_time is not None and previous_time < time_seconds:
            # Интегрируем только новый участок от previous_time до time_seconds
            # Early time contribution уже включена в previous_integral, не добавляем
            t_min = max(t_start, previous_time)
            integral_new, _ = integrate.quad(integrand, t_min, time_seconds,
                                              limit=_QUAD_LIMIT, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
            integral_result = previous_integral + integral_new
        else:
            # Интегрируем весь участок от t_start до time_seconds
            t_min = t_start
            integral_result, _ = integrate.quad(integrand, t_min, time_seconds,
                                                limit=_QUAD_LIMIT, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
            
            # ВАЖНО: Добавляем аналитический вклад от ранних времён (0 -> t_start)
            # ТОЛЬКО при первом интегрировании (когда нет previous_integral)
            # В эпоху доминирования материи: a(t) = a_start * (t/t_start)^(2/3)
            # Интеграл: ∫[0 to t_start] dt/a(t) = 3 * t_start / a_start
            early_time_contribution = 3.0 * t_start / a_start
            integral_result += early_time_contribution
        
        particle_r = c * scale_factor * integral_result
        return (max(0.0, particle_r), integral_result)
    except:
        return (0.0, 0.0)

def compute_event_horizon(time_years, scale_factor):
    """Вычислить космологический горизонт событий для заданного времени"""
    time_seconds = time_years * SECONDS_PER_YEAR
    omega_m = OMEGA_DM + OMEGA_B
    
    def da_dt_future(t, a_val):
        if a_val <= 0:
            a_val = _MIN_SCALE_FACTOR
        h = H0_s * np.sqrt(omega_m / (a_val**3) + OMEGA_LAMBDA)
        return a_val * h
    
    t_future = time_seconds + _EVENT_HORIZON_FUTURE_INTEGRATION_TIME_SECONDS
    
    sol_future = None
    try:
        result_future = integrate.solve_ivp(da_dt_future, [time_seconds, t_future], [scale_factor],
                                            method='RK45', dense_output=True,
                                            rtol=_ODE_FUTURE_RTOL, atol=_ODE_FUTURE_ATOL)
        if result_future.success:
            sol_future = result_future.sol
    except:
        pass
    
    def integrand(t_prime):
        if t_prime <= time_seconds:
            return 1.0 / scale_factor
        if sol_future is not None:
            try:
                a_at_t = float(sol_future(t_prime)[0])
                if a_at_t <= 0:
                    return 1.0 / _MIN_SCALE_FACTOR
                return 1.0 / a_at_t
            except:
                pass
        return 1.0 / _MIN_SCALE_FACTOR
    
    try:
        integral_result, _ = integrate.quad(integrand, time_seconds, t_future,
                                            limit=_QUAD_LIMIT, epsabs=_QUAD_EPSABS, epsrel=_QUAD_EPSREL)
        event_r = c * scale_factor * integral_result
        return max(0.0, event_r)
    except:
        return 0.0

def precompute_horizons():
    """Предвычислить горизонты для диапазона времен"""
    print("Precomputing cosmological horizons...")
    print(f"Time range: {_TIME_START_YEARS/1e9:.3f} - {_TIME_END_YEARS/1e9:.1f} billion years")
    print(f"Number of points: {_NUM_POINTS}")
    
    # ОПТИМИЗАЦИЯ: Добавляем точку для времени 0 (горизонт = 0)
    times_years = np.linspace(_TIME_START_YEARS, _TIME_END_YEARS, _NUM_POINTS - 1)
    times_years = np.concatenate([[0.0], times_years])  # Добавляем 0 в начало
    times_seconds = times_years * SECONDS_PER_YEAR
    
    particle_horizons = []
    event_horizons = []
    scale_factors = []
    
    # Для времени 0 горизонты равны 0
    particle_horizons.append(0.0)
    event_horizons.append(0.0)
    scale_factors.append(_MIN_SCALE_FACTOR)
    
    # ВАЖНО: Используем накопление интеграла для более точного вычисления
    previous_integral = 0.0
    previous_time_seconds = 0.0
    previous_comoving_horizon = 0.0
    
    print("Starting computation...", flush=True)
    
    for i, time_years in enumerate(times_years[1:], start=1):  # Пропускаем первый элемент (0)
        # Выводим прогресс каждый 1 миллиард лет или на первой/последней итерации
        current_billion = int(time_years / 1e9)
        previous_billion = int(times_years[i-1] / 1e9)
        if i == 1 or current_billion > previous_billion or i == _NUM_POINTS:
            time_billion_years = time_years / 1e9
            progress_percent = i * 100 / _NUM_POINTS
            print(f"Progress: {i}/{_NUM_POINTS} ({progress_percent:.1f}%) - Time: {time_billion_years:.2f} billion years", flush=True)
        
        # Вычисляем масштабный фактор
        scale_factor = calculate_scale_factor_at_time(time_years)
        scale_factors.append(scale_factor)
        
        # Вычисляем горизонты с накоплением интеграла
        time_seconds = time_years * SECONDS_PER_YEAR
        particle_r, integral_result = compute_particle_horizon(
            time_years, scale_factor, 
            previous_integral=previous_integral if previous_integral > 0 else None,
            previous_time=previous_time_seconds if previous_time_seconds > 0 else None
        )
        
        # ВАЖНО: Проверяем, что сопутствующий горизонт только растет
        comoving_horizon = particle_r / scale_factor if scale_factor > 0 else particle_r
        if comoving_horizon < previous_comoving_horizon and previous_comoving_horizon > 0:
            # Горизонт уменьшился - это ошибка! Используем предыдущее значение интеграла
            print(f"WARNING: Comoving horizon decreased at {time_years/1e9:.2f} billion years!")
            print(f"  Previous: {previous_comoving_horizon/9.461e15:.2f} billion light years")
            print(f"  Current: {comoving_horizon/9.461e15:.2f} billion light years")
            print(f"  Recomputing with higher precision...")
            # Пересчитываем без накопления, но с большей точностью
            particle_r, integral_result = compute_particle_horizon(time_years, scale_factor, None, None)
            comoving_horizon = particle_r / scale_factor if scale_factor > 0 else particle_r
            # Если все еще уменьшился, используем предыдущее значение
            if comoving_horizon < previous_comoving_horizon:
                print(f"  Still decreasing! Using previous value.")
                # Используем предыдущий интеграл и пересчитываем только физический радиус
                integral_result = previous_integral
                particle_r = c * scale_factor * integral_result
                comoving_horizon = previous_comoving_horizon
        
        # Обновляем предыдущие значения
        previous_integral = integral_result
        previous_time_seconds = time_seconds
        previous_comoving_horizon = comoving_horizon
        
        event_r = compute_event_horizon(time_years, scale_factor)
        
        particle_horizons.append(particle_r)
        event_horizons.append(event_r)
    
    # Сохраняем в отдельные файлы для каждого горизонта
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Файл для горизонта частиц
    particle_data = {
        'times_years': times_years.tolist(),
        'times_seconds': times_seconds.tolist(),
        'scale_factors': scale_factors,
        'horizons': particle_horizons,
    }
    particle_cache_file = os.path.join(data_dir, 'particle_horizon_cache.json')
    with open(particle_cache_file, 'w') as f:
        json.dump(particle_data, f)
    
    # Файл для горизонта событий
    event_data = {
        'times_years': times_years.tolist(),
        'times_seconds': times_seconds.tolist(),
        'scale_factors': scale_factors,
        'horizons': event_horizons,
    }
    event_cache_file = os.path.join(data_dir, 'event_horizon_cache.json')
    with open(event_cache_file, 'w') as f:
        json.dump(event_data, f)
    
    print(f"\nHorizons precomputed and saved:")
    print(f"  Particle horizon: {particle_cache_file}")
    print(f"  Event horizon: {event_cache_file}")
    print(f"Total points: {_NUM_POINTS}")
    return particle_cache_file, event_cache_file

if __name__ == "__main__":
    precompute_horizons()
