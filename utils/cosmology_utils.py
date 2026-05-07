"""
Утилиты для космологических вычислений
"""
import numpy as np
from scipy import integrate

from utils.constants import SECONDS_PER_YEAR, BILLION_LIGHT_YEARS_IN_METERS, OMEGA_LAMBDA, OMEGA_DM, OMEGA_B, H0_s


def calculate_scale_factor_at_time(time_years: float) -> float:
    """
    Вычисляет масштабный фактор a(t) при заданном времени (в годах).
    
    Args:
        time_years: Время в годах после Большого взрыва
        
    Returns:
        float: Масштабный фактор a(t)
    """
    # Вычисляем масштабный фактор для заданного времени
    time_seconds = time_years * SECONDS_PER_YEAR
    
    # Параметры космологии
    omega_m = OMEGA_DM + OMEGA_B
    omega_lambda = OMEGA_LAMBDA
    h0 = H0_s
    
    # Начинаем с очень маленького значения a и интегрируем до нужного времени
    initial_a = 1e-10
    initial_time = 0.0
    
    # Уравнение Фридмана: da/dt = a * H(t), где H(t) = H0 * sqrt(Ω_m/a³ + Ω_Λ)
    def da_dt(t, a_val):
        """Уравнение Фридмана: da/dt = a * H(t)"""
        if a_val <= 0:
            a_val = 1e-10
        h = h0 * np.sqrt(omega_m / (a_val**3) + omega_lambda)
        return a_val * h
    
    # Интегрируем от начального времени до целевого времени
    t_span = [initial_time, time_seconds]
    try:
        result = integrate.solve_ivp(da_dt, t_span, [initial_a], 
                                     method='RK45', dense_output=True, 
                                     rtol=1e-4, atol=1e-7)
        
        if result.success:
            return float(result.y[0, -1])
    except:
        pass
    
    # Если интегрирование не удалось, возвращаем начальное значение
    return initial_a
