"""
Расчёты, связанные с пределом Нараи (Schwarzschild–de Sitter / Kottler).

В проекте используется классическая «масса Нараи» в вакууме SdS.
Это *масса-избыток* (над фоном) чёрной дыры в метрике Шварцшильда–де Ситтера.
Она зависит только от космологической постоянной Λ:

    M_N_vac = c² / (3 G √Λ)
    r_N     = 1 / √Λ
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root, root_scalar

from config import DEBUG
from utils.constants import G, H0_s, OMEGA_LAMBDA, c


def cosmological_constant_lambda() -> float:
    """
    Космологическая постоянная Λ в единицах 1/м² из параметров ΛCDM проекта.

    Λ = 3 H0² Ω_Λ / c²
    """
    return 3.0 * (H0_s**2) * OMEGA_LAMBDA / (c**2)


def nariai_radius(lambda_const: float | None = None) -> float:
    """
    Радиус Нараи r_N (когда горизонты совпадают в SdS), м.

    r_N = 1 / √Λ
    """
    lam = cosmological_constant_lambda() if lambda_const is None else float(lambda_const)
    if lam <= 0.0:
        raise ValueError("Lambda must be positive for the Nariai radius.")
    return 1.0 / np.sqrt(lam)


def nariai_mass_vacuum(lambda_const: float | None = None) -> float:
    """
    Классическая «масса Нараи» (масса-избыток над фоном) в вакууме SdS, кг.

    M_N_vac = c² / (3 G √Λ)
    """
    lam = cosmological_constant_lambda() if lambda_const is None else float(lambda_const)
    if lam <= 0.0:
        raise ValueError("Lambda must be positive for the Nariai mass.")
    return (c**2) / (3.0 * G * np.sqrt(lam))


def schwarzschild_de_sitter_horizons(mass_kg: float, lambda_const: float | None = None) -> tuple[float, float]:
    """
    Вычисляет горизонты в метрике Шварцшильда-де Ситтера (SdS).
    
    Горизонты находятся из уравнения: 1 - 2GM/(c²r) - Λr²/3 = 0
    Это эквивалентно кубическому уравнению: r³ - (3/Λ)r + (6GM)/(Λc²) = 0
    
    Args:
        mass_kg: масса черной дыры (кг)
        lambda_const: космологическая постоянная Λ (1/м²). Если None, берётся из параметров проекта.
    
    Returns:
        tuple[float, float]: (r_inner, r_outer) - внутренний (горизонт событий ЧД) 
                             и внешний (космологический) горизонты в метрах.
                             Если M = 0, то r_inner = 0, r_outer = √(3/Λ) (классический горизонт де Ситтера).
                             Если M = M_N (масса Нарайи), то r_inner = r_outer = r_N.
    """
    lam = cosmological_constant_lambda() if lambda_const is None else float(lambda_const)
    mass = float(mass_kg)
    
    if lam <= 0.0:
        raise ValueError("Lambda must be positive for SdS horizons.")
    
    if mass < 0.0:
        raise ValueError("Mass must be non-negative.")
    
    # Случай без черной дыры: возвращаем классический горизонт де Ситтера
    if mass == 0.0:
        r_dS = np.sqrt(3.0 / lam)
        return (0.0, r_dS)
    
    # Масса Нарайи
    M_N = nariai_mass_vacuum(lam)
    r_N = nariai_radius(lam)
    
    # Физика SdS: при M >= M_N оба горизонта сливаются на r_N и далее не существуют
    # как вещественные. Для визуализации фиксируем оба на r_N и НЕ позволяем
    # «внутреннему» горизонту ЧД превышать «внешний» де-Ситтера.
    if mass >= M_N * 0.9999:
        if mass >= M_N:
            return (r_N, r_N)
        else:
            r_inner_calc, r_outer_calc = _solve_sds_horizons_numerically(mass, lam)
            interpolation_factor = (mass / M_N - 0.9999) / (1.0 - 0.9999)
            r_outer_smooth = r_outer_calc * (1.0 - interpolation_factor) + r_N * interpolation_factor
            r_inner_smooth = min(r_inner_calc, r_outer_smooth)
            return (r_inner_smooth, r_outer_smooth)
    
    # Используем численное решение для надежности
    # (аналитическое решение через формулу Кардано может быть сложным для проверки знаков)
    r_inner, r_outer = _solve_sds_horizons_numerically(mass, lam)
    return (r_inner, r_outer)



def _solve_sds_horizons_numerically(mass_kg: float, lambda_const: float) -> tuple[float, float]:
    """
    Численное решение уравнения горизонтов SdS.
    
    Уравнение: 1 - 2GM/(c²r) - Λr²/3 = 0
    """
    # Характерные радиусы
    r_N = nariai_radius(lambda_const)
    r_dS = np.sqrt(3.0 / lambda_const)
    r_schwarzschild = 2.0 * G * mass_kg / (c**2)
    M_N = nariai_mass_vacuum(lambda_const)
    mass_ratio = mass_kg / M_N if M_N > 0 else 0.0
    
    def horizon_equation(r):
        """Уравнение горизонтов SdS"""
        if r <= 0:
            return float('inf')
        return r - 2.0 * G * mass_kg / (c**2) - lambda_const * r**3 / 3.0
    
    # ===== ВНУТРЕННИЙ ГОРИЗОНТ (ЧД) =====
    r_inner = 0.0
    if r_schwarzschild > 0 and r_schwarzschild < r_N * 0.99:
        try:
            sol_inner = root_scalar(
                horizon_equation, 
                bracket=[r_schwarzschild * 0.5, r_N * 0.99], 
                method='brentq'
            )
            if sol_inner.converged and 0 < sol_inner.root < r_N:
                r_inner = float(sol_inner.root)
        except (ValueError, RuntimeError):
            # Если не получилось, оставляем r_inner = 0
            pass
    
    # ===== ВНЕШНИЙ ГОРИЗОНТ (космологический) =====
    # Используем аналитическое решение кубического уравнения (формула Кардано)
    r_outer = r_dS
    
    if mass_kg > 0:
        # Уравнение: r³ - (3/Λ)r + (6GM)/(Λc²) = 0
        # Форма: r³ + pr + q = 0
        p = -3.0 / lambda_const
        q = 6.0 * G * mass_kg / (lambda_const * c**2)
        
        # Дискриминант
        delta = (q / 2.0)**2 + (p / 3.0)**3
        
        if delta < 0:
            # Три вещественных корня (используем тригонометрическую формулу Виета)
            sqrt_neg_p3 = np.sqrt(-p / 3.0)
            acos_arg = np.clip(-q / (2.0 * sqrt_neg_p3**3), -1.0, 1.0)
            theta = np.arccos(acos_arg)
            
            # Три корня
            r1 = 2.0 * sqrt_neg_p3 * np.cos(theta / 3.0)
            r2 = 2.0 * sqrt_neg_p3 * np.cos((theta + 2.0 * np.pi) / 3.0)
            r3 = 2.0 * sqrt_neg_p3 * np.cos((theta + 4.0 * np.pi) / 3.0)
            
            positive_roots = sorted([r for r in [r1, r2, r3] if r > 0])
            
            if len(positive_roots) >= 2:
                # Внешний горизонт - наибольший корень
                r_outer_calc = positive_roots[-1]
                
                # Проверка: должен быть между r_N и r_dS
                if r_N * 0.95 < r_outer_calc < r_dS * 1.05:
                    # Проверяем, что это действительно решение
                    f_outer = abs(horizon_equation(r_outer_calc))
                    tolerance = 1e-5 * r_dS
                    if f_outer < tolerance:
                        r_outer = r_outer_calc
                
                # Внутренний горизонт - наименьший корень (если еще не найден)
                if r_inner == 0.0:
                    r_inner_calc = positive_roots[0]
                    if 0 < r_inner_calc < r_N * 1.5:
                        f_inner = abs(horizon_equation(r_inner_calc))
                        if f_inner < tolerance:
                            r_inner = r_inner_calc
    
    # Финальная проверка: r_outer должен быть >= r_inner и в разумных пределах
    r_outer = max(r_inner, min(r_dS, r_outer))
    r_inner = min(max(0.0, r_inner), r_outer)
    return (r_inner, r_outer)


