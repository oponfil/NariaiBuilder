"""
Apparent horizons в LTB-Λ-модели (чистый LTB, без Birkhoff-вычета фона).

Берём ПОЛНУЮ массу-энергию внутри сферы радиуса r:

    M(<r) = M_BH + M_matter_inside(r) + E_laser_inside(r)/c²

где
  • M_matter_inside(r) суммирует γ·m_rest всех активных оболочек внутри r,
  • E_laser_inside(r)/c² — масса-эквивалент лазерных фотонов в полёте.

Никакого вычитания ρ_m·V(r) НЕТ: вся материя (включая FRW-фоновую плотность,
представленную дискретными оболочками) учитывается честно в M(<r). Это даёт
неоднородную LTB-Λ метрику без проектирования на изолированную ЦЧД в SdS.

Уравнение apparent horizon в синхронной калибровке LTB-Λ:

    g(r) ≡ 2G·M(<r)/(c²·r) + Λ·r²/3 − 1 = 0
    trapped ⇔ g(r) ≥ 0

Для типичного режима «маленькая ЦЧД на FRW-фоне» g(r) имеет ДВА положительных
корня:
  • Внутренний (BH apparent horizon): первое пересечение g(r) = 0 при движении
    наружу от r → 0+. Это «горизонт ЧД» — именно туда мы поглощаем материю
    и фотоны.
  • Внешний (cosmological apparent horizon): последний корень g(r) = 0 перед
    тем, как g(r) уйдёт в +∞ на больших r. В чистом FRW без ЦЧД это
    космологический Hubble-радиус c/H(t); с ЦЧД он немного смещается внутрь.

Когда M_BH становится сравнимой с критической массой эпохи
M_crit ≈ c³/(G·H·3√3), внутренний и внешний корни сливаются — это GR-честный
признак того, что «локальная ЦЧД» больше неотличима от космологического
объекта. В таком режиме solve_apparent_inner_horizon вернёт верхнюю границу
скана (трапированная зона непрерывно тянется от r → 0 до внешнего горизонта).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import integrate

from utils.constants import G, H0_s, OMEGA_B, OMEGA_DM, OMEGA_LAMBDA, c as _C_LIGHT


_LTB_EVENT_FUTURE_YEARS = 100.0e9


def build_active_matter_state(
    matter_points,
    scale_factor: float,
):
    """
    Снимок активных (не поглощённых ЦЧД) точек материи: отсортированные
    comoving-расстояния и кумулятивная γ·m_rest по этому порядку.

    Возвращает None, если точек нет либо данные не готовы.

    Returns:
        dict | None с ключами:
          'd_sorted_comoving' — отсортированные comoving-расстояния (м),
          'cum_mass_sorted'   — кумулятивная γ·m_rest (кг),
          'active_indices'    — индексы активных точек в исходном массиве
                                (в порядке возрастания comoving-расстояния),
          'sort_idx'          — np.argsort применённый к расстояниям активных
                                (тот же порядок, что в active_indices/sorted).
    """
    if scale_factor is None or float(scale_factor) <= 0.0:
        return None
    cd = matter_points.comoving_distances
    masses_rest = matter_points.masses_per_point
    if cd is None or masses_rest is None or len(cd) == 0:
        return None
    if len(masses_rest) != len(cd):
        return None

    n_total = len(cd)
    active_mask = np.ones(n_total, dtype=bool)
    inside_arr = matter_points.get_inside_indices_arr()
    if inside_arr.size > 0:
        inside_arr = inside_arr[(inside_arr >= 0) & (inside_arr < n_total)]
        if inside_arr.size > 0:
            active_mask[inside_arr] = False
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return {
            'd_sorted_comoving': np.empty(0, dtype=np.float64),
            'cum_mass_sorted': np.empty(0, dtype=np.float64),
            'active_indices': active_indices,
            'sort_idx': np.empty(0, dtype=np.intp),
        }

    cd_a = cd[active_indices]
    sort_idx = np.argsort(cd_a)
    d_sorted_comoving = cd_a[sort_idx]

    if masses_rest.dtype != np.float64:
        masses_rest = masses_rest.astype(np.float64, copy=False)
    masses_active = masses_rest[active_indices]

    # γ·m_rest: используем предрассчитанный _v_sq_comoving, если он совпадает
    # по длине с cd. Иначе — без коррекции (при отсутствии релятивистских
    # скоростей это даёт точную m_rest).
    v_sq_full = matter_points._v_sq_comoving
    if (v_sq_full is not None
            and len(v_sq_full) == n_total
            and float(scale_factor) > 0.0):
        sf2_over_c2 = (float(scale_factor) ** 2) / (_C_LIGHT * _C_LIGHT)
        beta2_a = v_sq_full[active_indices] * sf2_over_c2
        np.clip(beta2_a, 0.0, 1.0 - 1e-12, out=beta2_a)
        gamma_a = 1.0 / np.sqrt(1.0 - beta2_a)
        m_eff_a = gamma_a * masses_active
    else:
        m_eff_a = masses_active

    cum_mass_sorted = np.cumsum(m_eff_a[sort_idx])

    return {
        'd_sorted_comoving': d_sorted_comoving,
        'cum_mass_sorted': cum_mass_sorted,
        'active_indices': active_indices,
        'sort_idx': sort_idx,
    }


def matter_mass_inside_phys(
    r_phys: float,
    matter_state: Optional[dict],
    scale_factor: float,
) -> float:
    """Сумма γ·m_rest активных точек внутри физического радиуса r_phys."""
    if matter_state is None or scale_factor <= 0.0 or r_phys <= 0.0:
        return 0.0
    d_sorted = matter_state.get('d_sorted_comoving')
    cum_mass = matter_state.get('cum_mass_sorted')
    if d_sorted is None or cum_mass is None or d_sorted.size == 0:
        return 0.0
    idx = int(np.searchsorted(d_sorted, r_phys / scale_factor, side='right'))
    if idx <= 0:
        return 0.0
    return float(cum_mass[idx - 1])


def laser_mass_inside_phys(
    r_phys: float,
    laser_state,
) -> float:
    """E/c² фотонов лазера в полёте внутри физического радиуса r_phys.

    Принимает laser_state в формате `MatterPoints.build_in_flight_laser_mass_state`.
    """
    if laser_state is None or r_phys <= 0.0:
        return 0.0
    r_sorted = laser_state.get('r_sorted')
    cum_mass = laser_state.get('cum_mass')
    if r_sorted is None or cum_mass is None or len(r_sorted) == 0:
        return 0.0
    idx = int(np.searchsorted(r_sorted, r_phys, side='right'))
    if idx <= 0:
        return 0.0
    return float(cum_mass[idx - 1])


def compute_ltb_event_horizon(
    time_now: float,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor_now: float,
    lambda_const: float,
    r_outer_now: float,
) -> float:
    """
    LTB event/null horizon from the same M(<r,t) used by apparent horizons.

    Event horizon is global in GR, so this estimates the radial null
    separatrix by evolving the current LTB mass profile into the future and
    integrating the null ray backward:

        dr/dt = R_dot(t,r) - c,
        R_dot² = 2G·M(<r,t)/r + Λc²r²/3.

    This keeps the scenario path LTB-consistent: no FLRW/SdS event-horizon
    helper is mixed into the displayed/dynamical LTB horizons.
    """
    t0 = float(time_now)
    a0 = float(scale_factor_now)
    if t0 <= 0.0 or a0 <= 0.0 or lambda_const <= 0.0:
        return 0.0

    seconds_per_year = 365.25 * 24 * 3600
    t_future = t0 + _LTB_EVENT_FUTURE_YEARS * seconds_per_year
    omega_m = OMEGA_DM + OMEGA_B

    def da_dt(_t, a_val):
        a = max(float(a_val[0]), 1e-30)
        h = H0_s * np.sqrt(omega_m / (a ** 3) + OMEGA_LAMBDA)
        return [a * h]

    try:
        sol_a = integrate.solve_ivp(
            da_dt,
            [t0, t_future],
            [a0],
            method='RK45',
            dense_output=True,
            rtol=1e-4,
            atol=1e-9,
            max_step=(t_future - t0) / 256.0,
        )
    except Exception:
        return 0.0
    if not sol_a.success:
        return 0.0

    def a_at(t_value: float) -> float:
        if t_value <= t0:
            return a0
        if t_value >= t_future:
            return max(float(sol_a.y[0, -1]), 1e-30)
        return max(float(sol_a.sol(t_value)[0]), 1e-30)

    a_future = a_at(t_future)
    r_outer_upper = np.sqrt(3.0 / lambda_const) * 1.001
    r_outer_future = solve_apparent_outer_horizon(
        M_bh_kg, matter_state, laser_state, a_future,
        0.0, r_outer_upper, lambda_const,
    )
    if not np.isfinite(r_outer_future) or r_outer_future <= 0.0:
        return 0.0

    def dr_dt(t_value, state):
        r = float(state[0])
        if r <= 0.0:
            return [0.0]
        a_t = a_at(t_value)
        M = (
            float(M_bh_kg)
            + matter_mass_inside_phys(r, matter_state, a_t)
            + laser_mass_inside_phys(r, laser_state)
        )
        arg = 2.0 * G * M / r + lambda_const * _C_LIGHT * _C_LIGHT * (r ** 2) / 3.0
        Rdot = np.sqrt(arg) if arg > 0.0 else 0.0
        return [Rdot - _C_LIGHT]

    r_init = float(r_outer_future) * (1.0 - 1.0e-5)
    try:
        sol_r = integrate.solve_ivp(
            dr_dt,
            [t_future, t0],
            [r_init],
            method='RK45',
            rtol=1e-4,
            atol=max(float(r_outer_now) * 1e-7, 1e7),
            max_step=(t_future - t0) / 256.0,
        )
    except Exception:
        return 0.0
    if not sol_r.success:
        return 0.0

    r_event = float(sol_r.y[0, -1])
    if not np.isfinite(r_event) or r_event <= 0.0:
        return 0.0
    return r_event


def effective_central_mass(
    r_phys: float,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
) -> float:
    """
    Полная LTB-масса внутри физического радиуса r (без вычета фона):

        M(<r) = M_BH + M_matter_inside(r) + E_laser_inside(r)/c²

    Это та масса, что входит в условие apparent horizon LTB-Λ:
    g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1 = 0.
    """
    r = float(r_phys)
    if r <= 0.0:
        return float(M_bh_kg)
    return (
        float(M_bh_kg)
        + matter_mass_inside_phys(r, matter_state, scale_factor)
        + laser_mass_inside_phys(r, laser_state)
    )


def make_effective_mass_callable(
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
):
    """Фабрика: возвращает callable r_phys -> M(<r_phys), замораживающий
    остальные аргументы. Удобно для решателей apparent horizon."""
    def _M_eff(r_phys: float) -> float:
        return effective_central_mass(
            r_phys, M_bh_kg, matter_state, laser_state, scale_factor,
        )
    return _M_eff


def effective_central_mass_array(
    r_array: np.ndarray,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
) -> np.ndarray:
    """
    Векторная версия `effective_central_mass`: считает M(<r) для массива
    физических радиусов одним проходом через searchsorted.
    """
    r = np.asarray(r_array, dtype=np.float64)
    M = np.full_like(r, float(M_bh_kg))

    if (matter_state is not None
            and float(scale_factor) > 0.0
            and matter_state.get('d_sorted_comoving') is not None):
        d_sorted = matter_state['d_sorted_comoving']
        cum_mass = matter_state['cum_mass_sorted']
        if d_sorted.size > 0:
            r_comoving = r / float(scale_factor)
            idx = np.searchsorted(d_sorted, r_comoving, side='right')
            has_inside = (idx > 0) & (r > 0.0)
            if np.any(has_inside):
                M[has_inside] = M[has_inside] + cum_mass[idx[has_inside] - 1]

    if laser_state is not None:
        r_sorted = laser_state.get('r_sorted')
        cum_mass_l = laser_state.get('cum_mass')
        if r_sorted is not None and cum_mass_l is not None and len(r_sorted) > 0:
            idx_l = np.searchsorted(r_sorted, r, side='right')
            has_inside_l = (idx_l > 0) & (r > 0.0)
            if np.any(has_inside_l):
                M[has_inside_l] = M[has_inside_l] + cum_mass_l[idx_l - 1][has_inside_l]

    return M


def _build_scan_radii(
    r_lo: float,
    r_hi: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    num_geom: int,
) -> np.ndarray:
    """Сборная сетка радиусов для скана g(r): геометрическая + узлы данных
    (физические радиусы оболочек и фотонов в полёте), всё ограничено
    [r_lo, r_hi]."""
    chunks = [np.geomspace(r_lo, r_hi, num=num_geom)]
    if matter_state is not None and float(scale_factor) > 0.0:
        d_sorted = matter_state.get('d_sorted_comoving')
        if d_sorted is not None and d_sorted.size > 0:
            chunks.append(d_sorted * float(scale_factor))
    if laser_state is not None:
        r_sorted = laser_state.get('r_sorted')
        if r_sorted is not None and len(r_sorted) > 0:
            chunks.append(np.asarray(r_sorted, dtype=np.float64))
    radii = np.unique(np.concatenate(chunks))
    radii = radii[(radii >= r_lo) & (radii <= r_hi)]
    return radii


def _g_function(
    r: np.ndarray,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    lambda_const: float,
) -> np.ndarray:
    """g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1."""
    M_arr = effective_central_mass_array(
        r, M_bh_kg, matter_state, laser_state, scale_factor,
    )
    return (
        (2.0 * G * M_arr) / (_C_LIGHT * _C_LIGHT * r)
        + lambda_const * (r ** 2) / 3.0
        - 1.0
    )


def _g_scalar(
    r_phys: float,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    lambda_const: float,
) -> float:
    """Скалярная версия g(r) для уточнения корней/минимума."""
    rr = float(r_phys)
    if rr <= 0.0:
        return -1.0
    M = effective_central_mass(
        rr, M_bh_kg, matter_state, laser_state, scale_factor,
    )
    return (2.0 * G * M) / (_C_LIGHT * _C_LIGHT * rr) \
        + lambda_const * (rr ** 2) / 3.0 - 1.0


def _refine_min_g_radius(
    radii: np.ndarray,
    g_values: np.ndarray,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    lambda_const: float,
) -> float:
    """
    Радиус ближайшего LTB-сближения горизонтов.

    Когда inner и outer apparent horizons сливаются, у g(r) появляется
    двойной корень: g(r*) = 0 и g'(r*) = 0. На дискретной сетке это лучше
    аппроксимировать минимумом g(r), чем возвращать искусственную верхнюю
    границу скана.
    """
    if radii.size == 0:
        return 0.0

    finite = np.isfinite(radii) & np.isfinite(g_values) & (radii > 0.0)
    if not np.any(finite):
        return 0.0

    r_valid = radii[finite]
    g_valid = g_values[finite]
    idx = int(np.argmin(g_valid))

    if idx == 0 or idx == r_valid.size - 1:
        return float(r_valid[idx])

    lo = float(r_valid[idx - 1])
    hi = float(r_valid[idx + 1])
    if hi <= lo:
        return float(r_valid[idx])

    # Golden-section search по g(r). Для ступенчатой M(<r) это всё равно
    # устойчивее, чем прыжок к r_upper: минимум уточняется внутри локального
    # интервала между соседними узлами скана.
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c1 = hi - inv_phi * (hi - lo)
    c2 = lo + inv_phi * (hi - lo)
    f1 = _g_scalar(c1, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const)
    f2 = _g_scalar(c2, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const)

    for _ in range(64):
        if f1 <= f2:
            hi = c2
            c2 = c1
            f2 = f1
            c1 = hi - inv_phi * (hi - lo)
            f1 = _g_scalar(c1, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const)
        else:
            lo = c1
            c1 = c2
            f1 = f2
            c2 = lo + inv_phi * (hi - lo)
            f2 = _g_scalar(c2, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const)

        if hi - lo <= 1e-9 * max(hi, 1.0):
            break

    return float(0.5 * (lo + hi))


def solve_apparent_inner_horizon(
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    r_classical: float,
    r_upper: float,
    lambda_const: float,
) -> float:
    """
    Apparent inner horizon ЦЧД в чистом LTB-Λ.

    Условие apparent horizon: g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1 = 0,
    trapped ⇔ g(r) ≥ 0.

    Внутренний горизонт ЦЧД — это ПЕРВОЕ пересечение g(r) = 0 при движении
    наружу от r → 0+:
      • При M_BH > 0: 2G·M_BH/(c²r) → +∞ при r → 0+ ⇒ g(r → 0+) → +∞ ⇒
        вблизи центра всегда trapped. Сканируем наружу, ищем первый r,
        где g(r) < 0. Это и есть apparent inner horizon.
      • При M_BH = 0 (затравочной ЦЧД нет): g(eps) ≈ 0 + Λ·eps²/3 − 1 < 0
        (eps мало, FRW-материя в окрестности — пренебрежимо). Inner trap
        отсутствует, возвращаем r_classical (≈ 0).
      • Если на всём [eps, r_upper] g(r) ≥ 0 — inner и outer apparent
        horizons слились (M_BH сравнима с критической массой эпохи).
        Возвращаем LTB-радиус слияния: минимум g(r), а не искусственную
        верхнюю границу скана.

    Returns:
        Физический радиус inner apparent horizon (м), ограниченный
        снизу r_classical (Шварцшильд от M_BH).
    """
    r_classical = max(float(r_classical), 0.0)
    r_hi = float(r_upper)
    if r_hi <= 0.0:
        return r_classical

    eps = max(r_classical * 1e-3, r_hi * 1e-9, 1.0)
    if eps >= r_hi:
        return r_classical

    radii = _build_scan_radii(
        eps, r_hi, matter_state, laser_state, scale_factor, num_geom=64,
    )
    if radii.size == 0:
        return r_classical

    g = _g_function(
        radii, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
    )
    trapped_mask = g >= 0.0

    if not trapped_mask[0]:
        return r_classical

    untrapped_mask = ~trapped_mask
    if not np.any(untrapped_mask):
        return _refine_min_g_radius(
            radii, g, M_bh_kg, matter_state, laser_state,
            scale_factor, lambda_const,
        )

    first_untrapped = int(np.argmax(untrapped_mask))
    if first_untrapped == 0:
        return r_classical

    lo = float(radii[first_untrapped - 1])
    hi = float(radii[first_untrapped])

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _g_scalar(
            mid, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
        ) >= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-9 * max(lo, 1.0):
            break

    return max(lo, r_classical)


def solve_apparent_outer_horizon(
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor: float,
    r_lower: float,
    r_upper: float,
    lambda_const: float,
) -> float:
    """
    Apparent outer (cosmological) horizon в чистом LTB-Λ.

    Тот же g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1 = 0, что и для inner AH,
    но мы ищем ПОСЛЕДНЕЕ пересечение untrapped → trapped при движении
    наружу. Это внешний (космологический) apparent horizon — там, где
    Hubble-расширение/Λ начинают доминировать настолько, что световые
    конусы наклоняются наружу.

    В чистом FRW без ЦЧД и без Λ-вклада это даёт r ≈ c/H(t). С Λ-доминантой
    (поздняя эпоха) — близко к √(3/Λ). С ЦЧД отрицательная поправка
    (горизонт чуть меньше) — это та же физика, что у SdS-внешнего корня,
    но с честным учётом всей материи.

    Returns:
        Физический радиус outer apparent horizon (м). Если на всём
        [r_lower, r_upper] g(r) < 0 — возвращает r_upper (нет внешнего
        трапа в этом диапазоне). Если всё в диапазоне trapped, возвращает
        LTB-радиус слияния: минимум g(r), а не нижнюю границу скана.
    """
    r_lo = max(float(r_lower), 0.0)
    r_hi = float(r_upper)
    if r_hi <= r_lo:
        return r_hi if r_hi > 0.0 else r_lo

    eps = max(r_lo, r_hi * 1e-9, 1.0)
    if eps >= r_hi:
        return r_hi

    radii = _build_scan_radii(
        eps, r_hi, matter_state, laser_state, scale_factor, num_geom=128,
    )
    if radii.size == 0:
        return r_hi

    g = _g_function(
        radii, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
    )
    trapped_mask = g >= 0.0

    # Если на внешней границе скана g(r_hi) < 0 — outer trap дальше r_hi,
    # вернём r_hi как нижнюю оценку.
    if not trapped_mask[-1]:
        return r_hi
    # Если всё в скане трапировано — inner и outer AH слились или корней уже
    # нет. Возвращаем радиус ближайшего LTB-сближения горизонтов.
    if np.all(trapped_mask):
        return _refine_min_g_radius(
            radii, g, M_bh_kg, matter_state, laser_state,
            scale_factor, lambda_const,
        )

    # Ищем ПОСЛЕДНЮЮ untrapped → trapped границу (наибольший r, где
    # trapped_mask переходит из False в True).
    # diff[i] = trapped_mask[i+1] - trapped_mask[i]; нас интересуют переходы
    # с diff = +1 (False → True), берём максимальный индекс.
    diff = trapped_mask[1:].astype(np.int8) - trapped_mask[:-1].astype(np.int8)
    pos_transitions = np.flatnonzero(diff == 1)
    if pos_transitions.size == 0:
        # Граница не найдена явно (например, единственный переход был
        # trapped → untrapped). Считаем outer trap отсутствует в [r_lo, r_hi].
        return r_hi

    last_idx = int(pos_transitions[-1])
    lo = float(radii[last_idx])
    hi = float(radii[last_idx + 1])

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _g_scalar(
            mid, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
        ) >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-9 * max(hi, 1.0):
            break

    return float(hi)


def matter_capture_mass_relativistic(
    matter_points,
    scale_factor: float,
    captured_indices: np.ndarray,
) -> float:
    """
    Релятивистская масса-эквивалент γ·m_rest для подмножества активных точек,
    задаваемого indices в исходном массиве. Используется при захвате
    в apparent horizon: эта масса прибавляется к accumulated_bh_mass.
    """
    if captured_indices is None or len(captured_indices) == 0:
        return 0.0
    masses_rest = matter_points.masses_per_point
    if masses_rest is None:
        return 0.0
    masses_rest_arr = np.asarray(masses_rest, dtype=np.float64)
    m_rest = masses_rest_arr[captured_indices]

    v_sq_full = matter_points._v_sq_comoving
    if (v_sq_full is not None
            and len(v_sq_full) == len(masses_rest_arr)
            and float(scale_factor) > 0.0):
        sf2_over_c2 = (float(scale_factor) ** 2) / (_C_LIGHT * _C_LIGHT)
        beta2 = v_sq_full[captured_indices] * sf2_over_c2
        np.clip(beta2, 0.0, 1.0 - 1e-12, out=beta2)
        gamma = 1.0 / np.sqrt(1.0 - beta2)
        return float(np.sum(gamma * m_rest))
    return float(np.sum(m_rest))


__all__ = [
    'build_active_matter_state',
    'matter_mass_inside_phys',
    'laser_mass_inside_phys',
    'effective_central_mass',
    'effective_central_mass_array',
    'make_effective_mass_callable',
    'matter_capture_mass_relativistic',
    'solve_apparent_inner_horizon',
    'solve_apparent_outer_horizon',
]
