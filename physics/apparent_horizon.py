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

import config
from utils.constants import (
    CAUSAL_HORIZON_COMOVING_METERS,
    G,
    SECONDS_PER_YEAR,
    c as _C_LIGHT,
)

_LTB_EVENT_SOLVE_RTOL = 1e-2
_LTB_EVENT_SOLVE_ATOL_REL = 1e-6
_LTB_EVENT_SOLVE_ATOL_MIN_M = 1e9
_LTB_EVENT_SOLVE_MAX_STEP_DIVISOR = 16.0
_LTB_EVENT_INITIAL_OUTER_FRACTION = 1.0 - 1.0e-5

_APPARENT_HORIZON_SCAN_EPS_REL = 1e-9
_APPARENT_HORIZON_SCAN_EPS_MIN_M = 1.0
_APPARENT_HORIZON_INNER_CLASSICAL_EPS_REL = 1e-3

_APPARENT_HORIZON_MIN_G_GOLDEN_ITERATIONS = 48
_APPARENT_HORIZON_MIN_G_REL_TOL = 1e-8

_APPARENT_HORIZON_INNER_SCAN_POINTS = 48
_APPARENT_HORIZON_INNER_BISECT_ITERATIONS = 32
_APPARENT_HORIZON_INNER_REL_TOL = 1e-7

_APPARENT_HORIZON_OUTER_SCAN_POINTS = 128
_APPARENT_HORIZON_OUTER_BISECT_ITERATIONS = 48
_APPARENT_HORIZON_OUTER_REL_TOL = 1e-9


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
    # ВАЖНО: β² = v_sq · a_stored² / c², где a_stored — это a(t) на момент
    # последнего обновления velocities_comoving, а НЕ текущий scale_factor.
    # Между update_positions и следующим вызовом cosmology.scale_factor
    # успевает уйти вперёд; если использовать его, β² для самой быстрой
    # точки переезжает через 1, clip → γ=10⁶ и одна точка раздувает M(<r)
    # на 10⁶·m_rest, что сдвигает корни g(r)=0 и даёт нефизичный прыжок
    # apparent horizon. См. _recompute_velocity_norms.
    v_sq_full = matter_points._v_sq_comoving
    a_for_gamma = getattr(matter_points, '_velocities_scale_factor', None)
    if a_for_gamma is None or float(a_for_gamma) <= 0.0:
        a_for_gamma = float(scale_factor)
    if (v_sq_full is not None
            and len(v_sq_full) == n_total
            and float(a_for_gamma) > 0.0):
        sf2_over_c2 = (float(a_for_gamma) ** 2) / (_C_LIGHT * _C_LIGHT)
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
    """Сглаженная сумма γ·m_rest активной пыли внутри r_phys.

    Точки материи в симуляции — это макро-частицы, представляющие
    непрерывную пыль. Для горизонтов нельзя трактовать каждую такую точку как
    бесконечно тонкую сферическую оболочку: при малом MATTER_NUM_POINTS это
    создаёт ступени M(<r) и скачки корней g(r)=0. Поэтому интерполируем
    кумулятивную массу по сопутствующему объёму chi^3.
    """
    if matter_state is None or scale_factor <= 0.0 or r_phys <= 0.0:
        return 0.0
    d_sorted = matter_state.get('d_sorted_comoving')
    cum_mass = matter_state.get('cum_mass_sorted')
    if d_sorted is None or cum_mass is None or d_sorted.size == 0:
        return 0.0
    x_nodes = np.asarray(d_sorted, dtype=np.float64) ** 3
    y_nodes = np.asarray(cum_mass, dtype=np.float64)
    shell_mass = np.diff(np.concatenate(([0.0], y_nodes)))
    y_nodes = y_nodes - 0.5 * shell_mass
    valid = np.isfinite(x_nodes) & np.isfinite(y_nodes) & (x_nodes > 0.0)
    if not np.any(valid):
        return 0.0
    x_nodes = x_nodes[valid]
    y_nodes = y_nodes[valid]
    # np.interp expects strictly increasing x. Exact duplicate radii are rare,
    # but possible after generated/added points; keep the last cumulative mass.
    keep_last = np.concatenate((np.diff(x_nodes) > 0.0, [True]))
    x_nodes = x_nodes[keep_last]
    y_nodes = y_nodes[keep_last]
    x = (float(r_phys) / float(scale_factor)) ** 3
    return float(np.interp(x, x_nodes, y_nodes, left=0.0, right=float(y_nodes[-1])))


def laser_mass_inside_phys(
    r_phys: float,
    laser_state,
) -> float:
    """Сглаженная сумма E/c² лазерных фотонов в полёте внутри r_phys.

    Лазерные пакеты — численная аппроксимация непрерывного потока энергии:
    каждый шаг dt создаёт дискретные «снаряды» вместо непрерывной струи.
    Если оставить M(<r) ступенчатой, у g(r) = 2GM/(c²r) + Λr²/3 − 1
    появляются дополнительные нули на радиусах пакетов; решатель apparent
    horizon перескакивает между ними и космологическая граница нефизически
    дёргается. Поэтому интерполируем кумулятивную массу как непрерывную
    функцию r³ — это аналог сглаживания макро-точек материи.

    Принимает laser_state в формате `MatterPoints.build_in_flight_laser_mass_state`.
    """
    if laser_state is None or r_phys <= 0.0:
        return 0.0
    r_sorted = laser_state.get('r_sorted')
    cum_mass = laser_state.get('cum_mass')
    if r_sorted is None or cum_mass is None or len(r_sorted) == 0:
        return 0.0
    x_nodes = np.asarray(r_sorted, dtype=np.float64) ** 3
    y_nodes = np.asarray(cum_mass, dtype=np.float64)
    shell_mass = np.diff(np.concatenate(([0.0], y_nodes)))
    y_nodes = y_nodes - 0.5 * shell_mass
    valid = np.isfinite(x_nodes) & np.isfinite(y_nodes) & (x_nodes > 0.0)
    if not np.any(valid):
        return 0.0
    x_nodes = x_nodes[valid]
    y_nodes = y_nodes[valid]
    keep_last = np.concatenate((np.diff(x_nodes) > 0.0, [True]))
    x_nodes = x_nodes[keep_last]
    y_nodes = y_nodes[keep_last]
    return float(np.interp(
        float(r_phys) ** 3, x_nodes, y_nodes,
        left=0.0, right=float(y_nodes[-1]),
    ))


def compute_ltb_event_horizon(
    time_now: float,
    M_bh_kg: float,
    matter_state: Optional[dict],
    laser_state,
    scale_factor_now: float,
    lambda_const: float,
    r_outer_now: float,
    a_at_seconds,
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

    ВНИМАНИЕ — обработка лазера в M(<r,t):
    Лазерные фотоны в этой симуляции летят ВНУТРЬ, к центральной ЦЧД
    (см. `MatterPoints._advance_laser_photons_chi`: chi = chi − c·dη), и
    поглощаются при r_photon ≤ r_BH (масса уходит в `accumulated_bh_mass`).
    Тем не менее `laser_state` сюда сознательно НЕ заводится в M(<r,t).
    Причины — три:

    (а) Snapshot `laser_state` хранит ФИЗИЧЕСКИЕ радиусы фотонов на t_now.
        Решатель `solve_apparent_outer_horizon(..., a_future, ...)`
        интерпретирует их как СТАТИЧЕСКУЮ массовую оболочку при a_future,
        что нефизично — за окно интегрирования (config.MAX_TIME_YEARS ≈
        100 Gyr) каждый фотон уже успел бы попасть в ЦЧД (время полёта
        r_emit/c ~ Myr–Gyr). Ступенька M(<r) на радиусе r_photon ещё и
        порождает дополнительные корни g(r)=0, между которыми решатель
        внешнего AH перепрыгивал от кадра к кадру.

    (б) Сумма E/c² фотонов в snapshot посчитана по их ТЕКУЩЕЙ энергии
        E(t_now) = E_emit·(a_emit/a_now). За время полёта до центра
        фотон испытает ещё один cosmological redshift на (a_now/a_absorb),
        так что в M_BH в итоге уйдёт не Σ E(t_now)/c², а
        Σ E_emit·(a_emit/a_absorb_i)/c² < Σ E(t_now)/c². Без явного
        отслеживания траекторий мы НЕ можем точно сказать, сколько массы
        реально доберётся до ЦЧД, и любое «пред-абсорбирование» в M_bh
        будет завышенной оценкой.

    (в) Даже если оставить (а) и (б) в стороне и попробовать вариант
        `M_bh_eff = M_bh_kg + Σ E_photon/c²` (наивный upper bound), в
        этой симуляции Σ M_laser_in_flight ≈ массе Нараи. Решатель
        apparent horizon с M_eff ≈ M_N оказывается у точки слияния (где
        inner и outer совпадают), и крошечные шумы в M_laser (от
        поглощения/эмиссии отдельных фотонов) приводят к
        НЕпропорционально большим скачкам стартовой точки
        backward-интегрирования сепаратрисы. На практике это давало
        видимые прыжки 20–30% за кадр.

    Поэтому используем самый стабильный путь: считаем event horizon как
    LTB-«фон» БЕЗ временной массы фотонов в полёте. Реальный (уже с учётом
    redshift'а) рост M_bh от их поглощения отражается в
    `accumulated_bh_mass` на следующих кадрах, и кривая горизонта плавно
    подстраивается. Расхождение с физически точной картиной — нижняя оценка
    на уровне (реально-поглощённая Σ E/c²) / M_BH(t→∞), которое мало для
    типичных сценариев и не растёт со временем.

    Args:
        laser_state: kept in signature для совместимости вызывающих, но
            игнорируется внутри (см. блок-комментарий выше).
        a_at_seconds: callable(t_seconds) → a(t). Прод-вызывающие передают
            ``cosmology._get_scale_factor_for_time``, у которого под капотом
            интерполяция `data/event_horizon_cache.json` (та же «обычная»
            ΛCDM-эволюция, что и `calculate_scale_factor_at_time`, только
            посчитанная один раз скриптом предвычисления). Раньше здесь
            висел отдельный scipy.solve_ivp(da_dt, ...) на ~70 мс/кадр.
    """
    del laser_state
    t0 = float(time_now)
    a0 = float(scale_factor_now)
    if t0 <= 0.0 or a0 <= 0.0 or lambda_const <= 0.0:
        return 0.0

    t_future = t0 + float(config.MAX_TIME_YEARS) * SECONDS_PER_YEAR

    def a_at(t_value: float) -> float:
        if t_value <= t0:
            return a0
        return max(float(a_at_seconds(t_value)), 1e-30)

    a_future = a_at(t_future)
    r_outer_upper = np.sqrt(3.0 / lambda_const) * 1.001
    r_outer_future = solve_apparent_outer_horizon(
        M_bh_kg, matter_state, None, a_future,
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
        )
        arg = 2.0 * G * M / r + lambda_const * _C_LIGHT * _C_LIGHT * (r ** 2) / 3.0
        Rdot = np.sqrt(arg) if arg > 0.0 else 0.0
        return [Rdot - _C_LIGHT]

    r_init = float(r_outer_future) * _LTB_EVENT_INITIAL_OUTER_FRACTION
    try:
        sol_r = integrate.solve_ivp(
            dr_dt,
            [t_future, t0],
            [r_init],
            method='RK45',
            rtol=_LTB_EVENT_SOLVE_RTOL,
            atol=max(
                float(r_outer_now) * _LTB_EVENT_SOLVE_ATOL_REL,
                _LTB_EVENT_SOLVE_ATOL_MIN_M,
            ),
            max_step=(t_future - t0) / _LTB_EVENT_SOLVE_MAX_STEP_DIVISOR,
        )
    except Exception:
        return 0.0
    if not sol_r.success:
        return 0.0

    r_event = float(sol_r.y[0, -1])
    if not np.isfinite(r_event) or r_event <= 0.0:
        return 0.0

    # ── Upper bound: r_event ≤ a(t) · CAUSAL_HORIZON_COMOVING_METERS ──
    #
    # Comoving causal horizon = (1/c)·∫_0^∞ c·dt/a(t) — полная сопутствующая
    # дистанция, до которой свет когда-либо может добраться в этой
    # ΛCDM-космологии (от Большого взрыва до t→∞). Аналитическая константа
    # лежит в utils.constants.CAUSAL_HORIZON_COMOVING_METERS = R_λ·∛(Ω_Λ/Ω_m)·I
    # ≈ 63.69 Gly (см. вывод там).
    #
    # Comoving event horizon в любой момент t равен ∫_t^∞ c·dt'/a(t') и по
    # построению ≤ ∫_0^∞ c·dt'/a(t') = causal comoving. Значит и физический
    # event horizon обязан удовлетворять
    #     r_event_phys(t)  ≤  a(t) · CAUSAL_HORIZON_COMOVING_METERS.
    #
    # В нашем LTB этот bound может нарушаться численно: вся масса Вселенной
    # (≈ 2.5e54 кг) сидит в маленьком сгустке, поэтому при backward-интегрировании
    # сепаратрисы dr/dt = R_dot − c член 2GM(<r)/r внутри R_dS делает R_dot ≫ c,
    # и null-траектория «убегает» дальше, чем позволяет глобальная FLRW-причинность.
    # Это артефакт overconcentrated-кластера, а не реальная физика — глобальная
    # причинная структура задаётся метрикой ВНЕ кластера, которая FLRW-подобна.
    #
    # Lower bound (r_event ≥ r_inner_AH) применяется в MassCalculator после
    # вычисления inner apparent horizon — там есть полный laser_state, который
    # мы здесь сознательно игнорируем (см. блок-комментарий в начале функции).
    r_causal_phys = a0 * float(CAUSAL_HORIZON_COMOVING_METERS)
    if r_event > r_causal_phys:
        r_event = r_causal_phys
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
            x_nodes = np.asarray(d_sorted, dtype=np.float64) ** 3
            y_nodes = np.asarray(cum_mass, dtype=np.float64)
            shell_mass = np.diff(np.concatenate(([0.0], y_nodes)))
            y_nodes = y_nodes - 0.5 * shell_mass
            valid = np.isfinite(x_nodes) & np.isfinite(y_nodes) & (x_nodes > 0.0)
            if np.any(valid):
                x_nodes = x_nodes[valid]
                y_nodes = y_nodes[valid]
                keep_last = np.concatenate((np.diff(x_nodes) > 0.0, [True]))
                x_nodes = x_nodes[keep_last]
                y_nodes = y_nodes[keep_last]
                r_comoving = r / float(scale_factor)
                x_query = np.maximum(r_comoving, 0.0) ** 3
                has_inside = r > 0.0
                if np.any(has_inside):
                    M[has_inside] = M[has_inside] + np.interp(
                        x_query[has_inside],
                        x_nodes,
                        y_nodes,
                        left=0.0,
                        right=float(y_nodes[-1]),
                    )

    if laser_state is not None:
        r_sorted = laser_state.get('r_sorted')
        cum_mass_l = laser_state.get('cum_mass')
        if r_sorted is not None and cum_mass_l is not None and len(r_sorted) > 0:
            x_nodes = np.asarray(r_sorted, dtype=np.float64) ** 3
            y_nodes = np.asarray(cum_mass_l, dtype=np.float64)
            shell_mass = np.diff(np.concatenate(([0.0], y_nodes)))
            y_nodes = y_nodes - 0.5 * shell_mass
            valid = np.isfinite(x_nodes) & np.isfinite(y_nodes) & (x_nodes > 0.0)
            if np.any(valid):
                x_nodes = x_nodes[valid]
                y_nodes = y_nodes[valid]
                keep_last = np.concatenate((np.diff(x_nodes) > 0.0, [True]))
                x_nodes = x_nodes[keep_last]
                y_nodes = y_nodes[keep_last]
                x_query = np.maximum(r, 0.0) ** 3
                has_inside_l = r > 0.0
                if np.any(has_inside_l):
                    M[has_inside_l] = M[has_inside_l] + np.interp(
                        x_query[has_inside_l],
                        x_nodes,
                        y_nodes,
                        left=0.0,
                        right=float(y_nodes[-1]),
                    )

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

    for _ in range(_APPARENT_HORIZON_MIN_G_GOLDEN_ITERATIONS):
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

        if hi - lo <= _APPARENT_HORIZON_MIN_G_REL_TOL * max(hi, 1.0):
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

    # ВАЖНО: eps НЕ должен зависеть от r_hi.
    # Для outer-горизонта r_hi ~ √(3/Λ) ≈ 1.6e26 м, и шкала `r_hi · 1e-9 ≈
    # 1.6e17 м` спокойно перепрыгивает через весь физически интересный
    # внутренний trapped-район (r_classical ~ км для пробных ЦЧД, фотоны
    # лазера в полёте сидят на r ~ 10⁹–10¹² м). На таком eps скан стартует
    # за пределами trapped-зоны, `trapped_mask[0] = False`, и функция
    # отдаёт r_classical, не «видя» накопленных фотонов внутри AH.
    # Берём чисто r_classical-релятивную шкалу + абсолютный пол: при любой
    # M_BH > 0 eps окажется глубоко под r_classical и `g(eps) → +∞` даст
    # trapped_mask[0] = True, после чего скан корректно ищет границу
    # trapped→untrapped с учётом узлов материи и фотонов из _build_scan_radii.
    eps = max(
        r_classical * _APPARENT_HORIZON_INNER_CLASSICAL_EPS_REL,
        _APPARENT_HORIZON_SCAN_EPS_MIN_M,
    )
    if eps >= r_hi:
        return r_classical

    radii = _build_scan_radii(
        eps, r_hi, matter_state, laser_state, scale_factor,
        num_geom=_APPARENT_HORIZON_INNER_SCAN_POINTS,
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

    for _ in range(_APPARENT_HORIZON_INNER_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if _g_scalar(
            mid, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
        ) >= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= _APPARENT_HORIZON_INNER_REL_TOL * max(lo, 1.0):
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

    eps = max(
        r_lo,
        r_hi * _APPARENT_HORIZON_SCAN_EPS_REL,
        _APPARENT_HORIZON_SCAN_EPS_MIN_M,
    )
    if eps >= r_hi:
        return r_hi

    radii = _build_scan_radii(
        eps, r_hi, matter_state, laser_state, scale_factor,
        num_geom=_APPARENT_HORIZON_OUTER_SCAN_POINTS,
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

    for _ in range(_APPARENT_HORIZON_OUTER_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if _g_scalar(
            mid, M_bh_kg, matter_state, laser_state, scale_factor, lambda_const,
        ) >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= _APPARENT_HORIZON_OUTER_REL_TOL * max(hi, 1.0):
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
