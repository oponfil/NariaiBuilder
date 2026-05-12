"""
Калькулятор масс для различных горизонтов и черной дыры
"""
import time

import numpy as np

import config
from config import DEBUG
from physics.apparent_horizon import (
    compute_ltb_event_horizon,
    laser_mass_inside_phys,
    matter_mass_inside_phys,
    solve_apparent_inner_horizon,
    solve_apparent_outer_horizon,
)
from physics.nariai import (
    cosmological_constant_lambda,
)
from utils.constants import (
    G,
    NARIAI_BLACK_HOLE_MASS_KG,
    OMEGA_B,
    OMEGA_DM,
    RHO_CRIT,
    c,
)
from utils.config_utils import get_dt
from utils.format_utils import format_velocity_m_per_s

# Пороги релятивистской γ-коррекции массы точек.
# Если β²ₘₐₓ < _BETA2_NEGLIGIBLE_THRESHOLD, γ-коррекция пропускается:
# при β² = 1e-8 ошибка γ ≈ β²/2 ≈ 5e-9, заведомо меньше численного шума,
# поэтому m_eff = m_rest без вычисления sqrt по N элементам.
_BETA2_NEGLIGIBLE_THRESHOLD = 1e-8
# Верхняя граница для clip β² перед 1/sqrt(1-β²): не даём γ уйти в бесконечность
# (β² == 1.0 → деление на ноль). 1 - 1e-12 соответствует γ ≈ 1e6 — заведомо
# больше любых физически осмысленных скоростей в симуляции.
_BETA2_CLIP_MAX = 1.0 - 1e-12

# Сглаживание dr_BH/dt (EMA): v = α·raw + (1-α)·prev; α ∈ (0,1], меньше α — сильнее сглаживание.
_BLACK_HOLE_VELOCITY_SMOOTHING_ALPHA = 0.1
class MassCalculator:
    """Класс для расчета масс в различных горизонтах и черной дыры"""
    
    def __init__(self):
        """Инициализация калькулятора масс"""
        self.previous_bh_horizon_radius = 0.0
        self.last_bh_horizon_update_time = None
        self.previous_bh_mass = 0.0
        self.last_calculated_time = None  # Время последнего расчета скорости
        self.last_velocity = 0.0  # Последняя вычисленная скорость
        self.smoothed_velocity = 0.0  # Сглаженная скорость (EMA)

        # Кэш горизонтов, которые не зависят от дискретного профиля материи.
        # LTB Hubble/outer apparent horizon считается ниже, после сборки
        # matter_state и laser_state.
        self._last_horizons_time = None
        self._last_horizons_mass = None
        self._cached_de_sitter_horizon = None

        # ОПТИМИЗАЦИЯ: кэш порядка сортировки активных точек по comoving distance.
        # Инвалидируется по версиям comoving_distances и points_inside_bh
        # из MatterPoints — порядок зависит только от них, не от scale_factor.
        self._sort_cache_cd_version = -1
        self._sort_cache_bh_version = -1
        self._sort_cache_active_mask = None
        self._sort_cache_active_indices = None
        self._sort_cache_sort_idx = None
        self._sort_cache_d_sorted_comoving = None
        self._sort_cache_masses_active = None
        self._sort_cache_masses_version = -1

        # ОПТИМИЗАЦИЯ B: кэш cum_mass_sorted (учитывая γ-коррекцию).
        # Инвалидируется по совокупному ключу (cd_v, bh_v, v_v, masses_v,
        # scale_factor_bin). При паузе или нерелятивистском режиме без новых
        # эмиссий кэш стабильно попадает и убирает O(N) cumsum + fancy index.
        self._cum_cache_key = None
        self._cum_cache_value = None

        # ОПТИМИЗАЦИЯ F: кэш build_in_flight_laser_mass_state по
        # (laser_photons_version, scale_factor_bin) — повторные обращения в
        # одном кадре или на паузе используют готовое состояние.
        self._laser_state_cache_key = None
        self._laser_state_cache_value = None

        # ОПТИМИЗАЦИЯ C: ослабленный ключ кэша горизонтов SdS — храним
        # последнюю массу ЦЧД, по которой уже считались горизонты, и
        # пересчитываем только при превышении относительного порога.
        self._horizons_mass_rel_tol = 1e-6

        # Последний вызов calculate_masses: solve outer / inner AH + LTB event
        # (для PERFORMANCE PROFILING в visualization.renderer).
        self._last_ltb_horizon_solve_seconds = 0.0
        self._last_calculate_masses_points_seconds = 0.0
        self._last_calculate_masses_laser_prep_seconds = 0.0

    def _get_laser_mass_state(self, matter_points, scale_factor: float):
        """Кэшированный build_in_flight_laser_mass_state.

        ОПТИМИЗАЦИЯ F: ключ — (laser_photons_version, scale_factor_bin).
        Между кадрами в активном режиме фотоны меняются (advance + emit), и
        кэш промахивается. На паузе и при повторных вызовах в одном кадре
        попадание в кэш убирает повторную сортировку радиусов фотонов и
        cumsum по ним.
        """
        photons_version = getattr(matter_points, '_laser_photons_version', None)
        if photons_version is None or scale_factor <= 0.0:
            return matter_points.build_in_flight_laser_mass_state(scale_factor)
        scale_bin = round(float(scale_factor), 12)
        key = (photons_version, scale_bin)
        if self._laser_state_cache_key == key:
            return self._laser_state_cache_value
        state = matter_points.build_in_flight_laser_mass_state(scale_factor)
        self._laser_state_cache_key = key
        self._laser_state_cache_value = state
        return state

    def _black_hole_radius_for_mass(self, mass_kg: float) -> float:
        """
        Физический горизонт ЧД в SdS-модели.

        Это только стартовая нижняя оценка для LTB-решателя apparent horizon.
        Не используем вакуумный Nariai cap в динамическом LTB-пути: если
        статического SdS-корня нет, геометрию всё равно должен определить
        корень/минимум g(r) с полной M(<r).
        """
        mass = float(mass_kg)
        if mass <= 0.0:
            return 0.0

        schwarzschild_factor = 2.0 * G / (c**2)
        r_s = mass * schwarzschild_factor
        return r_s

    def _advance_bh_horizon_velocity_ema(
        self,
        universe,
        r_black_hole_schwarzschild: float,
        paused: bool,
    ) -> float:
        """
        Единая точка расчёта dr_BH/dt и EMA (_BLACK_HOLE_VELOCITY_SMOOTHING_ALPHA).

        Используется обоими ветками calculate_masses (быстрый путь после
        начала коллапса и путь до коллапса). Возвращает сглаженную EMA-скорость
        роста радиуса ЧД (м/с). Side-effects:
          - обновляет self.smoothed_velocity, self.last_velocity;
          - обновляет self.previous_bh_horizon_radius,
            self.last_bh_horizon_update_time, self.last_calculated_time
            ровно один раз за шаг времени.
        """
        bh_growth_velocity_radius = float(self.smoothed_velocity)
        outer_ok = (
            (
                not paused
                or (
                    self.last_bh_horizon_update_time is not None
                    and abs(universe.time - self.last_bh_horizon_update_time) > 1e-6
                )
                or (
                    self.last_calculated_time is not None
                    and abs(universe.time - self.last_calculated_time) > 1e-6
                )
            )
            and r_black_hole_schwarzschild >= 0
        )
        if not outer_ok:
            return bh_growth_velocity_radius

        if not (
            self.last_bh_horizon_update_time is None
            or abs(universe.time - self.last_bh_horizon_update_time) > 1e-6
        ):
            return bh_growth_velocity_radius

        if (
            self.last_bh_horizon_update_time is not None
            and self.last_bh_horizon_update_time > 0
            and self.previous_bh_horizon_radius > 0
        ):
            actual_dt = universe.time - self.last_bh_horizon_update_time
            if actual_dt > 1e-6:
                radius_change = (
                    r_black_hole_schwarzschild - self.previous_bh_horizon_radius
                )
                raw_velocity = radius_change / actual_dt
                self.last_velocity = raw_velocity

                if self.smoothed_velocity == 0.0 and raw_velocity != 0.0:
                    self.smoothed_velocity = raw_velocity
                else:
                    self.smoothed_velocity = (
                        _BLACK_HOLE_VELOCITY_SMOOTHING_ALPHA * raw_velocity
                        + (
                            (1.0 - _BLACK_HOLE_VELOCITY_SMOOTHING_ALPHA)
                            * self.smoothed_velocity
                        )
                    )
                bh_growth_velocity_radius = float(self.smoothed_velocity)

                if DEBUG:
                    rvf = format_velocity_m_per_s(raw_velocity)
                    svf = format_velocity_m_per_s(bh_growth_velocity_radius)
                    print(
                        f"DEBUG BH velocity (EMA): time={universe.time:.2e}, "
                        f"actual_dt={actual_dt:.2e}, "
                        f"prev_radius={self.previous_bh_horizon_radius:.2e}, "
                        f"curr_radius={r_black_hole_schwarzschild:.2e}, "
                        f"raw={raw_velocity:.2e} ({rvf}), "
                        f"smoothed={bh_growth_velocity_radius:.2e} ({svf})"
                    )
            else:
                bh_growth_velocity_radius = float(self.smoothed_velocity)
        else:
            bh_growth_velocity_radius = 0.0
            self.last_velocity = 0.0
            self.smoothed_velocity = 0.0
            if DEBUG:
                print(
                    "DEBUG BH velocity (EMA): first step or prev_radius=0 "
                    f"(t={universe.time:.2e}, r={r_black_hole_schwarzschild:.2e})"
                )

        self.previous_bh_horizon_radius = float(r_black_hole_schwarzschild)
        self.last_bh_horizon_update_time = float(universe.time)
        self.last_calculated_time = float(universe.time)
        return bh_growth_velocity_radius
    
    def calculate_masses(
        self,
        universe,
        cosmology,
        matter_points,
        paused,
        initialize_matter_points_func,
    ):
        """
        Вычислить массы в различных радиусах с учётом коллапса материи.

        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            matter_points: Объект MatterPoints
            paused: Флаг паузы
            initialize_matter_points_func: Функция для ленивой инициализации
                точек материи (вызывается, если они ещё не инициализированы).

        Returns:
            dict: Словарь с массами и радиусами горизонтов.
        """
        t_method_start = time.perf_counter()
        self._last_calculate_masses_points_seconds = 0.0
        self._last_calculate_masses_laser_prep_seconds = 0.0
        self._last_ltb_horizon_solve_seconds = 0.0

        # ДЕТАЛЬНОЕ ПРОФИЛИРОВАНИЕ для поиска узких мест.
        # ОПТИМИЗАЦИЯ D: timestamps пишем только когда DEBUG включён —
        # в горячем пути иначе теряется ~10 perf_counter() / кадр впустую.
        debug_profile = bool(DEBUG)
        profile_times = {} if debug_profile else None
        if debug_profile:
            t_total_start = time.perf_counter()
            t0 = time.perf_counter()

        scale_factor = cosmology.scale_factor
        r_particle_horizon = cosmology.particle_horizon(universe.time)

        if debug_profile:
            profile_times['particle_horizon'] = time.perf_counter() - t0

        # Горизонты SdS зависят от текущей массы ЦЧД (с учётом аккумулированной).
        # ОПТИМИЗАЦИЯ C: M_BH в активном режиме растёт постоянно (поглощение
        # фотонов лазера), но скорость относительная — обычно <1e-9 за шаг.
        # Кэш по строгому равенству промахивался каждый кадр и зря дёргал
        # de_sitter_horizon (numerical). Используем относительный допуск.
        M_black_hole_initial = 0.0
        M_bh_current = M_black_hole_initial + float(matter_points.accumulated_bh_mass)

        if debug_profile:
            t0 = time.perf_counter()

        time_changed = (self._last_horizons_time is None or
                        abs(universe.time - self._last_horizons_time) > 1e-6)
        if self._last_horizons_mass is None:
            mass_changed = True
        else:
            ref = max(abs(self._last_horizons_mass), abs(M_bh_current), 1.0)
            mass_changed = (
                abs(M_bh_current - self._last_horizons_mass)
                > self._horizons_mass_rel_tol * ref
            )

        if time_changed or mass_changed:
            # Только справочная пустая Λ-шкала для верхней численной границы.
            # Она не зависит от M_BH и не используется как сценарный горизонт.
            r_de_sitter_horizon = cosmology.de_sitter_horizon(0.0)
            self._cached_de_sitter_horizon = r_de_sitter_horizon
            self._last_horizons_time = universe.time
            self._last_horizons_mass = M_bh_current
        else:
            r_de_sitter_horizon = self._cached_de_sitter_horizon

        if debug_profile:
            profile_times['initial_horizons'] = time.perf_counter() - t0
        
        # ОПТИМИЗАЦИЯ: Используем уже вычисленный r_particle_horizon вместо повторного вызова
        particle_horizon_physical = r_particle_horizon
        
        # Плотность материи при текущем масштабном факторе
        # ρ_m = (Ω_DM + Ω_B) * ρ_crit / a³
        # scale_factor уже определен выше
        if scale_factor > 0:
            rho_matter = (OMEGA_DM + OMEGA_B) * RHO_CRIT / (scale_factor**3)
        else:
            rho_matter = 0.0
        
        # Центральная ЧД начинается с нулевой массы и растёт только от поглощённой материи/фотонов.
        M_black_hole_initial = 0.0
        M_nariai = NARIAI_BLACK_HOLE_MASS_KG

        # Инициализируем точки материи, если они еще не инициализированы
        if matter_points.points_comoving is None:
            initialize_matter_points_func()

        if debug_profile:
            t_fast = time.perf_counter()
        M_black_hole = M_bh_current

        # «Классический» радиус ЦЧД от точечной массы (без учёта плотной
        # оболочки в окрестности): Шварцшильд / SdS-внутренний / Нараи. Это
        # нижняя оценка горизонта; ниже она будет уточнена самосогласованным
        # apparent inner horizon.
        r_black_hole_classical = self._black_hole_radius_for_mass(M_black_hole)
        r_black_hole_schwarzschild = r_black_hole_classical

        # Горизонты уже посчитаны/взяты из кэша по (time, M_bh_current)
        # выше — переиспользуем без повторных вызовов cosmology.

        # Массы в горизонтах: учитываем релятивистскую массу-эквивалент
        # E/c² = γ·m_rest для активных точек (поглощённые в ЦЧД уже учтены
        # через accumulated_bh_mass с γ на момент поглощения и добавляются
        # как M_black_hole, если центр внутри радиуса).
        M_BH_inside = float(M_black_hole)
        cd = matter_points.comoving_distances
        have_points_data = (
            cd is not None
            and len(cd) > 0
            and matter_points.masses_per_point is not None
            and len(matter_points.masses_per_point) == len(cd)
        )

        if have_points_data:
            # Версии нужны и для кэша порядка, и для кэша cum_mass_sorted.
            cd_v = matter_points._comoving_distances_version
            bh_v = matter_points._points_inside_bh_version
            v_v = matter_points._velocities_version
            masses_v = matter_points._masses_version

            # ОПТИМИЗАЦИЯ: переиспользуем sort_idx + active_mask + active_indices
            # + d_sorted_comoving + masses_active из кэша, пока comoving
            # distances / points_inside_bh / masses_per_point не изменились.
            sort_cache_valid = (
                self._sort_cache_cd_version == cd_v
                and self._sort_cache_bh_version == bh_v
                and self._sort_cache_d_sorted_comoving is not None
                and self._sort_cache_active_mask is not None
                and len(self._sort_cache_active_mask) == len(cd)
            )
            masses_cache_valid = (
                sort_cache_valid
                and self._sort_cache_masses_active is not None
                and self._sort_cache_masses_version == masses_v
            )

            if sort_cache_valid:
                active_mask = self._sort_cache_active_mask
                active_indices = self._sort_cache_active_indices
                sort_idx = self._sort_cache_sort_idx
                d_sorted_comoving = self._sort_cache_d_sorted_comoving
            else:
                n_total = len(cd)
                active_mask = np.ones(n_total, dtype=bool)
                inside_arr = matter_points.get_inside_indices_arr()
                if inside_arr.size > 0:
                    inside_arr = inside_arr[
                        (inside_arr >= 0) & (inside_arr < n_total)
                    ]
                    if inside_arr.size > 0:
                        active_mask[inside_arr] = False
                active_indices = np.flatnonzero(active_mask)
                if active_indices.size > 0:
                    cd_a = cd[active_indices]
                    sort_idx = np.argsort(cd_a)
                    d_sorted_comoving = cd_a[sort_idx]
                else:
                    sort_idx = np.empty(0, dtype=np.intp)
                    d_sorted_comoving = np.empty(0, dtype=np.float64)
                self._sort_cache_active_mask = active_mask
                self._sort_cache_active_indices = active_indices
                self._sort_cache_sort_idx = sort_idx
                self._sort_cache_d_sorted_comoving = d_sorted_comoving
                self._sort_cache_cd_version = cd_v
                self._sort_cache_bh_version = bh_v
                self._sort_cache_masses_active = None  # masses_v больше не валиден
                self._sort_cache_masses_version = -1

            if masses_cache_valid:
                masses_active = self._sort_cache_masses_active
            else:
                masses_rest = matter_points.masses_per_point
                if masses_rest.dtype != np.float64:
                    masses_rest = masses_rest.astype(np.float64, copy=False)
                if active_indices.size > 0:
                    masses_active = masses_rest[active_indices]
                else:
                    masses_active = np.empty(0, dtype=np.float64)
                self._sort_cache_masses_active = masses_active
                self._sort_cache_masses_version = masses_v
            n_active = masses_active.size

            # ОПТИМИЗАЦИЯ B: cum_mass_sorted полностью кэшируется по
            # совокупному ключу (cd_v, bh_v, v_v, masses_v, scale_bin).
            # На паузе или нерелятивистском режиме без изменений все
            # компоненты ключа стабильны → попадание в кэш убирает
            # cumsum + fancy indexing + γ-коррекцию (~1.5–2.5 мс).
            #
            # scale_factor квантуется до ~9 значащих цифр (ошибка <1e-9):
            # γ-зависимость от a² поглощается этим же бином.
            if scale_factor > 0:
                scale_bin = round(float(scale_factor), 12)
            else:
                scale_bin = 0.0
            cum_key = (cd_v, bh_v, v_v, masses_v, scale_bin)

            if (
                self._cum_cache_key == cum_key
                and self._cum_cache_value is not None
            ):
                cum_mass_sorted = self._cum_cache_value
            else:
                # ОПТИМИЗАЦИЯ A: γ-коррекция через предрассчитанный
                # _v_sq_comoving — никакого einsum по (N,3) каждый кадр.
                # Порог _BETA2_NEGLIGIBLE_THRESHOLD проверяется по скаляру
                # _v_sq_max за O(1).
                v_sq_full = matter_points._v_sq_comoving
                v_sq_max = float(matter_points._v_sq_max)
                if (n_active > 0 and v_sq_full is not None
                        and len(v_sq_full) == len(cd) and scale_factor > 0):
                    sf2_over_c2 = (scale_factor * scale_factor) / (c * c)
                    beta2_max = v_sq_max * sf2_over_c2
                    if beta2_max < _BETA2_NEGLIGIBLE_THRESHOLD:
                        m_eff_a = masses_active
                    else:
                        beta2_a = v_sq_full[active_indices] * sf2_over_c2
                        np.clip(beta2_a, 0.0, _BETA2_CLIP_MAX, out=beta2_a)
                        gamma_a = 1.0 / np.sqrt(1.0 - beta2_a)
                        m_eff_a = gamma_a * masses_active
                else:
                    m_eff_a = masses_active

                if n_active > 0:
                    cum_mass_sorted = np.cumsum(m_eff_a[sort_idx])
                else:
                    cum_mass_sorted = np.empty(0, dtype=np.float64)
                self._cum_cache_key = cum_key
                self._cum_cache_value = cum_mass_sorted
        else:
            d_sorted_comoving = np.empty(0, dtype=np.float64)
            cum_mass_sorted = np.empty(0, dtype=np.float64)

        # Массы внутри горизонтов считаются ниже, после LTB-расчёта
        # r_hubble_horizon. Старый c/H_FLRW уже недостаточен: радиус Хаббла
        # должен видеть тот же M(<r), что apparent/event диагностика.
        self._last_calculate_masses_points_seconds = (
            time.perf_counter() - t_method_start
        )
        t_laser_get = time.perf_counter()
        laser_mass_state = self._get_laser_mass_state(matter_points, scale_factor)
        self._last_calculate_masses_laser_prep_seconds = (
            time.perf_counter() - t_laser_get
        )

        if debug_profile:
            fast_path_time = time.perf_counter() - t_fast
            profile_times['fast_path_total'] = fast_path_time

        # === Apparent horizons (чистый LTB-Λ, без Birkhoff-вычета) ===
        # Берём ПОЛНУЮ массу-энергию внутри сферы радиуса r:
        #   M(<r) = M_BH
        #         + M_matter_inside(r)        (с γ·m_rest для активных)
        #         + E_laser_inside(r)/c²      (с космологическим redshift)
        # Условие apparent horizon: g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1 = 0.
        # Уравнение g(r) = 0 даёт ОБА apparent horizon: первый ноль при
        # движении наружу от r → 0+ — внутренний (ЦЧД), последний — внешний
        # (космологический Hubble/de Sitter). В режиме «маленькая ЦЧД на
        # FRW-фоне» они физически разделены; при сближении M_BH к
        # критической массе эпохи M_crit ≈ c³/(G·H·3√3) — сливаются.

        # Состояние материи в формате helpers'а (см. physics/apparent_horizon).
        matter_state = {
            'd_sorted_comoving': d_sorted_comoving,
            'cum_mass_sorted': cum_mass_sorted,
        }

        lam_apparent = cosmological_constant_lambda()
        _t_ltb_horizons = time.perf_counter()

        # ОПТИМИЗАЦИЯ: переиспользуем event/hubble горизонты, посчитанные ранее
        # за этот же кадр (например, в MatterSimulation._emission_boundary_radius
        # перед update_positions). Ключ — (universe.time, scale_factor),
        # инвариантные внутри одного кадра. Это убирает повторный scipy.solve_ivp
        # на ~150 мс с calculate_masses.
        cached_event_m = None
        cached_hubble_m = None
        cache_key = None
        try:
            cache_key = matter_points.ltb_horizons_cache_key(
                universe.time, scale_factor,
            )
            if matter_points._cached_ltb_horizons_key == cache_key:
                cev = matter_points._cached_ltb_event_horizon_m
                chu = matter_points._cached_ltb_hubble_horizon_m
                if cev is not None and cev > 0.0:
                    cached_event_m = float(cev)
                if chu is not None and chu > 0.0:
                    cached_hubble_m = float(chu)
        except Exception:
            cache_key = None

        # --- Apparent outer (cosmological) horizon ---
        # Последний ноль g(r) = 0; в чистом LTB это честный локальный аналог
        # Hubble-радиуса: в FLRW-пределе ≈ c/H(t), а при центральной массе и
        # дискретной материи смещается по полной M(<r).
        r_de_sitter_classical_m = float(r_de_sitter_horizon)
        if cached_hubble_m is not None:
            r_hubble_horizon = cached_hubble_m
        else:
            # Верхняя граница: чистый de Sitter (M=0) = √(3/Λ), это максимально
            # возможное значение внешнего AH в нашей космологии.
            r_outer_upper = max(
                float(r_de_sitter_horizon),
                float(cosmology.de_sitter_horizon(0.0)),
            ) * 1.001
            r_hubble_horizon = solve_apparent_outer_horizon(
                M_black_hole, matter_state, laser_mass_state, scale_factor,
                0.0, r_outer_upper, lam_apparent,
            )
        # Исторический ключ r_de_sitter_horizon_m сохраняем совместимым с уже
        # существующей диагностикой outer apparent horizon. Визуальный пунктир
        # «de Sitter (empty universe)» рисуется отдельно как √(3/Λ).
        r_de_sitter_horizon = r_hubble_horizon
        # Event horizon считаем отдельно как LTB null-сепаратрису, а не
        # приравниваем к локальному apparent/Hubble horizon.
        if cached_event_m is not None:
            r_event_horizon = cached_event_m
        else:
            r_event_horizon = compute_ltb_event_horizon(
                universe.time, M_black_hole, matter_state, laser_mass_state,
                scale_factor, lam_apparent, r_hubble_horizon,
            )

        # --- Apparent inner horizon (горизонт ЦЧД) ---
        # Первый ноль g(r) = 0 идя наружу от r → 0+. Ищем его только внутри
        # найденного LTB outer/Hubble AH. Если untrapped-зона исчезла, решатель
        # вернёт LTB-радиус слияния, а не вакуумный r_N.
        r_AH_upper = float(r_hubble_horizon)
        if r_black_hole_classical > 0.0 and r_AH_upper > r_black_hole_classical:
            r_apparent_inner_horizon = solve_apparent_inner_horizon(
                M_black_hole, matter_state, laser_mass_state,
                scale_factor,
                r_black_hole_classical, r_AH_upper, lam_apparent,
            )
        else:
            r_apparent_inner_horizon = 0.0
        if r_apparent_inner_horizon > 0.0:
            r_black_hole_schwarzschild = r_apparent_inner_horizon

        self._last_ltb_horizon_solve_seconds = (
            time.perf_counter() - _t_ltb_horizons
        )

        # ОПТИМИЗАЦИЯ: сохраняем r_event_horizon и r_hubble_horizon в кэш на
        # MatterPoints. MatterSimulation._ltb_event_horizon /
        # _ltb_hubble_horizon (вызываются из _emission_boundary_radius в
        # update_collapse каждый кадр) переиспользуют этот результат вместо
        # повторного scipy.solve_ivp на ~150 мс.
        if cache_key is not None:
            matter_points._cached_ltb_event_horizon_m = float(r_event_horizon)
            matter_points._cached_ltb_hubble_horizon_m = float(r_hubble_horizon)
            matter_points._cached_ltb_horizons_key = cache_key

        # NB: cosmology.cosmological_event_horizon остаётся справочным FLRW/SdS
        # helper'ом, но в сценарном словаре масс не используется: он не знает
        # дискретный LTB-профиль matter_state/laser_state и поэтому не должен
        # смешиваться с r_hubble_horizon_m.

        # Финальные массы внутри горизонтов. К массе внутри каждого горизонта
        # добавляется M_BH, активная γ·m_rest материя и E/c² лазерных фотонов.
        def _mass_inside_horizon(radius: float) -> float:
            if not np.isfinite(radius) or radius <= 0.0:
                return 0.0
            return (
                float(M_BH_inside)
                + matter_mass_inside_phys(radius, matter_state, scale_factor)
                + laser_mass_inside_phys(radius, laser_mass_state)
            )

        M_hubble_horizon = _mass_inside_horizon(r_hubble_horizon)
        M_particle_horizon = _mass_inside_horizon(r_particle_horizon)
        M_event_horizon = _mass_inside_horizon(r_event_horizon)
        M_de_sitter_horizon = _mass_inside_horizon(r_de_sitter_horizon)

        # Финальная диагностика внутри обновлённого outer AH.
        M_matter_inside_de_sitter = matter_mass_inside_phys(
            r_de_sitter_horizon, matter_state, scale_factor,
        )
        M_laser_inside_de_sitter = laser_mass_inside_phys(
            r_de_sitter_horizon, laser_mass_state,
        )
        # Справочно: масса «гладкого FRW-фона» внутри outer AH. В чистом
        # LTB-вычислении НЕ вычитается (вся материя честно входит в M(<r)),
        # показываем только для сравнения с M_matter_inside_de_sitter —
        # они должны быть близки при равномерном распределении оболочек.
        M_bg_inside_de_sitter = (
            rho_matter * (4.0 / 3.0) * np.pi * (r_de_sitter_horizon ** 3)
        )
        M_eff_de_sitter = (
            float(M_black_hole)
            + M_matter_inside_de_sitter
            + M_laser_inside_de_sitter
        )

        # Скорость роста ЦЧД через EMA-хелпер: считаем по apparent inner
        # horizon (это и есть «настоящий» горизонт). Когда apparent совпадает
        # с классическим — поведение прежнее; когда trapped surface обнимает
        # оболочку, EMA отразит реальную скорость расширения горизонта.
        bh_growth_velocity_radius = self._advance_bh_horizon_velocity_ema(
            universe,
            float(r_black_hole_schwarzschild),
            paused,
        )
        self.previous_bh_mass = float(M_black_hole)
        bh_growth_velocity_formatted = format_velocity_m_per_s(bh_growth_velocity_radius)

        return {
            'M_hubble_horizon_kg': M_hubble_horizon,
            'M_particle_horizon_kg': M_particle_horizon,
            'M_event_horizon_kg': M_event_horizon,
            'M_de_sitter_horizon_kg': M_de_sitter_horizon,
            'M_nariai_kg': M_nariai,
            'M_black_hole_kg': M_black_hole,
            'M_black_hole_initial_kg': M_black_hole_initial,
            # r_black_hole_schwarzschild_m — это apparent inner horizon (наш
            # «горизонт ЧД» для отрисовки и для cap'ов космологических
            # горизонтов). Имя сохранено для обратной совместимости.
            'r_black_hole_schwarzschild_m': r_black_hole_schwarzschild,
            'r_apparent_inner_horizon_m': r_apparent_inner_horizon,
            'r_black_hole_classical_m': r_black_hole_classical,
            'bh_growth_velocity_radius_m_per_s': bh_growth_velocity_radius,
            'bh_growth_velocity_formatted': bh_growth_velocity_formatted,
            'r_hubble_horizon_m': r_hubble_horizon,
            'r_particle_horizon_m': r_particle_horizon,
            'r_event_horizon_m': r_event_horizon,
            'r_de_sitter_horizon_m': r_de_sitter_horizon,
            # Диагностика apparent SdS-горизонта (модель Эйнштейна-Штрауса):
            'r_de_sitter_classical_m': r_de_sitter_classical_m,
            'M_eff_de_sitter_kg': M_eff_de_sitter,
            'M_matter_inside_de_sitter_kg': M_matter_inside_de_sitter,
            'M_laser_inside_de_sitter_kg': M_laser_inside_de_sitter,
            'M_bg_inside_de_sitter_kg': M_bg_inside_de_sitter,
        }
