"""
Калькулятор масс для различных горизонтов и черной дыры
"""
import time

import numpy as np

import config
from config import DEBUG
from physics.nariai import nariai_radius, schwarzschild_de_sitter_horizons
from utils.constants import (
    G,
    NARIAI_BLACK_HOLE_MASS_KG,
    OMEGA_B,
    OMEGA_DM,
    RHO_CRIT,
    c,
)
from utils.config_utils import get_collapse_start_time_seconds, get_dt, get_mass_per_point_kg
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
        self.previous_particle_horizon_mass = 0.0  # Предыдущая масса горизонта частиц
        self.previous_particle_horizon_radius = 0.0  # Предыдущий радиус горизонта частиц

        # Кэш горизонтов по (universe.time, M_bh_current)
        self._last_horizons_time = None
        self._last_horizons_mass = None
        self._cached_event_horizon = None
        self._cached_hubble_horizon = None
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

        До Нараи используем внутренний горизонт SdS. В пределе Нараи внутренний
        и космологический горизонты сливаются на r_N. Для M > M_N статического
        SdS-горизонта уже нет, поэтому не продолжаем радиус произвольной
        формулой: оставляем r_N как физический предельный индикатор, а масса ЧД
        может продолжать расти отдельно.
        """
        mass = float(mass_kg)
        if mass <= 0.0:
            return 0.0

        schwarzschild_factor = 2.0 * G / (c**2)
        if mass >= NARIAI_BLACK_HOLE_MASS_KG:
            try:
                return nariai_radius()
            except Exception:
                return NARIAI_BLACK_HOLE_MASS_KG * schwarzschild_factor

        r_inner, _ = schwarzschild_de_sitter_horizons(mass)
        r_s = mass * schwarzschild_factor
        return r_inner if r_inner > 0.0 else r_s

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
        get_physical_points_func,
        initialize_matter_points_func,
        add_matter_points_func,
    ):
        """
        Вычислить массы в различных радиусах с учетом коллапса материи
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            matter_points: Объект MatterPoints
            paused: Флаг паузы
            get_physical_points_func: Функция для получения физических координат точек
            initialize_matter_points_func: Функция для инициализации точек материи
            add_matter_points_func: Функция для добавления точек материи
        
        Returns:
            dict: Словарь с массами и радиусами
        """
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
            r_de_sitter_horizon = cosmology.de_sitter_horizon(M_bh_current)
            r_event_horizon = cosmology.cosmological_event_horizon(universe.time, M_bh_current)
            r_hubble_horizon = cosmology.hubble_horizon(universe.time, M_bh_current)
            self._cached_de_sitter_horizon = r_de_sitter_horizon
            self._cached_event_horizon = r_event_horizon
            self._cached_hubble_horizon = r_hubble_horizon
            self._last_horizons_time = universe.time
            self._last_horizons_mass = M_bh_current
        else:
            r_de_sitter_horizon = self._cached_de_sitter_horizon
            r_event_horizon = self._cached_event_horizon
            r_hubble_horizon = self._cached_hubble_horizon

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
        
        # Проверяем, начался ли коллапс
        collapse_started = universe.time >= get_collapse_start_time_seconds()

        # Инициализируем точки материи, если они еще не инициализированы
        if matter_points.points_comoving is None:
            initialize_matter_points_func()
        
        # Инициализируем радиус ЧД (будет переопределен ниже)
        r_black_hole_schwarzschild = 0.0
        M_black_hole = M_black_hole_initial
        
        # Инициализируем переменные масс (будут переопределены ниже)
        M_hubble_horizon = 0.0
        M_particle_horizon = 0.0
        M_event_horizon = 0.0
        M_de_sitter_horizon = 0.0
        M_nariai = NARIAI_BLACK_HOLE_MASS_KG
        
        def mass_in_radius(radius):
            """Масса материи в сфере заданного радиуса (по плотности ρ_m·a⁻³)."""
            volume = (4/3) * np.pi * radius**3
            return rho_matter * volume

        # ===== ОПТИМИЗАЦИЯ: Используем comoving_distances напрямую =====
        # Вместо преобразования всех точек в физические координаты,
        # пересчитываем горизонты в comoving и сравниваем с comoving_distances
        # Это экономит ~14ms на преобразование 245K точек
        comoving_distances = None
        scale_ratio = 1.0
        
        if matter_points.points_comoving is not None and len(matter_points.points_comoving) > 0:
            if debug_profile:
                t0 = time.perf_counter()

            if matter_points.comoving_distances is not None:
                comoving_distances = matter_points.comoving_distances
            else:
                comoving_distances = np.sqrt(np.sum(matter_points.points_comoving**2, axis=1))

            if debug_profile:
                profile_times['get_comoving_distances'] = time.perf_counter() - t0

            if comoving_distances is None:
                return None
        
        # ========== БЫСТРЫЙ ПУТЬ: Используем accumulated_bh_mass ==========
        # Если коллапс начался, считаем массу ЧД через accumulated_bh_mass и
        # массы в горизонтах через cumsum + searchsorted за O(N log N).
        if collapse_started:
            if debug_profile:
                t_fast = time.perf_counter()
            M_black_hole = M_bh_current

            # Радиус ЧД: SdS до Нараи, непрерывное продолжение после Нараи.
            r_black_hole_schwarzschild = self._black_hole_radius_for_mass(M_black_hole)

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

            # Векторизованный поиск массы внутри 4 радиусов одним searchsorted.
            # Сравниваем в comoving-пространстве, чтобы избежать умножения N
            # расстояний на scale_factor — делим только 4 радиуса.
            horizon_radii = np.array(
                [r_hubble_horizon, r_particle_horizon,
                 r_event_horizon, r_de_sitter_horizon],
                dtype=np.float64,
            )
            valid_radii_mask = np.isfinite(horizon_radii) & (horizon_radii > 0)
            add_from_points = np.zeros(4, dtype=np.float64)
            if d_sorted_comoving.size > 0 and np.any(valid_radii_mask) and scale_factor > 0:
                radii_comoving = horizon_radii / scale_factor
                idx = np.searchsorted(d_sorted_comoving, radii_comoving, side='right')
                has_inside = (idx > 0) & valid_radii_mask
                if np.any(has_inside):
                    add_from_points[has_inside] = cum_mass_sorted[idx[has_inside] - 1]

            M_BH_contrib = np.where(valid_radii_mask, M_BH_inside, 0.0)
            (M_hubble_horizon, M_particle_horizon,
             M_event_horizon, M_de_sitter_horizon) = (
                (M_BH_contrib + add_from_points).tolist()
            )

            # ВАЖНО: К массе внутри каждого горизонта добавляем массу-эквивалент
            # E/c² энергии лазерных фотонов, ещё летящих к ЦЧД и физически
            # находящихся внутри этого горизонта. Без этого учитывается только
            # «похудевшая» масса покоя точек, и часть энергии (фотоны в полёте)
            # пропадает из баланса масс внутри горизонтов.
            laser_mass_state = self._get_laser_mass_state(matter_points, scale_factor)
            laser_mass_in_horizons = matter_points.in_flight_laser_mass_inside_radii(
                horizon_radii, scale_factor, laser_mass_state,
            )
            M_hubble_horizon += float(laser_mass_in_horizons[0])
            M_particle_horizon += float(laser_mass_in_horizons[1])
            M_event_horizon += float(laser_mass_in_horizons[2])
            M_de_sitter_horizon += float(laser_mass_in_horizons[3])

            # Скорость роста ЧД — тот же EMA, что в медленном пути (быстрый путь делал раньше return без расчёта)
            bh_growth_velocity_radius = self._advance_bh_horizon_velocity_ema(
                universe,
                float(r_black_hole_schwarzschild),
                paused,
            )
            self.previous_bh_mass = float(M_black_hole)
            bh_growth_velocity_formatted = format_velocity_m_per_s(bh_growth_velocity_radius)

            if debug_profile:
                fast_path_time = time.perf_counter() - t_fast
                profile_times['fast_path_total'] = fast_path_time

            return {
                'M_hubble_horizon_kg': M_hubble_horizon,
                'M_particle_horizon_kg': M_particle_horizon,
                'M_event_horizon_kg': M_event_horizon,
                'M_de_sitter_horizon_kg': M_de_sitter_horizon,
                'M_nariai_kg': M_nariai,
                'M_black_hole_kg': M_black_hole,
                'M_black_hole_initial_kg': M_black_hole_initial,
                'r_black_hole_schwarzschild_m': r_black_hole_schwarzschild,
                'bh_growth_velocity_radius_m_per_s': bh_growth_velocity_radius,
                'bh_growth_velocity_formatted': bh_growth_velocity_formatted,
                'r_hubble_horizon_m': r_hubble_horizon,
                'r_particle_horizon_m': r_particle_horizon,
                'r_event_horizon_m': r_event_horizon,
                'r_de_sitter_horizon_m': r_de_sitter_horizon,
            }
        # ========== КОНЕЦ БЫСТРОГО ПУТИ ==========
        # Сюда мы попадаем только если коллапс ещё не начался (быстрый путь
        # выше уже сделал return). Считаем массы внутри горизонтов из точек
        # материи (если есть) либо аналитически из плотности.
        points_in_hubble = 0
        points_in_particle = 0
        points_in_event = 0
        points_in_de_sitter = 0
        M_black_hole = M_black_hole_initial
        r_black_hole_schwarzschild = 0.0

        if matter_points.points_comoving is not None and len(matter_points.points_comoving) > 0:
            if debug_profile:
                t0 = time.perf_counter()
            physical_points, distances_from_center, scale_ratio = get_physical_points_func(
                r_particle_horizon
            )
            if debug_profile:
                profile_times['get_physical_points'] = time.perf_counter() - t0

            if distances_from_center is not None and len(distances_from_center) > 0:
                if debug_profile:
                    t0 = time.perf_counter()
                points_in_hubble = np.sum(distances_from_center <= r_hubble_horizon)
                points_in_particle = np.sum(distances_from_center <= r_particle_horizon)
                points_in_event = np.sum(distances_from_center <= r_event_horizon)
                points_in_de_sitter = np.sum(distances_from_center <= r_de_sitter_horizon)
                if debug_profile:
                    profile_times['count_points'] = time.perf_counter() - t0

        if points_in_hubble > 0 or points_in_particle > 0 or points_in_event > 0 or points_in_de_sitter > 0:
            # До коллапса все точки имеют одинаковую массу покоя
            # (get_mass_per_point_kg(), см. matter_simulation.initialize_matter_points).
            mass_per_point = get_mass_per_point_kg()
            M_hubble_horizon = points_in_hubble * mass_per_point
            M_particle_horizon = points_in_particle * mass_per_point
            M_event_horizon = points_in_event * mass_per_point
            M_de_sitter_horizon = points_in_de_sitter * mass_per_point
        else:
            M_hubble_horizon = mass_in_radius(r_hubble_horizon)
            M_particle_horizon = mass_in_radius(r_particle_horizon)
            M_event_horizon = mass_in_radius(r_event_horizon)
            M_de_sitter_horizon = mass_in_radius(r_de_sitter_horizon)

        # ОТЛАДКА: В Lambda CDM масса внутри горизонта частиц должна только расти.
        # Проверяем только при движении времени вперёд.
        if (self.previous_particle_horizon_mass > 0 and
            M_particle_horizon < self.previous_particle_horizon_mass and
            self.last_calculated_time is not None and
            universe.time >= self.last_calculated_time):

            time_years = universe.time / 3.154e16
            rho_matter = (cosmology.omega_dm + cosmology.omega_b) * cosmology.rho_crit / (scale_factor**3)
            M_particle_horizon_theoretical = (4/3) * np.pi * (r_particle_horizon**3) * rho_matter

            print(f"\n{'='*80}")
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] Масса горизонта частиц УМЕНЬШАЕТСЯ! Время: {time_years:.2f} млрд лет")
            print(f"  Текущая масса: {M_particle_horizon:.2e} кг")
            print(f"  Предыдущая масса: {self.previous_particle_horizon_mass:.2e} кг")
            print(f"  Разница: {self.previous_particle_horizon_mass - M_particle_horizon:.2e} кг")
            print(f"  Текущий радиус горизонта (физический): {r_particle_horizon/9.461e15:.4f} млрд св. лет")
            print(f"  Предыдущий радиус: {self.previous_particle_horizon_radius/9.461e15:.4f} млрд св. лет")
            print(f"  Текущее количество точек: {points_in_particle}")
            print(f"  Теоретическая масса (из плотности): {M_particle_horizon_theoretical:.2e} кг")
            print(f"  Плотность материи: {rho_matter:.2e} кг/м³")
            print(f"  Масштабный фактор: {scale_factor:.6e}")
            print(f"  Горизонт частиц в сопутствующих координатах: {r_particle_horizon/scale_factor/9.461e15:.4f} млрд св. лет")
            print(f"{'='*80}\n")

        self.previous_particle_horizon_mass = M_particle_horizon
        self.previous_particle_horizon_radius = r_particle_horizon

        M_nariai = NARIAI_BLACK_HOLE_MASS_KG

        if M_black_hole > 0:
            r_black_hole_schwarzschild = self._black_hole_radius_for_mass(M_black_hole)
        else:
            r_black_hole_schwarzschild = 0.0

        # Скорость роста ЧД через единый EMA-хелпер (та же логика, что в
        # быстром пути). _black_hole_radius_for_mass всегда возвращает >= 0,
        # поэтому ветка с отрицательным радиусом удалена как мёртвая.
        bh_growth_velocity_radius = self._advance_bh_horizon_velocity_ema(
            universe,
            float(r_black_hole_schwarzschild),
            paused,
        )
        self.previous_bh_mass = float(M_black_hole)
        bh_growth_velocity_formatted = format_velocity_m_per_s(bh_growth_velocity_radius)
        
        # Выводим детальное профилирование каждые 60 кадров (НЕ во время паузы)
        if not hasattr(self, '_detail_profile_count'):
            self._detail_profile_count = 0
        self._detail_profile_count += 1
        
        if (debug_profile and self._detail_profile_count % 60 == 0
                and profile_times is not None and len(profile_times) > 0 and not paused):
            total_time = sum(profile_times.values())
            print("\n" + "="*70)
            print("DETAILED PROFILING calculate_masses (time in milliseconds):")
            print("="*70)
            sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
            for operation, elapsed in sorted_times:
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                print(f"  {operation:40s}: {elapsed*1000:7.2f} ms ({percentage:5.1f}%)")
            print(f"  {'TOTAL calculate_masses':40s}: {total_time*1000:7.2f} ms")
            if hasattr(cosmology, '_particle_interp_used') and hasattr(cosmology, '_event_interp_used'):
                print(f"  {'Interpolation stats':40s}: particle={cosmology._particle_interp_used}, event={cosmology._event_interp_used}")
            print("="*70 + "\n")
        
        # ВАЖНО: К массе внутри каждого горизонта добавляем массу-эквивалент
        # E/c² энергии лазерных фотонов, ещё летящих к ЦЧД и физически
        # находящихся внутри этого горизонта. До этого M_*_horizon учитывал
        # только «похудевшую» массу покоя точек (m_rest, уменьшенную из-за
        # излучения), а энергия уже излучённых, но ещё не дошедших до ЦЧД
        # фотонов выпадала из баланса. Эта поправка делает учёт массы внутри
        # горизонтов согласованным с законом сохранения энергии.
        laser_mass_state = self._get_laser_mass_state(matter_points, scale_factor)
        laser_mass_in_horizons = matter_points.in_flight_laser_mass_inside_radii(
            np.array(
                [r_hubble_horizon, r_particle_horizon,
                 r_event_horizon, r_de_sitter_horizon],
                dtype=np.float64,
            ),
            scale_factor,
            laser_mass_state,
        )
        M_hubble_horizon += float(laser_mass_in_horizons[0])
        M_particle_horizon += float(laser_mass_in_horizons[1])
        M_event_horizon += float(laser_mass_in_horizons[2])
        M_de_sitter_horizon += float(laser_mass_in_horizons[3])

        return {
            'M_hubble_horizon_kg': M_hubble_horizon,
            'M_particle_horizon_kg': M_particle_horizon,
            'M_event_horizon_kg': M_event_horizon,
            'M_de_sitter_horizon_kg': M_de_sitter_horizon,
            'M_nariai_kg': M_nariai,
            'M_black_hole_kg': M_black_hole,
            'M_black_hole_initial_kg': M_black_hole_initial,
            'r_black_hole_schwarzschild_m': r_black_hole_schwarzschild,
            'bh_growth_velocity_radius_m_per_s': bh_growth_velocity_radius,
            'bh_growth_velocity_formatted': bh_growth_velocity_formatted,
            # ОПТИМИЗАЦИЯ: Кэшируем радиусы горизонтов чтобы не пересчитывать в draw_horizons
            'r_hubble_horizon_m': r_hubble_horizon,
            'r_particle_horizon_m': r_particle_horizon,
            'r_event_horizon_m': r_event_horizon,
            'r_de_sitter_horizon_m': r_de_sitter_horizon,
        }
