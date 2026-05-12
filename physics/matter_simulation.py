"""
Модуль для симуляции материи во Вселенной
Содержит всю логику генерации, обновления и расчетов для точек материи
"""
import random
import time

import numpy as np

import config
from physics.apparent_horizon import (
    build_active_matter_state,
    compute_ltb_event_horizon,
    solve_apparent_inner_horizon,
    solve_apparent_outer_horizon,
)
from physics.cosmology import calculate_initial_horizons
from physics.matter_points import MatterPoints
from physics.nariai import (
    cosmological_constant_lambda,
)
from utils.config_utils import (
    get_collapse_start_time_seconds,
    get_dt,
    get_initial_time_seconds,
    get_mass_per_point_kg,
)
from utils.constants import (
    BILLION_LIGHT_YEARS_IN_METERS,
    CAUSAL_HORIZON_COMOVING_METERS,
    PARTICLE_HORIZON_MASS_LIMIT_KG,
    SECONDS_PER_YEAR,
    c,
)
from utils.cosmology_utils import calculate_scale_factor_at_time


class MatterSimulation:
    """Класс для управления симуляцией материи"""
    
    def __init__(self, mode=None):
        """Инициализация симуляции материи"""
        self.mode = mode
        self.matter_points = MatterPoints()
        self.matter_points_initialized = False
        self.current_num_points = 0
        self.last_added_radius_comoving = 0.0
        self.collapse_started = False
        self.last_collapse_time = 0.0
        self._initial_horizon_comoving = 0.0
        # solve_apparent_* внутри _capture_inside_apparent_inner_horizon за последний кадр
        self._last_capture_ltb_horizons_seconds = 0.0
    
    def generate_points_in_3d_sphere(self, num_points: int, radius: float, seed: int = None) -> np.ndarray:
        """
        Генерирует точки, равномерно распределенные в 3D сфере.
        
        Использует метод Marsaglia (rejection sampling) для равномерного
        распределения точек по объему сферы без артефактов сферических координат.
        
        Args:
            num_points: Количество точек для генерации
            radius: Радиус сферы (м)
            seed: Seed для генератора случайных чисел (опционально)
        
        Returns:
            np.ndarray: Массив точек в 3D координатах (N x 3)
        """
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] generate_points_in_3d_sphere: starting with num_points={num_points}, radius={radius:.2e}")
        start_time = time.perf_counter()
        
        # Используем независимый генератор случайных чисел для лучшей случайности
        # Это избегает проблем с глобальным состоянием np.random
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            # Если seed не задан, используем полностью случайный
            rng = np.random.default_rng(random.randint(0, 2**31 - 1))
        
        # ОПТИМИЗАЦИЯ: Ограничиваем максимальное количество точек
        max_points = int(1e6)  # Максимум 1 миллион точек
        if num_points > max_points:
            print(f"Warning: Limiting num_points from {num_points} to {max_points}")
            num_points = max_points
        
        num_points = int(num_points)
        if num_points <= 0:
            return np.array([], dtype=np.float64).reshape(0, 3)
        
        # УЛУЧШЕННЫЙ АЛГОРИТМ: Метод Marsaglia (rejection sampling)
        # Генерируем точки в кубе [-1, 1]^3 и отбрасываем точки вне сферы
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] Generating uniform points in 3D sphere using Marsaglia method...")
        t0 = time.perf_counter()
        
        points_generated = 0
        points_list = []
        batch_size = min(num_points * 2, int(1e6))  # Генерируем с запасом
        
        while points_generated < num_points:
            # Генерируем случайные точки в кубе [-1, 1]^3
            x = rng.uniform(-1, 1, batch_size)
            y = rng.uniform(-1, 1, batch_size)
            z = rng.uniform(-1, 1, batch_size)
            
            # Вычисляем расстояние от центра
            r_squared = x**2 + y**2 + z**2
            
            # Оставляем только точки внутри единичной сферы
            mask = r_squared <= 1.0
            x_valid = x[mask]
            y_valid = y[mask]
            z_valid = z[mask]
            r_squared_valid = r_squared[mask]
            
            if len(x_valid) == 0:
                continue
            
            # Нормализуем на радиус сферы и применяем распределение по объему
            # r = R * (u)^(1/3), где u - равномерно распределенная величина
            u = rng.uniform(0, 1, len(x_valid))
            scale = radius * np.cbrt(u) / np.sqrt(r_squared_valid)
            
            x_scaled = x_valid * scale
            y_scaled = y_valid * scale
            z_scaled = z_valid * scale
            
            # Добавляем точки в список
            points_batch = np.column_stack([x_scaled, y_scaled, z_scaled])
            points_list.append(points_batch)
            points_generated += len(points_batch)
            
            if points_generated >= num_points:
                break
        
        # Объединяем все точки и берем ровно num_points
        result = np.vstack(points_list)[:num_points]
        
        # КРИТИЧЕСКИ ВАЖНО: Перемешиваем точки случайным образом, чтобы устранить любые артефакты порядка
        # Это гарантирует, что точки будут выглядеть случайно распределенными, даже если алгоритм
        # создает их в определенном порядке
        if len(result) > 0:
            # Используем случайную перестановку индексов с тем же генератором
            shuffle_indices = rng.permutation(len(result))
            result = result[shuffle_indices]
        
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] Generated {num_points} uniform points in {(time.perf_counter() - t0)*1000:.2f} ms")
        
        total_time = time.perf_counter() - start_time
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] generate_points_in_3d_sphere: completed in {total_time*1000:.2f} ms")
        
        return result
    
    def generate_points_spiral_in_ball(
        self,
        num_points: int,
        radius: float,
    ) -> np.ndarray:
        """
        Плоская спираль в плоскости XY (z = 0), совпадающая с проекцией «на экран» в рендерере.
        Цилиндрический радиус ρ ∝ u^(1/3), азимут φ = 2π·u — ровно один оборот.
        """
        max_points = int(1e6)
        if num_points > max_points:
            print(f"Warning: Limiting num_points from {num_points} to {max_points}")
            num_points = max_points
        
        num_points = int(num_points)
        if num_points <= 0:
            return np.array([], dtype=np.float64).reshape(0, 3)
        
        # u ∈ (0, 1), без ρ = 0, чтобы начальная материя не лежала точно в центре.
        u = (np.arange(num_points, dtype=np.float64) + 0.5) / num_points
        rho = radius * np.cbrt(u)
        phi = 2.0 * np.pi * u
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        z = np.zeros(num_points, dtype=np.float64)
        return np.column_stack([x, y, z])
    
    def generate_points_in_spherical_shell(self, num_points: int, inner_radius: float, 
                                          outer_radius: float, seed: int = None) -> np.ndarray:
        """
        Генерирует точки в сферическом кольце (между двумя радиусами).
        
        Args:
            num_points: Количество точек для генерации
            inner_radius: Внутренний радиус кольца (м)
            outer_radius: Внешний радиус кольца (м)
            seed: Seed для генератора случайных чисел (опционально)
        
        Returns:
            np.ndarray: Массив точек в 3D координатах (N x 3)
        """
        # Используем независимый генератор случайных чисел
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(random.randint(0, 2**31 - 1))
        
        max_points = int(1e6)
        if num_points > max_points:
            print(f"Warning: Limiting num_points from {num_points} to {max_points}")
            num_points = max_points
        
        num_points = int(num_points)
        if num_points <= 0:
            return np.array([], dtype=np.float64).reshape(0, 3)
        
        # Метод Marsaglia для кольца
        new_points_list = []
        points_needed = num_points
        
        r_inner_cubed = inner_radius**3
        r_outer_cubed = outer_radius**3
        
        while len(new_points_list) < points_needed:
            batch_size = min(int(points_needed * 2), max_points)
            
            # Генерируем случайные точки в кубе [-1, 1]^3
            x = rng.uniform(-1, 1, batch_size)
            y = rng.uniform(-1, 1, batch_size)
            z = rng.uniform(-1, 1, batch_size)
            
            # Вычисляем квадрат расстояния от центра
            r_squared = x**2 + y**2 + z**2
            
            # Оставляем только точки внутри единичной сферы
            mask = r_squared <= 1.0
            x_valid = x[mask]
            y_valid = y[mask]
            z_valid = z[mask]
            r_squared_valid = r_squared[mask]
            
            if len(x_valid) == 0:
                continue
            
            # Масштабируем точки в кольцо [inner_radius, outer_radius]
            # r^3 распределено равномерно между inner^3 и outer^3
            u = rng.uniform(0, 1, len(x_valid))
            r_cubed = r_inner_cubed + u * (r_outer_cubed - r_inner_cubed)
            r_target = np.cbrt(r_cubed)
            
            # Масштабируем направление на целевой радиус
            scale = r_target / np.sqrt(r_squared_valid)
            x_scaled = x_valid * scale
            y_scaled = y_valid * scale
            z_scaled = z_valid * scale
            
            # Добавляем точки в список
            points_batch = np.column_stack([x_scaled, y_scaled, z_scaled])
            new_points_list.extend(points_batch)
            
            if len(new_points_list) >= points_needed:
                break
        
        # Берем ровно нужное количество точек
        result = np.array(new_points_list[:points_needed])
        
        # КРИТИЧЕСКИ ВАЖНО: Перемешиваем точки случайным образом
        if len(result) > 0:
            shuffle_indices = rng.permutation(len(result))
            result = result[shuffle_indices]
        
        return result
    
    def initialize_matter_points(self, universe, cosmology):
        """
        Инициализировать точки материи в сопутствующих координатах.
        Создаются один раз внутри сферы радиусом, равным горизонту частиц.
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
        """
        if self.matter_points_initialized and self.matter_points.points_comoving is not None:
            return
        
        if getattr(config, 'DEBUG', False):
            print("[DEBUG] initialize_matter_points: starting initialization...")
        init_start = time.perf_counter()
        
        # Получаем текущий масштабный фактор
        scale_factor = cosmology.scale_factor
        
        # Получаем горизонт частиц при времени старта симуляции
        initial_time = get_initial_time_seconds()
        initial_years = getattr(config, 'INITIAL_TIME_YEARS', config.DT_YEARS)
        initial_scale_factor = calculate_scale_factor_at_time(initial_years)
        
        # Вычисляем горизонт частиц при времени старта
        temp_scale = cosmology.scale_factor
        cosmology.scale_factor = initial_scale_factor
        particle_horizon_initial_physical = cosmology.particle_horizon(initial_time)
        cosmology.scale_factor = temp_scale
        
        # Горизонт частиц в сопутствующих координатах
        particle_horizon_initial_comoving = (
            particle_horizon_initial_physical / initial_scale_factor 
            if initial_scale_factor > 0 else particle_horizon_initial_physical
        )
        
        # Количество точек задаётся явно в config.MATTER_NUM_POINTS,
        # а масса одной точки выводится из полной массы материи Вселенной.
        total_mass = PARTICLE_HORIZON_MASS_LIMIT_KG
        
        num_points = max(int(getattr(config, 'MATTER_NUM_POINTS', 0)), 1)
        mass_per_point = get_mass_per_point_kg()
        
        # Ограничиваем максимальное количество точек
        max_total_points = int(1e6)
        if num_points > max_total_points:
            print(f"Warning: Limiting num_points from {num_points} to {max_total_points}")
            num_points = max_total_points
        
        if getattr(config, 'DEBUG', False):
            print(f"[DEBUG] initialize_matter_points: num_points={num_points}")
            print(f"[DEBUG] initialize_matter_points: total_mass={total_mass:.2e} kg (maximum)")
            print(f"[DEBUG] initialize_matter_points: radius={CAUSAL_HORIZON_COMOVING_METERS:.2e} m ({CAUSAL_HORIZON_COMOVING_METERS/9.461e24:.1f} Gly)")
        
        # Генерируем точки в ПОЛНОМ ПРИЧИННОМ РАДИУСЕ (сопутствующий)
        mode = self.mode
        if mode is None:
            mode = getattr(config, "MATTER_INITIAL_DISTRIBUTION", "uniform")
        if isinstance(mode, str):
            mode = mode.strip().lower()
        
        if mode == "spiral":
            if getattr(config, 'DEBUG', False):
                print("[DEBUG] initialize_matter_points: spiral mode, plane XY (z=0), 1 turn")
            generated_points = self.generate_points_spiral_in_ball(
                num_points,
                CAUSAL_HORIZON_COMOVING_METERS,
            )
        elif mode != "uniform":
            print(
                f"Warning: unknown MATTER_INITIAL_DISTRIBUTION={mode!r}, "
                'using "uniform"'
            )
            mode = "uniform"
        
        if mode == "uniform":
            # Используем seed из конфига для детерминированности
            matter_seed = getattr(config, 'MATTER_SEED', 42)
            if matter_seed == 0:
                matter_seed = int((time.time() * 1000000 + random.randint(0, 1000000)) % 2147483647)
                
            generated_points = self.generate_points_in_3d_sphere(
                num_points,
                CAUSAL_HORIZON_COMOVING_METERS,
                seed=matter_seed,
            )
        
        # Устанавливаем точки
        self.matter_points.points_comoving = generated_points

        # ОПТИМИЗАЦИЯ: Вычисляем comoving расстояния один раз при создании
        # (через _update_comoving_distances для согласованного инкремента версии).
        self.matter_points._update_comoving_distances()
        
        # Инициализируем массы точек (до коллапса каждая точка имеет полную массу)
        self.matter_points.masses_per_point = np.full(len(generated_points), get_mass_per_point_kg(), dtype=np.float64)
        
        # Инициализируем скорости нулями (до коллапса точки неподвижны в сопутствующих координатах)
        self.matter_points.velocities_comoving = np.zeros((len(generated_points), 3), dtype=np.float64)

        # ОПТИМИЗАЦИЯ: синхронизируем кэши норм скоростей и версии массива масс
        # (используются MassCalculator для пропуска γ-коррекции и кэша m_eff).
        self.matter_points._recompute_velocity_norms()
        self.matter_points._bump_masses_version()
        
        r_emission_boundary = self._emission_boundary_radius(universe, cosmology)

        self.matter_points.init_laser_emitter_mask(
            len(generated_points),
            r_emission_boundary,
            cosmology.scale_factor
        )
        self.matter_points.clear_photons()
        
        self.matter_points_initialized = True
        self.current_num_points = len(self.matter_points.points_comoving)
        self.last_added_radius_comoving = particle_horizon_initial_comoving
        
        init_time = time.perf_counter() - init_start
        if getattr(config, 'DEBUG', False):
            print(f"Initialized {self.current_num_points} matter points in {init_time*1000:.2f} ms")
            print(f"[DEBUG] Max comoving distance: {np.max(self.matter_points.comoving_distances):.2e} m = {np.max(self.matter_points.comoving_distances)/9.461e24:.1f} Gly")
    
    def add_matter_points(self, universe, cosmology, num_new_points: int, radius_physical: float):
        """
        Добавить новые точки материи в существующий массив.
        Новые точки появляются в кольце между старым и новым радиусом.
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            num_new_points: Количество новых точек для добавления
            radius_physical: Физический радиус текущего горизонта частиц (м)
        """
        if num_new_points <= 0:
            return
        
        num_new_points = int(num_new_points)
        scale_factor = cosmology.scale_factor
        radius_comoving = radius_physical / scale_factor if scale_factor > 0 else radius_physical
        
        # Инициализируем last_added_radius_comoving, если нужно
        if not hasattr(self, 'last_added_radius_comoving') or self.last_added_radius_comoving <= 0:
            self.last_added_radius_comoving = radius_comoving * 0.9
        
        # Генерируем точки в кольце между старым и новым радиусом
        inner_radius = self.last_added_radius_comoving
        outer_radius = radius_comoving
        
        # Если inner_radius >= outer_radius, генерируем точки во всей сфере
        seed = int((universe.time * 1000000 + random.randint(0, 1000000)) % 2147483647)
        
        if inner_radius >= outer_radius:
            new_points = self.generate_points_in_3d_sphere(
                num_new_points, outer_radius, seed=seed
            )
        else:
            new_points = self.generate_points_in_spherical_shell(
                num_new_points, inner_radius, outer_radius, seed=seed
            )
        
        r_emission_boundary = self._emission_boundary_radius(universe, cosmology)

        scale_ratio = self._calculate_scale_ratio(universe, cosmology, radius_physical)
        self.matter_points.add_points(new_points, scale_factor, scale_ratio, r_emission_boundary)
        self.current_num_points = len(self.matter_points.points_comoving)
        self.last_added_radius_comoving = outer_radius
    
    def _calculate_scale_ratio(self, universe, cosmology, particle_horizon_physical: float) -> float:
        """
        Вычислить коэффициент роста горизонта частиц.
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            particle_horizon_physical: Текущий физический радиус горизонта частиц
        
        Returns:
            float: Коэффициент scale_ratio
        """
        if not hasattr(self, '_initial_horizon_comoving') or self._initial_horizon_comoving <= 0:
            initial_time = get_initial_time_seconds()
            _, _, particle_horizon_initial = calculate_initial_horizons(
                initial_time / SECONDS_PER_YEAR, cosmology=cosmology
            )
            particle_horizon_initial_physical = particle_horizon_initial * BILLION_LIGHT_YEARS_IN_METERS
            initial_scale_factor = cosmology.scale_factor
            if initial_scale_factor > 0:
                self._initial_horizon_comoving = particle_horizon_initial_physical / initial_scale_factor
            else:
                self._initial_horizon_comoving = particle_horizon_initial_physical
        
        scale_factor = cosmology.scale_factor
        
        if self._initial_horizon_comoving > 0 and scale_factor > 0:
            particle_horizon_comoving = particle_horizon_physical / scale_factor
            scale_ratio = particle_horizon_comoving / self._initial_horizon_comoving
        else:
            scale_ratio = 1.0
        
        return scale_ratio

    def _ltb_hubble_horizon(self, time_seconds: float, cosmology) -> float:
        """
        LTB-Λ Hubble radius for this matter simulation state.

        This is the outer apparent horizon:
            2G·M(<r)/(c²r) + Λr²/3 = 1

        It is the same definition used by MassCalculator for
        r_hubble_horizon_m. If the state is not ready, fall back to the
        homogeneous FLRW helper c/H(t).
        """
        scale_factor = float(getattr(cosmology, 'scale_factor', 0.0) or 0.0)
        if scale_factor <= 0.0:
            return cosmology.hubble_horizon(time_seconds)

        # ОПТИМИЗАЦИЯ: hubble и event LTB-горизонты считаются также в
        # MassCalculator.calculate_masses; результат там кладётся в
        # MatterPoints._cached_ltb_*. Если кэш-ключ совпадает — без
        # повторного solve_apparent_outer_horizon.
        mp = self.matter_points
        try:
            cache_key = mp.ltb_horizons_cache_key(time_seconds, scale_factor)
            if (
                mp._cached_ltb_horizons_key == cache_key
                and mp._cached_ltb_hubble_horizon_m is not None
                and mp._cached_ltb_hubble_horizon_m > 0.0
            ):
                return float(mp._cached_ltb_hubble_horizon_m)
        except Exception:
            cache_key = None

        try:
            matter_state = build_active_matter_state(mp, scale_factor)
            laser_state = mp.build_in_flight_laser_mass_state(scale_factor)
            M_bh = float(getattr(mp, 'accumulated_bh_mass', 0.0))
            lam = cosmological_constant_lambda()
            r_outer_upper = np.sqrt(3.0 / lam) * 1.001
            radius = solve_apparent_outer_horizon(
                M_bh, matter_state, laser_state, scale_factor,
                0.0, r_outer_upper, lam,
            )
            if np.isfinite(radius) and radius > 0.0:
                if cache_key is not None:
                    mp._cached_ltb_hubble_horizon_m = float(radius)
                    mp._cached_ltb_horizons_key = cache_key
                return float(radius)
        except Exception as exc:
            if config.DEBUG:
                print(f"[MatterSimulation] LTB Hubble fallback to FLRW c/H: {exc}")

        return cosmology.hubble_horizon(time_seconds)

    def _ltb_event_horizon(self, time_seconds: float, cosmology) -> float:
        """
        LTB event/null horizon for emitter selection.

        Uses the same null-separatrix helper as MassCalculator, so the
        "event" laser boundary is not the FLRW/SdS helper and is not forced
        to equal the local apparent/Hubble boundary.
        """
        scale_factor = float(getattr(cosmology, 'scale_factor', 0.0) or 0.0)
        if scale_factor <= 0.0:
            return self._ltb_hubble_horizon(time_seconds, cosmology)

        # ОПТИМИЗАЦИЯ: ключевое узкое место — compute_ltb_event_horizon
        # (~150 мс scipy.solve_ivp). Если в этом кадре MassCalculator уже
        # посчитал event horizon для того же состояния, переиспользуем его
        # из MatterPoints._cached_ltb_event_horizon_m.
        mp = self.matter_points
        try:
            cache_key = mp.ltb_horizons_cache_key(time_seconds, scale_factor)
            if (
                mp._cached_ltb_horizons_key == cache_key
                and mp._cached_ltb_event_horizon_m is not None
                and mp._cached_ltb_event_horizon_m > 0.0
            ):
                return float(mp._cached_ltb_event_horizon_m)
        except Exception:
            cache_key = None

        try:
            matter_state = build_active_matter_state(mp, scale_factor)
            laser_state = mp.build_in_flight_laser_mass_state(scale_factor)
            M_bh = float(getattr(mp, 'accumulated_bh_mass', 0.0))
            lam = cosmological_constant_lambda()
            r_hubble = self._ltb_hubble_horizon(time_seconds, cosmology)
            radius = compute_ltb_event_horizon(
                time_seconds, M_bh, matter_state, laser_state,
                scale_factor, lam, r_hubble,
            )
            if np.isfinite(radius) and radius > 0.0:
                if cache_key is not None:
                    mp._cached_ltb_event_horizon_m = float(radius)
                    mp._cached_ltb_horizons_key = cache_key
                return float(radius)
        except Exception as exc:
            if config.DEBUG:
                print(f"[MatterSimulation] LTB Event fallback to LTB Hubble: {exc}")

        return self._ltb_hubble_horizon(time_seconds, cosmology)

    def _emission_boundary_radius(self, universe, cosmology) -> float:
        """Физический радиус границы эмиттеров для текущего config.EMISSION_BOUNDARY."""
        boundary_type = getattr(config, 'EMISSION_BOUNDARY', 'event')
        if boundary_type == 'hubble':
            return self._ltb_hubble_horizon(universe.time, cosmology)
        return self._ltb_event_horizon(universe.time, cosmology)
    
    def update_collapse(self, universe, cosmology, paused: bool = False, r_black_hole: float = None,
                       dt_step_signed: float | None = None):
        """
        Обновить состояние коллапса материи.

        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            paused: Флаг паузы симуляции
            r_black_hole: Радиус горизонта черной дыры в метрах (опционально)
            dt_step_signed: если задан (ручной шаг на паузе), заменяет get_dt(); отрицательный —
                только откат лазерных фотонов по конформному времени (без шага материи).
        """
        # Отмотка времени назад (стрелка влево на паузе): не зависит от t ≥ t_collapse.
        if (
            not paused
            and dt_step_signed is not None
            and dt_step_signed < 0
            and self.matter_points.points_comoving is not None
        ):
            scale_factor = cosmology.scale_factor
            particle_horizon_physical = cosmology.particle_horizon(universe.time)
            scale_ratio = self._calculate_scale_ratio(universe, cosmology, particle_horizon_physical)
            r_emission_boundary = self._emission_boundary_radius(universe, cosmology)
                
            self.matter_points.update_positions_and_velocities(
                dt_step_signed,
                scale_factor,
                scale_ratio,
                r_black_hole,
                universe_time_seconds=universe.time,
                r_emission_boundary=r_emission_boundary,
            )
            self.last_collapse_time = universe.time
            return

        # В новой модели коллапс = постоянное ускорение к центру (без разделения точек)
        if universe.time >= get_collapse_start_time_seconds():
            scale_factor = cosmology.scale_factor
            particle_horizon_physical = cosmology.particle_horizon(universe.time)
            scale_ratio = self._calculate_scale_ratio(universe, cosmology, particle_horizon_physical)
            
            if not self.collapse_started:
                self.collapse_started = True
                self.last_collapse_time = universe.time
                if config.DEBUG:
                    print(f"Collapse started at time {universe.time / 3.154e16:.2f} billion years")
            
            if not paused and self.matter_points.points_comoving is not None:
                dt = dt_step_signed if dt_step_signed is not None else get_dt()
                
                r_emission_boundary = self._emission_boundary_radius(universe, cosmology)
                    
                self.matter_points.update_positions_and_velocities(
                    dt, scale_factor, scale_ratio, r_black_hole,
                    universe_time_seconds=universe.time,
                    r_emission_boundary=r_emission_boundary,
                )

                self._capture_inside_apparent_inner_horizon(cosmology, scale_factor)
            
            self.last_collapse_time = universe.time

    # Максимум проходов AH-захвата за один шаг времени. Один проход
    # достаточен в стационаре; второй ловит случай, когда захват ближайшей
    # оболочки за этот шаг подрастил M_BH и r_classical настолько, что
    # соседняя оболочка тоже оказалась внутри нового AH.
    _AH_CAPTURE_MAX_ITER = 2
    # Ранний выход: прирост массы за проход мал относительно текущей M_BH.
    _AH_CAPTURE_REL_DM = 1e-4

    def _capture_inside_apparent_inner_horizon(self, cosmology, scale_factor: float) -> None:
        """
        Чистый LTB-Λ захват trapped-зоны вокруг ЦЧД. Всё, что физически лежит
        внутри inner apparent horizon, переходит в массу ЦЧД.

        Алгоритм inner AH (см. physics/apparent_horizon.solve_apparent_inner_horizon):
        ищем ПЕРВЫЙ переход trapped → untrapped, идя наружу от r → 0+,
        для g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1 = 0 c ПОЛНОЙ M(<r) (без
        Birkhoff-вычета): M(<r) = M_BH + Σ оболочек + Σ фотонов в полёте.

        Один проход обычно достаточен; до _AH_CAPTURE_MAX_ITER проходов
        ловят случай, когда захват одной оболочки расширил r_AH и
        накрыл следующую за один шаг времени.
        """
        a_now = float(scale_factor)
        if a_now <= 0.0:
            return
        mp = self.matter_points

        capture_horizons_seconds = 0.0
        for _ in range(self._AH_CAPTURE_MAX_ITER):
            M_bh = float(getattr(mp, 'accumulated_bh_mass', 0.0))

            matter_state = build_active_matter_state(mp, a_now)
            laser_state = mp.build_in_flight_laser_mass_state(a_now)

            lam = cosmological_constant_lambda()
            r_outer_upper = np.sqrt(3.0 / lam) * 1.001
            t_h0 = time.perf_counter()
            r_outer_ltb = solve_apparent_outer_horizon(
                M_bh, matter_state, laser_state, a_now,
                0.0, r_outer_upper, lam,
            )

            # В чистом LTB захват ограничен найденным outer apparent horizon,
            # а не вакуумным SdS/Nariai радиусом.
            r_lo = 0.0
            r_hi = float(r_outer_ltb)
            if not np.isfinite(r_hi) or r_hi <= 0.0:
                self._last_capture_ltb_horizons_seconds = capture_horizons_seconds
                return
            r_AH = solve_apparent_inner_horizon(
                M_bh, matter_state, laser_state, a_now,
                r_lo, r_hi, lam,
            )
            capture_horizons_seconds += time.perf_counter() - t_h0
            if r_AH <= 0.0 or not np.isfinite(r_AH):
                self._last_capture_ltb_horizons_seconds = capture_horizons_seconds
                return

            dM = mp.capture_inside_apparent_horizon(r_AH, a_now)
            if dM <= 0.0:
                self._last_capture_ltb_horizons_seconds = capture_horizons_seconds
                return
            ref = max(M_bh + dM, 1.0)
            if dM <= self._AH_CAPTURE_REL_DM * ref:
                self._last_capture_ltb_horizons_seconds = capture_horizons_seconds
                return

        self._last_capture_ltb_horizons_seconds = capture_horizons_seconds

    def get_physical_points(self, cosmology) -> np.ndarray:
        """
        Получить точки в физических координатах.
        
        Args:
            cosmology: Объект космологии
        
        Returns:
            np.ndarray: Массив точек в физических координатах (N x 3)
        """
        if self.matter_points.points_comoving is None:
            return np.array([]).reshape(0, 3)
        
        scale_factor = cosmology.scale_factor
        return self.matter_points.points_comoving * scale_factor
    
    def get_physical_points_and_distances(self, universe, cosmology, particle_horizon_physical):
        """
        Получить физические координаты точек и расстояния от центра.
        Физические расчеты без кэширования (кэширование - ответственность renderer).
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            particle_horizon_physical: Текущий физический радиус горизонта частиц
        
        Returns:
            tuple: (physical_points, distances_from_center, scale_ratio)
        """
        if self.matter_points.points_comoving is None or len(self.matter_points.points_comoving) == 0:
            return None, None, None
        
        scale_factor = cosmology.scale_factor
        scale_ratio = self._calculate_scale_ratio(universe, cosmology, particle_horizon_physical)
        
        # Преобразуем из сопутствующих координат в физические
        physical_points = self.matter_points.points_comoving * scale_factor
        
        # ОПТИМИЗАЦИЯ: Используем предвычисленные comoving_distances вместо пересчета sqrt
        # physical_distance = comoving_distance * scale_factor
        if self.matter_points.comoving_distances is not None:
            distances_from_center = self.matter_points.comoving_distances * scale_factor
        else:
            # Fallback: вычисляем если comoving_distances не инициализированы
            distances_squared = np.sum(physical_points**2, axis=1)
            distances_from_center = np.sqrt(distances_squared)
        
        # Отладочный вывод после коллапса
        if self.collapse_started and config.DEBUG:
            if not hasattr(self, '_debug_physical_points_printed'):
                num_points = len(physical_points)
                num_zero = np.sum(distances_from_center < 1e10)  # Точки в центре
                num_nonzero = num_points - num_zero
                min_dist = np.min(distances_from_center) if num_points > 0 else 0
                max_dist = np.max(distances_from_center) if num_points > 0 else 0
                print(f"[DEBUG Physical] После коллапса: total_points={num_points}, "
                      f"points_in_center={num_zero}, points_outside_center={num_nonzero}, "
                      f"min_dist={min_dist:.2e} м ({min_dist/9.461e24:.4f} млрд св. лет), "
                      f"max_dist={max_dist:.2e} м ({max_dist/9.461e24:.4f} млрд св. лет)")
                self._debug_physical_points_printed = True
        
        return physical_points, distances_from_center, scale_ratio
