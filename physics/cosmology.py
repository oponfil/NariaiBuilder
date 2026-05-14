"""
Lambda CDM космологические параметры и эффекты
"""
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

import config
from config import DEBUG
from physics.nariai import schwarzschild_de_sitter_horizons
from utils.config_utils import get_mass_per_point_kg
from utils.constants import (
    BILLION_LIGHT_YEARS_IN_METERS,
    DE_SITTER_HORIZON_BILLION_LY,
    EVENT_HORIZON_Z0_BILLION_LY,
    G,
    H0_s,
    HUBBLE_HORIZON_Z0_BILLION_LY,
    OMEGA_B,
    OMEGA_DM,
    OMEGA_LAMBDA,
    PARTICLE_HORIZON_Z0_BILLION_LY,
    RHO_CRIT,
    SECONDS_PER_YEAR,
    c,
)
from utils.cosmology_utils import calculate_scale_factor_at_time


_SCALE_FACTOR_CACHE_TOLERANCE_YEARS = 1.0e6
_SCALE_FACTOR_CACHE_TOLERANCE_SECONDS = _SCALE_FACTOR_CACHE_TOLERANCE_YEARS * 3.154e7

# Сообщения о кэше горизонтов — один раз за процесс (избегаем спама при множественных LambdaCDM()).
_logged_particle_horizon_cache_missing = False
_logged_event_horizon_cache_missing = False
_logged_particle_horizon_load_failure = False
_logged_event_horizon_load_failure = False
_logged_particle_horizon_loaded_ok = False
_logged_event_horizon_loaded_ok = False
_logged_horizon_precompute_summary = False


class LambdaCDM:
    """Lambda CDM космологическая модель"""
    
    def __init__(self):
        self.omega_lambda = OMEGA_LAMBDA
        self.omega_dm = OMEGA_DM
        self.omega_b = OMEGA_B
        self.h0 = H0_s
        self.rho_crit = RHO_CRIT
        
        # Масштабный фактор (a(t))
        # Начальное значение вычисляется на основе INITIAL_TIME_YEARS из config
        # Используем прямое вычисление, чтобы избежать циклических зависимостей
        try:
            initial_time = getattr(config, 'INITIAL_TIME_YEARS', config.DT_YEARS)
            self.scale_factor = calculate_scale_factor_at_time(initial_time)
        except (ImportError, AttributeError):
            # Если есть проблема с импортом, используем значение по умолчанию
            # Оно будет пересчитано позже
            self.scale_factor = 1e-10
        self.scale_factor_velocity = 0.0  # Начальная скорость изменения (будет вычислена)
        
        # История для численного интегрирования горизонтов
        self.time_history = []  # Список времен для накопления интеграла
        self.scale_factor_history = []  # Список масштабных факторов
        
        # ОПТИМИЗАЦИЯ: Загружаем предвычисленные горизонты из файла
        self._particle_interpolator = None
        self._event_interpolator = None
        self._particle_interp_used = 0
        self._event_interp_used = 0
        # ОПТИМИЗАЦИЯ: интерполятор a(t) из того же JSON-кэша. Подменяет
        # scipy.solve_ivp в _get_scale_factor_for_time для произвольных t
        # (например, t_future у compute_ltb_event_horizon). См.
        # _load_precomputed_horizons.
        self._scale_factor_interpolator = None
        self._horizon_time_min = None
        self._horizon_time_max = None
        self._load_precomputed_horizons()

        # ОПТИМИЗАЦИЯ: Кэш для scale_factor по времени
        self._scale_factor_cache_time = -1.0
        self._scale_factor_cache_value = 1.0

        # Кэш для honest LTB-Λ event horizon (грубая гранулярность по (t, M_BH)).
        self._eh_ltb_cache_key = None
        self._eh_ltb_cache_value = 0.0
    
    def _get_scale_factor_for_time(self, time: float) -> float:
        """Получить a(t) с кэшированием.

        Иерархия путей:
          1. Точечный кэш последнего вызова (O(1), пока |t − t_cached| < 1 Myr).
          2. Предвычисленный интерполятор a(t) из `data/event_horizon_cache.json`
             — np.interp на 150k-узловой сетке (~1 мкс/вызов). Та же «обычная»
             ΛCDM-эволюция, что и `calculate_scale_factor_at_time`, только
             посчитанная один раз скриптом precompute_horizons на сетке 1 Myr.
             Относительная точность ~1e-9, чего c запасом достаточно для всех
             потребителей (в т.ч. `compute_ltb_event_horizon` с rtol=1e-4).
          3. Численный `calculate_scale_factor_at_time` (scipy.solve_ivp от t=0)
             — резерв, если JSON-кэш не загружен или t вне его диапазона.
        """
        if abs(time - self._scale_factor_cache_time) < _SCALE_FACTOR_CACHE_TOLERANCE_SECONDS:
            return self._scale_factor_cache_value

        if (
            self._scale_factor_interpolator is not None
            and self._horizon_time_min is not None
            and self._horizon_time_max is not None
            and self._horizon_time_min <= time <= self._horizon_time_max
        ):
            a = float(self._scale_factor_interpolator(time))
            self._scale_factor_cache_time = time
            self._scale_factor_cache_value = a
            return a

        BILLION_YEARS_IN_SECONDS = 3.154e16
        time_years = time / BILLION_YEARS_IN_SECONDS * 1e9
        a = calculate_scale_factor_at_time(time_years)

        self._scale_factor_cache_time = time
        self._scale_factor_cache_value = a
        return a
    
    def _load_precomputed_horizons(self):
        """Загрузить предвычисленные горизонты из отдельных файлов"""
        global _logged_particle_horizon_cache_missing
        global _logged_event_horizon_cache_missing
        global _logged_particle_horizon_load_failure
        global _logged_event_horizon_load_failure
        global _logged_particle_horizon_loaded_ok
        global _logged_event_horizon_loaded_ok
        global _logged_horizon_precompute_summary
        load_start = time.perf_counter()
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        # Используем фиксированные имена файлов без суффикса
        particle_cache_file = os.path.join(data_dir, 'particle_horizon_cache.json')
        event_cache_file = os.path.join(data_dir, 'event_horizon_cache.json')
        
        # Загружаем горизонт частиц
        if os.path.exists(particle_cache_file):
            try:
                json_start = time.perf_counter()
                with open(particle_cache_file, 'r') as f:
                    particle_data = json.load(f)
                json_time_particle = time.perf_counter() - json_start
                
                interp_start = time.perf_counter()
                times_seconds = np.array(particle_data['times_seconds'])
                particle_horizons = np.array(particle_data['horizons'])
                
                # ВАЖНО: Используем fill_value='extrapolate' для экстраполяции за пределами данных
                # вместо использования последнего значения, чтобы избежать стабильных значений
                self._particle_interpolator = interp1d(times_seconds, particle_horizons, 
                                                       kind='linear', bounds_error=False, 
                                                       fill_value='extrapolate')
                interp_time_particle = time.perf_counter() - interp_start
                
                # Сохраняем границы для проверки
                self._horizon_time_min = times_seconds[0]
                self._horizon_time_max = times_seconds[-1]
                
                if not _logged_particle_horizon_loaded_ok:
                    print(f"Loaded particle horizon from {particle_cache_file}")
                    print(f"  JSON load: {json_time_particle*1000:.2f} ms, Interpolation setup: {interp_time_particle*1000:.2f} ms")
                    print(f"  Time range: {particle_data['times_years'][0]/1e9:.1f} - {particle_data['times_years'][-1]/1e9:.1f} billion years")
                    _logged_particle_horizon_loaded_ok = True
            except Exception as e:
                if not _logged_particle_horizon_load_failure:
                    print(f"Warning: Could not load particle horizon: {e}")
                    print("Will use numerical integration for particle horizon.")
                    _logged_particle_horizon_load_failure = True
        else:
            if not _logged_particle_horizon_cache_missing:
                print(f"Warning: Particle horizon cache file not found: {particle_cache_file}")
                print("Run scripts/precompute_horizons.py to generate it.")
                _logged_particle_horizon_cache_missing = True
        
        # Загружаем горизонт событий
        if os.path.exists(event_cache_file):
            try:
                json_start = time.perf_counter()
                with open(event_cache_file, 'r') as f:
                    event_data = json.load(f)
                json_time_event = time.perf_counter() - json_start
                
                interp_start = time.perf_counter()
                times_seconds_event = np.array(event_data['times_seconds'])
                event_horizons = np.array(event_data['horizons'])
                
                self._event_interpolator = interp1d(times_seconds_event, event_horizons,
                                                     kind='linear', bounds_error=False,
                                                     fill_value=(event_horizons[0], event_horizons[-1]))

                # ОПТИМИЗАЦИЯ: тот же JSON уже хранит `scale_factors` (см.
                # scripts/precompute_horizons.py: scale_factors.append(
                # calculate_scale_factor_at_time(...))). Поднимаем их как
                # интерполятор a(t) — нужен для compute_ltb_event_horizon,
                # чтобы не интегрировать ΛCDM-фон scipy.solve_ivp в каждом
                # кадре (~130 мс). Это та же «обычная» ΛCDM-эволюция, что
                # и calculate_scale_factor_at_time, только посчитанная один
                # раз скриптом предвычисления.
                if 'scale_factors' in event_data:
                    scale_factors_arr = np.array(event_data['scale_factors'])
                    if len(scale_factors_arr) == len(times_seconds_event):
                        self._scale_factor_interpolator = interp1d(
                            times_seconds_event,
                            scale_factors_arr,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(float(scale_factors_arr[0]),
                                        float(scale_factors_arr[-1])),
                        )
                        self._horizon_time_min = float(times_seconds_event[0])
                        self._horizon_time_max = float(times_seconds_event[-1])
                interp_time_event = time.perf_counter() - interp_start
                
                if not _logged_event_horizon_loaded_ok:
                    print(f"Loaded event horizon from {event_cache_file}")
                    print(f"  JSON load: {json_time_event*1000:.2f} ms, Interpolation setup: {interp_time_event*1000:.2f} ms")
                    print(f"  Time range: {event_data['times_years'][0]/1e9:.1f} - {event_data['times_years'][-1]/1e9:.1f} billion years")
                    _logged_event_horizon_loaded_ok = True
            except Exception as e:
                if not _logged_event_horizon_load_failure:
                    print(f"Warning: Could not load event horizon: {e}")
                    print("Will use numerical integration for event horizon.")
                    _logged_event_horizon_load_failure = True
        else:
            if not _logged_event_horizon_cache_missing:
                print(f"Warning: Event horizon cache file not found: {event_cache_file}")
                print("Run scripts/precompute_horizons.py to generate it.")
                _logged_event_horizon_cache_missing = True
        
        total_time = time.perf_counter() - load_start
        if not _logged_horizon_precompute_summary:
            print(f"  Total load time: {total_time*1000:.2f} ms")
            print(f"  Interpolators created: particle={self._particle_interpolator is not None}, event={self._event_interpolator is not None}")
            _logged_horizon_precompute_summary = True
        self._particle_integral_accumulated = 0.0  # Накопленный интеграл для горизонта частиц
        self._last_integration_time = 0.0
    
    def hubble_parameter(self, time: float) -> float:
        """
        Параметр Хаббла H(t)
        Для плоской Вселенной: H² = H₀²(Ωₘ/a³ + Ω_Λ)
        где Ωₘ = Ω_DM + Ω_B (темная материя + барионная материя)
        
        Args:
            time: время в секундах
            
        Returns:
            float: параметр Хаббла H(t) в единицах 1/с
        """
        # ОПТИМИЗАЦИЯ: Используем кэшированный scale_factor
        a = self._get_scale_factor_for_time(time)
        
        if a <= 0:
            a = 1e-10
        
        # Плотность материи убывает как a^-3
        # Включаем и темную материю, и барионную материю
        omega_m = self.omega_dm + self.omega_b
        # Темная энергия постоянна
        h_squared = self.h0**2 * (omega_m / (a**3) + self.omega_lambda)
        return np.sqrt(max(0, h_squared))
    
    def dark_energy_pressure(self) -> float:
        """
        Давление темной энергии (отрицательное)
        p_Λ = -ρ_Λ * c²
        """
        rho_lambda = self.omega_lambda * self.rho_crit
        return -rho_lambda * c**2
    
    def dark_energy_density(self) -> float:
        """Плотность темной энергии"""
        return self.omega_lambda * self.rho_crit
    
    def expansion_acceleration(self, time: float) -> float:
        """
        Ускорение расширения Вселенной
        ä/a = -4πG/3 * (ρ + 3p/c²)
        """
        a = self._get_scale_factor_for_time(time)
        if a <= 0:
            a = 1e-10
        
        # Плотность материи
        rho_m = (self.omega_dm + self.omega_b) * self.rho_crit / (a**3)
        
        # Плотность и давление темной энергии
        rho_lambda = self.omega_lambda * self.rho_crit
        p_lambda = self.dark_energy_pressure()
        
        # Ускорение расширения
        acceleration = -4 * np.pi * G / 3 * (rho_m + rho_lambda + 3 * p_lambda / (c**2))
        return acceleration
    
    def update_scale_factor(self, dt: float):
        """
        Обновить масштабный фактор с учетом расширения (уравнение Фридмана)
        Уравнение: da/dt = a * H(t), где H(t) = H₀ * sqrt(Ω_m/a³ + Ω_Λ)
        Используем метод Рунге-Кутты 4-го порядка для точности
        """
        a = self.scale_factor
        if a <= 0:
            a = 1e-10
        
        # Вспомогательная функция для вычисления da/dt
        def da_dt(a_val):
            if a_val <= 0:
                a_val = 1e-10
            omega_m = self.omega_dm + self.omega_b
            h = self.h0 * np.sqrt(omega_m / (a_val**3) + self.omega_lambda)
            return a_val * h
        
        # k1
        k1 = da_dt(a)
        
        # k2
        a_temp = a + 0.5 * dt * k1
        k2 = da_dt(a_temp)
        
        # k3
        a_temp = a + 0.5 * dt * k2
        k3 = da_dt(a_temp)
        
        # k4
        a_temp = a + dt * k3
        k4 = da_dt(a_temp)
        
        # Сохраняем старое значение для накопления интеграла
        old_scale_factor = self.scale_factor
        
        # Обновление по методу Рунге-Кутты 4-го порядка
        self.scale_factor += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.scale_factor_velocity = da_dt(self.scale_factor)
        
        # Предотвращаем слишком маленький масштабный фактор
        if self.scale_factor < 1e-5:
            self.scale_factor = 1e-5
        
        # Накопление интеграла для горизонта частиц (оптимизация)
        # Добавляем вклад от последнего шага: ∫[t_old to t_new] dt'/a(t')
        # Используем трапециевидное правило для накопления
        if not hasattr(self, '_particle_integral_accumulated'):
            self._particle_integral_accumulated = 0.0
        
        if old_scale_factor > 0 and self.scale_factor > 0:
            # Приближение: среднее 1/a на интервале (трапециевидное правило)
            avg_inv_a = 0.5 * (1.0/old_scale_factor + 1.0/self.scale_factor)
            self._particle_integral_accumulated += dt * avg_inv_a
        
        # ВАЖНО: НЕ обновляем здесь _scale_factor_cache_(time|value). Кэш — это
        # «последний запрошенный t → a(t)», и compute_ltb_event_horizon забивает
        # его произвольными t (например, t_future = t_now + 100 Гyr). Слепое
        # «cache_time += dt; cache_value = self.scale_factor» создавало
        # некорректную пару (≈t_future, a_now) — следующий вызов
        # _get_scale_factor_for_time(t_future_new) попадал в этот фейковый хит
        # и возвращал a_now вместо a_future. На космологическом event horizon
        # это вызывало нефизичный «прыжок в полтора раза» сразу после старта
        # лазера. Инвалидируем кэш — все обращения уйдут в interp1d (~1 мкс).
        self._scale_factor_cache_time = -1.0
    
    def comoving_to_physical(self, comoving_position: np.ndarray) -> np.ndarray:
        """Преобразовать сопутствующие координаты в физические"""
        return comoving_position * self.scale_factor
    
    def physical_to_comoving(self, physical_position: np.ndarray) -> np.ndarray:
        """Преобразовать физические координаты в сопутствующие"""
        if self.scale_factor > 0:
            return physical_position / self.scale_factor
        return physical_position
    
    def hubble_horizon(self, time: float, black_hole_mass_kg: float = None) -> float:
        """
        FLRW helper для радиуса Хаббла: r_H = c / H(t).

        Это кинематическая величина (расстояние, на котором скорость
        космологического разлёта равна c) для однородного FLRW-фона.
        Сценарный LTB Hubble radius с центральной массой и дискретной
        материей считается в MassCalculator как внешний apparent horizon:
        2G·M(<r)/(c²r) + Λr²/3 = 1.

        Параметр black_hole_mass_kg оставлен только для обратной
        совместимости сигнатуры.
        """
        h = self.hubble_parameter(time)
        return c / h
    
    def de_sitter_horizon(self, black_hole_mass_kg: float = 0.0) -> float:
        """
        Горизонт де Ситтера в метрике Шварцшильда-де Ситтера (SdS).
        
        Учитывает влияние центральной черной дыры на космологический горизонт.
        В метрике SdS внешний горизонт (космологический) уменьшается по мере роста массы ЧД.
        Когда масса ЧД достигает массы Нарайи, горизонты совпадают.

        При M=0 — чистый de Sitter: r = c/H_Λ. В визуализации пунктир
        «de Sitter (empty universe)» всегда использует это значение
        (вызов de_sitter_horizon(0.0)), независимо от массы ЦЧД.
        
        Args:
            black_hole_mass_kg: масса центральной черной дыры (кг). 
                               По умолчанию 0.0 (классический горизонт де Ситтера).
        
        Returns:
            float: внешний (космологический) горизонт в метрах.
                  При M=0: r_dS = c / H_Λ = √(3/Λ) (классический горизонт де Ситтера).
                  При M=M_N: r = r_N (горизонты совпадают).
                  При M>M_N: r = r_N (критический случай).
        """
        # Если масса не указана или равна 0, возвращаем классический горизонт де Ситтера
        if black_hole_mass_kg is None or black_hole_mass_kg <= 0.0:
            h_lambda = self.h0 * np.sqrt(self.omega_lambda)
            if h_lambda > 0:
                return c / h_lambda
            return float('inf')
        
        try:
            _, r_outer = schwarzschild_de_sitter_horizons(black_hole_mass_kg)
            return r_outer  # внешний (космологический) горизонт SdS
        except Exception as e:
            if config.DEBUG:
                print(f"[de_sitter_horizon] fallback to classical: {e}")
            h_lambda = self.h0 * np.sqrt(self.omega_lambda)
            if h_lambda > 0:
                return c / h_lambda
            return float('inf')
    
    def scale_factor_at_time(self, target_time: float, reference_time: float, reference_scale: float) -> float:
        """
        Вычислить масштабный фактор a(t) в заданный момент времени
        Использует аналитическое решение для Lambda CDM когда возможно, иначе численное
        
        Args:
            target_time: Время, для которого нужно вычислить a
            reference_time: Известное время
            reference_scale: Известный масштабный фактор в reference_time
        """
        if abs(target_time - reference_time) < 1e-10:
            return reference_scale
        
        omega_m = self.omega_dm + self.omega_b
        
        # Для будущего (target_time > reference_time) используем более эффективный метод
        if target_time > reference_time:
            # Используем решение ОДУ один раз для всего интервала с dense_output
            def da_dt(t, a_val):
                """Уравнение Фридмана: da/dt = a * H(t)"""
                if a_val <= 0:
                    a_val = 1e-10
                h = self.h0 * np.sqrt(omega_m / (a_val**3) + self.omega_lambda)
                return a_val * h
            
            try:
                # Решаем один раз с dense_output для интерполяции
                result = integrate.solve_ivp(da_dt, [reference_time, target_time], [reference_scale], 
                                             method='RK45', dense_output=True, 
                                             rtol=1e-4, atol=1e-7)  # Немного снижена точность для скорости
                
                if result.success:
                    return float(result.sol(target_time)[0])
            except:
                pass
        
        # Для прошлого или если не удалось - используем численное интегрирование
        def da_dt(t, a_val):
            if a_val <= 0:
                a_val = 1e-10
            h = self.h0 * np.sqrt(omega_m / (a_val**3) + self.omega_lambda)
            return a_val * h
        
        t_span = [reference_time, target_time] if target_time > reference_time else [target_time, reference_time]
        try:
            result = integrate.solve_ivp(da_dt, t_span, [reference_scale], 
                                         method='RK45', dense_output=True, 
                                         rtol=1e-4, atol=1e-7)
            
            if result.success:
                return float(result.y[0, -1])
        except:
            pass
        
        # Если интегрирование не удалось, используем приближение
        return reference_scale
    
    def particle_horizon(self, time: float) -> float:
        """
        Горизонт частиц: максимальное расстояние, которое свет мог пройти
        с момента Большого взрыва до времени t
        Для плоской Вселенной: r_p = c * a(t) * ∫[0 to t] dt'/a(t')
        
        ОПТИМИЗАЦИЯ: Использует предвычисленные значения с интерполяцией для ускорения.
        """
        if time <= 0:
            return 0.0
        
        # ОПТИМИЗАЦИЯ: Используем предвычисленные значения, если доступны
        # ПРИОРИТЕТ: Всегда используем интерполяцию, если она доступна
        # ВАЖНО: Данные в файле исправлены, поэтому интерполяция должна работать правильно
        # ВАЖНО: Используем экстраполяцию вместо численного интегрирования для избежания зависаний
        if self._particle_interpolator is not None:
            try:
                # Используем интерполяцию с экстраполяцией - это быстро и не зависает!
                result = float(self._particle_interpolator(time))
                # Проверяем только, что результат положительный
                if result >= 0:
                    # Используем интерполяцию - это быстро!
                    self._particle_interp_used += 1
                    return result
            except Exception as e:
                # ВАЖНО: НЕ используем численное интегрирование, чтобы избежать зависаний
                if config.DEBUG and not hasattr(self, '_particle_interp_fallback_printed'):
                    print(f"ERROR: particle_horizon interpolation failed: {e}")
                    self._particle_interp_fallback_printed = True
                return 0.0
        
        # ВАЖНО: Если интерполятор не загружен, НЕ используем численное интегрирование
        # чтобы избежать зависаний.
        # Если нет интерполятора, возвращаем 0
        # (численное интегрирование может зависать на больших временах, поэтому отключено)
        return 0.0
    
    def event_horizon_radius(self, mass: float) -> float:
        """
        Горизонт событий (радиус Шварцшильда) для объекта с массой M
        r_s = 2GM / c²
        """
        if mass <= 0:
            return 0.0
        return 2 * G * mass / (c ** 2)
    
    def cosmological_event_horizon(self, time: float, black_hole_mass_kg: float = None) -> float:
        """
        Космологический event horizon в LTB-Λ — собственная радиальная
        дальность, от которой свет, излучённый в момент `time`, асимптотически
        достигнет центрального наблюдателя при t → ∞.

        Считается ЧЕСТНО, интегрированием радиального нулевого геодезика в
        LTB-Λ с центральной ЦЧД и FRW-фоном материи. Радиальный inward null
        в LTB удовлетворяет:

            dr/dt = R_dot(t, r) − c,
            R_dot(t, r)² = 2G·M(<r, t)/r + Λc²r²/3,
            M(<r, t) = M_BH + ρ_m(t)·(4/3)π·r³,
            ρ_m(t) = ρ_m,0 / a(t)³.

        Event horizon = sup r₀ такой, что фотон, испущенный в (t, r₀) внутрь,
        достигает центра асимптотически. Находим бисекцией по r₀, для каждого
        кандидата интегрируем траекторию на ~100 Gyr вперёд.

        При M_BH = 0 интеграл вырождается до стандартного FLRW
        r_eh = c·a(t)·∫dt'/a(t'). При M_BH → M_крит (где inner и outer
        apparent horizons сливаются) event horizon стягивается к радиусу
        слияния (≈ r_N для асимптотической SdS).

        Параметр `black_hole_mass_kg`: None или 0 → FLRW fast-path через
        предрассчитанный интерполятор. Любое M_BH > 0 → честная LTB-Λ
        интеграция (с кэшем по (t, M_BH) на грубой сетке).
        """
        if time <= 0:
            return 0.0

        M_bh = float(black_hole_mass_kg) if black_hole_mass_kg is not None else 0.0
        if M_bh <= 0.0:
            return self._cosmological_event_horizon_flrw(time)

        # Кэш на грубой сетке: 100 Myr по t, ~0.5% M_N по массе.
        seconds_per_100myr = 1.0e8 * 365.25 * 24 * 3600
        mass_grain = 2.0e50  # ~0.5% от M_N ≈ 4e52 kg
        cache_key = (
            int(round(time / seconds_per_100myr)),
            int(round(M_bh / mass_grain)),
        )
        if cache_key == self._eh_ltb_cache_key:
            return self._eh_ltb_cache_value

        result = self._compute_honest_event_horizon_ltb(time, M_bh)
        self._eh_ltb_cache_key = cache_key
        self._eh_ltb_cache_value = result
        return result

    def _cosmological_event_horizon_flrw(self, time: float) -> float:
        """FLRW event horizon: r_eh = c·a(t)·∫[t,∞] dt'/a(t').

        Используется как fast-path при M_BH = 0 (через предрассчитанный
        интерполятор) и как fallback для honest-LTB при сбое интегрирования.
        """
        if time <= 0:
            return 0.0

        if self._event_interpolator is not None:
            try:
                r_event_flrw = float(self._event_interpolator(time))
                if r_event_flrw >= 0:
                    self._event_interp_used += 1
                    return r_event_flrw
            except Exception as e:
                if config.DEBUG and not hasattr(self, '_event_interp_fallback_printed'):
                    print(
                        f"DEBUG: event_horizon interpolation failed: {e}, "
                        "falling back to numerical integration"
                    )
                    self._event_interp_fallback_printed = True

        omega_m = self.omega_dm + self.omega_b
        a = self._get_scale_factor_for_time(time)

        def da_dt_future(t, a_val):
            if a_val <= 0:
                a_val = 1e-10
            h = self.h0 * np.sqrt(omega_m / (a_val**3) + self.omega_lambda)
            return a_val * h

        t_future = (
            time
            + float(config.MAX_TIME_YEARS) * SECONDS_PER_YEAR
        )

        sol_future = None
        try:
            result_future = integrate.solve_ivp(da_dt_future, [time, t_future], [a],
                                                method='RK45', dense_output=True,
                                                rtol=1e-3, atol=1e-6)
            if result_future.success:
                sol_future = result_future.sol
        except Exception:
            pass

        def integrand(t_prime):
            if t_prime <= time:
                return 1.0 / a
            if sol_future is not None:
                try:
                    a_at_t = float(sol_future(t_prime)[0])
                    if a_at_t <= 0:
                        return 1.0 / 1e-10
                    return 1.0 / a_at_t
                except Exception:
                    pass
            return 1.0 / 1e-10

        try:
            integral_result, _ = integrate.quad(integrand, time, t_future,
                                                limit=500, epsabs=1e-6, epsrel=1e-4)
            return c * a * integral_result
        except Exception:
            return 0.0

    def _compute_honest_event_horizon_ltb(self, time_now: float, M_bh: float) -> float:
        """Честный event horizon в LTB-Λ с центральной ЦЧД и FRW-материей.

        Идея: event horizon — это РАДИАЛЬНАЯ НУЛЕВАЯ СЕПАРАТРИСА в (t,r),
        разделяющая «фотоны, которые достигают центра» от «фотонов, которые
        улетают наружу». Сам фотон НА сепаратрисе удовлетворяет тому же
        уравнению, что и обычный inward null:

            dr/dt = R_dot(t,r) − c,   R_dot² = 2G·M(<r,t)/r + Λc²r²/3.

        При t → ∞ материя полностью разрежается под Λ-доминантой, геометрия
        → SdS(M_bh, Λ), и сепаратриса асимптотически сходится к внешнему
        корню SdS:  r_e(t→∞) = r_SdS_outer(M_bh).

        Прямое (forward) интегрирование сепаратрисы НЕУСТОЙЧИВО (малые
        отклонения растут экспоненциально). Обратное (backward) — УСТОЙЧИВО.
        Поэтому решаем dr/dt = R_dot − c **назад во времени** от
        (t_max, r_SdS_outer·(1-ε)) до (time_now, r_e_today) одним прогоном
        ODE. Это в ~10 раз быстрее, чем бисекция, и даёт точное значение.
        """
        omega_m = self.omega_dm + self.omega_b
        lam = 3.0 * (self.h0 ** 2) * self.omega_lambda / (c * c)
        rho_crit = self.rho_crit
        a_now = self._get_scale_factor_for_time(time_now)

        # t_max берём заведомо в глубокой Λ-эпохе; длина совпадает с
        # config.MAX_TIME_YEARS.
        t_max = time_now + float(config.MAX_TIME_YEARS) * SECONDS_PER_YEAR

        def da_dt(t, a_val):
            a_p = max(float(a_val[0]), 1e-30)
            H = self.h0 * np.sqrt(omega_m / (a_p ** 3) + self.omega_lambda)
            return [a_p * H]

        try:
            t_grid = np.linspace(time_now, t_max, 200)
            sol_a = integrate.solve_ivp(
                da_dt, [time_now, t_max], [a_now],
                method='RK45', t_eval=t_grid,
                rtol=1e-3, atol=1e-9,
            )
            if not sol_a.success:
                return self._cosmological_event_horizon_flrw(time_now)
        except Exception:
            return self._cosmological_event_horizon_flrw(time_now)

        a_grid = sol_a.y[0]
        a_final = float(a_grid[-1])

        def a_at(t):
            if t <= time_now:
                return a_now
            if t >= t_max:
                return a_final
            return max(float(np.interp(t, t_grid, a_grid)), 1e-30)

        r_sds_inner, r_sds_outer = schwarzschild_de_sitter_horizons(M_bh, lam)
        # При M_BH = M_N выраженных корней нет; nariai.py возвращает оба
        # корня = r_N. В этом случае event horizon формально схлопывается
        # к r_N — возвращаем его без интегрирования.
        r_upper = c / (self.h0 * np.sqrt(self.omega_lambda))  # √(3/Λ)
        if r_sds_outer <= r_sds_inner * 1.01:
            return r_sds_outer

        def dr_dt(t, state):
            r = float(state[0])
            if r <= 0.0:
                return [0.0]
            a_t = a_at(t)
            rho_m = rho_crit * omega_m / (a_t ** 3)
            M = M_bh + rho_m * (4.0 / 3.0) * np.pi * (r ** 3)
            arg = 2.0 * G * M / r + lam * c * c * (r ** 2) / 3.0
            Rdot = np.sqrt(arg) if arg > 0.0 else 0.0
            return [Rdot - c]

        # В точности на r_SdS_outer имеем dr/dt = 0 (фотон зафиксирован).
        # Чтобы backward-инеграция «двинулась» от этой стационарной точки,
        # стартуем чуть ВНУТРИ — там dr/dt < 0 (forward), backward даёт
        # r → больше. Сепаратриса (event horizon) — это устойчивое решение
        # обратной задачи, к которому сходятся все близкие траектории.
        r_init = r_sds_outer * (1.0 - 1.0e-4)

        try:
            sol = integrate.solve_ivp(
                dr_dt, [t_max, time_now], [r_init],
                method='RK45',
                rtol=1e-4, atol=max(r_sds_outer * 1.0e-6, 1.0e7),
                max_step=(t_max - time_now) / 100.0,
            )
            if sol.success:
                r_event = float(sol.y[0, -1])
                # Sanity: должен быть в [r_sds_inner, r_sds_outer], и
                # ≤ √(3/Λ) (никогда не превосходит чисто-dS).
                r_event = max(min(r_event, r_upper * 0.9999), 0.0)
                return r_event
        except Exception:
            pass
        return self._cosmological_event_horizon_flrw(time_now)
    
    def verify_z0_values(self, time: float, tolerance: float = 0.1) -> dict:
        """
        Проверка значений горизонтов при z=0 (a=1.0) с референсными значениями из config
        
        Args:
            time: Время для проверки (должно соответствовать z=0)
            tolerance: Допустимое отклонение в процентах (по умолчанию 0.1%)
        
        Returns:
            dict: Словарь с результатами проверки
        """
        if abs(self.scale_factor - 1.0) > 1e-6:
            return {"error": "Scale factor is not 1.0 (not at z=0)"}
        
        # Вычисляем горизонты
        # ВАЖНО: Для verify_z0_values используем минимальную массу
        h_h = self.hubble_horizon(time, get_mass_per_point_kg())
        e_h = self.cosmological_event_horizon(time)
        dS = self.de_sitter_horizon()
        p_h = self.particle_horizon(time)
        
        # Референсные значения из constants (в метрах)
        ref_hubble = HUBBLE_HORIZON_Z0_BILLION_LY * BILLION_LIGHT_YEARS_IN_METERS
        ref_event = EVENT_HORIZON_Z0_BILLION_LY * BILLION_LIGHT_YEARS_IN_METERS
        ref_de_sitter = DE_SITTER_HORIZON_BILLION_LY * BILLION_LIGHT_YEARS_IN_METERS
        ref_particle = PARTICLE_HORIZON_Z0_BILLION_LY * BILLION_LIGHT_YEARS_IN_METERS
        
        # Вычисляем отклонения
        def calc_deviation(computed, reference):
            if reference == 0:
                return float('inf')
            return abs((computed - reference) / reference) * 100
        
        results = {
            "hubble": {
                "computed": h_h / BILLION_LIGHT_YEARS_IN_METERS,
                "reference": HUBBLE_HORIZON_Z0_BILLION_LY,
                "deviation_percent": calc_deviation(h_h, ref_hubble),
                "ok": calc_deviation(h_h, ref_hubble) < tolerance
            },
            "event": {
                "computed": e_h / BILLION_LIGHT_YEARS_IN_METERS,
                "reference": EVENT_HORIZON_Z0_BILLION_LY,
                "deviation_percent": calc_deviation(e_h, ref_event),
                "ok": calc_deviation(e_h, ref_event) < tolerance
            },
            "de_sitter": {
                "computed": dS / BILLION_LIGHT_YEARS_IN_METERS,
                "reference": DE_SITTER_HORIZON_BILLION_LY,
                "deviation_percent": calc_deviation(dS, ref_de_sitter),
                "ok": calc_deviation(dS, ref_de_sitter) < tolerance
            },
            "particle": {
                "computed": p_h / BILLION_LIGHT_YEARS_IN_METERS,
                "reference": PARTICLE_HORIZON_Z0_BILLION_LY,
                "deviation_percent": calc_deviation(p_h, ref_particle),
                "ok": calc_deviation(p_h, ref_particle) < tolerance
            }
        }
        
        return results


def calculate_initial_horizons(
    time_years: float,
    cosmology: Optional[LambdaCDM] = None,
) -> Tuple[float, float, float]:
    """
    Вычисляет космологические горизонты при заданном времени (в годах).

    Args:
        time_years: Время в годах после Большого взрыва
        cosmology: Объект LambdaCDM симуляции; если None — создаётся временный экземпляр.

    Returns:
        (Горизонт Хаббла, Горизонт событий, Горизонт частиц) в миллиардах световых лет

    Примечание:
        Значения возвращаются как PROPER distance (физическая дистанция в момент t), а не комовинг.
        Определения/формулы для расстояний в FRW (и интегралов по z): D. W. Hogg (1999),
        "Distance measures in cosmology", arXiv: astro-ph/9905116: https://arxiv.org/abs/astro-ph/9905116
    """
    if cosmology is None:
        cosmology = LambdaCDM()

    time_seconds = time_years * SECONDS_PER_YEAR
    scale_factor = calculate_scale_factor_at_time(time_years)
    prev_scale_factor = cosmology.scale_factor
    try:
        cosmology.scale_factor = scale_factor

        hubble_horizon_m = cosmology.hubble_horizon(time_seconds)
        event_horizon_m = cosmology.cosmological_event_horizon(time_seconds)
        particle_horizon_m = cosmology.particle_horizon(time_seconds)
    finally:
        cosmology.scale_factor = prev_scale_factor

    hubble_horizon_bly = hubble_horizon_m / BILLION_LIGHT_YEARS_IN_METERS
    event_horizon_bly = event_horizon_m / BILLION_LIGHT_YEARS_IN_METERS
    particle_horizon_bly = particle_horizon_m / BILLION_LIGHT_YEARS_IN_METERS

    return (hubble_horizon_bly, event_horizon_bly, particle_horizon_bly)