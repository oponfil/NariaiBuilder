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
        self._load_precomputed_horizons()
        
        # ОПТИМИЗАЦИЯ: Кэш для scale_factor по времени
        self._scale_factor_cache_time = -1.0
        self._scale_factor_cache_value = 1.0
    
    def _get_scale_factor_for_time(self, time: float) -> float:
        """Получить scale_factor для времени с кэшированием"""
        # Если время совпадает с кэшированным, возвращаем кэш
        if abs(time - self._scale_factor_cache_time) < _SCALE_FACTOR_CACHE_TOLERANCE_SECONDS:
            return self._scale_factor_cache_value
        
        # Иначе вычисляем и кэшируем
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
        
        # ОПТИМИЗАЦИЯ: синхронизируем кэш _get_scale_factor_for_time с новым
        # состоянием. update_scale_factor RK4-проинтегрировал scale_factor на dt,
        # поэтому после вызова кэш можно считать актуальным на t_old + dt.
        # Без этого hubble_parameter(t)/hubble_horizon(t) каждый кадр уходил
        # в полную solve_ivp от t=0 (~5 мс на вызов) — доминирующая стоимость
        # горячего пути calculate_masses при бьющем cache miss.
        if self._scale_factor_cache_time >= 0:
            self._scale_factor_cache_time += float(dt)
            self._scale_factor_cache_value = self.scale_factor
    
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
        Горизонт Хаббла: расстояние, на котором скорость расширения равна скорости света
        r_H = c / H(t)
        
        В присутствии черной дыры горизонт Хаббла ограничен сверху внешним горизонтом
        из метрики SdS, который учитывает влияние черной дыры.
        
        Args:
            time: время в секундах
            black_hole_mass_kg: масса центральной черной дыры (кг).
                              Если None или не указан, используется 0.
        
        Returns:
            float: горизонт Хаббла в метрах, ограниченный сверху внешним горизонтом из SdS.
        """
        if black_hole_mass_kg is None:
            black_hole_mass_kg = 0.0
        
        h = self.hubble_parameter(time)
        r_hubble = c / h
        
        # Вычисляем внешний горизонт из метрики SdS (r_outer)
        r_outer_sds = self.de_sitter_horizon(black_hole_mass_kg)
        
        # Горизонт Хаббла ограничен сверху внешним горизонтом из SdS
        return min(r_hubble, r_outer_sds)
    
    def de_sitter_horizon(self, black_hole_mass_kg: float = 0.0) -> float:
        """
        Горизонт де Ситтера в метрике Шварцшильда-де Ситтера (SdS).
        
        Учитывает влияние центральной черной дыры на космологический горизонт.
        В метрике SdS внешний горизонт (космологический) уменьшается по мере роста массы ЧД.
        Когда масса ЧД достигает массы Нарайи, горизонты совпадают.
        
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
        Космологический горизонт событий в расширяющейся Вселенной с черной дырой.
        
        ВАЖНО: Космологический горизонт событий вычисляется из FLRW космологии как интеграл
        по будущему расширению: r_eh = c * a(t) * ∫[t to ∞] dt'/a(t')
        
        Этот горизонт растет со временем и стремится к внешнему горизонту из метрики SdS
        в будущем (t → ∞). В любой момент времени он ограничен сверху внешним горизонтом
        из метрики SdS, который учитывает влияние черной дыры.
        
        Args:
            time: время в секундах
            black_hole_mass_kg: масса центральной черной дыры (кг).
                              Если None или не указан, используется 0.
        
        Returns:
            float: космологический горизонт событий в метрах.
                  Вычисляется из FLRW (растет со временем), но ограничен сверху внешним горизонтом из SdS.
        """
        if black_hole_mass_kg is None:
            black_hole_mass_kg = 0.0
        if time <= 0:
            return 0.0

        # Сначала считаем SdS-ограничение: FLRW-горизонт не должен превышать
        # внешний горизонт в присутствии центральной ЧД.
        r_outer_sds = self.de_sitter_horizon(black_hole_mass_kg)

        # Горячий путь рендера: используем предвычисленный FLRW event horizon
        # вместо solve_ivp + quad на каждом кадре.
        if self._event_interpolator is not None:
            try:
                r_event_flrw = float(self._event_interpolator(time))
                if r_event_flrw >= 0:
                    self._event_interp_used += 1
                    return min(r_event_flrw, r_outer_sds)
            except Exception as e:
                if config.DEBUG and not hasattr(self, '_event_interp_fallback_printed'):
                    print(
                        f"DEBUG: event_horizon interpolation failed: {e}, "
                        "falling back to numerical integration"
                    )
                    self._event_interp_fallback_printed = True
        
        # Вычисляем классический космологический горизонт событий из FLRW
        # r_eh = c * a(t) * ∫[t to ∞] dt'/a(t')
        # Это учитывает расширение Вселенной и РАСТЕТ со временем
        # Fallback, если предрасчитанный кэш недоступен.
        
        # Вычисляем космологический горизонт событий через численное интегрирование
        # r_eh = c * a(t) * ∫[t to ∞] dt'/a(t')
        # Используем тот же подход, что и в precompute_horizons.py
        omega_m = self.omega_dm + self.omega_b
        
        # ОПТИМИЗАЦИЯ: Используем кэшированный scale_factor
        a = self._get_scale_factor_for_time(time)
        
        def da_dt_future(t, a_val):
            if a_val <= 0:
                a_val = 1e-10
            h = self.h0 * np.sqrt(omega_m / (a_val**3) + self.omega_lambda)
            return a_val * h
        
        t_future = time + 1000.0 * 365.25 * 24 * 3600 * 1e9  # 1000 млрд лет
        
        sol_future = None
        try:
            result_future = integrate.solve_ivp(da_dt_future, [time, t_future], [a],
                                                method='RK45', dense_output=True,
                                                rtol=1e-3, atol=1e-6)
            if result_future.success:
                sol_future = result_future.sol
        except:
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
                except:
                    pass
            return 1.0 / 1e-10
        
        try:
            integral_result, _ = integrate.quad(integrand, time, t_future,
                                                limit=500, epsabs=1e-6, epsrel=1e-4)
            r_event_flrw = c * a * integral_result
        except:
            r_event_flrw = 0.0
        
        # ВАЖНО: Космологический горизонт событий из FLRW учитывает расширение Вселенной
        # и РАСТЕТ со временем, стремясь к внешнему горизонту из SdS в будущем.
        # В любой момент времени он не может превышать этот горизонт, так как черная дыра
        # уменьшает доступное космологическое пространство.
        # Возвращаем минимум из них, чтобы горизонт расширялся, но был ограничен сверху.
        return min(r_event_flrw, r_outer_sds)
    
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