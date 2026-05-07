"""
Управление точками материи и их движением при коллапсе
"""
import numpy as np

import config
from physics.laser import (
    burnable_mass_kg,
    cosmological_redshifted_photon_mass_kg,
    emitted_photon_mass_coord_kg,
    emitted_rest_mass_for_step,
    laser_mass_floor,
    thrust_delta_gamma_v_per_second,
)
from utils.config_utils import get_mass_per_point_kg
from utils.constants import (
    G,
    c,
    H0_s,
    OMEGA_B,
    OMEGA_DM,
    OMEGA_LAMBDA,
    SECONDS_PER_YEAR,
)



def _scale_factor_lookup(years_arr: np.ndarray) -> np.ndarray:
    """
    Векторное a(t) для плоской ΛCDM (аналитическое решение Фридмана).

        a(t) = (Ω_m/Ω_Λ)^(1/3) · sinh^(2/3)(t / t_Λ),
        где t_Λ = 2 / (3·H0·√Ω_Λ).

    Точно совпадает с численным calculate_scale_factor_at_time, но мгновенно
        и векторно — годится для тысяч одновременных запросов на каждом шаге.
    """
    omega_m = OMEGA_DM + OMEGA_B
    t_lambda = 2.0 / (3.0 * H0_s * np.sqrt(OMEGA_LAMBDA))
    t_sec = np.maximum(np.asarray(years_arr, dtype=np.float64), 0.0) * SECONDS_PER_YEAR
    x = np.minimum(t_sec / t_lambda, 350.0)  # клип против overflow sinh при t→∞
    return (omega_m / OMEGA_LAMBDA) ** (1.0 / 3.0) * np.sinh(x) ** (2.0 / 3.0)


# Кэш предвычисленного конформного времени η(t) = ∫₀^t dt'/a(t').
# Используется для геометрии фотонов с учётом расширения пространства:
# фотон, испущенный в t_emit с сопутствующей координатой χ_emit, в момент
# t_now находится на χ(t_now) = χ_emit − c·[η(t_now) − η(t_emit)].
_ETA_CACHE = None


def _conformal_time_lookup(years_arr: np.ndarray) -> np.ndarray:
    """
    η(t) = ∫₀^t dt'/a(t') в секундах для плоской ΛCDM.

    Расчёт численный (трапеции) с предвычисленной таблицей; первый вызов
    инициализирует кэш, дальше — мгновенная векторная интерполяция.

    Опорная точка η(t_min)=0 при t_min ≈ 1 Myr (от Большого взрыва a→0,
    интеграл расходится; в наших задачах фотоны всегда испускаются после
    нескольких Гyr, так что разности η(t1)−η(t2) от выбора t_min не зависят).
    """
    global _ETA_CACHE
    if _ETA_CACHE is None:
        t_min_yr = 1.0e6
        t_max_yr = 200.0e9
        step_yr = 1.0e7  # 10 Myr; ~20k узлов, ошибка <<1% на наших масштабах
        years_grid = np.arange(t_min_yr, t_max_yr + step_yr, step_yr, dtype=np.float64)
        a_grid = _scale_factor_lookup(years_grid)
        sec_grid = years_grid * SECONDS_PER_YEAR
        inv_a = 1.0 / np.maximum(a_grid, 1e-300)
        # Cumulative trapezoid: η[i] = Σ_{k<i} ½(1/a_k + 1/a_{k+1})·Δt_k
        increments = 0.5 * (inv_a[:-1] + inv_a[1:]) * np.diff(sec_grid)
        eta_grid = np.concatenate(([0.0], np.cumsum(increments)))
        _ETA_CACHE = (years_grid, eta_grid)

    years_grid, eta_grid = _ETA_CACHE
    arr = np.asarray(years_arr, dtype=np.float64)
    return np.interp(arr, years_grid, eta_grid)


def _inverse_conformal_time_lookup(eta_seconds_arr: np.ndarray) -> np.ndarray:
    """t (годы) по заданному η(t) (секунды) — обратная функция к
    _conformal_time_lookup. Использует тот же кэш."""
    _conformal_time_lookup(np.array([0.0]))  # гарантировать инициализацию кэша
    years_grid, eta_grid = _ETA_CACHE  # type: ignore[misc]
    arr = np.asarray(eta_seconds_arr, dtype=np.float64)
    return np.interp(arr, eta_grid, years_grid)


# Кэш предвычисленного I(t) = ∫_0^t a(τ) dτ в секундах. Нужен для расчёта
# массы-эквивалента фотонов в полёте: энергия ещё не дошедших до ЦЧД фотонов,
# излучённых одной точкой в интервале τ ∈ [τ_min, τ_max], равна
#     E_in_flight = ∫ P · (a(τ)/a_now) dτ = (P/a_now) · [I(τ_max) − I(τ_min)],
# где P — мощность лазера, a_now — масштабный фактор сейчас.
_INT_A_CACHE = None


def _int_a_dt_lookup(years_arr: np.ndarray) -> np.ndarray:
    """I(t) = ∫_0^t a(t') dt' (в секундах) для плоской ΛCDM. Численный кэш на
    той же сетке, что и η(t); первый вызов инициализирует, далее — векторная
    интерполяция."""
    global _INT_A_CACHE
    if _INT_A_CACHE is None:
        _conformal_time_lookup(np.array([0.0]))  # гарантировать инициализацию сетки
        years_grid, _ = _ETA_CACHE  # type: ignore[misc]
        a_grid = _scale_factor_lookup(years_grid)
        sec_grid = years_grid * SECONDS_PER_YEAR
        increments = 0.5 * (a_grid[:-1] + a_grid[1:]) * np.diff(sec_grid)
        int_a_grid = np.concatenate(([0.0], np.cumsum(increments)))
        _INT_A_CACHE = (years_grid, int_a_grid)

    years_grid, int_a_grid = _INT_A_CACHE
    arr = np.asarray(years_arr, dtype=np.float64)
    return np.interp(arr, years_grid, int_a_grid)


class MatterPoints:
    """Класс для управления точками материи и их движением"""
    
    def __init__(self):
        """Инициализация"""
        self.points_comoving = None  # Координаты точек в базовых сопутствующих координатах (N x 3)
        self.comoving_distances = None  # ОПТИМИЗАЦИЯ: Расстояния от центра в comoving координатах (N,)
        self.velocities_comoving = None  # Скорости точек в базовых сопутствующих координатах (N x 3) в м/с
        self.masses_per_point = None  # Массы каждой точки (N,) в кг
        # Bool-маска (N,) точек, поглощённых ЦЧД. True — точка больше не
        # обновляется, её координаты обнулены, масса учтена в accumulated_bh_mass.
        # Было set[int]: O(N) проверки `idx in set` в горячем цикле. Bool-маска
        # даёт O(1) векторное `~mask`, `mask & ...`, и масштабируется на 200K+ точек.
        self.inside_bh_mask = None
        # ОПТИМИЗАЦИЯ: счётчики версий для кэширования внешних структур
        # (например, sort_idx по comoving_distances + active_mask). Любая мутация
        # comoving_distances / inside_bh_mask / velocities_comoving /
        # masses_per_point должна сопровождаться инкрементом соответствующей
        # версии — потребители сравнивают её с сохранённой.
        self._comoving_distances_version = 0
        self._points_inside_bh_version = 0
        self._velocities_version = 0
        self._masses_version = 0
        self._laser_photons_version = 0
        # Предрассчитанная норма² comoving-скоростей (N,) и её max — нужны для
        # γ-коррекции в MassCalculator без полного einsum по N точкам каждый
        # кадр. Обновляются вместе с velocities_comoving.
        self._v_sq_comoving = None
        self._v_sq_max = 0.0
        # Отсортированный массив индексов точек, поглощённых ЦЧД. Кэш для
        # инвалидации active_mask без пересоздания каждый раз.
        # Перестраивается лениво при cache miss по _points_inside_bh_version.
        self._inside_indices_arr = None
        self.accumulated_bh_mass = 0.0  # ОПТИМИЗАЦИЯ: Накопленная масса ЧД (прибавляем при попадании точек)
        # Время «включения» лазера. Все тяговые точки запускаются одновременно;
        # голова луча от каждой летит к центру со скоростью c в физ. координатах.
        # ЦЧД начинает поглощать энергию данной точки, когда c·(t_now - t_laser_start) ≥ r_initial,
        # где r_initial — её физ. расстояние в момент включения (НЕ текущее r_now,
        # т. к. точка может убегать наружу под отдачей со скоростью близкой к c).
        self.t_laser_start_seconds = None
        # Расстояние каждой точки в момент включения её лазера (м, физ.). NaN — лазер ещё
        # не включался для данного индекса (например, точка добавилась позже).
        self._laser_start_r_phys = None
        # Момент и физ. расстояние точки в момент выключения её лазера (m_rest стал ≤ 0).
        # NaN означает «лазер ещё активен» (или вообще не включался). Нужны для того,
        # чтобы уже излучённый «хвост» луча продолжал лететь к центру со скоростью c
        # после того, как точка-источник перестала излучать.
        self._laser_off_t_seconds = None
        self._laser_off_r_phys = None
        # Масса точки в момент первого включения лазера. Нужна, чтобы КПД
        # преобразования оставлял несгораемый остаток, а не превращал всю точку
        # в фотоны.
        self._laser_start_mass_kg = None
        # Накопленная масса-эквивалент энергии лазера, доставленной в ЦЧД (отдельно
        # от accumulated_bh_mass, который ещё включает поглощённые целые точки).
        self.laser_absorbed_mass = 0.0
        # Дискретные фотонные пакеты лазера. Каждый пакет создаётся при эмиссии
        # за один шаг dt и поглощается, когда пересекает горизонт ЧД.
        self._laser_photon_chi = np.empty(0, dtype=np.float64)
        self._laser_photon_mass_emit_kg = np.empty(0, dtype=np.float64)
        self._laser_photon_a_emit = np.empty(0, dtype=np.float64)
        self._laser_photon_r_emit_m = np.empty(0, dtype=np.float64)
        self._laser_photon_ux = np.empty(0, dtype=np.float64)
        self._laser_photon_uy = np.empty(0, dtype=np.float64)
        # Индекс материальной точки-источника (для UI: «первый» фотон дальнего испускателя).
        self._laser_photon_source_idx = np.empty(0, dtype=np.int64)
        # Космологическое время испускания пакета (с) — для отмотки назад (удаление «ещё не испущенных»).
        self._laser_photon_emit_t_seconds = np.empty(0, dtype=np.float64)
        self._laser_photon_last_time_seconds = None
        # True — точка участвует в лазерной тяге (внутри cosmological event horizon
        # в момент t_collapse; см. _emitter_comoving_radius_m /
        # _set_laser_emitter_mask_inside_event_horizon).
        self.laser_emitter_mask = None
        # Номинальная суммарная мощность за последний шаг: Σ(s·m покоя до вычитания dm).
        # UI после рендеринга видит уже уменьшенные массы; для отображения/пика нужно
        # значение в момент эмиссии (как P = MATTER_THRUST_POWER_PER_KG_W · m_rest в config).
        self._last_step_nominal_sigma_p_w = 0.0

    def clear_photons(self):
        """Очистить все лазерные фотоны и связанные с ними массивы"""
        self._laser_photon_chi = np.empty(0, dtype=np.float64)
        self._laser_photon_mass_emit_kg = np.empty(0, dtype=np.float64)
        self._laser_photon_a_emit = np.empty(0, dtype=np.float64)
        self._laser_photon_r_emit_m = np.empty(0, dtype=np.float64)
        self._laser_photon_ux = np.empty(0, dtype=np.float64)
        self._laser_photon_uy = np.empty(0, dtype=np.float64)
        self._laser_photon_source_idx = np.empty(0, dtype=np.int64)
        self._laser_photon_emit_t_seconds = np.empty(0, dtype=np.float64)
        self._laser_photon_last_time_seconds = None
        
        # Также сбрасываем параметры лазера для точек
        if self.points_comoving is not None:
            n = len(self.points_comoving)
            self._laser_start_r_phys = np.full(n, np.nan, dtype=np.float64)
            self._laser_off_t_seconds = np.full(n, np.nan, dtype=np.float64)
            self._laser_off_r_phys = np.full(n, np.nan, dtype=np.float64)
            self._laser_start_mass_kg = np.full(n, np.nan, dtype=np.float64)
        
        self.t_laser_start_seconds = None
        self.laser_absorbed_mass = 0.0
        self._last_step_nominal_sigma_p_w = 0.0
        self._laser_photons_version += 1

    def _update_comoving_distances(self):
        """ОПТИМИЗАЦИЯ: Пересчитать comoving расстояния от центра"""
        if self.points_comoving is not None and len(self.points_comoving) > 0:
            self.comoving_distances = np.sqrt(np.sum(self.points_comoving**2, axis=1))
        else:
            self.comoving_distances = None
        self._comoving_distances_version += 1

    def _recompute_velocity_norms(self):
        """Пересчитать norm² comoving-скоростей и обновить max + версию.

        Вынесено сюда, чтобы MassCalculator не запускал einsum по полному
        массиву (N,3) каждый кадр. _v_sq_comoving имеет ту же длину, что
        velocities_comoving; _v_sq_max используется для O(1) проверки порога
        γ-коррекции (_BETA2_NEGLIGIBLE_THRESHOLD в mass_calculator).
        """
        v = self.velocities_comoving
        if v is None or len(v) == 0:
            self._v_sq_comoving = None
            self._v_sq_max = 0.0
        else:
            self._v_sq_comoving = np.einsum('ij,ij->i', v, v)
            self._v_sq_max = float(self._v_sq_comoving.max()) if self._v_sq_comoving.size > 0 else 0.0
        self._velocities_version += 1

    def _bump_masses_version(self):
        """Сообщить, что masses_per_point мог измениться (для кэша m_eff)."""
        self._masses_version += 1

    def _ensure_inside_bh_mask(self):
        """Гарантировать, что inside_bh_mask имеет длину len(points_comoving).

        Расширяется (значения False для новых индексов), если в points_comoving
        были добавлены новые точки. Сохраняет «уже поглощённые» флаги.
        """
        n = 0 if self.points_comoving is None else len(self.points_comoving)
        if self.inside_bh_mask is None or len(self.inside_bh_mask) != n:
            new = np.zeros(n, dtype=bool)
            if self.inside_bh_mask is not None:
                k = min(len(self.inside_bh_mask), n)
                new[:k] = self.inside_bh_mask[:k]
            self.inside_bh_mask = new

    def get_inside_indices_arr(self):
        """Отсортированные индексы точек, поглощённых ЦЧД, кэшированно.

        Отдаёт np.ndarray int64 — пересоздаётся только когда инвалидируется
        кэш по _points_inside_bh_version. Используется MassCalculator для
        быстрого построения active_mask без np.fromiter(set, ...) каждый кадр.
        """
        if self._inside_indices_arr is None:
            if self.inside_bh_mask is not None and self.inside_bh_mask.any():
                self._inside_indices_arr = np.flatnonzero(self.inside_bh_mask).astype(
                    np.int64, copy=False
                )
            else:
                self._inside_indices_arr = np.empty(0, dtype=np.int64)
        return self._inside_indices_arr

    def mark_point_inside_bh(self, idx_int: int):
        """Зарегистрировать точку как поглощённую ЦЧД (атомарно с инкрементом версии)."""
        self._ensure_inside_bh_mask()
        idx_int = int(idx_int)
        if 0 <= idx_int < len(self.inside_bh_mask) and not self.inside_bh_mask[idx_int]:
            self.inside_bh_mask[idx_int] = True
            self._points_inside_bh_version += 1
            self._inside_indices_arr = None

    def _mark_inside_bh_vectorized(self, newly_inside_mask: np.ndarray) -> None:
        """Отметить сразу множество точек как поглощённые ЦЧД (по bool-маске длины N).

        Используется в горячем цикле update_positions_and_velocities вместо
        Python-цикла `for idx_int: mark_point_inside_bh(idx_int)`.
        """
        self._ensure_inside_bh_mask()
        if newly_inside_mask.any():
            self.inside_bh_mask |= newly_inside_mask
            self._points_inside_bh_version += 1
            self._inside_indices_arr = None

    def _append_laser_photons(
        self,
        chi_emit,
        mass_emit_kg,
        scale_factor: float,
        r_emit_m,
        ux_emit,
        uy_emit,
        source_point_idx,
        emit_time_seconds: float,
    ):
        """Добавить дискретные фотоны, испущенные за текущий шаг dt."""
        mass_emit = np.asarray(mass_emit_kg, dtype=np.float64)
        valid = mass_emit > 0.0
        if not np.any(valid):
            return
        chi = np.asarray(chi_emit, dtype=np.float64)[valid]
        r_emit = np.asarray(r_emit_m, dtype=np.float64)[valid]
        src_idx = np.asarray(source_point_idx, dtype=np.int64).reshape(-1)
        if len(src_idx) != len(mass_emit):
            return
        src_idx = src_idx[valid]
        self._laser_photon_chi = np.concatenate([self._laser_photon_chi, chi])
        self._laser_photon_mass_emit_kg = np.concatenate([
            self._laser_photon_mass_emit_kg,
            mass_emit[valid],
        ])
        self._laser_photon_a_emit = np.concatenate([
            self._laser_photon_a_emit,
            np.full(int(np.sum(valid)), float(scale_factor), dtype=np.float64),
        ])
        self._laser_photon_r_emit_m = np.concatenate([
            self._laser_photon_r_emit_m,
            r_emit,
        ])
        self._laser_photon_ux = np.concatenate([
            self._laser_photon_ux,
            np.asarray(ux_emit, dtype=np.float64)[valid],
        ])
        self._laser_photon_uy = np.concatenate([
            self._laser_photon_uy,
            np.asarray(uy_emit, dtype=np.float64)[valid],
        ])
        self._laser_photon_source_idx = np.concatenate([
            self._laser_photon_source_idx,
            src_idx,
        ])
        self._laser_photon_emit_t_seconds = np.concatenate([
            self._laser_photon_emit_t_seconds,
            np.full(int(np.sum(valid)), float(emit_time_seconds), dtype=np.float64),
        ])
        self._laser_photons_version += 1

    def _sync_laser_photon_emit_times_length(self) -> None:
        """Длина _laser_photon_emit_t_seconds совпадает с числом пакетов (-∞ — без метки времени)."""
        n = len(self._laser_photon_chi)
        m = len(self._laser_photon_emit_t_seconds)
        if m == n:
            return
        if n == 0:
            self._laser_photon_emit_t_seconds = np.empty(0, dtype=np.float64)
            return
        if m == 0:
            self._laser_photon_emit_t_seconds = np.full(n, -np.inf, dtype=np.float64)
            return
        if m < n:
            self._laser_photon_emit_t_seconds = np.concatenate([
                self._laser_photon_emit_t_seconds,
                np.full(n - m, -np.inf, dtype=np.float64),
            ])
        else:
            self._laser_photon_emit_t_seconds = self._laser_photon_emit_t_seconds[:n]

    def _apply_laser_photon_keep_mask(self, keep: np.ndarray) -> None:
        """Отфильтровать все массивы лазерных фотонов одной маской keep."""
        self._laser_photon_chi = self._laser_photon_chi[keep]
        self._laser_photon_mass_emit_kg = self._laser_photon_mass_emit_kg[keep]
        self._laser_photon_a_emit = self._laser_photon_a_emit[keep]
        self._laser_photon_r_emit_m = self._laser_photon_r_emit_m[keep]
        self._laser_photon_ux = self._laser_photon_ux[keep]
        self._laser_photon_uy = self._laser_photon_uy[keep]
        self._laser_photon_source_idx = self._laser_photon_source_idx[keep]
        self._laser_photon_emit_t_seconds = self._laser_photon_emit_t_seconds[keep]

    def _advance_and_absorb_laser_photons(
        self,
        scale_factor: float,
        universe_time_seconds: float | None,
        r_black_hole: float | None,
    ) -> None:
        """Сдвиг χ по Δη между последним кадром и текущим t; вперёд — поглощение у ЧД; назад — откат χ."""
        if universe_time_seconds is None or scale_factor <= 0.0:
            return

        current_time = float(universe_time_seconds)
        if self._laser_photon_last_time_seconds is None:
            self._laser_photon_last_time_seconds = current_time
            return

        if len(self._laser_photon_chi) == 0:
            self._laser_photon_last_time_seconds = current_time
            return

        self._sync_laser_photon_emit_times_length()

        t_prev_yr = float(self._laser_photon_last_time_seconds) / SECONDS_PER_YEAR
        t_now_yr = current_time / SECONDS_PER_YEAR
        eta_pair = _conformal_time_lookup(np.array([t_prev_yr, t_now_yr]))
        delta_eta = float(eta_pair[1]) - float(eta_pair[0])
        self._laser_photon_chi = self._laser_photon_chi - c * delta_eta

        if delta_eta > 0.0:
            r_photon = self._laser_photon_chi * float(scale_factor)
            absorption_radius = max(float(r_black_hole or 0.0), 0.0)
            absorbed = r_photon <= absorption_radius
            if absorption_radius <= 0.0:
                absorbed = r_photon <= 0.0

            if np.any(absorbed):
                absorbed_mass = cosmological_redshifted_photon_mass_kg(
                    self._laser_photon_mass_emit_kg[absorbed],
                    self._laser_photon_a_emit[absorbed],
                    scale_factor,
                )
                dM = float(np.sum(absorbed_mass))
                self.accumulated_bh_mass += dM
                self.laser_absorbed_mass += dM

            keep = (~absorbed) & np.isfinite(self._laser_photon_chi) & (self._laser_photon_chi > 0.0)
            npho = int(self._laser_photon_chi.shape[0])
            if int(self._laser_photon_source_idx.shape[0]) != npho:
                self._laser_photon_source_idx = np.full(npho, -1, dtype=np.int64)
            self._apply_laser_photon_keep_mask(keep)
        else:
            a_em = np.maximum(self._laser_photon_a_emit, 1e-300)
            chi_cap = self._laser_photon_r_emit_m / a_em
            self._laser_photon_chi = np.minimum(self._laser_photon_chi, chi_cap)
            emit_keep = self._laser_photon_emit_t_seconds <= current_time + 1e-9
            finite_keep = np.isfinite(self._laser_photon_chi) & (self._laser_photon_chi > 0.0)
            keep = emit_keep & finite_keep
            npho = int(self._laser_photon_chi.shape[0])
            if int(self._laser_photon_source_idx.shape[0]) != npho:
                self._laser_photon_source_idx = np.full(npho, -1, dtype=np.int64)
            self._apply_laser_photon_keep_mask(keep)

        self._laser_photon_last_time_seconds = current_time
        self._laser_photons_version += 1

    @staticmethod
    def _emitter_comoving_radius_m() -> float:
        """
        Пороговое комовинг-расстояние от центра, до которого точка считается
        полезным лазерным эмиттером.

        Физический критерий: фотон, испущенный в момент `t_collapse` из точки
        на комовинг-расстоянии `χ` от центра, асимптотически достигает центра
        тогда и только тогда, когда

            χ ≤ χ_event(t_collapse) = c · ∫_{t_collapse}^{∞} dt'/a(t')
                                    = c · (η(∞) − η(t_collapse)),

        где `χ_event` — комовинг-радиус cosmological event horizon в момент
        включения лазера. За этим горизонтом фотоны уносятся расширением и
        в ЦЧД попасть не могут — такие точки бесполезно теряли бы массу.

        Возвращает:
            χ_event = c·(η(t_max) − η(t_collapse)) [м, комовинг].
            `η(t_max)` берётся как асимптота η(∞): сетка `_ETA_CACHE` идёт до
            ≈ 200 Гyr, что соответствует ~12·t_Hubble в Λ-эре, и оставшийся
            «хвост» интеграла даёт пренебрежимо малую поправку.
        """
        t_collapse_yr = float(getattr(config, "LASER_START_TIME_YEARS", 0.0))
        _conformal_time_lookup(np.array([0.0]))
        years_grid, eta_grid = _ETA_CACHE
        t_max_yr = float(years_grid[-1])
        eta_pair = _conformal_time_lookup(
            np.array([t_collapse_yr, t_max_yr], dtype=np.float64)
        )
        chi_event = c * float(eta_pair[1] - eta_pair[0])
        return max(chi_event, 0.0)

    def _set_laser_emitter_mask_inside_event_horizon(self) -> None:
        """
        Авто-выбор эмиттеров: маска `True` для точек, чьё комовинг-расстояние
        от центра меньше `χ_event(t_collapse)`.
        Подробнее о критерии — см. `_emitter_comoving_radius_m`.
        """
        if self.points_comoving is None:
            self.laser_emitter_mask = None
            return
        n = len(self.points_comoving)
        self.laser_emitter_mask = np.zeros(n, dtype=bool)
        if n == 0:
            return
        chi_threshold = self._emitter_comoving_radius_m()
        if chi_threshold <= 0.0:
            return
        d = np.sqrt(np.sum(self.points_comoving ** 2, axis=1))
        self.laser_emitter_mask = d < chi_threshold

    def init_laser_emitter_mask(self, n: int) -> None:
        """Заполнить маску после инициализации координат (ожидается len(points_comoving) == n)."""
        if self.points_comoving is None or len(self.points_comoving) != n:
            return
        self._set_laser_emitter_mask_inside_event_horizon()

    def total_laser_emitters_power_w(
        self,
        scale_factor: float,
        r_black_hole_m: float | None = None,
    ) -> float:
        """
        Номинальная суммарная мощность лазера после последнего шага симуляции (Вт):

            ΣP_nom = Σ_i ( s · m_i ),

        где слагаются только активные источники (те же условия, что thrust_active:
        маска эмиттера, вне горизонта ЧД, burnable_mass > 0), а ``m_i`` — масса
        покоя **на начале эмиссии в этом шаге**, до вычитания ``dm``. Так совпадает
        с задачей P = (Вт/кг)·масса точки без занижения после большого ``s·Δt``.

        Интегрирование кадра задаёт последнее сохранённое значение;
        параметры ``scale_factor`` и ``r_black_hole_m`` остаются для совместимости
        со старым вызовом UI.
        """
        return float(max(getattr(self, '_last_step_nominal_sigma_p_w', 0.0), 0.0))

    def total_in_flight_laser_photon_mass_kg(self, scale_factor: float) -> tuple[float, int]:
        """
        Суммарная масса-эквивалент E/c² всех дискретных лазерных фотонов в полёте и их число.
        Для каждого пакета: m(t_now) = m_emit * a_emit / a_now (космологический redshift энергии).
        """
        a_now = float(scale_factor)
        if a_now <= 0.0:
            return 0.0, 0
        chi = self._laser_photon_chi
        if chi is None or len(chi) == 0:
            return 0.0, 0
        m_emit = np.asarray(self._laser_photon_mass_emit_kg, dtype=np.float64)
        a_emit = np.asarray(self._laser_photon_a_emit, dtype=np.float64)
        n = int(m_emit.shape[0])
        if n == 0 or a_emit.shape[0] != n or int(np.asarray(chi).shape[0]) != n:
            return 0.0, 0
        m_sum = float(np.sum(cosmological_redshifted_photon_mass_kg(m_emit, a_emit, a_now)))
        return m_sum, n

    def _build_in_flight_laser_state(
        self,
        scale_factor: float,
        universe_time_seconds: float,
        absorption_radius_m: float = 0.0,
    ):
        """
        Подготовить per-point массивы, инвариантные относительно радиуса горизонта R,
        для подсчёта массы-эквивалента энергии лазерных фотонов, ещё летящих к ЦЧД
        (и физически находящихся внутри радиуса R на момент t_now).

        Источник в этой модели приближённо считается зафиксированным в сопутствующей
        координате χ_initial = r_initial / a(t_start). Фотон, испущенный в момент τ,
        находится в момент t_now в χ(τ) = χ_initial − c·[η(t_now) − η(τ)] и
        несёт энергию E = (P·dτ)·a(τ)/a_now (single-photon redshift).

        absorption_radius_m — радиус поглощения, обычно горизонт ЧД. Фотоны,
        уже пересёкшие этот радиус, считаются поглощёнными ЦЧД и НЕ входят
        в in-flight массу.

        Возвращает:
            dict с ключами 'eta_min', 'eta_max', 'chi_initial', 'eta_now',
            'a_now', 'thrust_power_per_kg', либо None — если лазер ещё не включался,
            нет валидных точек, или удельная мощность = 0.
        """
        if scale_factor <= 0 or universe_time_seconds is None:
            return None
        if self.t_laser_start_seconds is None or self._laser_start_r_phys is None:
            return None

        thrust_spec = float(getattr(config, 'MATTER_THRUST_POWER_PER_KG_W', 0.0))
        if thrust_spec <= 0.0:
            return None

        t_now_yr = float(universe_time_seconds) / SECONDS_PER_YEAR
        t_start_yr = float(self.t_laser_start_seconds) / SECONDS_PER_YEAR
        if t_now_yr <= t_start_yr:
            return None

        a_start = float(_scale_factor_lookup(np.array([t_start_yr]))[0])
        eta_now = float(_conformal_time_lookup(np.array([t_now_yr]))[0])
        eta_start = float(_conformal_time_lookup(np.array([t_start_yr]))[0])

        # Лазер действительно стартовал только для точек с зафиксированным r_initial > 0.
        # Без второго условия NaN-ы и нули могли бы вырождать интеграл и давать
        # ложный «мгновенный» вклад на первых кадрах.
        valid = (~np.isnan(self._laser_start_r_phys)) & (self._laser_start_r_phys > 0.0)
        if not np.any(valid):
            return None

        r_init_all = self._laser_start_r_phys[valid]
        chi_initial = r_init_all / max(a_start, 1e-30)

        chi_absorb = max(float(absorption_radius_m), 0.0) / float(scale_factor)
        travel_to_absorb = np.maximum(chi_initial - chi_absorb, 0.0)
        eta_arrived = eta_now - travel_to_absorb / c
        eta_min = np.maximum(eta_arrived, eta_start)

        if (
            self._laser_off_t_seconds is not None
            and len(self._laser_off_t_seconds) == len(self._laser_start_r_phys)
        ):
            t_off_arr = self._laser_off_t_seconds[valid]
        else:
            t_off_arr = np.full(len(r_init_all), np.nan)

        eta_max = np.full_like(chi_initial, eta_now)
        is_off = ~np.isnan(t_off_arr)
        if np.any(is_off):
            t_off_yr = t_off_arr[is_off] / SECONDS_PER_YEAR
            eta_off = _conformal_time_lookup(t_off_yr)
            eta_max[is_off] = eta_off

        return {
            'eta_min': eta_min,
            'eta_max': eta_max,
            'chi_initial': chi_initial,
            'r_initial_m': r_init_all,
            'eta_now': eta_now,
            'a_now': float(scale_factor),
            'thrust_power_per_kg': thrust_spec,
            'black_hole_mass_kg': float(getattr(self, 'accumulated_bh_mass', 0.0)),
        }

    def in_flight_laser_mass_inside_radii(
        self,
        radii_m,
        scale_factor: float,
        precomputed_state=None,
    ) -> np.ndarray:
        """
        Масса-эквивалент E/c² фотонов лазерных лучей, находящихся в полёте
        ВНУТРИ каждого из переданных физических радиусов на момент t_now.

        Считается одним np.searchsorted по уже отсортированным радиусам
        фотонных пакетов (precomputed_state = build_in_flight_laser_mass_state),
        что позволяет получать массу для всех 4 космологических горизонтов
        одним вызовом за O((K+M) log K), где K — число пакетов фотонов,
        M — число радиусов.

        Args:
            radii_m: Последовательность физических радиусов (м).
            scale_factor: a(t_now).
            precomputed_state: Опциональный кэш build_in_flight_laser_mass_state.

        Returns:
            np.ndarray (float64) той же длины, что и radii_m: масса-эквивалент
            (кг) внутри каждого радиуса. Невалидные / нулевые радиусы дают 0.
        """
        radii = np.asarray(radii_m, dtype=np.float64)
        out = np.zeros_like(radii)
        if scale_factor <= 0.0 or len(self._laser_photon_chi) == 0:
            return out
        state = (
            precomputed_state
            if precomputed_state is not None
            else self.build_in_flight_laser_mass_state(scale_factor)
        )
        if state is None:
            return out
        r_sorted = state.get('r_sorted')
        cum_mass = state.get('cum_mass')
        if r_sorted is None or cum_mass is None or len(r_sorted) == 0:
            return out
        valid = np.isfinite(radii) & (radii > 0.0)
        if not np.any(valid):
            return out
        idx = np.searchsorted(r_sorted, radii, side='right')
        has_inside = (idx > 0) & valid
        if np.any(has_inside):
            out[has_inside] = cum_mass[idx[has_inside] - 1]
        return out

    def build_in_flight_laser_mass_state(self, scale_factor: float):
        """
        Предрассчитать отсортированные радиусы и кумулятивные массы фотонов.

        Это позволяет получить массу внутри нескольких горизонтов за O(log N)
        на каждый радиус вместо повторного полного прохода по пакетам фотонов.
        """
        if scale_factor <= 0.0 or len(self._laser_photon_chi) == 0:
            return None

        r_photon = self._laser_photon_chi * float(scale_factor)
        valid = np.isfinite(r_photon) & (r_photon > 0.0)
        if not np.any(valid):
            return None

        r_valid = r_photon[valid]
        photon_mass = cosmological_redshifted_photon_mass_kg(
            self._laser_photon_mass_emit_kg[valid],
            self._laser_photon_a_emit[valid],
            scale_factor,
        )
        order = np.argsort(r_valid)
        r_sorted = r_valid[order]
        cum_mass = np.cumsum(photon_mass[order])
        return {
            'r_sorted': r_sorted,
            'cum_mass': cum_mass,
        }
    
    def update_positions_and_velocities(
        self,
        dt: float,
        scale_factor: float,
        scale_ratio: float,
        r_black_hole: float = None,
        universe_time_seconds: float = None,
        r_event_horizon: float = None,
    ):
        """
        Обновить позиции точек на основе их скоростей и пересчитать скорости.

        Гибридная модель: ΛCDM/FRW-фон + полная Schwarzschild-de Sitter
        радиальная геодезика для ЦЧД-локальной физики. Λ-член остаётся
        в FRW (через a(t) и Hubble drag), и НЕ дублируется в локальном
        f(r) ЦЧД, поэтому Schwarzschild-часть SdS даёт f(r) = 1 − r_s/r.

        Все формулы строго самосогласованы (см. блок-комментарий перед
        радиальной динамикой), s — собственная Лоренц-инвариантная удельная
        мощность лазера в Вт/кг:
          • d(γv)/dt = (s/c)·η − (GM/r²)·√f/γ  — SdS-радиальная геодезика
            пробной частицы в координатном времени; в пределе r≫r_s, v≪c
            восстанавливается −GM/r².
          • dm_rest/dt_lab = −s·m_rest·η·√f/(γc²)  — полная дилатация
            (SR через 1/γ + GR через √f).
          • E_photon_∞/c² = γ(1−β_r)·dm_rest·√f(r_emit)  — координатная
            энергия фотона (Schwarzschild conserved energy along null geodesic),
            учитывает SR-Doppler от движения источника (γ(1−β_r): redshift
            для убегающей точки β_r>0, blueshift для β_r<0) и gravitational
            переход от локальной энергии к «энергии на бесконечности» √f(r_emit).
          • η ∈ [0,1] — доля dt, в которую лазер реально излучает (η<1 если
            burnable mass закончилась до конца dt);
          • поглощаемая ЦЧД масса с космологическим redshift по пути:
            dM_bh_arrival = (E_phot_∞/c²) · a(t_emit)/a(t_arrival).
            Gravitational blueshift у горизонта учтён АВТОМАТИЧЕСКИ, так как
            мы храним именно E_∞ — она инвариантна вдоль нулевой SdS-геодезики
            и равна ADM-вкладу в массу ЦЧД.

        Космологическое трение (Hubble drag): между шагами a(t) меняется, и
        peculiar momentum точки масштабируется как p_pec ∝ 1/a (стандартный
        результат геодезической массивной частицы в FRW: a·γ·v_pec = const).
        В нерелятивистском пределе это даёт v_pec ∝ 1/a. Применяется операторным
        расщеплением: сначала Hubble drag по Δa, потом локальная динамика сил.

        Args:
            dt: Шаг времени в секундах
            scale_factor: Текущий масштабный фактор Вселенной a(t_now)
            scale_ratio: Коэффициент роста горизонта частиц (не используется тут)
            r_black_hole: Радиус горизонта черной дыры в метрах (опционально)
            universe_time_seconds: Космологическое время сейчас, нужно для
                расчёта redshift фотонов в полёте до центра.
        """
        if dt < 0:
            self._last_step_nominal_sigma_p_w = 0.0
            self._advance_and_absorb_laser_photons(
                scale_factor, universe_time_seconds, r_black_hole
            )
            return

        if self.velocities_comoving is None or len(self.velocities_comoving) == 0:
            self._last_step_nominal_sigma_p_w = 0.0
            return

        previous_scale_factor = getattr(self, '_previous_scale_factor', scale_factor)
        if (
            scale_factor > 0.0
            and previous_scale_factor > 0.0
            and abs(previous_scale_factor - scale_factor) > 1e-10
        ):
            # Hubble drag: для свободной массивной частицы в FRW-метрике
            # сохраняется a·γ·v_pec = const (геодезическая, p_χ-инвариант).
            # Поэтому peculiar momentum p_phys = γ·v_pec затухает как 1/a:
            #     p_phys_new = p_phys_old · (a_old/a_new),
            # и в нерелятивистском пределе v_pec ∝ 1/a. Реализуем релятивистски
            # через импульс, чтобы корректно работать и при v→c (например,
            # после длительной лазерной тяги).
            ratio_a = previous_scale_factor / scale_factor
            v_phys_old = self.velocities_comoving * previous_scale_factor
            v2 = np.einsum('ij,ij->i', v_phys_old, v_phys_old)
            beta2 = np.clip(v2 / (c * c), 0.0, 1.0 - 1e-12)
            gamma = 1.0 / np.sqrt(1.0 - beta2)
            p_phys = gamma[:, np.newaxis] * v_phys_old
            p_phys *= ratio_a
            p2_over_c2 = np.einsum('ij,ij->i', p_phys, p_phys) / (c * c)
            gamma_new = np.sqrt(1.0 + p2_over_c2)
            v_phys_new = p_phys / gamma_new[:, np.newaxis]
            self.velocities_comoving[:] = v_phys_new / scale_factor
        self._previous_scale_factor = scale_factor

        self._ensure_inside_bh_mask()

        def mark_laser_stopped_for_indices(indices, radii_by_index=None):
            """Остановить дальнейшую эмиссию у источников, поглощённых ЧД.

            indices — array-like глобальных индексов (np.ndarray или list).
            radii_by_index — массив длины N с физ. расстояниями (может быть None).
            """
            if universe_time_seconds is None:
                return
            n_idx = 0 if indices is None else len(indices)
            if n_idx == 0:
                return
            n_total = len(self.points_comoving)
            if self._laser_off_t_seconds is None or len(self._laser_off_t_seconds) != n_total:
                new_t = np.full(n_total, np.nan, dtype=np.float64)
                new_r = np.full(n_total, np.nan, dtype=np.float64)
                if self._laser_off_t_seconds is not None:
                    old_t = self._laser_off_t_seconds
                    old_r = self._laser_off_r_phys
                    k = min(len(old_t), n_total)
                    new_t[:k] = old_t[:k]
                    if old_r is not None:
                        new_r[:k] = old_r[:k]
                self._laser_off_t_seconds = new_t
                self._laser_off_r_phys = new_r

            idx_arr = np.asarray(indices, dtype=np.int64).ravel()
            valid = (idx_arr >= 0) & (idx_arr < n_total)
            idx_arr = idx_arr[valid]
            if idx_arr.size == 0:
                return
            fresh = np.isnan(self._laser_off_t_seconds[idx_arr])
            fresh_idx = idx_arr[fresh]
            if fresh_idx.size == 0:
                return
            self._laser_off_t_seconds[fresh_idx] = float(universe_time_seconds)
            if radii_by_index is not None:
                self._laser_off_r_phys[fresh_idx] = np.asarray(radii_by_index, dtype=np.float64)[fresh_idx]
            else:
                self._laser_off_r_phys[fresh_idx] = 0.0

        self._advance_and_absorb_laser_photons(
            scale_factor, universe_time_seconds, r_black_hole
        )
        self._last_step_nominal_sigma_p_w = 0.0

        # ОПТИМИЗАЦИЯ: physical_pts и dist считаются ОДИН раз и переиспользуются
        # ниже (детекция поглощений, радиальная динамика, эмиссия фотонов).
        # Раньше это было 3 одинаковых прохода по N точкам подряд.
        physical_pts = self.points_comoving * scale_factor
        dist = np.sqrt(np.einsum('ij,ij->i', physical_pts, physical_pts))

        # 1) Точки, попавшие в ЦЧД В ЭТОМ ШАГЕ.
        # Релятивистский учёт энергии: ЦЧД получает E/c² = γ·m_rest, где
        # γ = 1/√(1-β²) для пекулярной скорости точки в момент поглощения.
        if r_black_hole is not None and r_black_hole > 0:
            newly_inside = (dist <= r_black_hole) & ~self.inside_bh_mask
            if np.any(newly_inside):
                new_indices = np.flatnonzero(newly_inside)
                mark_laser_stopped_for_indices(new_indices, dist)
                if (
                    self.masses_per_point is not None
                    and self.velocities_comoving is not None
                ):
                    v_phys_in = self.velocities_comoving[new_indices] * scale_factor
                    v2_in = np.einsum('ij,ij->i', v_phys_in, v_phys_in)
                    beta2_in = np.clip(v2_in / (c * c), 0.0, 1.0 - 1e-12)
                    gamma_in = 1.0 / np.sqrt(1.0 - beta2_in)
                    valid = new_indices < len(self.masses_per_point)
                    if np.any(valid):
                        m_rest = self.masses_per_point[new_indices[valid]].astype(
                            np.float64, copy=False
                        )
                        self.accumulated_bh_mass += float(np.sum(gamma_in[valid] * m_rest))
                self._mark_inside_bh_vectorized(newly_inside)

        # 2) Все поглощённые точки (новые + старые) перемещаются в центр.
        # Векторное обнуление вместо Python-цикла по set().
        if self.inside_bh_mask.any():
            self.points_comoving[self.inside_bh_mask] = 0.0

        # 3) mask_outside_bh используется и в радиальной динамике, и в шаге позиций.
        mask_outside_bh = ~self.inside_bh_mask

        # Радиальная динамика (знаковая, + наружу) с полной Schwarzschild-de Sitter
        # геодезикой ЦЧД, наложенной на FRW-фон (гибрид PN+FRW). Λ-член НЕ входит
        # в локальное f(r): его эффект уже учтён через эволюцию a(t) и Hubble drag
        # (двойного учёта быть не должно, как в классическом McVittie weak-field).
        #
        # Локальный лапс ЦЧД:
        #     f(r) = 1 − r_s/r,   r_s = 2·G·M_BH/c²   (Schwarzschild часть SdS)
        # √f(r) — это `lapse function` static observer'а: dt_local = √f·dt_coord,
        # E_coord_∞ = E_local·√f. На горизонте f → 0 (ставится численный floor).
        #
        # Уравнения:
        #   • Точка (массивная пробная частица) в радиальном движении, координатное t:
        #         d(γv)/dt = (s/c)·η − (GM/r²)·√f/γ,
        #     где γ = γ_local = 1/√(1−v_pec²/c²) (как в плоском пределе),
        #     η ∈ [0,1] — доля dt, в которую лазер реально излучает,
        #     s — собственная (Лоренц-инвариантная) удельная мощность лазера.
        #     Гравитационный член — это полная SdS-радиальная геодезика для
        #     пробной частицы в координатном времени; в пределе r≫r_s, v≪c
        #     восстанавливается ньютоновское −GM/r². У горизонта (f→0) или
        #     для ультра-релятивистских частиц (γ→∞) гравитационное «торможение»
        #     ослабевает — это известное свойство SdS-геодезик.
        #   • dm_rest/dt_lab = −s·m·η/(γc²)·√f (полная дилатация: SR + GR).
        #   • Координатная энергия фотона на эмиссии (E_∞ = E_local·√f):
        #         E_phot_∞/c² = γ(1−β_r)·dm_rest_emitted·√f(r_emit).
        #     E_∞ инвариантна вдоль нулевой SdS-геодезики (gravitational
        #     blueshift при подлёте к ЦЧД учтён автоматически: static observer
        #     near horizon видит E_local = E_∞/√f → ∞, но в ADM-массу ЦЧД
        #     добавляется именно E_∞/c²).
        #   • При поглощении ЦЧД: dM_bh = (E_phot_∞/c²)·a(t_emit)/a(t_now)
        #     с учётом дополнительного космологического redshift по пути.
        thrust_spec = float(getattr(config, 'MATTER_THRUST_POWER_PER_KG_W', 0.0))
        M_bh_total = float(getattr(self, 'accumulated_bh_mass', 0.0))
        any_force = (thrust_spec > 0.0) or (M_bh_total > 0.0)
        if any_force and scale_factor > 0.0:
            radial_ok = mask_outside_bh & (dist > 1e-10)
            if np.any(radial_ok):
                # Единичный вектор НАРУЖУ от центра (только для radial_ok).
                u_out = np.zeros_like(physical_pts)
                inv_d = np.zeros_like(dist)
                inv_d[radial_ok] = 1.0 / dist[radial_ok]
                u_out[radial_ok] = physical_pts[radial_ok] * inv_d[radial_ok, np.newaxis]

                # Тяга есть только пока есть сжигаемая масса выше остатка,
                # заданного КПД преобразования массы в фотоны.
                thrust_active = np.zeros(len(physical_pts), dtype=bool)
                laser_mass_floor_arr = None
                burnable_mass = None
                if thrust_spec > 0.0 and self.masses_per_point is not None:
                    masses_current = np.asarray(self.masses_per_point, dtype=np.float64)
                    n_total = len(masses_current)
                    if (
                        self._laser_start_mass_kg is None
                        or len(self._laser_start_mass_kg) != n_total
                    ):
                        new_start_mass = np.full(n_total, np.nan, dtype=np.float64)
                        if self._laser_start_mass_kg is not None:
                            old = self._laser_start_mass_kg
                            new_start_mass[: min(len(old), n_total)] = old[: min(len(old), n_total)]
                        self._laser_start_mass_kg = new_start_mass

                    le = self.laser_emitter_mask
                    emitter_mask = np.zeros(n_total, dtype=bool)
                    if le is not None and len(le) == n_total:
                        emitter_mask = np.asarray(le, dtype=bool)

                    if r_event_horizon is not None and r_event_horizon > 0:
                        outside_eh = (dist > r_event_horizon) & emitter_mask
                        if np.any(outside_eh):
                            emitter_mask &= ~outside_eh
                            stopped_indices = np.flatnonzero(outside_eh)
                            mark_laser_stopped_for_indices(stopped_indices, dist)

                    fresh_laser = radial_ok & emitter_mask & np.isnan(self._laser_start_mass_kg)
                    if np.any(fresh_laser):
                        self._laser_start_mass_kg[fresh_laser] = masses_current[fresh_laser]

                    laser_remaining_frac = float(getattr(config, 'MATTER_LASER_REMAINING_FRACTION', 1e-4))
                    efficiency = float(
                        np.clip(1.0 - laser_remaining_frac, 0.0, 1.0)
                    )
                    laser_mass_floor_arr = laser_mass_floor(
                        self._laser_start_mass_kg,
                        efficiency,
                    )
                    burnable_mass = burnable_mass_kg(masses_current, laser_mass_floor_arr)
                    thrust_active = radial_ok & emitter_mask & (burnable_mass > 0.0)
                    if thrust_spec > 0.0 and np.any(thrust_active):
                        self._last_step_nominal_sigma_p_w = float(
                            thrust_spec * float(np.sum(masses_current[thrust_active]))
                        )

                # Релятивистский расчёт скоростей и β_radial для всех точек.
                # Считаем здесь, чтобы (а) использовать γ для фактора 1/γ в темпе
                # выгорания, (б) использовать β_r для Doppler-сдвига энергии
                # фотона при эмиссии, (в) применить тот же γ при обновлении p_phys
                # ниже без повторного вычисления.
                v_phys = self.velocities_comoving * scale_factor
                v2 = np.einsum('ij,ij->i', v_phys, v_phys)
                beta2 = np.clip(v2 / (c * c), 0.0, 1.0 - 1e-12)
                gamma = 1.0 / np.sqrt(1.0 - beta2)
                # β_radial: компонента β источника в направлении +r̂ (signed).
                # Положительная если точка движется наружу, отрицательная — внутрь.
                beta_radial = np.zeros(len(physical_pts), dtype=np.float64)
                if np.any(radial_ok):
                    beta_radial[radial_ok] = np.einsum(
                        'ij,ij->i', v_phys[radial_ok], u_out[radial_ok]
                    ) / c

                # Schwarzschild lapse √f(r) для гравитации ЦЧД (Λ — через FRW,
                # сюда не входит, чтобы не дублировать). Floor около горизонта,
                # чтобы избежать бесконечного blueshift и/или 0 в ускорении:
                # √f_floor = 0.01 (соответствует r ≳ r_s·1.0001). Точки внутри
                # r_s всё равно отсекаются ранее по dist <= r_black_hole.
                if M_bh_total > 0.0:
                    r_s_bh = 2.0 * G * M_bh_total / (c * c)
                    f_floor = 1.0e-4
                    r_for_f = np.maximum(dist, r_s_bh / (1.0 - f_floor))
                    f_lapse = np.maximum(1.0 - r_s_bh / r_for_f, f_floor)
                    sqrt_f = np.sqrt(f_lapse)
                else:
                    sqrt_f = np.ones(len(physical_pts), dtype=np.float64)

                # Запрос потери массы покоя за лаб. dt с полной дилатацией:
                #   dm_request = s·m·dt·√f/(γc²)
                # (1/γ — SR, √f — gravitational time dilation у источника).
                # Если burnable_mass < dm_request, лазер реально работает только
                # часть dt; effective_fraction = emitted/request ∈ (0, 1].
                emitted_dm_active = None
                effective_fraction_thrust = None
                if (
                    thrust_spec > 0.0
                    and self.masses_per_point is not None
                    and np.any(thrust_active)
                    and burnable_mass is not None
                ):
                    masses_thrust = np.asarray(
                        self.masses_per_point, dtype=np.float64
                    )[thrust_active]
                    emitted_dm_active, effective_fraction_thrust = emitted_rest_mass_for_step(
                        thrust_spec,
                        masses_thrust,
                        dt,
                        gamma[thrust_active],
                        sqrt_f[thrust_active],
                        burnable_mass[thrust_active],
                    )

                # Знаковое радиальное ускорение: +наружу (тяга), -к центру (гравитация).
                a_radial = np.zeros(len(physical_pts), dtype=np.float64)
                if np.any(thrust_active) and effective_fraction_thrust is not None:
                    # d(γv)/dt = (s/c)·η — собственное ускорение, масштабированное
                    # эффективной долей dt. Лаб. dp/dt при этом получает фактор
                    # (1−β_r) автоматически через согласованный 1/γ-расход массы.
                    a_radial[thrust_active] += thrust_delta_gamma_v_per_second(
                        thrust_spec,
                        effective_fraction_thrust,
                    )
                if M_bh_total > 0.0:
                    r_floor = float(r_black_hole) if (r_black_hole is not None and r_black_hole > 0) else 1e-10
                    # SdS-радиальное ускорение пробной частицы в координатном
                    # времени:  d(γv)/dt = −(GM/r²)·√f/γ. В пределе r≫r_s, v≪c
                    # → классическое −GM/r². У горизонта (√f→0) и для
                    # ультра-релятивистских частиц (γ→∞) гравитационное
                    # «торможение» в координатах ослабевает — это полная
                    # SdS-геодезика, а не нерелятивистский McVittie-предел.
                    d_safe = np.maximum(dist[radial_ok], r_floor)
                    sqrt_f_radial = sqrt_f[radial_ok]
                    gamma_radial = gamma[radial_ok]
                    a_radial[radial_ok] -= (
                        (G * M_bh_total) / (d_safe * d_safe)
                        * sqrt_f_radial / gamma_radial
                    )

                p_phys = gamma[:, np.newaxis] * v_phys

                # Импульс добавляем только на radial_ok-подмножестве, без аллокации dp размера (N,3).
                p_phys[radial_ok] += (a_radial[radial_ok] * dt)[:, np.newaxis] * u_out[radial_ok]

                p2_over_c2 = np.einsum('ij,ij->i', p_phys, p_phys) / (c * c)
                gamma_new = np.sqrt(1.0 + p2_over_c2)
                v_phys_new = p_phys / gamma_new[:, np.newaxis]

                self.velocities_comoving[:] = v_phys_new / scale_factor

                # Применяем потерю массы покоя, используя уже посчитанный
                # релятивистский emitted_dm_active (с фактором 1/γ).
                if (
                    thrust_spec > 0.0
                    and self.masses_per_point is not None
                    and np.any(thrust_active)
                    and emitted_dm_active is not None
                ):
                    masses_arr = np.asarray(self.masses_per_point, dtype=np.float64)
                    if laser_mass_floor_arr is None or len(laser_mass_floor_arr) != len(masses_arr):
                        laser_mass_floor_arr = np.zeros(len(masses_arr), dtype=np.float64)
                    burnable_now = burnable_mass[thrust_active]
                    was_above_floor = burnable_now > 0.0
                    emitted_dm = emitted_dm_active
                    masses_arr[thrust_active] -= emitted_dm
                    masses_arr[thrust_active] = np.maximum(
                        masses_arr[thrust_active],
                        laser_mass_floor_arr[thrust_active],
                    )
                    self.masses_per_point = masses_arr

                    # Зафиксировать момент и физ. позицию точки в момент,
                    # когда её m_rest впервые опустился до 0. Эти данные
                    # нужны, чтобы рендер мог рисовать «уходящий хвост» уже
                    # испущенного луча после выгорания источника.
                    n_total = len(self.points_comoving)
                    if (
                        self._laser_off_t_seconds is None
                        or len(self._laser_off_t_seconds) != n_total
                    ):
                        new_t = np.full(n_total, np.nan, dtype=np.float64)
                        new_r = np.full(n_total, np.nan, dtype=np.float64)
                        if self._laser_off_t_seconds is not None:
                            old_t = self._laser_off_t_seconds
                            old_r = self._laser_off_r_phys
                            k = min(len(old_t), n_total)
                            new_t[:k] = old_t[:k]
                            new_r[:k] = old_r[:k]
                        self._laser_off_t_seconds = new_t
                        self._laser_off_r_phys = new_r
                    burnable_after = np.zeros(len(masses_arr), dtype=np.float64)
                    burnable_after[thrust_active] = np.maximum(
                        masses_arr[thrust_active] - laser_mass_floor_arr[thrust_active],
                        0.0,
                    )
                    just_off = np.zeros(len(masses_arr), dtype=bool)
                    just_off[thrust_active] = was_above_floor & (burnable_after[thrust_active] <= 0.0)
                    just_off &= np.isnan(self._laser_off_t_seconds)
                    if np.any(just_off) and universe_time_seconds is not None:
                        self._laser_off_t_seconds[just_off] = float(universe_time_seconds)
                        self._laser_off_r_phys[just_off] = dist[just_off]

                    # «Время включения лазера» фиксируем при первой эмиссии. Также
                    # для каждой точки запоминаем её исходное физ. расстояние r_initial:
                    # голова первого фотона стартует ИМЕННО оттуда. Условие долёта
                    # головы до центра: c·(t_now − t_laser_start) ≥ r_initial,
                    # независимо от того, куда успела улететь сама точка.
                    if self.t_laser_start_seconds is None and universe_time_seconds is not None:
                        self.t_laser_start_seconds = float(universe_time_seconds)

                    n_total = len(self.points_comoving)
                    if self._laser_start_r_phys is None or len(self._laser_start_r_phys) != n_total:
                        new_arr = np.full(n_total, np.nan, dtype=np.float64)
                        if self._laser_start_r_phys is not None:
                            old = self._laser_start_r_phys
                            new_arr[: min(len(old), n_total)] = old[: min(len(old), n_total)]
                        self._laser_start_r_phys = new_arr
                    fresh_thrust = thrust_active & np.isnan(self._laser_start_r_phys)
                    if np.any(fresh_thrust):
                        self._laser_start_r_phys[fresh_thrust] = dist[fresh_thrust]

                    # Дискретный фотонный пакет за этот dt. Хранится
                    # КООРДИНАТНАЯ масса-эквивалент E_∞/c² (Schwarzschild
                    # conserved energy along null geodesic):
                    #     E_phot_∞/c² = γ·(1−β_r)·dm_rest·√f(r_emit).
                    # Здесь γ(1−β_r) — релятивистский Doppler от движения
                    # источника, √f(r_emit) — gravitational переход
                    # от локальной энергии static observer'а в r_emit
                    # к энергии «на бесконечности» (ADM-эквивалент).
                    # Cosmological redshift по a(t) применяется при
                    # поглощении ЦЧД — см. _advance_and_absorb_laser_photons.
                    active_indices = np.where(thrust_active)[0]
                    if len(active_indices) > 0:
                        mass_emit_coord = emitted_photon_mass_coord_kg(
                            emitted_dm,
                            gamma[active_indices],
                            beta_radial[active_indices],
                            sqrt_f[active_indices],
                        )
                        self._append_laser_photons(
                            dist[active_indices] / float(scale_factor),
                            mass_emit_coord,
                            scale_factor,
                            dist[active_indices],
                            u_out[active_indices, 0],
                            u_out[active_indices, 1],
                            active_indices.astype(np.int64, copy=False),
                            float(universe_time_seconds),
                        )

        # 4) Шаг позиций для точек ВНЕ ЦЧД с защитой от перескока через центр.
        # Используем уже посчитанные physical_pts/dist как «before»: для outside-точек
        # они валидны (мы не меняли их points_comoving), для inside — отфильтрованы.
        if mask_outside_bh.any():
            physical_points_before = physical_pts[mask_outside_bh]
            distances_before = dist[mask_outside_bh]

            displacement = self.velocities_comoving[mask_outside_bh] * dt
            self.points_comoving[mask_outside_bh] += displacement

            physical_points_after = self.points_comoving[mask_outside_bh] * scale_factor
            distances_after = np.sqrt(
                np.einsum('ij,ij->i', physical_points_after, physical_points_after)
            )

            # Перескок через центр: точка летела К центру (v·r < 0 в физ. координатах),
            # но за шаг dt оказалась дальше, чем была. Точки, удаляющиеся естественно
            # (v·r ≥ 0, например под действием лазерной отдачи), к перескоку не относятся.
            v_phys_subset = self.velocities_comoving[mask_outside_bh] * scale_factor
            v_dot_r_before = np.einsum('ij,ij->i', v_phys_subset, physical_points_before)
            mask_overshot = (distances_after > distances_before) & (v_dot_r_before < 0.0)

            if mask_overshot.any():
                global_indices_overshot = np.flatnonzero(mask_outside_bh)[
                    np.flatnonzero(mask_overshot)
                ]
                # Релятивистский учёт энергии при поглощении: ЦЧД получает γ·m_rest.
                self.points_comoving[global_indices_overshot] = 0.0

                new_overshot = global_indices_overshot[
                    ~self.inside_bh_mask[global_indices_overshot]
                ]
                mark_laser_stopped_for_indices(new_overshot)
                if (
                    new_overshot.size > 0
                    and self.masses_per_point is not None
                    and self.velocities_comoving is not None
                ):
                    v_phys_o = self.velocities_comoving[new_overshot] * scale_factor
                    v2_o = np.einsum('ij,ij->i', v_phys_o, v_phys_o)
                    beta2_o = np.clip(v2_o / (c * c), 0.0, 1.0 - 1e-12)
                    gamma_o = 1.0 / np.sqrt(1.0 - beta2_o)
                    valid = new_overshot < len(self.masses_per_point)
                    if np.any(valid):
                        m_rest_o = self.masses_per_point[new_overshot[valid]].astype(
                            np.float64, copy=False
                        )
                        self.accumulated_bh_mass += float(np.sum(gamma_o[valid] * m_rest_o))

                if new_overshot.size > 0:
                    overshot_mask = np.zeros_like(self.inside_bh_mask)
                    overshot_mask[new_overshot] = True
                    self._mark_inside_bh_vectorized(overshot_mask)

        self._previous_scale_factor = scale_factor

        # ОПТИМИЗАЦИЯ: Пересчитываем comoving distances после движения точек.
        self._update_comoving_distances()
        # ОПТИМИЗАЦИЯ: Пересчитываем v² и его max — это позволяет
        # MassCalculator избегать полного einsum по (N,3) при γ-коррекции
        # и проверять порог _BETA2_NEGLIGIBLE_THRESHOLD (mass_calculator) за O(1).
        self._recompute_velocity_norms()
        # masses_per_point могли быть «прожжены» лазером в этом шаге — поднимаем
        # версию (дёшево; служит ключом инвалидации внешних кэшей m_eff).
        self._bump_masses_version()

    def add_points(self, new_points: np.ndarray, scale_factor: float = None, scale_ratio: float = None):
        """
        Добавить новые точки к существующим.
        
        Args:
            new_points: Массив новых точек (N x 3)
            scale_factor: Текущий масштабный фактор (оставлен для совместимости вызовов)
            scale_ratio: Коэффициент роста горизонта частиц (оставлен для совместимости вызовов)
        """
        # ВАЖНО: Новые точки уже перемешаны в matter_simulation после генерации
        # Не нужно перемешивать их снова или перемешивать все точки вместе
        # Это предотвращает визуальные артефакты (точки не будут "прыгать" на экране)
        
        if len(new_points) == 0:
            return
        
        new_velocities = np.zeros((len(new_points), 3), dtype=np.float64)
        
        # Инициализируем массы для новых точек
        new_masses = np.full(len(new_points), get_mass_per_point_kg(), dtype=np.float64)
        
        # Добавляем точки, скорости и массы
        if self.points_comoving is None or len(self.points_comoving) == 0:
            self.points_comoving = new_points
            self.velocities_comoving = new_velocities
            self.masses_per_point = new_masses
        else:
            self.points_comoving = np.vstack([self.points_comoving, new_points])
            if self.velocities_comoving is None:
                self.velocities_comoving = new_velocities
            else:
                self.velocities_comoving = np.vstack([self.velocities_comoving, new_velocities])
            if self.masses_per_point is None:
                self.masses_per_point = new_masses
            else:
                self.masses_per_point = np.concatenate([self.masses_per_point, new_masses])
        
        self._set_laser_emitter_mask_inside_event_horizon()

        # inside_bh_mask должен расти вместе с points_comoving (новые точки → False).
        self._ensure_inside_bh_mask()

        # ОПТИМИЗАЦИЯ: Пересчитываем comoving distances и нормы скоростей
        # после добавления точек, поднимаем версию масс — внешние кэши
        # (MassCalculator) корректно инвалидируются.
        self._update_comoving_distances()
        self._recompute_velocity_norms()
        self._bump_masses_version()
