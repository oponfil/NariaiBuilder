"""
Визуализация симуляции с помощью pygame
"""

import math
import time

import numpy as np
import pygame

import config
from config import DEBUG
import visualization.ui as ui
from physics.matter_simulation import MatterSimulation
from physics.mass_calculator import MassCalculator
from visualization.input_handler import InputHandler
from visualization.info_panel import InfoPanel
from visualization.horizons_renderer import HorizonsRenderer
from utils.constants import CAUSAL_HORIZON_COMOVING_METERS, NARIAI_BLACK_HOLE_MASS_KG
from utils.config_utils import (
    get_one_billion_ly,
    get_ten_billion_ly,
    get_pixel_to_meter,
    is_comoving_display,
)

# Внутренние константы для photon_rgb_blue_green_red
_PHOTON_LASER_RGB_CHANNEL_MAX = 255.0
_PHOTON_LASER_COLOR_P_MIDPOINT = 0.5
# p = u^γ, u = a_emit/a_now. γ=1 — равные половины по u; γ<1 растягивает синий↔зелёный по u.
_MATTER_LASER_PHOTON_COLOR_GAMMA = 0.8


def photon_rgb_blue_green_red(factor_a_emit_over_now: np.ndarray) -> np.ndarray:
    """
    Цвет по космологическому покраснению: u = a_emit / a_now.

    Два линейных участка в параметре p = u^γ (γ — _MATTER_LASER_PHOTON_COLOR_GAMMA):
      • p ∈ [½, 1]: зелёный → синий (p=1 свежий фотон);
      • p ∈ [0, ½]: красный → зелёный (p=0 сильнее ушедшая энергия).

    При γ<1 диапазон u у «свежих» фотонов занимает большую долю пути синий↔зелёный
    (переход выглядит плавнее), а для малых u половина оттенков красный↔зелёный
    укладывается в узкий интервал по u — оттенки сильнее меняются от шага к шагу.

    Возвращает (N, 3) с RGB в диапазоне [0, 255].
    """
    u = np.clip(np.asarray(factor_a_emit_over_now, dtype=np.float64), 0.0, 1.0).ravel()
    gamma = float(_MATTER_LASER_PHOTON_COLOR_GAMMA)
    p = np.power(u, gamma)
    n = int(p.size)
    r = np.zeros(n)
    g = np.zeros(n)
    b = np.zeros(n)
    half = _PHOTON_LASER_COLOR_P_MIDPOINT
    mx = _PHOTON_LASER_RGB_CHANNEL_MAX
    hi = p >= half
    lo = ~hi
    s_hi = (p[hi] - half) / half
    g[hi] = mx * (1.0 - s_hi)
    b[hi] = mx * s_hi
    s_lo = np.clip(p[lo] / half, 0.0, 1.0)
    r[lo] = mx * (1.0 - s_lo)
    g[lo] = mx * s_lo
    return np.stack([r, g, b], axis=1)


_PHOTON_DISK_OFFSETS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _photon_disk_offsets_int32(radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Целочисленные смещения для заполненного круга радиуса radius_px (как у pygame.draw.circle).

    Для radius_px == 1 используется отдельная ветка «один пиксель» в отрисовке.
    """
    r = int(radius_px)
    if r < 2:
        raise ValueError("disk offsets require radius_px >= 2")
    if r not in _PHOTON_DISK_OFFSETS_CACHE:
        dx_list: list[int] = []
        dy_list: list[int] = []
        r2 = r * r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    dx_list.append(dx)
                    dy_list.append(dy)
        _PHOTON_DISK_OFFSETS_CACHE[r] = (
            np.asarray(dx_list, dtype=np.int32),
            np.asarray(dy_list, dtype=np.int32),
        )
    return _PHOTON_DISK_OFFSETS_CACHE[r]


_MATTER_POINT_SQUARE_OFFSETS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _matter_point_square_offsets_int32(px_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Смещения для квадратного штампа точки материи размера px_size×px_size (как в draw.rect).
    px_size >= 2; при px_size <= 1 штамп не используется.
    """
    ps = int(px_size)
    if ps < 2:
        raise ValueError("square offsets require px_size >= 2")
    if ps not in _MATTER_POINT_SQUARE_OFFSETS_CACHE:
        ox = np.arange(ps, dtype=np.int32)
        dx, dy = np.meshgrid(ox, ox, indexing="ij")
        _MATTER_POINT_SQUARE_OFFSETS_CACHE[ps] = (dx.ravel(), dy.ravel())
    return _MATTER_POINT_SQUARE_OFFSETS_CACHE[ps]


def _write_density_points_to_rgb_a_buffers(
    pix: np.ndarray,
    pix_alpha: np.ndarray,
    x_screen: np.ndarray,
    y_screen: np.ndarray,
    colors_screen: np.ndarray,
    px_size: int,
    width: int,
    height: int,
) -> None:
    """Запись точек плотности в buffers surfarray (RGB + отдельная альфа-плоскость)."""
    if px_size <= 1:
        pix[x_screen, y_screen, 0] = colors_screen[:, 0]
        pix[x_screen, y_screen, 1] = colors_screen[:, 1]
        pix[x_screen, y_screen, 2] = colors_screen[:, 2]
        pix_alpha[x_screen, y_screen] = 255
        return
    dx, dy = _matter_point_square_offsets_int32(px_size)
    xs = x_screen[:, None] + dx[None, :]
    ys = y_screen[:, None] + dy[None, :]
    xs = xs.ravel()
    ys = ys.ravel()
    cols = np.repeat(colors_screen, len(dx), axis=0)
    m = (
        (xs >= 0) & (xs < width) &
        (ys >= 0) & (ys < height)
    )
    if np.any(m):
        pix[xs[m], ys[m], 0] = cols[m, 0]
        pix[xs[m], ys[m], 1] = cols[m, 1]
        pix[xs[m], ys[m], 2] = cols[m, 2]
        pix_alpha[xs[m], ys[m]] = 255


def _write_density_points_to_pixels3d_screen(
    pixels: np.ndarray,
    x_screen: np.ndarray,
    y_screen: np.ndarray,
    colors_screen: np.ndarray,
    px_size: int,
    width: int,
    height: int,
) -> None:
    """Прямая запись в pixels3d экрана (без отдельной альфы). Резервный путь."""
    if px_size <= 1:
        pixels[x_screen, y_screen] = colors_screen
        return
    dx, dy = _matter_point_square_offsets_int32(px_size)
    for k in range(int(dx.shape[0])):
        xk = x_screen + int(dx[k])
        yk = y_screen + int(dy[k])
        m = (xk >= 0) & (xk < width) & (yk >= 0) & (yk < height)
        if not m.any():
            continue
        pixels[xk[m], yk[m]] = colors_screen[m]


def _pack_rgb_for_surface(rgb: np.ndarray, surface: pygame.Surface) -> np.ndarray:
    """
    Упаковать (N,3) uint8 RGB в (N,) uint32 в формате surface (учитывает
    Surface.get_shifts/get_masks). Альфа-канал, если есть, не выставляется —
    экран обычно XRGB8888, и нулевой альфа-байт не виден.
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    shifts = surface.get_shifts()
    r_shift = int(shifts[0])
    g_shift = int(shifts[1])
    b_shift = int(shifts[2])
    return (
        (arr[:, 0].astype(np.uint32) << r_shift)
        | (arr[:, 1].astype(np.uint32) << g_shift)
        | (arr[:, 2].astype(np.uint32) << b_shift)
    )


def _write_points_packed_to_pixels2d(
    pix2d: np.ndarray,
    x_screen: np.ndarray,
    y_screen: np.ndarray,
    packed: np.ndarray,
    offsets_xy,
    width: int,
    height: int,
) -> None:
    """
    Прямая запись точек в pixels2d (uint32) без альфа-смешивания и без blit.
    offsets_xy=None → один пиксель; иначе кортеж (dx, dy) штампа (квадрат/диск).
    Координаты x_screen/y_screen уже считаются в пределах экрана для центра штампа,
    поля штампа фильтруются по границам.
    """
    if x_screen.size == 0:
        return
    if offsets_xy is None:
        pix2d[x_screen, y_screen] = packed
        return
    dx, dy = offsets_xy
    for k in range(int(dx.shape[0])):
        xk = x_screen + int(dx[k])
        yk = y_screen + int(dy[k])
        m = (xk >= 0) & (xk < width) & (yk >= 0) & (yk < height)
        if not m.any():
            continue
        pix2d[xk[m], yk[m]] = packed[m]


class UniverseRenderer:
    """Рендерер для визуализации Вселенной"""
    
    def __init__(
        self,
        width: int = None,
        height: int = None,
        background_color: tuple = None
    ):
        """
        Args:
            width: Ширина окна в пикселях (по умолчанию из config)
            height: Высота окна в пикселях (по умолчанию из config)
            background_color: RGB цвет фона (по умолчанию из config)
        """
        print("[DEBUG] UniverseRenderer.__init__: starting...")
        pygame.init()
        print("[DEBUG] UniverseRenderer.__init__: pygame initialized")
        self.width = width if width is not None else ui.WINDOW_WIDTH
        self.height = height if height is not None else ui.WINDOW_HEIGHT
        self.background_color = background_color if background_color is not None else ui.BACKGROUND_COLOR
        
        # ОПТИМИЗАЦИЯ: Используем SCALED для автоматического GPU ускорения в pygame 2.0+
        try:
            # Пробуем использовать SCALED для GPU ускорения (pygame 2.0+)
            # SCALED автоматически использует аппаратный рендерер SDL2
            self.screen = pygame.display.set_mode((self.width, self.height), 
                                                   pygame.RESIZABLE | pygame.SCALED | pygame.DOUBLEBUF)
            self._using_hw_surface = True
            print("[INFO] Using SCALED mode for GPU acceleration")
        except:
            try:
                # Fallback на HWSURFACE (старый метод)
                self.screen = pygame.display.set_mode((self.width, self.height), 
                                                      pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
                self._using_hw_surface = True
                print("[INFO] Using HWSURFACE mode")
            except:
                # Fallback на software surface, если GPU недоступен
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.SWSURFACE)
                self._using_hw_surface = False
                print("[INFO] Using SWSURFACE mode (CPU only)")
        pygame.display.set_caption(ui.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        
        # Параметры камеры
        self.camera_x = self.width // 2
        self.camera_y = self.height // 2
        self.zoom = 1.0  # Зум не используется, фиксированное значение
        
        # Флаги отображения
        self.show_info = True
        self.show_horizons = True
        
        # Состояние паузы
        self.paused = False
        # На паузе стрелки сдвигают t и a(t); один шаг физики за кадр
        # (иначе step_matter_simulation на паузе выходит сразу).
        self._manual_cosmic_step_this_frame = False
        self._manual_cosmic_dt_signed: float | None = None
        
        # Симуляция материи (все расчеты и логика вынесены сюда)
        print("[DEBUG] UniverseRenderer.__init__: creating MatterSimulation...")
        self.matter_simulation = MatterSimulation()
        print("[DEBUG] UniverseRenderer.__init__: MatterSimulation created")
        
        # Калькулятор масс
        print("[DEBUG] UniverseRenderer.__init__: creating MassCalculator...")
        self.mass_calculator = MassCalculator()
        print("[DEBUG] UniverseRenderer.__init__: MassCalculator created")
        
        # ОПТИМИЗАЦИЯ: Кэш для физических координат и расстояний
        # Пересчитываем только если время изменилось
        self._cached_physical_points = None
        self._cached_distances = None
        self._cached_time = None
        self._cached_scale_factor = None
        self._cached_scale_ratio = None
        self._cached_distances_squared = None

        # ОПТИМИЗАЦИЯ: Кэш результата calculate_masses на уровне кадра.
        # Ключ — (universe.time, accumulated_bh_mass). За один кадр функция
        # зовётся 8 раз (info_panel, horizons_renderer, draw, selection и т.д.),
        # благодаря кэшу 7 из 8 вызовов становятся O(1). На паузе кэш живёт
        # между кадрами; step_matter_simulation сбрасывает его в начале каждого
        # активного кадра, потому что состояние симуляции меняется.
        self._cached_masses = None
        self._cached_masses_key = None
        
        # Пакетная отрисовка точек/фотонов: SRCALPHA-поверхность размера экрана (переиспользуется)
        self._batch_pixel_overlay: pygame.Surface | None = None
        self._batch_pixel_overlay_wh: tuple[int, int] | None = None
        
        # Шрифт для текста
        print("[DEBUG] UniverseRenderer.__init__: creating fonts...")
        self.font = pygame.font.Font(None, ui.FONT_SIZE_LARGE)
        self.small_font = pygame.font.Font(None, ui.FONT_SIZE_SMALL)
        print("[DEBUG] UniverseRenderer.__init__: fonts created")
        
        # Обработчик ввода (выделен в отдельный модуль)
        self.input_handler = InputHandler(self)
        
        # Панель информации (выделена в отдельный модуль)
        self.info_panel = InfoPanel(self)
        
        # Рендерер горизонтов (выделен в отдельный модуль)
        self.horizons_renderer = HorizonsRenderer(self)
        
        print("[DEBUG] UniverseRenderer.__init__: initialization complete!")

    def _black_hole_color_for_mass(self, mass_kg: float) -> tuple:
        """Цвет ЧД: зеленый, когда масса достигла порога Нараи."""
        if float(mass_kg) >= NARIAI_BLACK_HOLE_MASS_KG:
            return ui.HORIZON_BLACK_HOLE_NARIAI_COLOR
        return ui.HORIZON_BLACK_HOLE_COLOR

    def draw_central_black_hole_point(self, masses=None):
        """Всегда рисовать одну центральную точку ЧД-затравки."""
        mass_kg = masses.get('M_black_hole_kg', 0.0) if masses else 0.0
        color = self._black_hole_color_for_mass(mass_kg)
        center = (self.width // 2, self.height // 2)
        size = max(1, int(getattr(config, 'MATTER_POINT_SCREEN_PX', 3)))
        pygame.draw.rect(self.screen, color, (center[0], center[1], size, size))

    def draw_info(self, universe, cosmology, masses, fps: float):
        """Шкала масштаба, горизонты, маркер ЦЧД и основная информационная панель."""
        try:
            self.draw_scale_ruler(cosmology)
        except Exception as e:
            if not getattr(self, "_ruler_warned", False):
                print(f"[WARN] draw_scale_ruler failed: {type(e).__name__}: {e}")
                self._ruler_warned = True
        self.draw_horizons(universe, cosmology, masses)
        self.draw_central_black_hole_point(masses)
        self.draw_info_panel(universe, cosmology, fps, masses=masses)

    # ОБРАТНАЯ СОВМЕСТИМОСТЬ: Property-делегаты для доступа к данным из matter_simulation
    # Это позволяет старому коду работать без изменений
    
    @property
    def matter_points(self):
        """Доступ к объекту MatterPoints из симуляции"""
        return self.matter_simulation.matter_points
    
    @property
    def matter_points_initialized(self):
        """Флаг инициализации точек материи"""
        return self.matter_simulation.matter_points_initialized
    
    @matter_points_initialized.setter
    def matter_points_initialized(self, value):
        """Установка флага инициализации"""
        self.matter_simulation.matter_points_initialized = value
    
    @property
    def current_num_points(self):
        """Текущее количество точек"""
        return self.matter_simulation.current_num_points
    
    @current_num_points.setter
    def current_num_points(self, value):
        """Установка количества точек"""
        self.matter_simulation.current_num_points = value
    
    @property
    def last_added_radius_comoving(self):
        """Радиус последнего добавления точек"""
        return self.matter_simulation.last_added_radius_comoving
    
    @last_added_radius_comoving.setter
    def last_added_radius_comoving(self, value):
        """Установка радиуса последнего добавления"""
        self.matter_simulation.last_added_radius_comoving = value
    
    @property
    def collapse_started(self):
        """Флаг начала коллапса"""
        return self.matter_simulation.collapse_started
    
    @collapse_started.setter
    def collapse_started(self, value):
        """Установка флага коллапса"""
        self.matter_simulation.collapse_started = value
    
    @property
    def last_collapse_time(self):
        """Время последнего обновления коллапса"""
        return self.matter_simulation.last_collapse_time
    
    @last_collapse_time.setter
    def last_collapse_time(self, value):
        """Установка времени последнего обновления"""
        self.matter_simulation.last_collapse_time = value
    
    def world_to_screen(self, world_pos: np.ndarray) -> tuple:
        """Преобразовать мировые координаты в экранные"""
        pixel_to_meter = get_pixel_to_meter()
        x = float(world_pos[0]) / pixel_to_meter * self.zoom + self.camera_x
        y = float(world_pos[1]) / pixel_to_meter * self.zoom + self.camera_y
        return (int(float(x)), int(float(y)))
    
    def _calculate_scale_ratio(self, universe, cosmology, particle_horizon_physical):
        """
        Вычислить scale_ratio (коэффициент роста горизонта частиц).
        ДЕЛЕГИРУЕТ к matter_simulation.
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            particle_horizon_physical: Текущий физический радиус горизонта частиц
        
        Returns:
            scale_ratio: Коэффициент масштабирования
        """
        return self.matter_simulation._calculate_scale_ratio(universe, cosmology, particle_horizon_physical)
    
    def _get_physical_points_and_distances(self, universe, cosmology, particle_horizon_physical):
        """
        Получить физические координаты точек и расстояния от центра.
        Обертка с кэшированием для оптимизации визуализации.
        Физические расчеты делегируются к matter_simulation.
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            particle_horizon_physical: Текущий физический радиус горизонта частиц
        
        Returns:
            tuple: (physical_points, distances_from_center, scale_ratio)
        """
        scale_factor = cosmology.scale_factor
        
        # ВАЖНО: Проверяем, не изменилось ли количество точек (например, после разделения)
        current_num_points = len(self.matter_simulation.matter_points.points_comoving) if self.matter_simulation.matter_points.points_comoving is not None else 0
        cached_num_points = len(self._cached_physical_points) if self._cached_physical_points is not None else 0
        
        # ВАЖНО: Если количество точек изменилось, сбрасываем весь кэш
        if current_num_points != cached_num_points and cached_num_points > 0:
            if config.DEBUG and self.collapse_started:
                print(f"[DEBUG Cache] Сброс кэша: количество точек изменилось с {cached_num_points} на {current_num_points}")
            # Сбрасываем весь кэш
            self._cached_physical_points = None
            self._cached_distances = None
            self._cached_distances_squared = None
            self._cached_time = None
            self._cached_scale_factor = None
            self._cached_scale_ratio = None
        
        # ОПТИМИЗАЦИЯ: Кэшируем физические координаты и расстояния
        # Пересчитываем только если время изменилось ИЛИ количество точек изменилось
        # ВАЖНО: На паузе ИСПОЛЬЗУЕМ кэш (не пересчитываем каждый кадр!)
        if (self._cached_physical_points is not None and 
            self._cached_time is not None and 
            abs(universe.time - self._cached_time) < 1e-6 and
            abs(scale_factor - self._cached_scale_factor) < 1e-10 and
            current_num_points == cached_num_points):
            # Используем кэшированные значения
            if not hasattr(self, '_cached_distances_squared') or self._cached_distances_squared is None:
                self._cached_distances_squared = self._cached_distances**2
            return self._cached_physical_points, self._cached_distances, self._cached_scale_ratio
        
        # Делегируем физические расчеты к matter_simulation
        physical_points, distances_from_center, scale_ratio = self.matter_simulation.get_physical_points_and_distances(
            universe, cosmology, particle_horizon_physical
        )
        
        if physical_points is None:
            return None, None, None
        
        # Кэшируем результаты для оптимизации визуализации
        distances_squared = np.sum(physical_points**2, axis=1)
        self._cached_physical_points = physical_points
        self._cached_distances = distances_from_center
        self._cached_distances_squared = distances_squared
        self._cached_time = universe.time
        self._cached_scale_factor = scale_factor
        self._cached_scale_ratio = scale_ratio
        
        return physical_points, distances_from_center, scale_ratio
    
    def draw_scale_ruler(self, cosmology):
        """
        Две шкалы по центру снизу:
        — физическая: 10 млрд св. лет (proper), длина в px;
        — комовинг: χ = 10 млрд св. лет, длина в px.

        В режиме "physical" 1 пиксель = фиксированное физическое расстояние:
        физическая линейка постоянной длины (RULER_LENGTH_PX),
        комовинг-линейка проявляет 10 Гсв.лет comoving в физических единицах,
        её длина = a(t) · RULER_LENGTH_PX (при a<1 короче, при a>1 длиннее).

        В режиме "comoving" 1 пиксель = фиксированное сопутствующее расстояние:
        комовинг-линейка постоянной длины, а физическая линейка показывает,
        сколько комовинг-метров занимает 10 Гсв.лет proper. Поскольку
        χ = r_proper / a, её длина = RULER_LENGTH_PX / a — при a<1 длиннее
        комовинг-линейки, при a>1 короче.
        """
        TEN_BILLION_LY = get_ten_billion_ly()
        ref_px = int(ui.RULER_LENGTH_PX)
        lw = max(1, int(ui.HORIZON_LINE_WIDTH))
        tick = int(ui.RULER_TICK_HEIGHT)
        label_pad = 5
        gap_after_phys_caption = int(ui.RULER_COMOVING_OFFSET_PX)

        # Защита от «битого» состояния окна: берём актуальный размер прямо из surface
        # и приводим к нормальным int. Если что-то не так — выходим тихо.
        try:
            sw, sh = self.screen.get_size()
            width = int(sw if sw else (self.width or 0))
            height = int(sh if sh else (self.height or 0))
        except Exception:
            return TEN_BILLION_LY, ref_px
        if width <= 0 or height <= 0:
            return TEN_BILLION_LY, ref_px

        if cosmology is not None and getattr(cosmology, "scale_factor", None):
            try:
                a = float(cosmology.scale_factor)
            except (TypeError, ValueError):
                a = 1.0
            if not np.isfinite(a) or a <= 0.0:
                a = 1.0
        else:
            a = 1.0

        text = self.small_font.render(ui.RULER_PHYSICAL_TEXT, True, ui.RULER_COLOR)
        comov_col = ui.RULER_COMOVING_COLOR
        tex_c = self.small_font.render(ui.RULER_COMOVING_TEXT, True, comov_col)

        margin_bottom = int(ui.RULER_Y_OFFSET)
        tex_c_y = height - margin_bottom - tex_c.get_height()
        y_comov = tex_c_y - label_pad - tick
        text_y = y_comov - tick - gap_after_phys_caption - text.get_height()
        y_phys = text_y - label_pad - tick

        # Длины линеек в пикселях. Симметрия:
        #   physical-mode: phys = ref_px, comov = a · ref_px;
        #   comoving-mode: comov = ref_px, phys = ref_px / a.
        # Ограничим сверху диагональю экрана + запас, чтобы при a→0 не уходить за миллионы пикселей.
        max_len = int(np.hypot(width, height)) + 1000
        if is_comoving_display():
            comov_len_px = ref_px
            if a > 0.0:
                phys_len_px = int(max(2, min(max_len, round(ref_px / a))))
            else:
                phys_len_px = max_len
        else:
            phys_len_px = ref_px
            comov_len_px = int(max(2, min(max_len, round(a * ref_px))))

        ruler_x = int((width - phys_len_px) // 2)
        pygame.draw.line(
            self.screen,
            ui.RULER_COLOR,
            (ruler_x, y_phys),
            (ruler_x + phys_len_px, y_phys),
            width=lw,
        )
        pygame.draw.line(
            self.screen,
            ui.RULER_COLOR,
            (ruler_x, y_phys - tick),
            (ruler_x, y_phys + tick),
            width=lw,
        )
        pygame.draw.line(
            self.screen,
            ui.RULER_COLOR,
            (ruler_x + phys_len_px, y_phys - tick),
            (ruler_x + phys_len_px, y_phys + tick),
            width=lw,
        )

        comov_x = int((width - comov_len_px) // 2)
        pygame.draw.line(
            self.screen,
            comov_col,
            (comov_x, y_comov),
            (comov_x + comov_len_px, y_comov),
            width=lw,
        )
        pygame.draw.line(
            self.screen,
            comov_col,
            (comov_x, y_comov - tick),
            (comov_x, y_comov + tick),
            width=lw,
        )
        pygame.draw.line(
            self.screen,
            comov_col,
            (comov_x + comov_len_px, y_comov - tick),
            (comov_x + comov_len_px, y_comov + tick),
            width=lw,
        )

        text_x = int((width - text.get_width()) // 2)
        self.screen.blit(text, (text_x, text_y))
        tex_c_x = int((width - tex_c.get_width()) // 2)
        self.screen.blit(tex_c, (tex_c_x, tex_c_y))

        return TEN_BILLION_LY, ref_px
    
    def draw_horizons(self, universe, cosmology, masses=None):
        """Отрисовать космологические горизонты и горизонт событий центральной ЧД"""
        if not self.show_horizons:
            return
        
        # Космологические горизонты привязаны к наблюдателю (центру экрана)
        # В космологии горизонты определяются относительно наблюдателя, а не абсолютного центра Вселенной
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Фиксированная шкала: 10 млрд св. лет = RULER_LENGTH_PX пикселей
        TEN_BILLION_LY = get_ten_billion_ly()
        RULER_LENGTH_PX = ui.RULER_LENGTH_PX
        
        # Вычисляем все горизонты для правильного масштабирования
        # Кэширование уже реализовано внутри методов cosmology
        # ВАЖНО: Проверяем на NaN и Inf, чтобы избежать зависаний
        # ВАЖНО: Горизонт де Ситтера зависит от массы ЧД (метрика SdS)
        # Получаем массу ЧД из переданных масс или вычисляем
        if masses is None:
            masses_temp = self.calculate_masses(universe, cosmology)
            M_black_hole_kg = masses_temp.get('M_black_hole_kg', 0.0) if masses_temp else 0.0
        else:
            M_black_hole_kg = masses.get('M_black_hole_kg', 0.0) if masses else 0.0
        
        # Получаем scale_factor для причинного горизонта
        scale_factor = cosmology.scale_factor

        
        try:
            # ОПТИМИЗАЦИЯ: Используем кэшированные радиусы из masses (уже вычислены в calculate_masses)
            if masses and 'r_hubble_horizon_m' in masses:
                hubble_r = masses.get('r_hubble_horizon_m', 0.0)
                de_sitter_r = masses.get('r_de_sitter_horizon_m', 0.0)
                particle_r = masses.get('r_particle_horizon_m', 0.0)
                event_r = masses.get('r_event_horizon_m', 0.0)
            else:
                # Fallback: вычисляем горизонты (старый путь, только если masses не переданы)
                hubble_r = float(str(cosmology.hubble_horizon(universe.time, M_black_hole_kg)))
                de_sitter_r = float(str(cosmology.de_sitter_horizon(M_black_hole_kg)))
                particle_r = float(str(cosmology.particle_horizon(universe.time)))
                event_r = float(str(cosmology.cosmological_event_horizon(universe.time, M_black_hole_kg)))
            
            # Проверяем на NaN и Inf
            if not np.isfinite(hubble_r) or hubble_r < 0:
                hubble_r = 0.0
            if not np.isfinite(de_sitter_r) or de_sitter_r < 0:
                de_sitter_r = 0.0
            if not np.isfinite(particle_r) or particle_r < 0:
                particle_r = 0.0
            if not np.isfinite(event_r) or event_r < 0:
                event_r = 0.0
            
            # Масштабируем горизонты относительно фиксированной шкалы (1 млрд св. лет = RULER_LENGTH_PX пикселей)
            # В режиме "comoving" физический радиус сначала делим на a(t),
            # чтобы получить сопутствующий радиус: r_co = r_phys / a.
            display_a = scale_factor if (is_comoving_display() and scale_factor > 0) else 1.0

            def to_display_radius(physical_radius: float) -> float:
                """Перевод физического радиуса в радиус отображения (комовинг или физический)."""
                if is_comoving_display():
                    return float(physical_radius) / display_a
                return float(physical_radius)

            def scale_radius(physical_radius):
                """Масштабировать физический радиус в пиксели относительно фиксированной шкалы"""
                if physical_radius <= 0 or physical_radius >= float('inf'):
                    return 0
                # 10 млрд св. лет = RULER_LENGTH_PX пикселей.
                # В режиме "comoving" делим на a, чтобы перейти в сопутствующие.
                display_radius = physical_radius / display_a if is_comoving_display() else physical_radius
                pixels = (display_radius / TEN_BILLION_LY) * RULER_LENGTH_PX
                radius_int = int(pixels)
                # ВАЖНО: Ограничиваем максимальный радиус, чтобы pygame.draw.circle не зависал
                # Максимальный радиус = диагональ экрана + запас
                # Также проверяем на NaN и Inf
                if not np.isfinite(radius_int) or radius_int < 0:
                    return 0
                max_radius = int(np.sqrt(self.width**2 + self.height**2)) + 1000
                if radius_int > max_radius:
                    radius_int = max_radius
                return radius_int
            
            # Горизонт Хаббла (голубой) - самый маленький
            if hubble_r < float('inf') and hubble_r > 0:
                radius_int = scale_radius(hubble_r)
                if radius_int > 5:
                    center_x_int = int(float(str(center_x)))
                    center_y_int = int(float(str(center_y)))
                    # ВАЖНО: Проверяем, виден ли круг на экране перед рисованием
                    # Круг виден, если его центр находится в пределах экрана + радиус
                    if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                        center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                        pygame.draw.circle(
                        self.screen, 
                        ui.HORIZON_HUBBLE_COLOR, 
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                    # Название горизонта - справа от круга
                    label_x = center_x_int + radius_int + 5
                    label_y = center_y_int + ui.HORIZON_HUBBLE_OFFSET_Y
                    if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                        text = self.small_font.render(ui.HORIZON_HUBBLE_LABEL, True, ui.HORIZON_HUBBLE_COLOR)
                        self.screen.blit(text, (label_x, label_y))
                        
                        # Радиус горизонта - прямо под названием
                        radius_text = f"{to_display_radius(hubble_r) / 9.461e24:.2f}"
                        radius_label = self.small_font.render(radius_text, True, ui.HORIZON_HUBBLE_COLOR)
                        radius_y = label_y + 18  # Под названием
                        if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                            self.screen.blit(radius_label, (label_x, radius_y))
            
            # Горизонт де Ситтера (желтый) - второй по размеру
            if de_sitter_r < float('inf') and de_sitter_r > 0:
                radius_int = scale_radius(de_sitter_r)
                if radius_int > 5:
                    center_x_int = int(float(str(center_x)))
                    center_y_int = int(float(str(center_y)))
                    # ВАЖНО: Проверяем, виден ли круг на экране перед рисованием
                    if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                        center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                        pygame.draw.circle(
                        self.screen, 
                        ui.HORIZON_DE_SITTER_COLOR, 
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                    # Название горизонта - справа от круга
                    label_x = center_x_int + radius_int + 5
                    label_y = center_y_int + ui.HORIZON_DE_SITTER_OFFSET_Y
                    if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                        text = self.small_font.render(ui.HORIZON_DE_SITTER_LABEL, True, ui.HORIZON_DE_SITTER_COLOR)
                        self.screen.blit(text, (label_x, label_y))
                        
                        # Радиус горизонта - прямо под названием
                        radius_text = f"{to_display_radius(de_sitter_r) / 9.461e24:.2f}"
                        radius_label = self.small_font.render(radius_text, True, ui.HORIZON_DE_SITTER_COLOR)
                        radius_y = label_y + 18  # Под названием
                        if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                            self.screen.blit(radius_label, (label_x, radius_y))
            
            # Горизонт событий (серый) - третий по размеру
            if event_r < float('inf') and event_r > 0:
                radius_int = scale_radius(event_r)
                if radius_int > 5:
                    center_x_int = int(float(str(center_x)))
                    center_y_int = int(float(str(center_y)))
                    # ВАЖНО: Проверяем, виден ли круг на экране перед рисованием
                    if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                        center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                        pygame.draw.circle(
                        self.screen, 
                        ui.HORIZON_EVENT_COLOR, 
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                    # Название горизонта - справа от круга
                    label_x = center_x_int + radius_int + 5
                    label_y = center_y_int + ui.HORIZON_EVENT_OFFSET_Y
                    if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                        text = self.small_font.render(ui.HORIZON_EVENT_LABEL, True, ui.HORIZON_EVENT_COLOR)
                        self.screen.blit(text, (label_x, label_y))
                        
                        # Радиус горизонта - прямо под названием
                        radius_text = f"{to_display_radius(event_r) / 9.461e24:.2f}"
                        radius_label = self.small_font.render(radius_text, True, ui.HORIZON_EVENT_COLOR)
                        radius_y = label_y + 18  # Под названием
                        if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                            self.screen.blit(radius_label, (label_x, radius_y))
            
            # Горизонт частиц (красный) - самый большой
            if particle_r < float('inf') and particle_r > 0:
                radius_int = scale_radius(particle_r)
                
                # ОТЛАДКА: проверяем радиус горизонта частиц при отрисовке
                if DEBUG and not hasattr(self, '_debug_horizon_drawn'):
                    print("=" * 60)
                    print("DEBUG: Drawing particle horizon")
                    print("=" * 60)
                    print(f"Particle horizon (physical): {particle_r/9.461e24:.4f} billion ly")
                    print(f"Particle horizon (screen radius in pixels): {radius_int} px")
                    print(f"Screen center: ({center_x}, {center_y})")
                    print(f"Scale: 10 billion ly = {RULER_LENGTH_PX} px")
                    print(f"Calculation: ({particle_r/9.461e24:.4f} / 10.0) * {RULER_LENGTH_PX} = {radius_int} px")
                    print("=" * 60)
                    self._debug_horizon_drawn = True
                
                if radius_int > 5:
                    center_x_int = int(float(str(center_x)))
                    center_y_int = int(float(str(center_y)))
                    # ВАЖНО: Проверяем, виден ли круг на экране перед рисованием
                    if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                        center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                        pygame.draw.circle(
                            self.screen, 
                            ui.HORIZON_PARTICLE_COLOR, 
                            (center_x_int, center_y_int), 
                            radius_int, 
                            width=ui.HORIZON_LINE_WIDTH
                        )
                    # Название горизонта - справа от круга
                    label_x = center_x_int + radius_int + 5
                    label_y = center_y_int + ui.HORIZON_PARTICLE_OFFSET_Y
                    if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                        text = self.small_font.render(ui.HORIZON_PARTICLE_LABEL, True, ui.HORIZON_PARTICLE_COLOR)
                        self.screen.blit(text, (label_x, label_y))
                        
                        # Радиус горизонта - прямо под названием
                        radius_text = f"{to_display_radius(particle_r) / 9.461e24:.2f}"
                        radius_label = self.small_font.render(radius_text, True, ui.HORIZON_PARTICLE_COLOR)
                        radius_y = label_y + 18  # Под названием
                        if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                            self.screen.blit(radius_label, (label_x, radius_y))
            
            # Причинный горизонт (белый пунктир) - самый большой
            # Это полный сопутствующий радиус chi_p(∞) = 63.6 Gly
            causal_r = CAUSAL_HORIZON_COMOVING_METERS * scale_factor
            if causal_r < float('inf') and causal_r > 0:
                radius_int = scale_radius(causal_r)
                if radius_int > 5:
                    center_x_int = int(float(str(center_x)))
                    center_y_int = int(float(str(center_y)))
                    # Рисуем пунктирную линию
                    if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                        center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                        # Пунктирная окружность (рисуем сегментами)
                        num_segments = 60
                        dash_length = 2 * math.pi * radius_int / (num_segments * 2)
                        for i in range(0, num_segments, 2):  # Рисуем каждый второй сегмент
                            angle1 = 2 * math.pi * i / num_segments
                            angle2 = 2 * math.pi * (i + 1) / num_segments
                            x1 = int(center_x_int + radius_int * math.cos(angle1))
                            y1 = int(center_y_int + radius_int * math.sin(angle1))
                            x2 = int(center_x_int + radius_int * math.cos(angle2))
                            y2 = int(center_y_int + radius_int * math.sin(angle2))
                            pygame.draw.line(self.screen, (255, 255, 255), (x1, y1), (x2, y2), ui.HORIZON_LINE_WIDTH)
                    
                    # Название горизонта - справа от круга
                    label_x = center_x_int + radius_int + 5
                    label_y = center_y_int - 40  # Выше других
                    if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                        text = self.small_font.render("Causal", True, (255, 255, 255))
                        self.screen.blit(text, (label_x, label_y))
                        
                        # Радиус горизонта - прямо под названием
                        radius_text = f"{to_display_radius(causal_r) / 9.461e24:.2f}"
                        radius_label = self.small_font.render(radius_text, True, (255, 255, 255))
                        radius_y = label_y + 18  # Под названием
                        if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                            self.screen.blit(radius_label, (label_x, radius_y))

            
            # Горизонт событий центральной черной дыры (пурпурный)
            # Отображается только если коллапс начался и ЧД образовалась
            if masses is not None and 'r_black_hole_schwarzschild_m' in masses:
                r_black_hole = masses['r_black_hole_schwarzschild_m']
                black_hole_color = self._black_hole_color_for_mass(M_black_hole_kg)
                if r_black_hole > 0 and r_black_hole < float('inf'):
                    radius_int = scale_radius(r_black_hole)
                    if radius_int > 1:  # Отображаем даже очень маленькие горизонты
                        center_x_int = int(float(str(center_x)))
                        center_y_int = int(float(str(center_y)))
                        # ВАЖНО: Проверяем, виден ли круг на экране перед рисованием
                        if (center_x_int + radius_int >= 0 and center_x_int - radius_int < self.width and
                            center_y_int + radius_int >= 0 and center_y_int - radius_int < self.height):
                            # ВАЖНО: Сначала закрашиваем круг сплошным цветом (width=0 означает заливку)
                            pygame.draw.circle(
                                self.screen, 
                                black_hole_color,
                                (center_x_int, center_y_int), 
                                radius_int, 
                                width=0  # width=0 означает сплошную заливку
                            )
                            # Затем рисуем контур поверх заливки (если нужен более темный/яркий контур)
                            # Можно использовать более темный оттенок для контура, но пока используем тот же цвет
                            pygame.draw.circle(
                                self.screen, 
                                black_hole_color,
                                (center_x_int, center_y_int), 
                                radius_int, 
                                width=ui.HORIZON_LINE_WIDTH
                            )
                        # Название горизонта - справа от круга
                        label_x = center_x_int + radius_int + 5
                        label_y = center_y_int + ui.HORIZON_BLACK_HOLE_OFFSET_Y
                        if 0 < label_x < self.width - 100 and 0 < label_y < self.height:
                            text = self.small_font.render(ui.HORIZON_BLACK_HOLE_LABEL, True, black_hole_color)
                            self.screen.blit(text, (label_x, label_y))
                            
                            # Радиус горизонта - прямо под названием
                            # Форматируем так же, как другие горизонты (в млрд св. лет, 2 знака после запятой)
                            radius_text = f"{to_display_radius(r_black_hole) / get_one_billion_ly():.2f}"
                            radius_label = self.small_font.render(radius_text, True, black_hole_color)
                            radius_y = label_y + 18  # Под названием
                            if 0 < label_x < self.width - 100 and 0 < radius_y < self.height:
                                self.screen.blit(radius_label, (label_x, radius_y))
        except (ValueError, TypeError) as e:
            pass
    
    def invalidate_mass_cache(self):
        """Инвалидировать кэш масс (например, при ручном изменении времени)"""
        self._cached_masses = None
        self._cached_masses_key = None

    def calculate_masses(self, universe, cosmology):
        """Вычислить массы в различных радиусах с учетом коллапса материи.

        ОПТИМИЗАЦИЯ: результат кэшируется по ключу (universe.time,
        accumulated_bh_mass). За кадр функция зовётся ~8 раз из разных мест UI;
        благодаря кэшу 7 вызовов становятся O(1). На паузе ключ не меняется и
        кэш живёт между кадрами; в активном кадре step_matter_simulation
        сбрасывает кэш до первого вызова, потому что состояние симуляции уже
        изменилось.
        """
        matter_points = self.matter_simulation.matter_points
        cache_key = (universe.time, float(matter_points.accumulated_bh_mass))
        if (self._cached_masses is not None
                and self._cached_masses_key == cache_key):
            return self._cached_masses

        # Создаем обертки для функций, которые нужны MassCalculator
        def get_physical_points_wrapper(particle_horizon_physical):
            return self._get_physical_points_and_distances(universe, cosmology, particle_horizon_physical)

        def initialize_matter_points_wrapper():
            self.matter_simulation.initialize_matter_points(universe, cosmology)

        def add_matter_points_wrapper(num_new_points, radius_physical):
            self.matter_simulation.add_matter_points(universe, cosmology, num_new_points, radius_physical)

        # Вызываем MassCalculator для вычисления масс
        result = self.mass_calculator.calculate_masses(
            universe,
            cosmology,
            matter_points,
            self.paused,
            get_physical_points_wrapper,
            initialize_matter_points_wrapper,
            add_matter_points_wrapper,
        )

        self._cached_masses = result
        self._cached_masses_key = cache_key

        return result

    def step_matter_simulation(self, universe, cosmology):
        """Один шаг симуляции материи: ленивая инициализация, продвижение
        физики (тяга/гравитация/фотоны) и инвалидация кэша масс на новый кадр.

        Вызывается ровно один раз в начале draw() — до calculate_masses(),
        чтобы массы и радиусы соответствовали уже обновлённому состоянию.
        """
        if not self.collapse_started:
            self.initialize_matter_points(universe, cosmology)
        elif not self.matter_simulation.matter_points_initialized:
            self.initialize_matter_points(universe, cosmology)

        manual_cosmic = self._manual_cosmic_step_this_frame
        dt_signed = self._manual_cosmic_dt_signed
        self._manual_cosmic_step_this_frame = False
        self._manual_cosmic_dt_signed = None

        # На паузе физика не двигается, кроме ручного шага времени стрелками влево/вправо.
        if self.paused and not manual_cosmic:
            return

        r_black_hole = None
        cached_masses = self._cached_masses
        if cached_masses:
            r_black_hole = cached_masses.get('r_black_hole_schwarzschild_m', 0.0)
            if r_black_hole <= 0:
                r_black_hole = None

        physics_paused = self.paused and not manual_cosmic
        self.matter_simulation.update_collapse(
            universe,
            cosmology,
            physics_paused,
            r_black_hole,
            dt_step_signed=(dt_signed if manual_cosmic else None),
        )

        # Состояние материи изменилось — сбрасываем кэш масс, чтобы первый
        # calculate_masses в кадре пересчитал.
        self._cached_masses = None
        self._cached_masses_key = None
    
    def initialize_matter_points(self, universe, cosmology):
        """Инициализация точек материи. ДЕЛЕГИРУЕТ к matter_simulation."""
        self.matter_simulation.initialize_matter_points(universe, cosmology)
    
    def draw_info_panel(self, universe, cosmology, fps: float, masses=None):
        """Только текстовая панель. ДЕЛЕГИРУЕТ к info_panel."""
        self.info_panel.draw_info(universe, cosmology, fps, masses)
    
    
    def generate_points_in_3d_sphere(self, num_points: int, radius: float, seed: int = None) -> np.ndarray:
        """Генерация точек в 3D сфере. ДЕЛЕГИРУЕТ к matter_simulation."""
        return self.matter_simulation.generate_points_in_3d_sphere(num_points, radius, seed)
    
    def _add_matter_points(self, universe, cosmology, num_new_points: int, radius_physical: float):
        """Добавление новых точек материи. ДЕЛЕГИРУЕТ к matter_simulation."""
        self.matter_simulation.add_matter_points(universe, cosmology, num_new_points, radius_physical)
    
    def initialize_matter_points(self, universe, cosmology):
        """Инициализация точек материи. ДЕЛЕГИРУЕТ к matter_simulation."""
        self.matter_simulation.initialize_matter_points(universe, cosmology)
    
    def draw_density_points(self, universe, cosmology, masses=None):
        """
        Отрисовать точки плотности материи (размер на экране — MATTER_POINT_SCREEN_PX в config)
        Точки создаются один раз в сопутствующих координатах и расширяются вместе с Вселенной
        
        Args:
            universe: Объект вселенной
            cosmology: Объект космологии
            masses: Кэшированные результаты calculate_masses (опционально, для оптимизации)
        """
        scale_factor = cosmology.scale_factor
        if scale_factor <= 0:
            return
        
        # Инициализируем точки один раз
        # ВАЖНО: после начала коллапса не переинициализируем точки, чтобы сохранить изменения
        if not self.collapse_started:
            self.initialize_matter_points(universe, cosmology)
        else:
            # Если коллапс уже начался, просто проверяем, что точки инициализированы
            if not self.matter_simulation.matter_points_initialized:
                self.initialize_matter_points(universe, cosmology)
        
        if self.matter_simulation.matter_points.points_comoving is None or len(self.matter_simulation.matter_points.points_comoving) == 0:
            return
        
        # Радиус круга = горизонт частиц в физических координатах (расширяется со временем)
        particle_horizon_physical = cosmology.particle_horizon(universe.time)
        
        # ВАЖНО: горизонт частиц расширяется быстрее, чем просто масштабный фактор
        # Вычисляем scale_ratio (коэффициент роста горизонта частиц)
        scale_ratio = self._calculate_scale_ratio(universe, cosmology, particle_horizon_physical)
        
        # ===== ОПТИМИЗАЦИЯ: Фильтруем точки по видимому радиусу В COMOVING координатах =====
        # Это избегает преобразования всех 245K точек в физические координаты
        
        # Получаем comoving_distances (уже предвычислены)
        comoving_distances = self.matter_simulation.matter_points.comoving_distances
        if comoving_distances is None or len(comoving_distances) == 0:
            return
        
        # Вычисляем максимальный видимый радиус в comoving координатах
        TEN_BILLION_LY = get_ten_billion_ly()
        RULER_LENGTH_PX = ui.RULER_LENGTH_PX
        # Максимальное "экранное" расстояние в единицах отображения
        # (физических — в режиме "physical", комовинг — в режиме "comoving").
        max_screen_display = max(self.width, self.height) * TEN_BILLION_LY / RULER_LENGTH_PX
        if is_comoving_display():
            # 1 пиксель = (TEN_BILLION_LY/RULER_LENGTH_PX) комовинг-метров →
            # max_visible в комовинге равен max_screen_display напрямую.
            max_visible_comoving = max_screen_display
        else:
            max_visible_comoving = (
                max_screen_display / scale_factor if scale_factor > 0 else max_screen_display
            )
        
        # ИЗМЕНЕНО: Рисуем ВСЕ точки (не ограничиваем горизонтом частиц)
        # Используем только экранный фильтр для оптимизации
        max_comoving = max_visible_comoving
        
        # Фильтруем по comoving расстоянию (очень быстро - одно сравнение)
        visible_mask = comoving_distances <= max_comoving
        
        # Получаем только видимые точки
        points_comoving = self.matter_simulation.matter_points.points_comoving
        visible_points_comoving = points_comoving[visible_mask]
        visible_comoving_distances = comoving_distances[visible_mask]
        visible_indices = np.where(visible_mask)[0]
        
        if len(visible_points_comoving) == 0:
            return
        
        # Преобразуем ТОЛЬКО видимые точки в физические координаты
        physical_points = visible_points_comoving * scale_factor
        distances_from_center = visible_comoving_distances * scale_factor
        
        # Для совместимости с остальным кодом - создаем circle_mask для visible точек
        circle_mask = np.ones(len(physical_points), dtype=bool)
        circle_indices = np.arange(len(physical_points))
        
        if len(physical_points) == 0:
            return
        
        # Для совместимости с остальным кодом:
        points_in_circle = physical_points
        distances_in_circle = distances_from_center
        
        # DEBUG: Первый раз выводим статистику
        if DEBUG and not hasattr(self, '_debug_points_printed'):
            print(f"Drawing points: total_comoving={len(comoving_distances)}, visible={len(physical_points)}")
            print(f"Center: (0, 0, 0) in physical coordinates (3D)")
            print(f"Scale factor: {scale_factor}")
            self._debug_points_printed = True
        
        # ОПТИМИЗАЦИЯ: Векторизованное преобразование координат вместо list comprehension
        # Фильтруем точки по видимой области экрана
        # Преобразуем все точки в экранные координаты и проверяем, попадают ли они на экран
        
        # ВАЖНО: Определяем center_x и center_y ДО блока if, чтобы они были доступны везде
        # ВАЖНО: Используем ТОЧНО ТЕ ЖЕ значения, что и в draw_horizons() для совпадения позиций
        TEN_BILLION_LY = get_ten_billion_ly()
        RULER_LENGTH_PX = ui.RULER_LENGTH_PX
        # Используем self.width и self.height, как в draw_horizons() (строка 277-278)
        center_x = self.width // 2  # Центр экрана (тот же, что используется для горизонтов)
        center_y = self.height // 2  # Центр экрана (тот же, что используется для горизонтов)
        
        # ОПТИМИЗАЦИЯ: Вычисляем максимальный физический радиус, видимый на экране
        scale_to_physical = TEN_BILLION_LY / RULER_LENGTH_PX
        max_display_dist = max(self.width, self.height) * scale_to_physical
        # distances_in_circle — физические; в режиме комовинг 1 пиксель соответствует
        # большему физическому расстоянию (в `a` раз), поэтому max-radius умножается на a.
        if is_comoving_display():
            max_screen_radius_physical = max_display_dist * scale_factor
        else:
            max_screen_radius_physical = max_display_dist

        # ОПТИМИЗАЦИЯ: Фильтр по физическому радиусу экрана эквивалентен уже
        # проделанному фильтру по комовингу (max_screen_radius_physical =
        # max_visible_comoving · a). Поэтому не дублируем — просто работаем
        # дальше с уже отфильтрованными массивами как с «потенциально видимыми».
        points_potentially_visible = points_in_circle
        circle_indices_visible = circle_indices
        distances_potentially_visible = distances_in_circle
        
        if len(points_potentially_visible) > 0:
            # ОПТИМИЗАЦИЯ: ВЕКТОРИЗАЦИЯ - Преобразуем ТОЛЬКО потенциально видимые точки
            # ВАЖНО: Используем ту же формулу, что и для горизонтов, чтобы точки были в центре
            # Формула: x_px = (point[0] / TEN_BILLION_LY) * RULER_LENGTH_PX + center_x
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Перемешиваем точки случайным образом перед отрисовкой
            # Это устраняет любые артефакты порядка точек в массиве
            if not hasattr(self, '_points_shuffled_once'):
                # Перемешиваем точки только один раз при первой отрисовке
                shuffle_indices = np.random.permutation(len(points_potentially_visible))
                points_potentially_visible = points_potentially_visible[shuffle_indices]
                circle_indices_visible = circle_indices_visible[shuffle_indices]
                distances_potentially_visible = distances_potentially_visible[shuffle_indices]
                self._points_shuffled_once = True
                if DEBUG:
                    print("[DEBUG] Points shuffled randomly to eliminate ordering artifacts")
            
            # ОПТИМИЗАЦИЯ: Вычисляем масштаб один раз и используем in-place операции
            scale_to_px = RULER_LENGTH_PX / TEN_BILLION_LY
            # В режиме "comoving" делим физические координаты на a(t),
            # чтобы получить отображение в сопутствующих координатах:
            #   r_phys = a · χ ⇒ x_px = (r_phys / a) · scale_to_px = χ · scale_to_px.
            if is_comoving_display() and scale_factor > 0:
                display_scale_to_px = scale_to_px / scale_factor
            else:
                display_scale_to_px = scale_to_px
            # ОПТИМИЗАЦИЯ: Используем более эффективное преобразование координат
            x_px_all = np.multiply(points_potentially_visible[:, 0], display_scale_to_px, out=np.empty(len(points_potentially_visible), dtype=np.float32))
            x_px_all += center_x
            y_px_all = np.multiply(points_potentially_visible[:, 1], display_scale_to_px, out=np.empty(len(points_potentially_visible), dtype=np.float32))
            y_px_all += center_y
            
            # ОПТИМИЗАЦИЯ: Фильтруем точки, которые попадают в видимую область экрана
            # Упрощаем проверки - для координат на экране проверка int32 не нужна
            visible_mask = (
                (x_px_all >= 0) & (x_px_all < self.width) &
                (y_px_all >= 0) & (y_px_all < self.height)
            )
            points_to_draw = points_potentially_visible[visible_mask]
            # Сохраняем экранные координаты для видимых точек
            x_px_visible = x_px_all[visible_mask]
            y_px_visible = y_px_all[visible_mask]
            # Сохраняем индексы точек для рисования (для сопоставления со скоростями)
            points_to_draw_indices = circle_indices_visible[visible_mask]
            # ОПТИМИЗАЦИЯ: Используем кэшированные расстояния вместо пересчета
            # Сохраняем расстояния от центра для видимых точек (уже вычислены выше)
            distances_visible = distances_potentially_visible[visible_mask]
        else:
            points_to_draw = np.array([]).reshape(0, 3)
            points_to_draw_indices = np.array([], dtype=int)
            x_px_visible = np.array([])
            y_px_visible = np.array([])
            distances_visible = np.array([])
        
        # Отладочный вывод для первых нескольких точек
        if not hasattr(self, '_debug_draw_printed') and len(points_to_draw) > 0:
            test_point = points_to_draw[0]
            test_screen = self.world_to_screen(test_point)
            print(f"First point to draw: world=({test_point[0]/9.461e24:.2f}, {test_point[1]/9.461e24:.2f}) billion ly")
            print(f"Screen coordinates: ({test_screen[0]}, {test_screen[1]})")
            print(f"Screen size: {self.width}x{self.height}, Camera: ({self.camera_x}, {self.camera_y}), Zoom: {self.zoom}")
            print(f"Points in circle: {len(points_in_circle)}, Points in visible area: {len(points_to_draw)}")
            self._debug_draw_printed = True
        
        # Отладочный вывод после отрисовки
        if not hasattr(self, '_debug_drawn_printed'):
            self._debug_drawn_printed = True
        
        # ОТЛАДКА: проверяем радиусы точек при отрисовке относительно горизонта частиц
        if DEBUG and not hasattr(self, '_debug_draw_radius_printed') and len(points_to_draw) > 0:
            # Вычисляем радиусы точек в физических координатах (3D)
            point_distances = np.sqrt(
                points_to_draw[:, 0]**2 + 
                points_to_draw[:, 1]**2 +
                points_to_draw[:, 2]**2
            )
            max_point_radius = np.max(point_distances) if len(point_distances) > 0 else 0
            
            # Вычисляем радиус горизонта частиц в пикселях для сравнения
            TEN_BILLION_LY = get_ten_billion_ly()
            RULER_LENGTH_PX = ui.RULER_LENGTH_PX
            horizon_radius_px = int((particle_horizon_physical / TEN_BILLION_LY) * RULER_LENGTH_PX)
            
            # Вычисляем радиус максимальной точки в пикселях
            max_point_radius_px = int((max_point_radius / TEN_BILLION_LY) * RULER_LENGTH_PX)
            
            print("=" * 60)
            print("DEBUG: Point radii when drawing (comparison with horizon)")
            print("=" * 60)
            print(f"Particle horizon (physical): {particle_horizon_physical/9.461e24:.4f} billion ly = {horizon_radius_px} px")
            print(f"Max point radius (physical): {max_point_radius/9.461e24:.4f} billion ly = {max_point_radius_px} px")
            print(f"Points to draw: {len(points_to_draw)}")
            if max_point_radius > particle_horizon_physical:
                print(f"ERROR: Max point radius ({max_point_radius/9.461e24:.4f}) EXCEEDS horizon ({particle_horizon_physical/9.461e24:.4f})!")
                print(f"Difference: {(max_point_radius - particle_horizon_physical)/9.461e24:.4f} billion ly")
            else:
                print(f"OK: Max point radius is within horizon")
                print(f"Ratio: {max_point_radius/particle_horizon_physical:.4f} (should be <= 1.0)")
            print("=" * 60)
            self._debug_draw_radius_printed = True
        
        # Рисуем точки материи (площадь зависит от MATTER_POINT_SCREEN_PX)
        # ВАЖНО: используем тот же масштаб, что и для горизонта частиц
        # Фиксированная шкала: 10 млрд св. лет = RULER_LENGTH_PX пикселей
        # center_x и center_y уже определены выше
        
        # ОТЛАДКА: получаем радиус горизонта частиц в пикселях для сравнения
        horizon_radius_px = int((particle_horizon_physical / TEN_BILLION_LY) * RULER_LENGTH_PX)
        
        # ОТЛАДКА: проверяем координаты точек и горизонта частиц на экране
        if DEBUG and not hasattr(self, '_debug_screen_coords_printed') and len(points_to_draw) > 0:
            # Вычисляем радиусы точек в пикселях
            point_distances_px = []
            point_screen_coords = []
            for point in points_to_draw[:100]:  # Первые 100 точек для отладки
                x_px = (point[0] / TEN_BILLION_LY) * RULER_LENGTH_PX + center_x
                y_px = (point[1] / TEN_BILLION_LY) * RULER_LENGTH_PX + center_y
                point_screen_coords.append((x_px, y_px))
                # Расстояние от центра экрана в пикселях
                dist_px = np.sqrt((x_px - center_x)**2 + (y_px - center_y)**2)
                point_distances_px.append(dist_px)
            
            max_point_dist_px = np.max(point_distances_px) if len(point_distances_px) > 0 else 0
            min_point_dist_px = np.min(point_distances_px) if len(point_distances_px) > 0 else 0
            mean_point_dist_px = np.mean(point_distances_px) if len(point_distances_px) > 0 else 0
            
            print("=" * 60)
            print("DEBUG: Screen coordinates comparison")
            print("=" * 60)
            print(f"Screen center: ({center_x}, {center_y})")
            print(f"PARTICLE HORIZON on screen:")
            print(f"  Center: ({center_x}, {center_y})")
            print(f"  Radius: {horizon_radius_px} px")
            print(f"  Physical radius: {particle_horizon_physical/9.461e24:.4f} billion ly")
            print(f"")
            print(f"POINTS on screen (first 100):")
            print(f"  Min distance from center: {min_point_dist_px:.2f} px")
            print(f"  Max distance from center: {max_point_dist_px:.2f} px")
            print(f"  Mean distance from center: {mean_point_dist_px:.2f} px")
            print(f"  Horizon radius: {horizon_radius_px} px")
            print(f"")
            if max_point_dist_px > horizon_radius_px:
                print(f"ERROR: Max point distance ({max_point_dist_px:.2f} px) EXCEEDS horizon radius ({horizon_radius_px} px)!")
                print(f"Difference: {max_point_dist_px - horizon_radius_px:.2f} px")
            else:
                print(f"OK: Max point distance ({max_point_dist_px:.2f} px) is within horizon radius ({horizon_radius_px} px)")
                print(f"Ratio: {max_point_dist_px/horizon_radius_px:.4f} (should be <= 1.0)")
            print(f"")
            print(f"Example point coordinates on screen:")
            for i in range(min(5, len(point_screen_coords))):
                dist = point_distances_px[i]
                print(f"  Point {i+1}: screen=({point_screen_coords[i][0]:.1f}, {point_screen_coords[i][1]:.1f}), "
                      f"distance={dist:.2f} px, horizon_radius={horizon_radius_px} px")
            print("=" * 60)
            self._debug_screen_coords_printed = True
        
        # Вычисляем радиус горизонта частиц в пикселях для проверки
        # ВАЖНО: Используем тот же метод масштабирования, что и для точек (с учетом zoom)
        pixel_to_meter = get_pixel_to_meter()
        # Радиус в пикселях = (радиус в метрах / pixel_to_meter) * zoom
        # Но для горизонтов используется фиксированная шкала без zoom
        # Поэтому используем ту же формулу, что и в draw_horizons()
        horizon_radius_px = (particle_horizon_physical / TEN_BILLION_LY) * RULER_LENGTH_PX
        
        white_color = (255, 255, 255)
        particle_color = ui.HORIZON_PARTICLE_COLOR  # Красный цвет для точек внутри горизонта частиц
        points_drawn = 0
        
        # ОПТИМИЗАЦИЯ: Используем переданные массы вместо повторного вызова calculate_masses
        if masses is None:
            masses = self.calculate_masses(universe, cosmology)
        r_black_hole = masses.get('r_black_hole_schwarzschild_m', 0.0) if masses else 0.0
        M_black_hole_kg = masses.get('M_black_hole_kg', 0.0) if masses else 0.0
        black_hole_color = self._black_hole_color_for_mass(M_black_hole_kg)
        
        # Получаем радиусы горизонтов для проверки цвета точек
        # ВАЖНО: Горизонт де Ситтера зависит от массы ЧД (метрика SdS)
        hubble_horizon_radius = cosmology.hubble_horizon(universe.time)  # в метрах
        de_sitter_horizon_radius = float(str(cosmology.de_sitter_horizon(M_black_hole_kg)))  # в метрах
        
        # ОПТИМИЗАЦИЯ: Прямая запись в pixels2d(screen) без SRCALPHA-оверлея и blit.
        # Цвета упаковываются один раз в uint32 под формат поверхности → одна запись
        # вместо трёх (R/G/B) и без полноэкранного fill+alpha-blend каждый кадр.
        points_drawn = 0
        if len(x_px_visible) > 0:
            xi = x_px_visible.astype(np.int32, copy=False)
            yi = y_px_visible.astype(np.int32, copy=False)

            # Один общий фильтр границ экрана (на случай FP-граничных значений после astype).
            in_bounds = (
                (xi >= 0) & (xi < self.width) &
                (yi >= 0) & (yi < self.height)
            )
            if not in_bounds.all():
                xi = xi[in_bounds]
                yi = yi[in_bounds]
                distances_valid = distances_visible[in_bounds]
            else:
                distances_valid = distances_visible

            # Исключаем точки внутри ЦЧД (она рисуется отдельной точкой/диском).
            if r_black_hole > 0.0 and xi.size > 0:
                outside_bh = distances_valid > r_black_hole
                if not outside_bh.all():
                    xi = xi[outside_bh]
                    yi = yi[outside_bh]
                    distances_valid = distances_valid[outside_bh]

                    if config.DEBUG and self.collapse_started:
                        if not hasattr(self, '_debug_bh_filter_count'):
                            self._debug_bh_filter_count = 0
                        self._debug_bh_filter_count += 1
                        if self._debug_bh_filter_count <= 20:
                            inside = int(np.size(outside_bh) - np.count_nonzero(outside_bh))
                            print(f"[DEBUG Render] Фильтрация по ЧД: points_inside_bh={inside}, "
                                  f"r_bh={r_black_hole:.2e} м ({r_black_hole/9.461e24:.4f} млрд св. лет)")

            if xi.size > 0:
                # Цвет: белый внутри де Ситтера, particle_color снаружи.
                color_mask_ds = distances_valid <= de_sitter_horizon_radius
                white_packed = np.uint32(self.screen.map_rgb(white_color))
                particle_packed = np.uint32(self.screen.map_rgb(particle_color))
                packed = np.where(color_mask_ds, white_packed, particle_packed).astype(np.uint32, copy=False)

                px_size = int(getattr(config, 'MATTER_POINT_SCREEN_PX', 1))
                offsets = _matter_point_square_offsets_int32(px_size) if px_size > 1 else None

                try:
                    pix2d = pygame.surfarray.pixels2d(self.screen)
                    try:
                        _write_points_packed_to_pixels2d(
                            pix2d, xi, yi, packed, offsets,
                            self.width, self.height,
                        )
                        points_drawn = int(xi.size)
                    finally:
                        del pix2d
                except (AttributeError, ValueError, IndexError, pygame.error):
                    # Резервный путь: pixels3d (uint8 ×3).
                    try:
                        rgb = np.where(
                            color_mask_ds[:, None],
                            np.asarray(white_color, dtype=np.uint8),
                            np.asarray(particle_color, dtype=np.uint8),
                        ).astype(np.uint8, copy=False)
                        pixels = pygame.surfarray.pixels3d(self.screen)
                        try:
                            _write_density_points_to_pixels3d_screen(
                                pixels, xi, yi, rgb, px_size,
                                self.width, self.height,
                            )
                            points_drawn = int(xi.size)
                        finally:
                            del pixels
                    except (AttributeError, ValueError, IndexError, pygame.error):
                        # Самый медленный фолбэк — попиксельная запись.
                        for idx in range(int(xi.size)):
                            try:
                                c = white_color if color_mask_ds[idx] else particle_color
                                self.screen.set_at((int(xi[idx]), int(yi[idx])), c)
                            except (TypeError, ValueError, IndexError):
                                continue
                        points_drawn = int(xi.size)
            elif config.DEBUG and self.collapse_started:
                if not hasattr(self, '_debug_no_points_printed'):
                    print(f"[DEBUG Render] Все точки внутри ЧД r_bh={r_black_hole:.2e} м")
                    self._debug_no_points_printed = True
        
        # Центральная точка ЧД уже входит в массив точек материи (первая точка в массиве)
        # и отрисовывается вместе с остальными точками, поэтому отдельная отрисовка не нужна
        
        # Если точек не нарисовано, выводим информацию
        if DEBUG and points_drawn == 0 and len(points_in_circle) > 0:
            test_point = points_in_circle[0]
            test_screen = self.world_to_screen(test_point)
            print(f"Warning: No points drawn. Test point: world={test_point}, screen={test_screen}")
            print(f"Points in circle: {len(points_in_circle)}, Visible on screen: {len(x_px_visible)}")
    
    def render(self, universe, cosmology, fps: float):
        """Отрисовать весь кадр"""
        # ОПТИМИЗАЦИЯ: Профилирование времени выполнения для поиска узких мест
        profile_times = {}
        
        # ВАЖНО: Синхронизируем self.width и self.height с актуальным размером экрана
        # Это гарантирует, что координаты всегда вычисляются с правильными значениями
        screen_width, screen_height = self.screen.get_size()
        if self.width != screen_width or self.height != screen_height:
            self.width = screen_width
            self.height = screen_height
            self.camera_x = self.width // 2
            self.camera_y = self.height // 2
        
        # Очистка экрана
        t0 = time.perf_counter()
        self.screen.fill(self.background_color)
        profile_times['fill_screen'] = time.perf_counter() - t0
        
        # Сначала продвигаем физику материи, чтобы массы/радиусы соответствовали
        # уже обновлённому состоянию этого кадра.
        t0 = time.perf_counter()
        self.step_matter_simulation(universe, cosmology)
        profile_times['step_matter_simulation'] = time.perf_counter() - t0

        # ВАЖНО: Вычисляем массы ОДИН РАЗ за кадр и кэшируем результат
        # Это критично для производительности, так как calculate_masses очень тяжелый
        t0 = time.perf_counter()
        masses = self.calculate_masses(universe, cosmology)
        profile_times['calculate_masses'] = time.perf_counter() - t0
        
        # Рисуем белые точки плотности материи (всегда видимы)
        # Передаем кэшированные массы, чтобы избежать повторного вызова calculate_masses
        t0 = time.perf_counter()
        self.draw_density_points(universe, cosmology, masses=masses)
        profile_times['draw_density_points'] = time.perf_counter() - t0

        # Рисуем фотоны (излучение материя→ЦЧД) с redshift-градиентом
        t0 = time.perf_counter()
        self.draw_photons(universe, cosmology)
        profile_times['draw_photons'] = time.perf_counter() - t0

        # Шкала, горизонты, маркер ЦЧД и панель информации — один таймер профилирования.
        t0 = time.perf_counter()
        self.draw_info(universe, cosmology, masses, fps)
        profile_times['draw_info'] = time.perf_counter() - t0

        # В Lambda CDM модели нет отдельных объектов - только однородная жидкость
        # Объекты не рисуются, только карта плотности (если включена клавишей P)
        
        # Рисуем количество точек внутри горизонтов в правом нижнем углу (передаем кэшированные массы)
        t0 = time.perf_counter()
        self.horizon_point_counts(universe, cosmology, masses=masses)
        profile_times['horizon_point_counts'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        pygame.display.flip()
        profile_times['flip_screen'] = time.perf_counter() - t0
        
        # Выводим профилирование каждые 30 кадров (примерно раз в 1 секунду при 30 FPS)
        # НЕ выводим во время паузы, чтобы не засорять логи
        if not hasattr(self, '_profile_frame_count'):
            self._profile_frame_count = 0
        self._profile_frame_count += 1
        
        if self._profile_frame_count % 30 == 0 and not self.paused:
            total_time = sum(profile_times.values())
            # Используем безопасный вывод для Windows консоли
            try:
                print("\n" + "="*60)
                print("PERFORMANCE PROFILING (time in milliseconds):")
                print("="*60)
                # Сортируем по времени выполнения
                sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
                for method, elapsed in sorted_times:
                    percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                    print(f"  {method:30s}: {elapsed*1000:7.2f} ms ({percentage:5.1f}%)")
                print(f"  {'TOTAL':30s}: {total_time*1000:7.2f} ms")
                print("="*60 + "\n")
            except UnicodeEncodeError:
                # Fallback для консолей без поддержки Unicode
                print("\n" + "="*60)
                print("PERFORMANCE PROFILING:")
                print("="*60)
                sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
                for method, elapsed in sorted_times:
                    percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                    print(f"  {method:30s}: {elapsed*1000:7.2f} ms ({percentage:5.1f}%)")
                print(f"  {'TOTAL':30s}: {total_time*1000:7.2f} ms")
                print("="*60 + "\n")
    
    def handle_input(self, universe=None, cosmology=None) -> bool:
        """
        Обработать ввод пользователя. ДЕЛЕГИРУЕТ к input_handler.
        Returns: True если нужно продолжить, False если выйти
        """
        return self.input_handler.handle_input(universe, cosmology)

    def _get_batch_pixel_overlay_surface(self) -> pygame.Surface:
        """Переиспользуемая RGBA-поверхность размера экрана (точки плотности, фотоны)."""
        wh = (int(self.width), int(self.height))
        if self._batch_pixel_overlay is None or self._batch_pixel_overlay_wh != wh:
            self._batch_pixel_overlay = pygame.Surface(wh, pygame.SRCALPHA, 32)
            self._batch_pixel_overlay_wh = wh
        return self._batch_pixel_overlay

    def draw_photons(self, universe, cosmology):
        """
        Отрисовать фотоны излучения материя→ЦЧД с учётом расширения пространства.

        Геометрия фотона в FLRW: dχ/dt = −c/a(t) (движение к центру). Фотон,
        испущенный в момент t_emit из сопутствующей координаты χ_emit, в момент
        t_now находится на

            χ_photon(t_now) = χ_emit − c·[η(t_now) − η(t_emit)],
            r_phys(t_now)   = a(t_now) · χ_photon(t_now),

        где η(t) = ∫₀^t dt'/a(t') — конформное время. Между двумя фотонами
        одного луча Δχ постоянна, поэтому Δr_phys = a(t)·Δχ растёт вместе с
        a(t) — луч физически растягивается с расширением Вселенной.

        Концы луча:
          • голова (самый старый фотон, испущен в t_start с r_initial):
                χ_head = r_initial/a(t_start) − c·[η(t_now) − η(t_start)],
                r_head = max(a(t_now) · χ_head, 0).
          • хвост — текущая позиция точки, ПОКА m_rest > 0;
                после выгорания (m_rest=0, точка была на r_off в t_off):
                χ_tail = r_off/a(t_off) − c·[η(t_now) − η(t_off)],
                r_tail = max(a(t_now) · χ_tail, 0).

        Когда r_head ≥ r_tail, луч полностью поглощён ЦЧД.

        Цвет фотона: factor = a_emit/a_now — синий (свежий) → через зелёный → красный
        (сильнее покрасневший при меньшем factor).
        """
        mp = getattr(self.matter_simulation, 'matter_points', None)
        if mp is None or mp.points_comoving is None or len(mp.points_comoving) == 0:
            return
        scale_factor = cosmology.scale_factor
        if scale_factor <= 0:
            return

        photon_chi = getattr(mp, '_laser_photon_chi', None)
        if photon_chi is None or len(photon_chi) == 0:
            return

        TEN_BILLION_LY = get_ten_billion_ly()
        scale_to_px = ui.RULER_LENGTH_PX / TEN_BILLION_LY
        # В режиме "comoving" делим все физические радиусы на a(t) при отрисовке.
        if is_comoving_display() and scale_factor > 0:
            display_scale_to_px = scale_to_px / float(scale_factor)
        else:
            display_scale_to_px = scale_to_px
        center_x = self.width // 2
        center_y = self.height // 2
        point_radius = max(1, int(getattr(config, 'MATTER_LASER_PHOTON_RADIUS_PX', 1)))

        chi = np.asarray(photon_chi, dtype=np.float64)
        r_photon = chi * float(scale_factor)

        masses_cached = self._cached_masses
        r_black_hole = 0.0
        if masses_cached:
            r_black_hole = float(masses_cached.get('r_black_hole_schwarzschild_m', 0.0))

        visible = np.isfinite(r_photon) & (r_photon > max(r_black_hole, 0.0))
        if not np.any(visible):
            return

        # Аккуратно достаём ux/uy/a_emit без аллокации zeros_like в default-аргументе getattr.
        ux_attr = getattr(mp, '_laser_photon_ux', None)
        uy_attr = getattr(mp, '_laser_photon_uy', None)
        a_emit_attr = getattr(mp, '_laser_photon_a_emit', None)
        if ux_attr is None or uy_attr is None or a_emit_attr is None:
            return
        ux_v = np.asarray(ux_attr, dtype=np.float64)[visible]
        uy_v = np.asarray(uy_attr, dtype=np.float64)[visible]
        a_emit_v = np.asarray(a_emit_attr, dtype=np.float64)[visible]
        r_v = r_photon[visible] * display_scale_to_px

        x = ux_v * r_v + center_x
        y = uy_v * r_v + center_y
        on_screen = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        if not np.any(on_screen):
            return

        factor = np.clip(a_emit_v[on_screen] / float(scale_factor), 0.0, 1.0)
        colors = np.clip(photon_rgb_blue_green_red(factor), 0, 255).astype(np.uint8)
        xi = np.clip(np.floor(x[on_screen]), 0, self.width - 1).astype(np.int32)
        yi = np.clip(np.floor(y[on_screen]), 0, self.height - 1).astype(np.int32)

        # ОПТИМИЗАЦИЯ: упаковываем RGB в uint32 под формат screen и пишем
        # напрямую в pixels2d — без SRCALPHA-оверлея, fill и blit.
        offsets = _photon_disk_offsets_int32(point_radius) if point_radius > 1 else None
        try:
            packed = _pack_rgb_for_surface(colors, self.screen)
            pix2d = pygame.surfarray.pixels2d(self.screen)
            try:
                _write_points_packed_to_pixels2d(
                    pix2d, xi, yi, packed, offsets,
                    self.width, self.height,
                )
            finally:
                del pix2d
        except (AttributeError, ValueError, IndexError, TypeError, pygame.error):
            # Резервный путь — pygame.draw.circle/set_at попиксельно.
            for i in range(int(xi.size)):
                try:
                    pos = (int(xi[i]), int(yi[i]))
                    color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
                    if point_radius <= 1:
                        self.screen.set_at(pos, color)
                    else:
                        pygame.draw.circle(self.screen, color, pos, point_radius)
                except (TypeError, ValueError):
                    continue
        return
    
    def horizon_point_counts(self, universe, cosmology, masses=None):
        """Счётчики точек по горизонтам (HUD). ДЕЛЕГИРУЕТ к info_panel."""
        self.info_panel.horizon_point_counts(universe, cosmology, masses)
    
    def tick(self, fps: int = 60):
        """Ограничить FPS"""
        self.clock.tick(fps)
        return self.clock.get_fps()
