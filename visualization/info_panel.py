"""
Модуль UI панели информации для симуляции
"""
import numpy as np
import pygame

import config
from config import DEBUG
import visualization.ui as ui
from utils.constants import (
    G,
    NARIAI_BLACK_HOLE_MASS_KG,
    PARTICLE_HORIZON_MASS_LIMIT_KG,
    PLANCK_POWER_W,
    SECONDS_PER_BILLION_YEARS,
    c,
)
from utils.config_utils import get_dt, get_mass_per_point_kg, get_coordinate_display_mode
from utils.format_utils import format_velocity_m_per_s


class InfoPanel:
    """Класс для отображения информации о симуляции"""
    
    def __init__(self, renderer):
        """
        Args:
            renderer: Ссылка на UniverseRenderer для доступа к состоянию и экрану
        """
        self.renderer = renderer
        self._max_laser_emitters_power_w = 0.0
        self._photon_mass_hud_last: tuple[str, tuple[int, int, int]] | None = None

    def reset_photon_mass_label_hold(self) -> None:
        """Сброс удерживаемой подписи массы фотона (renderer.reset)."""
        self._photon_mass_hud_last = None

    def draw_info(self, universe, cosmology, fps: float, masses=None):
        """Отрисовать информацию о симуляции"""
        renderer = self.renderer
        
        if not renderer.show_info:
            return
        
        # ОПТИМИЗАЦИЯ: Используем переданные массы вместо повторного вызова calculate_masses
        if masses is None:
            masses = renderer.calculate_masses(universe, cosmology)
        
        time_billion_years = universe.time / SECONDS_PER_BILLION_YEARS
        
        # Вычисляем красное смещение z = 1/a - 1
        scale_factor = cosmology.scale_factor
        if scale_factor > 0:
            redshift = 1.0 / scale_factor - 1.0
        else:
            redshift = 0.0
        
        # Температура реликтового излучения (микроволнового фона)
        # T = T0 * (1 + z), где T0 = 2.725 K - современная температура
        T0_CMB = 2.725  # Кельвины
        cmb_temperature = T0_CMB * (1.0 + redshift)
        
        # Вычисляем плотности материи и темной энергии
        omega_m = cosmology.omega_dm + cosmology.omega_b
        # Плотность материи убывает как a^-3
        rho_m = cosmology.rho_crit * omega_m / (scale_factor**3) if scale_factor > 0 else 0.0
        # Плотность темной энергии постоянна
        rho_lambda = cosmology.dark_energy_density()
        
        # Вычисляем текущий процент материи и темной энергии
        # от текущей полной плотности энергии
        total_density = rho_m + rho_lambda
        if total_density > 0:
            percent_matter = (rho_m / total_density) * 100.0
            percent_dark_energy = (rho_lambda / total_density) * 100.0
        else:
            percent_matter = 0.0
            percent_dark_energy = 0.0
            
        v_line, m_line = self._format_point_kinematics_lines(scale_factor)
        photon_mass_line, photon_mass_color = (
            self._format_single_photon_mass_line_with_color(scale_factor)
        )
        peak_laser_line = self._format_laser_peak_power_line(scale_factor, masses)
        all_photons_line = self._format_all_in_flight_photons_line(scale_factor)
        
        # Остальная информация (сверху)
        # Не показываем FPS во время паузы (чтобы не обновлялся)
        fps_line = "" if renderer.paused else f"FPS: {fps:.1f}"
        info_rows: list[tuple[str, tuple[int, int, int] | None]] = [
            (fps_line, None),
            (f"Time: {time_billion_years:.2f} billion years", None),
            (f"Scale factor a: {scale_factor:.4f}", None),
            (f"Redshift z: {redshift:.4f}", None),
            (f"CMB Temperature: {cmb_temperature:.3f} K", None),
            (f"H(t): {cosmology.hubble_parameter(universe.time) * 3.086e19:.2f} km/s/Mpc", None),
            ("", None),
            (f"Matter: {percent_matter:.3f}%", None),
            (f"Dark Energy: {percent_dark_energy:.3f}%", None),
            ("", None),
            (v_line, None),
            (m_line, None),
            (photon_mass_line, photon_mass_color),
            (peak_laser_line, None),
            (all_photons_line, None),
        ]
        
        # Отрисовка интерактивных кнопок вверху по центру
        mouse_pos = pygame.mouse.get_pos()
        
        # Общие размеры слайдеров
        slider_w = 200
        slider_h = 30
        knob_w = slider_w // 2
        
        total_top_width = ui.BUTTON_WIDTH * 2 + slider_w + ui.BUTTON_MARGIN * 2
        start_x = (renderer.width - total_top_width) // 2
        
        # 1. Play/Pause
        play_rect = pygame.Rect(start_x, ui.BUTTON_Y, ui.BUTTON_WIDTH, ui.BUTTON_HEIGHT)
        play_color = ui.BUTTON_PAUSED_COLOR if renderer.paused else ui.BUTTON_PLAYING_COLOR
        if play_rect.collidepoint(mouse_pos):
            play_color = (min(255, play_color[0]+30), min(255, play_color[1]+30), min(255, play_color[2]+30))
        pygame.draw.rect(renderer.screen, play_color, play_rect, border_radius=5)
        play_text_str = "PLAY" if renderer.paused else "PAUSE"
        play_text = renderer.font.render(play_text_str, True, ui.BUTTON_TEXT_COLOR)
        play_text_rect = play_text.get_rect(center=play_rect.center)
        renderer.screen.blit(play_text, play_text_rect)
        
        # 2. Reset
        reset_x = start_x + ui.BUTTON_WIDTH + ui.BUTTON_MARGIN
        reset_rect = pygame.Rect(reset_x, ui.BUTTON_Y, ui.BUTTON_WIDTH, ui.BUTTON_HEIGHT)
        reset_color = ui.BUTTON_RESET_HOVER_COLOR if reset_rect.collidepoint(mouse_pos) else ui.BUTTON_RESET_COLOR
        pygame.draw.rect(renderer.screen, reset_color, reset_rect, border_radius=5)
        reset_text = renderer.font.render("RESET", True, ui.BUTTON_TEXT_COLOR)
        reset_text_rect = reset_text.get_rect(center=reset_rect.center)
        renderer.screen.blit(reset_text, reset_text_rect)
        
        # 3. Слайдер Distribution Mode (вверху экрана, справа от кнопок)
        import config
        is_spiral = getattr(config, "MATTER_INITIAL_DISTRIBUTION", "spiral").strip().lower() == "spiral"
        dist_slider_x = reset_x + ui.BUTTON_WIDTH + ui.BUTTON_MARGIN
        dist_slider_y = ui.BUTTON_Y + (ui.BUTTON_HEIGHT - slider_h) // 2
        
        dist_rect = pygame.Rect(dist_slider_x, dist_slider_y, slider_w, slider_h)
        pygame.draw.rect(renderer.screen, (40, 40, 60), dist_rect, border_radius=15)
        
        dist_knob_x = dist_slider_x if is_spiral else dist_slider_x + knob_w
        dist_knob_rect = pygame.Rect(dist_knob_x, dist_slider_y, knob_w, slider_h)
        dist_knob_color = (150, 100, 200) if is_spiral else (100, 200, 150)
        
        if dist_rect.collidepoint(mouse_pos):
            dist_knob_color = (min(255, dist_knob_color[0]+30), min(255, dist_knob_color[1]+30), min(255, dist_knob_color[2]+30))
            
        pygame.draw.rect(renderer.screen, dist_knob_color, dist_knob_rect, border_radius=15)
        
        spiral_text = renderer.small_font.render("Spiral", True, (255, 255, 255))
        uniform_text = renderer.small_font.render("Uniform", True, (255, 255, 255))
        
        spiral_rect = spiral_text.get_rect(center=(dist_slider_x + knob_w//2, dist_slider_y + slider_h//2))
        uniform_rect = uniform_text.get_rect(center=(dist_slider_x + slider_w - knob_w//2, dist_slider_y + slider_h//2))
        
        renderer.screen.blit(spiral_text, spiral_rect)
        renderer.screen.blit(uniform_text, uniform_rect)
        
        # 4. Слайдер Coordinate Mode (внизу экрана)
        is_comoving = get_coordinate_display_mode() == "comoving"
        slider_x = (renderer.width - slider_w) // 2
        slider_y = renderer.height - 40
        
        mode_rect = pygame.Rect(slider_x, slider_y, slider_w, slider_h)
        # Фон слайдера
        pygame.draw.rect(renderer.screen, (40, 40, 60), mode_rect, border_radius=15)
        
        # Ползунок (активная часть)
        knob_x = slider_x + knob_w if is_comoving else slider_x
        knob_rect = pygame.Rect(knob_x, slider_y, knob_w, slider_h)
        knob_color = (80, 120, 200) if is_comoving else (200, 120, 80)
        
        # Подсветка при наведении
        if mode_rect.collidepoint(mouse_pos):
            knob_color = (min(255, knob_color[0]+30), min(255, knob_color[1]+30), min(255, knob_color[2]+30))
            
        pygame.draw.rect(renderer.screen, knob_color, knob_rect, border_radius=15)
        
        # Тексты
        phys_text = renderer.small_font.render("Physical", True, (255, 255, 255))
        comov_text = renderer.small_font.render("Comoving", True, (255, 255, 255))
        
        phys_rect = phys_text.get_rect(center=(slider_x + knob_w//2, slider_y + slider_h//2))
        comov_rect = comov_text.get_rect(center=(slider_x + slider_w - knob_w//2, slider_y + slider_h//2))
        
        renderer.screen.blit(phys_text, phys_rect)
        renderer.screen.blit(comov_text, comov_rect)
        
        renderer.ui_rects = {
            "play": play_rect,
            "reset": reset_rect,
            "mode": mode_rect,
            "dist": dist_rect,
        }

        y_offset = 10
        for line, color_override in info_rows:
            if line:
                line_color = (
                    color_override if color_override is not None else ui.INFO_TEXT_COLOR
                )
                text = renderer.small_font.render(line, True, line_color)
                renderer.screen.blit(text, (10, y_offset))
            y_offset += 20

        renderer.info_text_end_y = y_offset + 6

        # Чекбокс (правый верх): вкл — показывать фотоны за r_dS; выкл — обрезка по пунктиру.
        sq = 16
        gap = 8
        show_ds = getattr(renderer, "show_laser_photons_outside_ds_ref", True)
        label_ds = renderer.small_font.render(
            "Фотоны за горизонтом de Sitter",
            True,
            ui.INFO_TEXT_COLOR,
        )
        label_w = int(label_ds.get_width())
        label_h = int(label_ds.get_height())
        cb_y = ui.BUTTON_Y + (ui.BUTTON_HEIGHT - sq) // 2
        marg = ui.BUTTON_MARGIN
        cb_x = int(renderer.width) - marg - sq
        label_x = cb_x - gap - label_w
        label_y = cb_y + (sq - label_h) // 2

        pad_i = 6
        ds_hit_w = pad_i + label_w + gap + sq + pad_i
        ds_hit_x = label_x - pad_i
        if ds_hit_x < 2:
            ds_hit_x = 2
        ds_hit = pygame.Rect(
            ds_hit_x,
            cb_y - 4,
            ds_hit_w,
            max(label_h, sq) + 8,
        )
        hovered = ds_hit.collidepoint(mouse_pos)

        cb_rect = pygame.Rect(cb_x, cb_y, sq, sq)
        bg_cb = (90, 140, 220) if show_ds else (42, 42, 58)
        if hovered:
            bg_cb = tuple(min(255, c + 40) for c in bg_cb)
        pygame.draw.rect(renderer.screen, bg_cb, cb_rect, border_radius=3)
        pygame.draw.rect(renderer.screen, (210, 210, 235), cb_rect, 1, border_radius=3)
        if show_ds:
            pygame.draw.line(
                renderer.screen,
                (255, 255, 255),
                (cb_x + 3, cb_y + 8),
                (cb_x + 6, cb_y + 11),
                2,
            )
            pygame.draw.line(
                renderer.screen,
                (255, 255, 255),
                (cb_x + 6, cb_y + 11),
                (cb_x + 13, cb_y + 4),
                2,
            )
        renderer.screen.blit(label_ds, (label_x, label_y))

        renderer.ui_rects["photons_outside_ds"] = ds_hit

        # Панель с массами внизу слева
        self._draw_mass_panel(renderer, universe, cosmology, masses)
    
    def _nearest_laser_emitter_index(self, scale_factor: float) -> int | None:
        """Индекс самой ближней точки среди испускателей лазера (маска laser_emitter_mask)."""
        renderer = self.renderer
        try:
            mp = renderer.matter_points
            points_comoving = getattr(mp, 'points_comoving', None)
            laser_mask = getattr(mp, 'laser_emitter_mask', None)
        except AttributeError:
            return None
        if (
            points_comoving is None
            or len(points_comoving) == 0
            or scale_factor <= 0
        ):
            return None
        points = np.asarray(points_comoving, dtype=np.float64)
        distances = np.sqrt(np.einsum('ij,ij->i', points, points))
        candidate_mask = np.ones(len(distances), dtype=bool)
        if laser_mask is not None and len(laser_mask) == len(candidate_mask):
            laser_candidates = candidate_mask & np.asarray(laser_mask, dtype=bool)
            if np.any(laser_candidates):
                candidate_mask = laser_candidates
        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            return None
        return int(candidate_indices[np.argmin(distances[candidate_indices])])

    def _farthest_laser_emitter_index(self, scale_factor: float) -> int | None:
        """Индекс самой дальней точки среди испускателей лазера (маска laser_emitter_mask)."""
        renderer = self.renderer
        try:
            mp = renderer.matter_points
            points_comoving = getattr(mp, 'points_comoving', None)
            laser_mask = getattr(mp, 'laser_emitter_mask', None)
        except AttributeError:
            return None
        if (
            points_comoving is None
            or len(points_comoving) == 0
            or scale_factor <= 0
        ):
            return None
        points = np.asarray(points_comoving, dtype=np.float64)
        distances = np.sqrt(np.einsum('ij,ij->i', points, points))
        candidate_mask = np.ones(len(distances), dtype=bool)
        if laser_mask is not None and len(laser_mask) == len(candidate_mask):
            laser_candidates = candidate_mask & np.asarray(laser_mask, dtype=bool)
            if np.any(laser_candidates):
                candidate_mask = laser_candidates
        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            return None
        return int(candidate_indices[np.argmax(distances[candidate_indices])])
    
    def _m_point_pct_of_initial_suffix(self, m_rest_kg: float) -> str:
        """Скобки: процент массы покоя точки от get_mass_per_point_kg()."""
        m0 = float(get_mass_per_point_kg())
        if m0 <= 0.0:
            return ""
        pct = 100.0 * float(m_rest_kg) / m0
        return f" ({pct:.2f}%)"

    def _format_point_kinematics_lines(self, scale_factor: float):
        """Скорость и эффективная масса одной ближней лазерной точки."""
        renderer = self.renderer
        try:
            mp = renderer.matter_points
            v_com = mp.velocities_comoving
            masses_per_point = mp.masses_per_point
            points_comoving = getattr(mp, 'points_comoving', None)
        except AttributeError:
            v_com = None
            masses_per_point = None
            points_comoving = None
        
        m_rest = float(get_mass_per_point_kg())
        if masses_per_point is not None and len(masses_per_point) > 0:
            masses_arr = np.asarray(masses_per_point, dtype=np.float64)
            m_rest = float(masses_arr[-1])
        
        v_unknown = (
            v_com is None
            or len(v_com) == 0
            or points_comoving is None
            or len(points_comoving) == 0
            or scale_factor <= 0
        )
        if v_unknown:
            return (
                "Point v: —",
                f"Point m: {m_rest:.2e} kg{self._m_point_pct_of_initial_suffix(m_rest)}",
            )

        v_phys = np.asarray(v_com, dtype=np.float64) * float(scale_factor)
        speeds = np.sqrt(np.einsum('ij,ij->i', v_phys, v_phys))
        points = np.asarray(points_comoving, dtype=np.float64)
        if len(points) != len(speeds):
            return (
                "Point v: —",
                f"Point m: {m_rest:.2e} kg{self._m_point_pct_of_initial_suffix(m_rest)}",
            )

        distances = np.sqrt(np.einsum('ij,ij->i', points, points))
        if len(distances) == 0:
            return (
                "Point v: 0.0000 c",
                f"Point m: {m_rest:.2e} kg{self._m_point_pct_of_initial_suffix(m_rest)}",
            )

        nearest_idx = self._nearest_laser_emitter_index(scale_factor)
        if nearest_idx is None:
            return (
                "Point v: —",
                f"Point m: {m_rest:.2e} kg{self._m_point_pct_of_initial_suffix(m_rest)}",
            )

        v_point = float(speeds[nearest_idx])
        if masses_per_point is not None and len(masses_per_point) == len(speeds):
            m_rest = float(np.asarray(masses_per_point, dtype=np.float64)[nearest_idx])
        beta2 = min(v_point * v_point / (c * c), 1.0 - 1e-15)
        gamma = 1.0 / np.sqrt(1.0 - beta2)
        m_eff = gamma * m_rest
        return (
            f"Point v: {v_point / c:.4f} c",
            f"Point m: {m_eff:.2e} kg{self._m_point_pct_of_initial_suffix(m_rest)}",
        )

    def _format_single_photon_mass_line_with_color(
        self, scale_factor: float
    ) -> tuple[str, tuple[int, int, int]]:
        """Масса-эквивалент пакета с мин. a_emit; цвет как у маркера (a_emit/a_now)."""
        renderer = self.renderer
        try:
            mp = renderer.matter_points
            photon_masses = getattr(mp, '_laser_photon_mass_emit_kg', None)
            photon_a_emit = getattr(mp, '_laser_photon_a_emit', None)
            photon_r_emit = getattr(mp, '_laser_photon_r_emit_m', None)
        except AttributeError:
            photon_masses = None
            photon_a_emit = None
            photon_r_emit = None

        picked = False
        a_emit_picked = 0.0
        if (
            photon_masses is not None
            and photon_a_emit is not None
            and photon_r_emit is not None
            and len(photon_masses) > 0
            and len(photon_a_emit) == len(photon_masses)
            and len(photon_r_emit) == len(photon_masses)
            and scale_factor > 0
        ):
            a_all = np.asarray(photon_a_emit, dtype=np.float64)
            m_all = np.asarray(photon_masses, dtype=np.float64)
            r_emit_all = np.asarray(photon_r_emit, dtype=np.float64)
            
            # Получаем радиус черной дыры
            masses = renderer._cached_masses
            r_bh = masses.get('r_black_hole_schwarzschild_m', 0.0) if masses else 0.0
            
            idx = int(np.argmin(a_all))
            m_emit = float(m_all[idx])
            a_emit = float(a_all[idx])
            a_emit_picked = a_emit
            a_eff = float(scale_factor)
            r_emit = float(r_emit_all[idx])
            
            # Строгий расчет сохраняющейся энергии на бесконечности E_infinity:
            # m_photon = m_emit * (a_emit / a_eff) * sqrt(f(r_emit))
            f_emit = max(0.0, 1.0 - r_bh / max(r_emit, 1e-10))
            m_photon = m_emit * (a_emit / max(a_eff, 1e-300)) * np.sqrt(f_emit)
            picked = True

        if not picked:
            m_photon = 0.0
            m_emit = 0.0

        min_f_cfg = float(
            getattr(config, 'MATTER_LASER_PHOTON_DRAW_MIN_REMAINING_FRACTION', 0.0)
        )
        neutral = ui.INFO_TEXT_COLOR

        if not picked:
            if self._photon_mass_hud_last is not None:
                return self._photon_mass_hud_last
            return ("Photon m: 0.00e+00 kg (0.00%)", neutral)

        # Пол надписи по min_f: только космологическая доля × m_emit (без √f), чтобы
        # при min_f=0.1 скобки давали ~10%, а не 100×min_f×√f.
        m_min_disp = float(m_emit) * min_f_cfg if min_f_cfg > 0.0 else 0.0
        m_show = max(float(m_photon), m_min_disp)
        pct_of_emit = (
            100.0 * m_show / float(m_emit) if float(m_emit) > 0.0 else 0.0
        )

        if m_show <= 0.0:
            if self._photon_mass_hud_last is not None:
                return self._photon_mass_hud_last
            return ("Photon m: 0.00e+00 kg (0.00%)", neutral)

        # Для надписи: множитель a_emit/a_now не «краснеет» сильнее уровня порога (точки
        # с долей ниже порога не рисуются в draw_photons).
        from visualization.renderer import photon_rgb_blue_green_red

        cosmo_factor = float(a_emit_picked) / max(float(scale_factor), 1e-300)
        if min_f_cfg > 0.0:
            cosmo_factor = max(cosmo_factor, min_f_cfg)
        rgb = photon_rgb_blue_green_red(np.asarray([cosmo_factor], dtype=np.float64))[0]
        rgb_i = (
            int(np.clip(rgb[0], 0.0, 255.0)),
            int(np.clip(rgb[1], 0.0, 255.0)),
            int(np.clip(rgb[2], 0.0, 255.0)),
        )
        line = f"Photon m: {m_show:.2e} kg ({pct_of_emit:.2f}%)"
        out = (line, rgb_i)
        self._photon_mass_hud_last = out
        return out

    def _format_laser_peak_power_line(
        self, scale_factor: float, masses: dict | None
    ) -> str:
        """
        Только накопленный пик номинальной ΣP (Вт): s·(сумма масс покоя активных
        источников в момент эмиссии на шаге, до выгорания mass). См. MatterPoints.total_laser_emitters_power_w.
        """
        limit_to_planck = getattr(config, 'LIMIT_TOTAL_POWER_TO_PLANCK', False)
        spec = float(getattr(config, 'MATTER_THRUST_POWER_PER_KG_W', 0.0))
        pp_planck = float(PLANCK_POWER_W)
        if not limit_to_planck and spec <= 0.0:
            return "Peak ΣP: 0 W (0.00 × Planck)"
        try:
            mp = self.renderer.matter_points
            r_bh = None
            if masses is not None:
                r_bh = masses.get('r_black_hole_schwarzschild_m', 0.0)
            p_w = float(mp.total_laser_emitters_power_w(scale_factor, r_bh))
        except (AttributeError, TypeError, ValueError):
            p_w = 0.0
        self._max_laser_emitters_power_w = max(self._max_laser_emitters_power_w, p_w)
        p_mx = self._max_laser_emitters_power_w
        return (
            f"Peak ΣP: {p_mx:.2e} W ({p_mx / pp_planck:.2f} × Planck)"
        )

    def _format_all_in_flight_photons_line(self, scale_factor: float) -> str:
        """Суммарная масса-эквивалент всех лазерных фотонов в полёте и число пакетов."""
        try:
            mp = self.renderer.matter_points
            m_kg, n = mp.total_in_flight_laser_photon_mass_kg(scale_factor)
        except (AttributeError, TypeError, ValueError):
            m_kg, n = 0.0, 0
        return f"All photons: {m_kg:.2e} kg ({n})"

    def _draw_mass_panel(self, renderer, universe, cosmology, masses):
        """Отрисовать панель с массами внизу слева"""
        def format_mass_kg(mass_kg):
            """Форматировать массу в килограммах"""
            return f"{mass_kg:.2e} kg"
        
        # Проверяем, что masses не None
        if masses is None:
            masses = {
                'bh_growth_velocity_formatted': '0.00 m/s',
                'M_nariai_kg': 0.0,
                'M_black_hole_kg': 0.0,
                'M_hubble_horizon_kg': 0.0,
                'M_event_horizon_kg': 0.0,
                'M_de_sitter_horizon_kg': 0.0,
                'M_particle_horizon_kg': 0.0,
            }
        
        bh_growth_velocity_formatted = masses.get('bh_growth_velocity_formatted', '0.00 m/s')
        
        # Получаем масштабный фактор для вычисления comoving координат
        scale_factor = cosmology.scale_factor if cosmology.scale_factor > 0 else 1.0
        
        # Для горизонта частиц вычисляем радиус Шварцшильда
        M_black_hole_kg = masses.get('M_black_hole_kg', 0.0)
        M_nariai_kg = float(masses.get('M_nariai_kg', 0.0))
        if M_nariai_kg > 0.0:
            bh_frac_nariai = 100.0 * float(M_black_hole_kg) / M_nariai_kg
            central_bh_mass_str = (
                f"Central BH: {format_mass_kg(M_black_hole_kg)} "
                f"({bh_frac_nariai:.3f}%)"
            )
        else:
            central_bh_mass_str = f"Central BH: {format_mass_kg(M_black_hole_kg)}"
        black_hole_color = (
            ui.HORIZON_BLACK_HOLE_NARIAI_COLOR
            if M_black_hole_kg >= NARIAI_BLACK_HOLE_MASS_KG
            else ui.HORIZON_BLACK_HOLE_COLOR
        )
        M_particle = masses.get('M_particle_horizon_kg', 0.0)
        particle_horizon_r = cosmology.particle_horizon(universe.time)
        particle_horizon_billion_ly = particle_horizon_r / 9.461e24 if particle_horizon_r > 0 and particle_horizon_r < float('inf') else 0.0
        # Comoving горизонт частиц: chi_p = r_p / a
        chi_particle_bly = particle_horizon_billion_ly / scale_factor
        
        # Горизонт событий
        event_horizon_r = masses.get('r_event_horizon_m', 0.0)
        event_horizon_billion_ly = event_horizon_r / 9.461e24 if event_horizon_r > 0 and event_horizon_r < float('inf') else 0.0
        # Comoving горизонт событий: chi_e = r_e / a
        chi_event_bly = event_horizon_billion_ly / scale_factor
        
        particle_horizon_lines = []
        if M_particle > 0:
            r_schwarzschild_particle = 2 * G * M_particle / (c**2)
            r_schwarzschild_particle_bly = r_schwarzschild_particle / 9.461e24
            
            # Вычисляем процент от максимальной массы
            percentage = (M_particle / PARTICLE_HORIZON_MASS_LIMIT_KG) * 100
            
            particle_horizon_lines = [
                (f"Particle horizon: {format_mass_kg(M_particle)} ({percentage:.1f}%)", ui.HORIZON_PARTICLE_COLOR),
                (f"  r: {particle_horizon_billion_ly:.2f} billion ly", ui.HORIZON_PARTICLE_COLOR),
                (f"  χ: {chi_particle_bly:.2f} billion ly", ui.HORIZON_PARTICLE_COLOR),
                (f"  r_S: {r_schwarzschild_particle_bly:.4f} billion ly", ui.HORIZON_PARTICLE_COLOR),
            ]
        else:
            particle_horizon_lines = [
                (f"Particle horizon: {format_mass_kg(M_particle)}", ui.HORIZON_PARTICLE_COLOR),
                (f"  r: {particle_horizon_billion_ly:.2f} billion ly", ui.HORIZON_PARTICLE_COLOR),
                (f"  χ: {chi_particle_bly:.2f} billion ly", ui.HORIZON_PARTICLE_COLOR),
            ]
        
        # Сумма comoving горизонтов (должна быть константой ≈ 63.7 Gly)
        chi_sum = chi_particle_bly + chi_event_bly
        chi_sum_line = [
            (f"χ_p + χ_e = {chi_particle_bly:.2f} + {chi_event_bly:.2f} = {chi_sum:.2f}", ui.INFO_TEXT_COLOR),
        ]
        
        # Диагностика apparent outer (cosmological) horizon в чистом LTB-Λ:
        # M_eff = M_BH + M_matter_inside + M_laser_inside (полная масса
        # внутри outer AH, без Birkhoff-вычета). Показываем компоненты,
        # чтобы видеть, какой вклад двигает горизонт.
        ds_diag_lines = []
        if 'M_eff_de_sitter_kg' in masses:
            M_laser_in_ds = float(masses.get('M_laser_inside_de_sitter_kg', 0.0))
            M_eff_ds = float(masses.get('M_eff_de_sitter_kg', 0.0))
            r_ds_class = float(masses.get('r_de_sitter_classical_m', 0.0))
            r_ds_now = float(masses.get('r_de_sitter_horizon_m', 0.0))
            shrink_bly = (r_ds_class - r_ds_now) / 9.461e24
            ds_diag_lines = [
                (f"  M_laser in dS: {format_mass_kg(M_laser_in_ds)}", ui.HORIZON_DE_SITTER_COLOR),
                (f"  M_eff (LTB): {format_mass_kg(M_eff_ds)}", ui.HORIZON_DE_SITTER_COLOR),
                (f"  shrink: {shrink_bly:.3f} bly", ui.HORIZON_DE_SITTER_COLOR),
            ]

        # Диагностика apparent inner horizon ЦЧД (trapped surface).
        # Показываем разницу между «классическим» 2GM/c² и фактическим
        # apparent horizon — это та оболочка, которая поглощается ЦЧД сверх
        # точечной массы (см. physics/apparent_horizon.py).
        bh_diag_lines = []
        if 'r_apparent_inner_horizon_m' in masses:
            r_ah = float(masses.get('r_apparent_inner_horizon_m', 0.0))
            r_bh_class = float(masses.get('r_black_hole_classical_m', 0.0))
            shell_bly = max(r_ah - r_bh_class, 0.0) / 9.461e24
            r_ah_bly = r_ah / 9.461e24
            r_bh_class_bly = r_bh_class / 9.461e24
            bh_diag_lines = [
                (f"  r_AH (inner): {r_ah_bly:.3e} bly", black_hole_color),
                (f"  r_BH classical: {r_bh_class_bly:.3e} bly", black_hole_color),
                (f"  Δr (capture shell): {shell_bly:.3e} bly", black_hole_color),
            ]

        mass_lines = [
            ("MASSES:", ui.INFO_TEXT_COLOR),
            (central_bh_mass_str, black_hole_color),
            (f"  Growth speed: {bh_growth_velocity_formatted}", black_hole_color),
        ] + bh_diag_lines + [
            (f"Hubble horizon: {format_mass_kg(masses['M_hubble_horizon_kg'])}", ui.HORIZON_HUBBLE_COLOR),
            (f"Event horizon: {format_mass_kg(masses['M_event_horizon_kg'])}", ui.HORIZON_EVENT_COLOR),
            (f"de Sitter horizon: {format_mass_kg(masses['M_de_sitter_horizon_kg'])}", ui.HORIZON_DE_SITTER_COLOR),
        ] + ds_diag_lines + particle_horizon_lines + chi_sum_line
        
        # Вычисляем позицию для масс внизу экрана
        mass_panel_height = len([l for l, _ in mass_lines if l]) * 20
        y_offset_mass = renderer.height - mass_panel_height - 10
        
        for line, text_color in mass_lines:
            if line:
                text = renderer.small_font.render(line, True, text_color)
                renderer.screen.blit(text, (10, y_offset_mass))
            y_offset_mass += 20
    
    def horizon_point_counts(self, universe, cosmology, masses=None):
        """Счётчики числа точек внутри каждого горизонта — текст в правом нижнем углу."""
        renderer = self.renderer
        
        if renderer.matter_points.points_comoving is None or len(renderer.matter_points.points_comoving) == 0:
            return
        
        try:
            # ОПТИМИЗАЦИЯ: Используем переданные массы
            if masses is None:
                masses = renderer.calculate_masses(universe, cosmology)
            r_black_hole = masses.get('r_black_hole_schwarzschild_m', 0.0) if masses else 0.0
            M_black_hole_kg = masses.get('M_black_hole_kg', 0.0) if masses else 0.0
            
            # ОПТИМИЗАЦИЯ: Используем кэшированные радиусы из masses
            if masses and 'r_particle_horizon_m' in masses:
                particle_horizon_physical = masses.get('r_particle_horizon_m', 0.0)
                hubble_r = masses.get('r_hubble_horizon_m', 0.0)
                event_r = masses.get('r_event_horizon_m', 0.0)
            else:
                # Fallback: вычисляем горизонты (старый путь)
                particle_horizon_physical = cosmology.particle_horizon(universe.time)
                hubble_r = float(str(cosmology.hubble_horizon(universe.time, M_black_hole_kg)))
                event_r = float(str(cosmology.cosmological_event_horizon(universe.time, M_black_hole_kg)))
            # Счётчик «de Sitter» совпадает с пунктиром: пустая Λ, не apparent r_dS(M).
            de_sitter_r = float(str(cosmology.de_sitter_horizon(0.0)))
            
            # Получаем физические координаты точек и расстояния
            physical_points, distances_from_center, scale_ratio = renderer._get_physical_points_and_distances(
                universe, cosmology, particle_horizon_physical
            )
            
            if physical_points is None:
                return
            
            # Подсчитываем точки внутри каждого горизонта
            total_points = len(distances_from_center)
            
            # Подсчитываем количество точек внутри ЧД
            if r_black_hole > 0 and r_black_hole < float('inf'):
                points_in_black_hole = np.sum(distances_from_center <= r_black_hole)
            else:
                points_in_black_hole = np.sum(distances_from_center == 0.0)
            
            # Подсчёт точек в горизонтах
            if len(distances_from_center) > 0:
                points_in_hubble = np.sum(distances_from_center <= hubble_r) if hubble_r > 0 and hubble_r < float('inf') else 0
                points_in_event = np.sum(distances_from_center <= event_r) if event_r > 0 and event_r < float('inf') else 0
                points_in_de_sitter = np.sum(distances_from_center <= de_sitter_r) if de_sitter_r > 0 and de_sitter_r < float('inf') else 0
                points_in_particle = np.sum(distances_from_center <= particle_horizon_physical) if particle_horizon_physical > 0 and particle_horizon_physical < float('inf') else 0
            else:
                points_in_hubble = points_in_event = points_in_de_sitter = points_in_particle = 0
            
            # Отрисовка
            text_color = ui.INFO_TEXT_COLOR
            white_color = (255, 255, 255)
            line_height = 20
            x_offset = 10
            
            texts = [
                (f"Central BH: {points_in_black_hole}", ui.HORIZON_BLACK_HOLE_COLOR),
                (f"Hubble: {points_in_hubble}", ui.HORIZON_HUBBLE_COLOR),
                (f"Event: {points_in_event}", ui.HORIZON_EVENT_COLOR),
                (f"de Sitter: {points_in_de_sitter}", white_color),
                (f"Particle: {points_in_particle}", ui.HORIZON_PARTICLE_COLOR),
                (f"Total: {total_points}", text_color),
            ]
            
            points_panel_height = (1 + len(texts)) * line_height
            start_y = renderer.height - points_panel_height - 10
            
            # Заголовок
            try:
                header_text = renderer.small_font.render("Points count:", True, text_color)
                header_x = renderer.width - header_text.get_width() - x_offset
                renderer.screen.blit(header_text, (header_x, start_y))
                start_y += line_height
            except:
                pass
            
            y_pos = start_y
            for text_str, color in texts:
                try:
                    text_surface = renderer.small_font.render(text_str, True, color)
                    text_x = renderer.width - text_surface.get_width() - x_offset
                    renderer.screen.blit(text_surface, (text_x, y_pos))
                    y_pos += line_height
                except:
                    pass
                    
        except Exception as e:
            if DEBUG:
                print(f"Error drawing horizon point counts: {e}")
