"""
Модуль отрисовки космологических горизонтов
"""
import numpy as np
import pygame

from config import DEBUG
import visualization.ui as ui
from utils.config_utils import get_one_billion_ly, get_ten_billion_ly, is_comoving_display


class HorizonsRenderer:
    """Класс для отрисовки космологических горизонтов"""
    
    def __init__(self, renderer):
        """
        Args:
            renderer: Ссылка на UniverseRenderer для доступа к экрану и шрифтам
        """
        self.renderer = renderer
    
    def draw_horizons(self, universe, cosmology, masses=None):
        """Отрисовать космологические горизонты и горизонт событий центральной ЧД"""
        renderer = self.renderer
        
        if not renderer.show_horizons:
            return
        
        # Космологические горизонты привязаны к наблюдателю (центру экрана)
        center_x = renderer.width // 2
        center_y = renderer.height // 2
        
        # Фиксированная шкала
        TEN_BILLION_LY = get_ten_billion_ly()
        RULER_LENGTH_PX = ui.RULER_LENGTH_PX
        
        # Получаем массу ЧД
        if masses is None:
            masses_temp = renderer.calculate_masses(universe, cosmology)
            M_black_hole_kg = masses_temp.get('M_black_hole_kg', 0.0) if masses_temp else 0.0
        else:
            M_black_hole_kg = masses.get('M_black_hole_kg', 0.0) if masses else 0.0
        
        try:
            # Вычисляем все горизонты
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

            # В режиме "comoving" переводим физический радиус в комовинг (делим на a).
            scale_factor = getattr(cosmology, "scale_factor", 1.0) or 1.0
            comoving = is_comoving_display()
            display_a = scale_factor if (comoving and scale_factor > 0) else 1.0

            # Функция масштабирования
            def scale_radius(physical_radius):
                """Масштабировать физический радиус в пиксели"""
                if physical_radius <= 0 or physical_radius >= float('inf'):
                    return 0
                display_radius = (
                    physical_radius / display_a if comoving else physical_radius
                )
                pixels = (display_radius / TEN_BILLION_LY) * RULER_LENGTH_PX
                radius_int = int(pixels)
                if not np.isfinite(radius_int) or radius_int < 0:
                    return 0
                max_radius = int(np.sqrt(renderer.width**2 + renderer.height**2)) + 1000
                if radius_int > max_radius:
                    radius_int = max_radius
                return radius_int
            
            # Рисуем горизонты (в порядке от меньшего к большему)
            self._draw_horizon(center_x, center_y, hubble_r, scale_radius, 
                             ui.HORIZON_HUBBLE_COLOR, ui.HORIZON_HUBBLE_LABEL, 
                             ui.HORIZON_HUBBLE_OFFSET_Y, cosmology=cosmology)
            
            self._draw_horizon(center_x, center_y, de_sitter_r, scale_radius,
                             ui.HORIZON_DE_SITTER_COLOR, ui.HORIZON_DE_SITTER_LABEL,
                             ui.HORIZON_DE_SITTER_OFFSET_Y, cosmology=cosmology)
            
            self._draw_horizon(center_x, center_y, event_r, scale_radius,
                             ui.HORIZON_EVENT_COLOR, ui.HORIZON_EVENT_LABEL,
                             ui.HORIZON_EVENT_OFFSET_Y, cosmology=cosmology)
            
            self._draw_particle_horizon(center_x, center_y, particle_r, scale_radius, RULER_LENGTH_PX, cosmology=cosmology)
            
            # Горизонт событий ЧД
            if masses is not None and 'r_black_hole_schwarzschild_m' in masses:
                r_black_hole = masses['r_black_hole_schwarzschild_m']
                self._draw_black_hole_horizon(
                    center_x, center_y, r_black_hole, scale_radius, M_black_hole_kg,
                    cosmology=cosmology,
                )
        
        except (ValueError, TypeError) as e:
            pass
    
    def _draw_horizon(self, center_x, center_y, radius_physical, scale_func, color, label, offset_y, cosmology=None):
        """Отрисовать один космологический горизонт"""
        renderer = self.renderer
        
        if radius_physical < float('inf') and radius_physical > 0:
            radius_int = scale_func(radius_physical)
            if radius_int > 5:
                center_x_int = int(float(str(center_x)))
                center_y_int = int(float(str(center_y)))
                
                # Проверяем видимость
                if (center_x_int + radius_int >= 0 and center_x_int - radius_int < renderer.width and
                    center_y_int + radius_int >= 0 and center_y_int - radius_int < renderer.height):
                    pygame.draw.circle(
                        renderer.screen, 
                        color, 
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                
                # Название и радиус
                label_x = center_x_int + radius_int + 5
                label_y = center_y_int + offset_y
                if 0 < label_x < renderer.width - 100 and 0 < label_y < renderer.height:
                    text = renderer.small_font.render(label, True, color)
                    renderer.screen.blit(text, (label_x, label_y))
                    
                    display_radius = self._physical_to_display(radius_physical, cosmology)
                    radius_text = f"{display_radius / 9.461e24:.2f}"
                    radius_label = renderer.small_font.render(radius_text, True, color)
                    radius_y = label_y + 18
                    if 0 < label_x < renderer.width - 100 and 0 < radius_y < renderer.height:
                        renderer.screen.blit(radius_label, (label_x, radius_y))

    def _physical_to_display(self, physical_radius: float, cosmology=None) -> float:
        """Перевод физического радиуса в радиус отображения (комовинг в режиме "comoving")."""
        if not is_comoving_display():
            return float(physical_radius)
        a = getattr(cosmology, "scale_factor", 1.0) if cosmology is not None else 1.0
        a = float(a) if a else 1.0
        if a <= 0:
            return float(physical_radius)
        return float(physical_radius) / a
    
    def _draw_particle_horizon(self, center_x, center_y, particle_r, scale_func, ruler_length_px, cosmology=None):
        """Отрисовать горизонт частиц с отладочным выводом"""
        renderer = self.renderer
        
        if particle_r < float('inf') and particle_r > 0:
            radius_int = scale_func(particle_r)
            
            # Отладка
            if DEBUG and not hasattr(self, '_debug_horizon_drawn'):
                print("=" * 60)
                print("DEBUG: Drawing particle horizon")
                print("=" * 60)
                print(f"Particle horizon (physical): {particle_r/9.461e24:.4f} billion ly")
                print(f"Particle horizon (screen radius in pixels): {radius_int} px")
                print(f"Screen center: ({center_x}, {center_y})")
                print(f"Scale: 10 billion ly = {ruler_length_px} px")
                print(f"Calculation: ({particle_r/9.461e24:.4f} / 10.0) * {ruler_length_px} = {radius_int} px")
                print("=" * 60)
                self._debug_horizon_drawn = True
            
            if radius_int > 5:
                center_x_int = int(float(str(center_x)))
                center_y_int = int(float(str(center_y)))
                
                if (center_x_int + radius_int >= 0 and center_x_int - radius_int < renderer.width and
                    center_y_int + radius_int >= 0 and center_y_int - radius_int < renderer.height):
                    pygame.draw.circle(
                        renderer.screen, 
                        ui.HORIZON_PARTICLE_COLOR, 
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                
                # Название и радиус
                label_x = center_x_int + radius_int + 5
                label_y = center_y_int + ui.HORIZON_PARTICLE_OFFSET_Y
                if 0 < label_x < renderer.width - 100 and 0 < label_y < renderer.height:
                    text = renderer.small_font.render(ui.HORIZON_PARTICLE_LABEL, True, ui.HORIZON_PARTICLE_COLOR)
                    renderer.screen.blit(text, (label_x, label_y))
                    
                    radius_text = f"{self._physical_to_display(particle_r, cosmology) / 9.461e24:.2f}"
                    radius_label = renderer.small_font.render(radius_text, True, ui.HORIZON_PARTICLE_COLOR)
                    radius_y = label_y + 18
                    if 0 < label_x < renderer.width - 100 and 0 < radius_y < renderer.height:
                        renderer.screen.blit(radius_label, (label_x, radius_y))
    
    def _draw_black_hole_horizon(self, center_x, center_y, r_black_hole, scale_func, mass_kg, cosmology=None):
        """Отрисовать горизонт событий центральной черной дыры"""
        renderer = self.renderer
        if hasattr(renderer, '_black_hole_color_for_mass'):
            black_hole_color = renderer._black_hole_color_for_mass(mass_kg)
        else:
            black_hole_color = ui.HORIZON_BLACK_HOLE_COLOR
        
        if r_black_hole > 0 and r_black_hole < float('inf'):
            radius_int = scale_func(r_black_hole)
            if radius_int > 1:
                center_x_int = int(float(str(center_x)))
                center_y_int = int(float(str(center_y)))
                
                if (center_x_int + radius_int >= 0 and center_x_int - radius_int < renderer.width and
                    center_y_int + radius_int >= 0 and center_y_int - radius_int < renderer.height):
                    # Заливка
                    pygame.draw.circle(
                        renderer.screen, 
                        black_hole_color,
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=0
                    )
                    # Контур
                    pygame.draw.circle(
                        renderer.screen, 
                        black_hole_color,
                        (center_x_int, center_y_int), 
                        radius_int, 
                        width=ui.HORIZON_LINE_WIDTH
                    )
                
                # Название и радиус
                label_x = center_x_int + radius_int + 5
                label_y = center_y_int + ui.HORIZON_BLACK_HOLE_OFFSET_Y
                if 0 < label_x < renderer.width - 100 and 0 < label_y < renderer.height:
                    text = renderer.small_font.render(ui.HORIZON_BLACK_HOLE_LABEL, True, black_hole_color)
                    renderer.screen.blit(text, (label_x, label_y))
                    
                    radius_text = f"{self._physical_to_display(r_black_hole, cosmology) / get_one_billion_ly():.2f}"
                    radius_label = renderer.small_font.render(radius_text, True, black_hole_color)
                    radius_y = label_y + 18
                    if 0 < label_x < renderer.width - 100 and 0 < radius_y < renderer.height:
                        renderer.screen.blit(radius_label, (label_x, radius_y))
