"""
Модуль обработки пользовательского ввода для симуляции
"""
import pygame
import config
from utils.constants import SECONDS_PER_YEAR
from utils.cosmology_utils import calculate_scale_factor_at_time



class InputHandler:
    """Класс для обработки ввода пользователя"""
    
    def __init__(self, renderer):
        """
        Args:
            renderer: Ссылка на UniverseRenderer для доступа к состоянию
        """
        self.renderer = renderer
    
    def handle_input(self, universe=None, cosmology=None) -> bool:
        """
        Обработать ввод пользователя
        Returns: True если нужно продолжить, False если выйти
        """
        renderer = self.renderer
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Пробел или P для паузы/старта
                elif event.key == pygame.K_SPACE or event.key == pygame.K_p:
                    renderer.paused = not renderer.paused
            # Обработка изменения размера окна
            elif event.type == pygame.VIDEORESIZE:
                # ВАЖНО: Пересоздаем поверхность экрана с новым размером
                renderer.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                # Обновляем размеры после пересоздания поверхности
                renderer.width = event.w
                renderer.height = event.h
                renderer.camera_x = renderer.width // 2
                renderer.camera_y = renderer.height // 2
        
        # Перемотка времени клавишами влево/вправо на паузе
        if renderer.paused and universe is not None and cosmology is not None:
            keys = pygame.key.get_pressed()
            dt_seconds = config.DT_YEARS * SECONDS_PER_YEAR
            
            if keys[pygame.K_LEFT]:
                # Отматываем назад. ВАЖНО: сначала меняем время, потом обновляем космологию
                # (или наоборот, но важно соблюдать порядок как в основном цикле, но с отрицательным dt)
                # В main.py: time += dt, затем update_scale_factor(dt). Повторяем эту логику.
                # Но проверяем на < 0
                if universe.time > dt_seconds:
                    dt = -dt_seconds
                    universe.time = max(dt_seconds, universe.time + dt)
                    cosmology.update_scale_factor(dt)
                    renderer.invalidate_mass_cache()
                    renderer._manual_cosmic_step_this_frame = True
                    renderer._manual_cosmic_dt_signed = dt
                
            if keys[pygame.K_RIGHT]:
                dt = dt_seconds
                universe.time += dt
                cosmology.update_scale_factor(dt)
                renderer.invalidate_mass_cache()
                renderer._manual_cosmic_step_this_frame = True
                renderer._manual_cosmic_dt_signed = dt
        
        return True
