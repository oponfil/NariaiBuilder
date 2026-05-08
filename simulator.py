"""
Главный файл запуска симуляции Вселенной
"""
import os
import sys
import traceback
# ОПТИМИЗАЦИЯ: Пробуем использовать GPU ускорение для более быстрой отрисовки
# Закомментировано: windib драйвер (GDI) - используем по умолчанию DirectX/OpenGL для Windows
# os.environ['SDL_VIDEODRIVER'] = 'windib'  # Раскомментируйте, если нужен CPU рендеринг

from physics.objects import Universe
from physics.cosmology import LambdaCDM
from visualization.renderer import UniverseRenderer
from utils.config_utils import get_initial_time_seconds, get_dt
import config


def create_test_universe() -> tuple[Universe, LambdaCDM]:
    """Создать тестовую Вселенную с однородной жидкостью (космологический принцип Lambda CDM)"""
    universe = Universe()
    cosmology = LambdaCDM()
    
    # В Lambda CDM модели Вселенная заполнена однородной жидкостью
    # Нет отдельных объектов - только однородное распределение материи
    # Плотность материи будет визуализироваться через карту плотности (клавиша P)
    
    # Начинаем с времени: 1 миллиард лет после Большого взрыва
    # Параметры старта симуляции берутся из config
    universe.time = get_initial_time_seconds()
    
    return universe, cosmology


def safe_print(text):
    """Безопасный вывод текста с обработкой кодировки"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Если не удается вывести, пробуем без кириллицы
        try:
            print(text.encode('ascii', 'ignore').decode('ascii'))
        except:
            pass


def main():
    """Главная функция"""
    safe_print("Starting Universe 2D simulation...")
    safe_print("Lambda CDM + GR")
    
    # Создаем Вселенную
    universe, cosmology = create_test_universe()
    
    # Создаем рендерер (параметры из config)
    safe_print("Creating renderer...")
    renderer = UniverseRenderer()
    safe_print("Renderer created")
    
    # Параметры симуляции (из config)
    max_fps = config.MAX_FPS
    dt = get_dt()
    
    safe_print("Simulation started!")
    safe_print("Press ESC or close window to exit")
    
    running = True
    while running:
        # Обработка ввода
        action = renderer.handle_input(universe, cosmology)
        if action == "QUIT":
            break
        elif action == "RESET":
            universe, cosmology = create_test_universe()
            renderer.reset()
            continue

        
        # Обновление времени и космологии только если не на паузе
        if not renderer.paused:
            # Обновление времени (сначала обновляем время)
            universe.time += dt
            
            # Остановка, если достигнут предел времени
            from utils.constants import SECONDS_PER_YEAR
            from scripts.precompute_horizons import PRECOMPUTE_TIME_END_YEARS
            max_time_seconds = PRECOMPUTE_TIME_END_YEARS * SECONDS_PER_YEAR
            if universe.time >= max_time_seconds:
                universe.time = max_time_seconds
                renderer.paused = True
                safe_print(f"Time limit reached: {PRECOMPUTE_TIME_END_YEARS/1e9:.1f} billion years. Simulation paused.")
            
            # Обновление космологии (расширение Вселенной)
            cosmology.update_scale_factor(dt)
            
            # Сохраняем историю для численного интегрирования горизонтов
            cosmology.time_history.append(universe.time)
            cosmology.scale_factor_history.append(cosmology.scale_factor)
        
        # В Lambda CDM модели нет отдельных объектов для интегрирования
        # Вселенная заполнена однородной жидкостью, которая расширяется согласно космологии
        
        # Рендеринг (всегда, даже на паузе)
        fps = renderer.tick(max_fps)
        renderer.render(universe, cosmology, fps)
    
    safe_print("Simulation finished.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        safe_print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
