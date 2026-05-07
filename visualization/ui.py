"""
Параметры внешнего вида и раскладки интерфейса (окно, шкалы, горизонты, подписи).

Симуляция и физика остаются в config.py; здесь только то, что видит пользователь.
"""
from typing import Tuple

# Окно
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 20)
WINDOW_TITLE = "Universe 2D Simulation - Lambda CDM + GR"

# Шрифты
FONT_SIZE_LARGE = 24
FONT_SIZE_SMALL = 18

# Масштабная линейка
RULER_LENGTH_PX = 100
RULER_X = 20
RULER_Y_OFFSET = 45
RULER_TICK_HEIGHT = 5
RULER_COLOR: Tuple[int, int, int] = (255, 255, 255)
RULER_PHYSICAL_TEXT = "10 billion ly (physical)"
RULER_COMOVING_COLOR: Tuple[int, int, int] = (160, 200, 255)
RULER_COMOVING_OFFSET_PX = 8
RULER_COMOVING_TEXT = "10 billion ly (comoving)"

# Горизонты (линии и подписи на экране)
HORIZON_LINE_WIDTH = 1
HORIZON_HUBBLE_COLOR: Tuple[int, int, int] = (255, 200, 0)
HORIZON_DE_SITTER_COLOR: Tuple[int, int, int] = (0, 255, 255)
HORIZON_EVENT_COLOR: Tuple[int, int, int] = (150, 150, 150)
HORIZON_PARTICLE_COLOR: Tuple[int, int, int] = (255, 100, 100)
HORIZON_BLACK_HOLE_COLOR: Tuple[int, int, int] = (200, 0, 200)
HORIZON_BLACK_HOLE_NARIAI_COLOR: Tuple[int, int, int] = (0, 150, 0)
HORIZON_HUBBLE_LABEL = "Hubble"
HORIZON_DE_SITTER_LABEL = "de Sitter"
HORIZON_EVENT_LABEL = "Event"
HORIZON_PARTICLE_LABEL = "Particle"
HORIZON_BLACK_HOLE_LABEL = "BH Event"
HORIZON_HUBBLE_OFFSET_Y = -35
HORIZON_DE_SITTER_OFFSET_Y = 0
HORIZON_EVENT_OFFSET_Y = 35
HORIZON_PARTICLE_OFFSET_Y = 70
HORIZON_BLACK_HOLE_OFFSET_Y = -70

# Инфопанель
INFO_TEXT_COLOR: Tuple[int, int, int] = (200, 200, 200)

# Кнопки UI
BUTTON_WIDTH = 110
BUTTON_HEIGHT = 30
BUTTON_Y = 10
BUTTON_MARGIN = 10
BUTTON_COLOR: Tuple[int, int, int] = (50, 50, 70)
BUTTON_HOVER_COLOR: Tuple[int, int, int] = (80, 80, 110)
BUTTON_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
BUTTON_PLAYING_COLOR: Tuple[int, int, int] = (40, 140, 40)
BUTTON_PAUSED_COLOR: Tuple[int, int, int] = (160, 140, 0)
BUTTON_RESET_COLOR: Tuple[int, int, int] = (140, 40, 40)
BUTTON_RESET_HOVER_COLOR: Tuple[int, int, int] = (180, 60, 60)
