"""
Утилиты для форматирования физических величин для отображения
"""
from utils.constants import c


def format_velocity_m_per_s(velocity_m_per_s: float) -> str:
    """
    Форматировать скорость в долях от скорости света
    
    Args:
        velocity_m_per_s: Скорость в метрах в секунду
    
    Returns:
        Отформатированная строка со скоростью в долях от скорости света (формат: .000 c)
    """
    fraction_of_c = velocity_m_per_s / c
    return f"{fraction_of_c:.3f} c"
