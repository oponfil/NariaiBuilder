"""
Утилиты для вычисления производных значений из конфигурации
"""
import config
from visualization.ui import RULER_LENGTH_PX
from utils.constants import (
    SECONDS_PER_YEAR,
    LIGHT_YEAR_METERS,
    TOTAL_MATTER_MASS_KG,
)


def get_initial_time_seconds():
    """Возвращает начальное время в секундах"""
    initial_years = getattr(config, 'INITIAL_TIME_YEARS', config.DT_YEARS)
    return initial_years * SECONDS_PER_YEAR


def get_collapse_start_time_seconds():
    """Возвращает время начала коллапса в секундах"""
    return config.LASER_START_TIME_YEARS * SECONDS_PER_YEAR


def get_dt():
    """Возвращает шаг времени в секундах"""
    return config.DT_YEARS * SECONDS_PER_YEAR


def get_one_billion_ly():
    """Возвращает 1 миллиард световых лет в метрах"""
    return 1e9 * LIGHT_YEAR_METERS


def get_ten_billion_ly():
    """Возвращает 10 миллиардов световых лет в метрах"""
    return 10e9 * LIGHT_YEAR_METERS


def get_pixel_to_meter():
    """Возвращает масштаб преобразования: метры на пиксель"""
    return get_ten_billion_ly() / RULER_LENGTH_PX


def get_mass_per_point_kg() -> float:
    """Масса одной точки материи (кг): TOTAL_MATTER_MASS_KG / MATTER_NUM_POINTS."""
    return TOTAL_MATTER_MASS_KG / config.MATTER_NUM_POINTS


_COORD_MODE_PHYSICAL = "physical"
_COORD_MODE_COMOVING = "comoving"
_COORD_MODES_VALID = frozenset({_COORD_MODE_PHYSICAL, _COORD_MODE_COMOVING})

# Кэш нормализованного режима + предупреждённое значение, чтобы не печатать спам каждый кадр.
_coord_mode_cache: dict = {"raw": object(), "mode": _COORD_MODE_PHYSICAL, "warned": None}


def get_coordinate_display_mode() -> str:
    """Текстовый режим отображения координат: "physical" или "comoving".

    Читает `config.COORDINATE_DISPLAY_MODE`, нормализует регистр и возвращает
    "physical" при некорректном значении. Предупреждение печатается один раз
    на каждое некорректное значение.
    """
    raw = getattr(config, "COORDINATE_DISPLAY_MODE", _COORD_MODE_PHYSICAL)
    if raw is _coord_mode_cache["raw"]:
        return _coord_mode_cache["mode"]
    mode = str(raw).strip().lower()
    if mode not in _COORD_MODES_VALID:
        if _coord_mode_cache["warned"] != raw:
            print(
                f"[WARN] config.COORDINATE_DISPLAY_MODE={raw!r} is invalid; "
                f"falling back to {_COORD_MODE_PHYSICAL!r}. "
                f"Allowed values: {sorted(_COORD_MODES_VALID)}"
            )
            _coord_mode_cache["warned"] = raw
        mode = _COORD_MODE_PHYSICAL
    _coord_mode_cache["raw"] = raw
    _coord_mode_cache["mode"] = mode
    return mode


def is_comoving_display() -> bool:
    """True, если конфиг просит отображение в сопутствующих координатах."""
    return get_coordinate_display_mode() == _COORD_MODE_COMOVING

def toggle_coordinate_display_mode() -> str:
    """Переключить режим отображения координат и вернуть новый режим."""
    current = get_coordinate_display_mode()
    new_mode = _COORD_MODE_PHYSICAL if current == _COORD_MODE_COMOVING else _COORD_MODE_COMOVING
    
    # Переписываем значение в config.py
    setattr(config, "COORDINATE_DISPLAY_MODE", new_mode)
    
    # Сбрасываем кэш
    _coord_mode_cache["raw"] = new_mode
    _coord_mode_cache["mode"] = new_mode
    
    return new_mode

def toggle_matter_distribution_mode() -> str:
    """Переключить режим распределения точек материи."""
    current = getattr(config, "MATTER_INITIAL_DISTRIBUTION", "spiral").strip().lower()
    new_mode = "uniform" if current == "spiral" else "spiral"
    setattr(config, "MATTER_INITIAL_DISTRIBUTION", new_mode)
    return new_mode
