"""
JSON-кэш с секциями для скриптов-симуляций.

Все скрипты в `scripts/` пишут результаты в один общий файл
(`data/nariai_simulation_data.json`), но каждый владеет своей секцией
верхнего уровня (например `successful_runs`, `mass_vs_power`,
`planck_limit_runs`). Этот класс инкапсулирует загрузку, безопасное
обновление и сохранение таких секций, чтобы один скрипт не затирал
данные другого.
"""
import json
import os
from typing import Any


class JsonCache:
    """Тонкая обёртка над JSON-файлом со секциями верхнего уровня.

    Использование:
        cache = JsonCache(path)
        section = cache.get_section("successful_runs", default={"x": [], "y": []})
        section["x"].append(...)
        cache.set_section("successful_runs", section)  # save() вызывается автоматически
    """

    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    print(f"[JsonCache] {self.path}: ожидался dict, получен {type(loaded).__name__}. Кэш будет пересоздан.")
                    return {}
                return loaded
        except Exception as e:
            print(f"[JsonCache] Не удалось прочитать {self.path}: {e}. Кэш будет пересоздан.")
            return {}

    def get_section(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = {}
        return self._data.get(key, default)

    def set_section(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def reset_section(self, key: str, template: Any) -> None:
        """Затирает секцию шаблоном и сохраняет файл (используется при --force)."""
        self._data[key] = template
        self.save()

    def has_section(self, key: str) -> bool:
        return key in self._data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=4)
