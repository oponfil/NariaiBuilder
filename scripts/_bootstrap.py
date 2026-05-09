"""
Добавляет корень проекта в sys.path, чтобы CLI-скрипты в этой папке могли
импортировать пакеты `utils`, `physics`, `config` без ручной правки PYTHONPATH.

Использование (первой строкой в каждом скрипте, до импортов из `utils`/`physics`):

    try:
        import _bootstrap  # noqa: F401  -- python scripts/<name>.py
    except ModuleNotFoundError:
        from scripts import _bootstrap  # noqa: F401  -- from scripts.<name> import ...

Двойной импорт нужен потому, что директория `scripts/` попадает в `sys.path`
автоматически только при прямом запуске файла. При импорте `scripts.<name>`
из другого модуля (например, из `simulator.py`) приходится тянуть bootstrap
через сам пакет `scripts`.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
