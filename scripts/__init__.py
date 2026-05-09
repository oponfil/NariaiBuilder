"""
Пакет с CLI-скриптами и утилитами для них.

Существует только для того, чтобы `from scripts.<name> import ...` работало
гарантированно (например, `simulator.py` импортирует
`scripts.precompute_horizons`). Сами скрипты остаются точками входа и
запускаются как `python scripts/<name>.py`.
"""
