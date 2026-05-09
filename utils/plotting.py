"""
Шаблоны для графиков, которые рисуют CLI-скрипты.

Все эти графики делят одни и те же настройки (фигура 10×6, лог-шкала по Y,
сетка по обеим осям, маркер 'o', сортировка по X). Раньше этот блок копировался
в каждый скрипт, теперь вызывается через `plot_vs_time`.
"""
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FIGSIZE = (10, 6)
DEFAULT_XLABEL = 'Laser Start Time (Billion Years)'


def plot_vs_time(
    times: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: str,
    color: str = 'b',
    yscale: str = 'log',
    xlabel: str = DEFAULT_XLABEL,
    marker: str = 'o',
    label: str | None = None,
    extra: callable = None,
) -> None:
    """Нарисовать «Y vs время» с сортировкой по X и сохранить в файл.

    `extra` — необязательный колбэк, получает текущий axes и может добавить
    дополнительные элементы (например, вертикальную линию предела Нариаи).
    """
    times_arr = np.asarray(times, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    sort_idx = np.argsort(times_arr)
    sorted_times = times_arr[sort_idx]
    sorted_values = values_arr[sort_idx]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(sorted_times, sorted_values, marker=marker, linestyle='-', color=color, label=label)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    if label is not None:
        ax.legend()
    if extra is not None:
        extra(ax)

    fig.savefig(out_path)
    print(f"График сохранён: {out_path}")


def show_open_figures() -> None:
    """Показать все открытые фигуры; молча игнорирует отсутствие GUI-бэкенда."""
    try:
        plt.show()
    except Exception:
        pass
