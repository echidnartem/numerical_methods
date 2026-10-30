"""Графики для лабораторной работы по нелинейным уравнениям."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.nonlinear import system_values
from utils.plotting import save_png


# Показывает функцию, отрезок локализации, начальную точку и найденные корни.
def save_scalar_plot(function, left, right, initial, simple_root, newton_root, path):
    points = [left + (right - left) * index / 400 for index in range(401)]
    values = [function(point) for point in points]

    figure, axes = plt.subplots(figsize = (8, 5))
    axes.plot(points, values, label = "f(x)")
    axes.axhline(0, color = "black", linewidth = 0.8)
    axes.axvline(left, color = "gray", linestyle = ":", label = "Границы отрезка")
    axes.axvline(right, color = "gray", linestyle = ":")
    axes.scatter(
        [initial], [function(initial)], color = "black", zorder = 5,
        label = "Начальное приближение",
    )
    axes.scatter(
        [simple_root], [0], color = "tab:orange", zorder = 6,
        label = "Простые итерации",
    )
    axes.scatter(
        [newton_root], [0], marker = "x", s = 90, color = "tab:red",
        zorder = 7, label = "Ньютон",
    )
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.grid(True, alpha = 0.3)
    axes.legend()
    figure.tight_layout()
    save_png(figure, path)
    plt.close(figure)


# Показывает линии обоих уравнений и путь итераций от начальной точки.
def save_system_plot(
    functions, parameters, initial, simple_history, newton_history, reference, path
):
    all_points = [initial, reference]
    all_points.extend(item["value"] for item in simple_history)
    all_points.extend(item["value"] for item in newton_history)

    bounds = []
    for coordinate in (0, 1):
        values = [point[coordinate] for point in all_points]
        minimum = min(values)
        maximum = max(values)
        margin = max(0.3 * (maximum - minimum), 0.25 * max(abs(maximum), 1.0))
        left = minimum - margin
        if minimum > 0:
            left = max(1e-6, left)
        bounds.append((left, maximum + margin))

    x_values = np.linspace(*bounds[0], 160)
    y_values = np.linspace(*bounds[1], 160)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    first_values = np.empty_like(x_grid)
    second_values = np.empty_like(x_grid)

    for row, x2 in enumerate(y_values):
        for column, x1 in enumerate(x_values):
            try:
                first, second = system_values(functions, [x1, x2], parameters)
            except (ValueError, ZeroDivisionError, OverflowError):
                first, second = np.nan, np.nan
            first_values[row, column] = first
            second_values[row, column] = second

    figure, axes = plt.subplots(figsize = (8, 6))
    axes.contour(
        x_grid, y_grid, first_values, levels = [0],
        colors = ["tab:blue"], linewidths = 2,
    )
    axes.contour(
        x_grid, y_grid, second_values, levels = [0],
        colors = ["tab:green"], linewidths = 2,
    )
    axes.plot([], [], color = "tab:blue", label = "f₁(x₁, x₂) = 0")
    axes.plot([], [], color = "tab:green", label = "f₂(x₁, x₂) = 0")

    for history, color, label in (
        (simple_history, "tab:orange", "Простые итерации"),
        (newton_history, "tab:red", "Ньютон"),
    ):
        points = [initial] + [item["value"] for item in history]
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            "o--",
            color = color,
            markersize = 3,
            alpha = 0.8,
            label = label,
        )

    axes.scatter(
        [initial[0]], [initial[1]], color = "black", zorder = 6,
        label = "Начальная точка",
    )
    axes.scatter(
        [reference[0]], [reference[1]], marker = "*", s = 160,
        color = "purple", zorder = 7, label = "Решение",
    )
    axes.set_xlabel("x₁")
    axes.set_ylabel("x₂")
    axes.set_xlim(*bounds[0])
    axes.set_ylim(*bounds[1])
    axes.grid(True, alpha = 0.3)
    axes.legend()
    figure.tight_layout()
    save_png(figure, path)
    plt.close(figure)


# Сравнивает абсолютные ошибки и невязки двух методов по итерациям.
def save_convergence_plot(simple_history, newton_history, epsilon, path):
    figure, axes = plt.subplots(1, 2, figsize = (11, 4.5))

    for history, label in (
        (simple_history, "Простые итерации"),
        (newton_history, "Ньютон"),
    ):
        iterations = [item["iteration"] for item in history]
        errors = [max(item["error"], 1e-16) for item in history]
        residuals = [max(item["residual"], 1e-16) for item in history]
        axes[0].semilogy(iterations, errors, "o-", label = label)
        axes[1].semilogy(iterations, residuals, "o-", label = label)

    for axis, title in zip(axes, ("Ошибка относительно эталона", "Невязка")):
        axis.axhline(epsilon, color = "gray", linestyle = ":", label = "Заданная точность")
        axis.set_title(title)
        axis.set_xlabel("Номер итерации")
        axis.grid(True, which = "both", alpha = 0.3)
        axis.legend()

    axes[0].set_ylabel("Абсолютная ошибка")
    axes[1].set_ylabel("Невязка")
    figure.tight_layout()
    save_png(figure, path)
    plt.close(figure)
