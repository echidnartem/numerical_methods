"""Лабораторная 2.1. Решение одного нелинейного уравнения."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.io import fmt
from utils.nonlinear import (
    bisection_reference,
    evaluate,
    scalar_newton,
    scalar_simple_iterations,
)
from utils.nonlinear_plots import save_convergence_plot, save_scalar_plot
from utils.test_runner import load_data


# Печатает зависимость погрешности от номера итерации.
def print_history(name, history):
    print(f"\n{name}")
    print(f"{'k':>4}  {'x':>14}  {'|f(x)|':>14}  {'погрешность':>14}")
    for item in history:
        print(
            f"{item['iteration']:>4}  "
            f"{fmt(item['value']):>14}  "
            f"{fmt(item['residual']):>14}  "
            f"{fmt(item['error']):>14}"
        )


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Укажите путь к JSON-файлу с данными.")

    data = load_data(sys.argv[1])
    plots_dir = pathlib.Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(exist_ok = True)

    for test_number, test in enumerate(data.get("tests", [data]), start = 1):
        expression = test["function"]
        parameters = test.get("parameters", {})
        function = lambda x: evaluate(expression, {"x": x} | parameters)
        left, right = test["interval"]
        initial = test.get("initial", (left + right) / 2)
        epsilon = test.get("eps", data.get("eps", 1e-6))
        reference = bisection_reference(function, left, right)

        simple_root, simple_history, parameter = scalar_simple_iterations(
            function, initial, left, right, epsilon, reference
        )
        newton_root, newton_history = scalar_newton(
            function, initial, epsilon, reference
        )
        variant = test.get("number", test_number)
        function_plot = plots_dir / f"2.1_variant_{variant}_function.png"
        convergence_plot = plots_dir / f"2.1_variant_{variant}_convergence.png"
        save_scalar_plot(
            function, left, right, initial, simple_root, newton_root, function_plot
        )
        save_convergence_plot(simple_history, newton_history, epsilon, convergence_plot)

        print(f"\n=== Вариант {variant} ===")
        print("Уравнение: f(x) =", expression)
        print("Отрезок локализации:", [left, right])
        print("Начальное приближение:", initial)
        print("Точность:", epsilon)
        print("Параметр простой итерации:", fmt(parameter))
        print("Эталонный корень:", fmt(reference))
        print("Корень методом простых итераций:", fmt(simple_root))
        print("Количество простых итераций:", len(simple_history))
        print("Корень методом Ньютона:", fmt(newton_root))
        print("Количество итераций Ньютона:", len(newton_history))
        print_history("Метод простых итераций", simple_history)
        print_history("Метод Ньютона", newton_history)
        print("График функции:", function_plot)
        print("График сходимости:", convergence_plot)


if __name__ == "__main__":
    main()
