"""Лабораторная 2.2. Решение системы нелинейных уравнений."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.io import fmt, print_vector
from utils.nonlinear import system_newton, system_simple_iterations
from utils.nonlinear_plots import save_convergence_plot, save_system_plot
from utils.test_runner import load_data


# Печатает зависимость погрешности от номера итерации.
def print_history(name, history):
    print(f"\n{name}")
    print(f"{'k':>4}  {'приближение':>30}  {'невязка':>14}  {'погрешность':>14}")
    for item in history:
        vector = "[ " + "  ".join(fmt(value) for value in item["value"]) + " ]"
        print(
            f"{item['iteration']:>4}  "
            f"{vector:>30}  "
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
        functions = test["functions"]
        iteration_functions = test["iteration_functions"]
        initial = test["initial"]
        parameters = test.get("parameters", {})
        epsilon = test.get("eps", data.get("eps", 1e-6))

        reference, _ = system_newton(
            functions,
            initial,
            1e-12,
            parameters = parameters,
        )
        simple_solution, simple_history = system_simple_iterations(
            functions,
            iteration_functions,
            initial,
            epsilon,
            reference,
            parameters,
        )
        newton_solution, newton_history = system_newton(
            functions,
            initial,
            epsilon,
            reference,
            parameters,
        )
        variant = test.get("number", test_number)
        system_plot = plots_dir / f"2.2_variant_{variant}_system.png"
        convergence_plot = plots_dir / f"2.2_variant_{variant}_convergence.png"
        if len(initial) == 2 and len(functions) == 2:
            save_system_plot(
                functions,
                parameters,
                initial,
                simple_history,
                newton_history,
                reference,
                system_plot,
            )
        save_convergence_plot(simple_history, newton_history, epsilon, convergence_plot)

        print(f"\n=== Вариант {variant} ===")
        print("Система:")
        for expression in functions:
            print("  ", expression, "= 0")
        print("Параметры:", parameters)
        print_vector("Начальное приближение", initial)
        print("Точность:", epsilon)
        print_vector("Эталонное решение", reference)
        print_vector("Решение методом простых итераций", simple_solution)
        print("Количество простых итераций:", len(simple_history))
        print_vector("Решение методом Ньютона", newton_solution)
        print("Количество итераций Ньютона:", len(newton_history))
        print_history("Метод простых итераций", simple_history)
        print_history("Метод Ньютона", newton_history)
        if len(initial) == 2 and len(functions) == 2:
            print("График системы:", system_plot)
        else:
            print("График системы строится только для двух неизвестных.")
        print("График сходимости:", convergence_plot)


if __name__ == "__main__":
    main()
