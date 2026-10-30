import math

from utils.linalg import norm, solve_lu


ALLOWED_FUNCTIONS = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "lg": math.log10,
    "ln": math.log,
    "log": math.log,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


# Вычисляет математическое выражение с заданными значениями переменных.
def evaluate(expression, variables):
    namespace = ALLOWED_FUNCTIONS | variables
    return eval(expression, {"__builtins__": {}}, namespace)


# Численно вычисляет производную функции в точке.
def numerical_derivative(function, point, step = 1e-6):
    return (function(point + step) - function(point - step)) / (2 * step)


# Находит эталонный корень на отрезке методом деления пополам.
def bisection_reference(function, left, right, eps = 1e-12, max_iter = 10000):
    left_value = function(left)
    right_value = function(right)

    if left_value * right_value > 0:
        raise ValueError("На концах отрезка функция должна иметь разные знаки")

    for _ in range(max_iter):
        middle = (left + right) / 2
        middle_value = function(middle)

        if abs(middle_value) < eps or (right - left) / 2 < eps:
            return middle

        if left_value * middle_value <= 0:
            right = middle
        else:
            left = middle
            left_value = middle_value

    raise RuntimeError("Эталонный метод не сошёлся")


# Выбирает постоянный параметр итерации по оценкам производной на отрезке.
def relaxation_parameter(function, left, right, samples = 1000):
    derivatives = []

    for index in range(samples + 1):
        point = left + (right - left) * index / samples
        derivatives.append(numerical_derivative(function, point))

    minimum = min(derivatives)
    maximum = max(derivatives)
    if minimum <= 0 <= maximum:
        raise ValueError("Производная меняет знак на выбранном отрезке")

    return 2 / (minimum + maximum)


# Решает одно нелинейное уравнение методом простых итераций.
def scalar_simple_iterations(
    function,
    initial,
    left,
    right,
    eps,
    reference,
    max_iter = 10000,
):
    parameter = relaxation_parameter(function, left, right)
    current = initial
    history = []

    for iteration in range(1, max_iter + 1):
        next_value = current - parameter * function(current)
        history.append(
            {
                "iteration": iteration,
                "value": next_value,
                "residual": abs(function(next_value)),
                "error": abs(next_value - reference),
            }
        )

        if abs(next_value - current) < eps and abs(function(next_value)) < eps:
            return next_value, history, parameter

        current = next_value

    raise RuntimeError("Метод простых итераций не сошёлся")


# Решает одно нелинейное уравнение методом Ньютона.
def scalar_newton(function, initial, eps, reference, max_iter = 10000):
    current = initial
    history = []

    for iteration in range(1, max_iter + 1):
        derivative = numerical_derivative(function, current)
        if abs(derivative) < 1e-14:
            raise ZeroDivisionError("Производная близка к нулю")

        next_value = current - function(current) / derivative
        history.append(
            {
                "iteration": iteration,
                "value": next_value,
                "residual": abs(function(next_value)),
                "error": abs(next_value - reference),
            }
        )

        if abs(next_value - current) < eps and abs(function(next_value)) < eps:
            return next_value, history

        current = next_value

    raise RuntimeError("Метод Ньютона не сошёлся")


# Вычисляет значения функций системы в заданной точке.
def system_values(functions, point, parameters = None):
    variables = {f"x{index + 1}": value for index, value in enumerate(point)}
    variables.update(parameters or {})
    return [evaluate(expression, variables) for expression in functions]


# Численно строит матрицу Якоби системы в заданной точке.
def numerical_jacobian(functions, point, parameters = None, step = 1e-6):
    size = len(point)
    jacobian = [[0.0] * size for _ in range(size)]

    for column in range(size):
        point_left = point[:]
        point_right = point[:]
        point_left[column] -= step
        point_right[column] += step
        values_left = system_values(functions, point_left, parameters)
        values_right = system_values(functions, point_right, parameters)

        for row in range(size):
            jacobian[row][column] = (
                values_right[row] - values_left[row]
            ) / (2 * step)

    return jacobian


# Решает систему нелинейных уравнений методом простых итераций.
def system_simple_iterations(
    functions,
    iteration_functions,
    initial,
    eps,
    reference,
    parameters = None,
    max_iter = 10000,
):
    current = initial[:]
    history = []

    for iteration in range(1, max_iter + 1):
        variables = {f"x{index + 1}": value for index, value in enumerate(current)}
        variables.update(parameters or {})
        next_value = [
            evaluate(expression, variables) for expression in iteration_functions
        ]
        residual = norm(system_values(functions, next_value, parameters))
        error = norm(
            [next_value[index] - reference[index] for index in range(len(current))]
        )
        history.append(
            {
                "iteration": iteration,
                "value": next_value,
                "residual": residual,
                "error": error,
            }
        )

        difference = norm(
            [next_value[index] - current[index] for index in range(len(current))]
        )
        if difference < eps and residual < eps:
            return next_value, history

        current = next_value

    raise RuntimeError("Метод простых итераций не сошёлся")


# Решает систему нелинейных уравнений методом Ньютона.
def system_newton(
    functions,
    initial,
    eps,
    reference = None,
    parameters = None,
    max_iter = 10000,
):
    current = initial[:]
    history = []

    for iteration in range(1, max_iter + 1):
        values = system_values(functions, current, parameters)
        jacobian = numerical_jacobian(functions, current, parameters)
        correction = solve_lu(jacobian, [-value for value in values])
        next_value = [
            current[index] + correction[index] for index in range(len(current))
        ]
        residual = norm(system_values(functions, next_value, parameters))
        error = None

        if reference is not None:
            error = norm(
                [next_value[index] - reference[index] for index in range(len(current))]
            )

        history.append(
            {
                "iteration": iteration,
                "value": next_value,
                "residual": residual,
                "error": error,
            }
        )

        if norm(correction) < eps and residual < eps:
            return next_value, history

        current = next_value

    raise RuntimeError("Метод Ньютона не сошёлся")
