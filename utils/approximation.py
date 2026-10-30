"""Приближение табличной функции многочленом методом наименьших квадратов."""

from utils.interpolation import polynomial_value
from utils.linalg import solve_lu


# Строит нормальную систему и находит коэффициенты многочлена МНК.
def least_squares_polynomial(nodes, values, degree):
    if len(nodes) != len(values) or len(nodes) < degree + 1:
        raise ValueError("Для МНК нужны согласованные таблицы и достаточно точек")
    if len(set(nodes)) < degree + 1:
        raise ValueError("Для МНК нужны различные значения аргумента")

    size = degree + 1
    matrix = [
        [sum(node ** (row + column) for node in nodes) for column in range(size)]
        for row in range(size)
    ]
    right_side = [
        sum(value * node ** row for node, value in zip(nodes, values))
        for row in range(size)
    ]
    return solve_lu(matrix, right_side)


# Считает сумму квадратов отклонений на заданных узлах.
def squared_error_sum(nodes, values, coefficients):
    return sum(
        (polynomial_value(coefficients, node) - value) ** 2
        for node, value in zip(nodes, values)
    )


# Считает средний квадрат отклонений по всем исходным точкам.
def mean_squared_error(nodes, values, coefficients):
    return squared_error_sum(nodes, values, coefficients) / len(nodes)
