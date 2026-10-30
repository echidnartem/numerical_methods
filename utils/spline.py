"""Натуральный кубический сплайн по табличным данным."""

from utils.interpolation import validate_nodes
from utils.linalg import tridiagonal_solve


# Строит коэффициенты сплайна на каждом интервале относительно левого узла.
def natural_cubic_spline(nodes, values):
    validate_nodes(nodes, values)
    if len(nodes) < 2:
        raise ValueError("Для сплайна нужны хотя бы два узла")
    if any(nodes[index + 1] <= nodes[index] for index in range(len(nodes) - 1)):
        raise ValueError("Узлы сплайна должны идти по возрастанию")

    steps = [nodes[index + 1] - nodes[index] for index in range(len(nodes) - 1)]
    inner_count = len(nodes) - 2
    second_derivatives = [0.0] * len(nodes)

    if inner_count:
        diagonal = [2 * (steps[i - 1] + steps[i]) for i in range(1, len(nodes) - 1)]
        off_diagonal = [steps[i] for i in range(1, len(nodes) - 2)]
        right_side = [
            6 * (
                (values[i + 1] - values[i]) / steps[i]
                - (values[i] - values[i - 1]) / steps[i - 1]
            )
            for i in range(1, len(nodes) - 1)
        ]

        if inner_count == 1:
            inner_values = [right_side[0] / diagonal[0]]
        else:
            _, _, inner_values = tridiagonal_solve(
                off_diagonal, diagonal, off_diagonal, right_side
            )
        second_derivatives[1:-1] = inner_values

    segments = []
    for index, step in enumerate(steps):
        left_second = second_derivatives[index]
        right_second = second_derivatives[index + 1]
        a = values[index]
        b = (
            (values[index + 1] - values[index]) / step
            - step * (2 * left_second + right_second) / 6
        )
        c = left_second / 2
        d = (right_second - left_second) / (6 * step)
        segments.append((a, b, c, d))

    return segments, second_derivatives


# Находит номер участка сплайна для точки внутри диапазона узлов.
def spline_interval(nodes, point):
    if point < nodes[0] or point > nodes[-1]:
        raise ValueError("Точка должна лежать внутри диапазона узлов")

    for index in range(len(nodes) - 1):
        if point <= nodes[index + 1]:
            return index

    return len(nodes) - 2


# Вычисляет значение сплайна на заданном участке.
def spline_segment_value(segment, left_node, point):
    a, b, c, d = segment
    distance = point - left_node
    return a + b * distance + c * distance ** 2 + d * distance ** 3


# Вычисляет значение сплайна и возвращает номер содержащего точку интервала.
def spline_value(nodes, segments, point):
    index = spline_interval(nodes, point)
    return spline_segment_value(segments[index], nodes[index], point), index
