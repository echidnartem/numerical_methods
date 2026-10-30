"""Построение интерполяционных многочленов по табличным данным."""


# Проверяет, что узлы и значения образуют корректную таблицу.
def validate_nodes(nodes, values):
    if len(nodes) != len(values) or not nodes:
        raise ValueError("Число узлов должно совпадать с числом значений")
    if len(set(nodes)) != len(nodes):
        raise ValueError("Узлы интерполяции должны быть различными")


# Складывает многочлены, заданные коэффициентами при x в степенях 0, 1, ...
def add_polynomials(first, second):
    result = [0.0] * max(len(first), len(second))
    for index, value in enumerate(first):
        result[index] += value
    for index, value in enumerate(second):
        result[index] += value
    return result


# Перемножает два многочлена в степенном базисе.
def multiply_polynomials(first, second):
    result = [0.0] * (len(first) + len(second) - 1)
    for i, first_value in enumerate(first):
        for j, second_value in enumerate(second):
            result[i + j] += first_value * second_value
    return result


# Строит коэффициенты многочлена Лагранжа в степенном базисе.
def lagrange_coefficients(nodes, values):
    validate_nodes(nodes, values)
    coefficients = [0.0] * len(nodes)

    for i, value in enumerate(values):
        basis = [1.0]
        denominator = 1.0
        for j, node in enumerate(nodes):
            if i == j:
                continue
            basis = multiply_polynomials(basis, [-node, 1.0]) # домножаем на (x - x_j)
            denominator *= nodes[i] - node
        coefficients = add_polynomials(
            coefficients, [value * coefficient / denominator for coefficient in basis]
        )

    return coefficients


# Вычисляет многочлен Лагранжа по базисным произведениям в заданной точке.
def lagrange_value(nodes, values, point):
    validate_nodes(nodes, values)
    result = 0.0

    for i, value in enumerate(values):
        basis = 1.0
        for j, node in enumerate(nodes):
            if i != j:
                basis *= (point - node) / (nodes[i] - node)
        result += value * basis

    return result


# Вычисляет разделённые разности для многочлена Ньютона.
def divided_differences(nodes, values):
    return divided_difference_table(nodes, values)[0]


# Строит треугольную таблицу разделённых разностей и коэффициенты Ньютона.
def divided_difference_table(nodes, values):
    validate_nodes(nodes, values)
    table = [values[:]]

    for order in range(1, len(nodes)):
        previous = table[-1]
        current = []
        for i in range(len(previous) - 1):
            difference = (
                previous[i + 1] - previous[i]
            ) / (nodes[i + order] - nodes[i])
            current.append(difference)
        table.append(current)

    coefficients = [column[0] for column in table]
    return coefficients, table


# Переводит форму Ньютона в коэффициенты при степенях x.
def newton_power_coefficients(nodes, differences):
    coefficients = [0.0] * len(nodes)
    basis = [1.0]

    for index, difference in enumerate(differences):
        coefficients = add_polynomials(
            coefficients, [difference * coefficient for coefficient in basis]
        )
        if index < len(nodes) - 1:
            basis = multiply_polynomials(basis, [-nodes[index], 1.0])

    return coefficients


# Вычисляет значение многочлена по схеме Горнера.
def polynomial_value(coefficients, point):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


# Вычисляет многочлен Ньютона без перехода к степенному базису.
def newton_value(nodes, differences, point):
    result = differences[-1]
    for index in range(len(differences) - 2, -1, -1):
        result = differences[index] + (point - nodes[index]) * result
    return result
