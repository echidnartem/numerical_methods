import numbers


# Преобразует число в строку с удобным форматом.
def fmt(value):
    if isinstance(value, complex):
        return f"{value.real:.6f}{value.imag:+.6f}i"

    if isinstance(value, numbers.Real):
        if abs(value) < 1e-12:
            value = 0.0
        return f"{float(value):.6f}"

    return str(value)


# Печатает матрицу построчно.
def print_matrix(name, matrix):
    formatted_rows = [[fmt(value) for value in row] for row in matrix]
    column_count = len(formatted_rows[0])
    column_widths = [
        max(len(row[column]) for row in formatted_rows)
        for column in range(column_count)
    ]

    print(f"{name}:")
    for row in formatted_rows:
        values = "  ".join(
            value.rjust(column_widths[index])
            for index, value in enumerate(row)
        )
        print(f"| {values} |")
    print()


# Печатает вектор в едином числовом формате.
def print_vector(name, vector):
    values = "  ".join(fmt(value) for value in vector)
    print(f"{name}: [ {values} ]")


# Форматирует коэффициенты многочлена для печати.
def format_polynomial(coefficients, variable = "x"):
    terms = []
    for degree in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[degree]
        if abs(coefficient) < 1e-12:
            continue

        factor = f"{abs(coefficient):.6f}"
        if degree > 0:
            factor += f" * {variable}"
        if degree > 1:
            factor += f"^{degree}"
        sign = "-" if coefficient < 0 else "+"
        terms.append((sign, factor))

    if not terms:
        return "0"

    first_sign, first_factor = terms[0]
    result = f"-{first_factor}" if first_sign == "-" else first_factor
    for sign, factor in terms[1:]:
        result += f" {sign} {factor}"
    return result
