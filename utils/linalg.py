import cmath
import math

import numpy as np


# Создаёт нулевую матрицу размера rows x columns.
def zeros(rows, columns):
    return [[0.0 for _ in range(columns)] for _ in range(rows)]

# Создаёт единичную матрицу размера size x size.
def eye(size):
    matrix = zeros(size, size)
    for index in range(size):
        matrix[index][index] = 1.0
    return matrix

# Транспонирует матрицу.
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Вычисляет евклидову норму вектора.
def norm(vector):
    return math.sqrt(sum(abs(value) ** 2 for value in vector))

# Находит максимальный внедиагональный элемент симметричной матрицы.
def max_offdiag(matrix):
    size = len(matrix)
    row_index = 0
    column_index = 0
    maximum = 0.0

    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j]) > maximum:
                maximum = abs(matrix[i][j])
                row_index = i
                column_index = j

    return row_index, column_index, maximum

# Выполняет LU-разложение с частичным выбором главного элемента.
def lu_decompose(matrix, eps = 1e-14):
    size = len(matrix)
    upper = [row[:] for row in matrix]
    lower = eye(size)
    permutation = eye(size)
    permutation_sign = 1.0  # знак перестановок для вычисления определителя

    for k in range(size):
        pivot = max(range(k, size), key=lambda i: abs(upper[i][k]))
        if abs(upper[pivot][k]) < eps:
            raise ValueError("Матрица вырожденная")

        if pivot != k:
            upper[k], upper[pivot] = upper[pivot], upper[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
            permutation_sign *= -1

            for j in range(k):
                # меняем не полностью строку в L, а её уже собранную часть
                lower[k][j], lower[pivot][j] = lower[pivot][j], lower[k][j]

        for i in range(k + 1, size):
            lower[i][k] = upper[i][k] / upper[k][k]
            for j in range(k, size):
                upper[i][j] -= lower[i][k] * upper[k][j]

    return lower, upper, permutation, permutation_sign

# Решает систему уравнений через LU-разложение.
def solve_lu(matrix, right_side, decomposition = None):
    if decomposition is None:
        decomposition = lu_decompose(matrix)

    lower, upper, permutation, _ = decomposition
    permuted_right_side = np.matmul(permutation, right_side).tolist()
    size = len(right_side)
    intermediate = [0.0] * size # промежуточное решение для Ly = Pb
    solution = [0.0] * size

    for i in range(size): # Ly = Pb
        intermediate[i] = permuted_right_side[i] - sum(
            lower[i][j] * intermediate[j] for j in range(i)
        )

    for i in range(size - 1, -1, -1): # Ux = y
        solution[i] = (
            intermediate[i]
            - sum(upper[i][j] * solution[j] for j in range(i + 1, size))
        ) / upper[i][i]

    return solution

# Строит обратную матрицу через решение систем с единичными правыми частями.
def inverse_lu(matrix, decomposition = None):
    if decomposition is None:
        decomposition = lu_decompose(matrix)

    lower, upper, permutation, _ = decomposition
    size = len(matrix)
    columns = []

    for column in range(size):
        right_side = [1.0 if i == column else 0.0 for i in range(size)]
        columns.append(solve_lu(matrix, right_side, decomposition))

    return transpose(columns) # в columns наши столбцы хранились как строки, поэтому транспонируем

# Вычисляет определитель матрицы через LU-разложение.
def determinant_lu(matrix, decomposition = None):
    if decomposition is None:
        decomposition = lu_decompose(matrix)

    _, upper, _, permutation_sign = decomposition
    determinant = permutation_sign

    for i in range(len(matrix)):
        determinant *= upper[i][i]

    return determinant

# Решает трёхдиагональную систему методом прогонки.
def tridiagonal_solve(lower, diagonal, upper, right_side):
    size = len(diagonal)
    alpha = [0.0] * size
    beta = [0.0] * size

    alpha[0] = -upper[0] / diagonal[0]
    beta[0] = right_side[0] / diagonal[0]

    for i in range(1, size):
        denominator = diagonal[i] + lower[i - 1] * alpha[i - 1]
        alpha[i] = -upper[i] / denominator if i < size - 1 else 0.0
        beta[i] = (right_side[i] - lower[i - 1] * beta[i - 1]) / denominator

    solution = [0.0] * size
    solution[-1] = beta[-1]

    for i in range(size - 2, -1, -1):
        solution[i] = alpha[i] * solution[i + 1] + beta[i]

    return alpha, beta, solution

# Решает СЛАУ методом простых итераций (методом Якоби).
def simple_iterations(matrix, right_side, eps = 1e-8, max_iter = 10000):
    size = len(matrix)
    solution = [0.0] * size

    for iteration in range(1, max_iter + 1):
        next_solution = [
            (
                right_side[i] - 
                sum(matrix[i][j] * solution[j] for j in range(size) if j != i)
            ) / matrix[i][i] for i in range(size)
        ]

        if norm([next_solution[i] - solution[i] for i in range(size)]) < eps:
            return next_solution, iteration
        solution = next_solution

    return solution, max_iter

# Решает СЛАУ методом последовательных приближений Зейделя.
def seidel(matrix, right_side, eps = 1e-8, max_iter = 10000):
    size = len(matrix)
    solution = [0.0] * size

    for iteration in range(1, max_iter + 1):
        old_solution = solution[:]

        for i in range(size):
            before = sum(matrix[i][j] * solution[j] for j in range(i))
            after = sum(matrix[i][j] * solution[j] for j in range(i + 1, size))
            solution[i] = (right_side[i] - before - after) / matrix[i][i]

        if norm([solution[i] - old_solution[i] for i in range(size)]) < eps:
            return solution, iteration

    return solution, max_iter

# Находит собственные значения и векторы симметричной матрицы методом Якоби.
def jacobi_eigen(matrix, eps = 1e-10, max_iter = 10000):
    size = len(matrix)
    diagonalized = [row[:] for row in matrix]
    eigenvectors = eye(size)

    for iteration in range(max_iter):
        p, q, maximum = max_offdiag(diagonalized)
        if maximum < eps:
            break

        app = diagonalized[p][p]
        aqq = diagonalized[q][q]
        apq = diagonalized[p][q]

        # После выбора максимального внедиагонального элемента apq != 0:
        # если бы apq был равен нулю, то maximum был бы равен нулю,
        # и срабатывало бы условие остановки выше.
        # Величина (aqq - app) может быть равна нулю, поэтому нельзя
        # использовать формулу, в которой она стоит в знаменателе.
        tau = (aqq - app) / (2 * apq)

        # Для вычисления t используем формулы:
        # tg(2 * phi) = 2 * tg(phi) / (1 - tg(phi)^2),
        # ctg(2 * phi) = 1 / tg(2 * phi)
        #              = (1 - tg(phi)^2) / (2 * tg(phi)).
        # При t = tg(phi) и tau = ctg(2 * phi) получаем
        # устойчивую форму корня:
        # t = sign(tau) / (abs(tau) + sqrt(1 + tau^2)).
        t = (1 if tau >= 0 else -1) / (
            abs(tau) + math.sqrt(1 + tau * tau)
        )

        # Считаем sin(phi) и cos(phi) через t напрямую.
        # Это не требует вычисления phi и обращения к sin/cos, что
        # уменьшает накопление ошибок округления при разложении в ряд Тейлора.
        cosine = 1 / math.sqrt(1 + t * t)
        sine = t * cosine

        # Из формулы поворота:
        # a'pp = c^2 * app - 2 * c * s * apq + s^2 * aqq.
        # При t = s / c и условии зануления apq это сокращается до
        # a'pp = app - t * apq; аналогично a'qq = aqq + t * apq.
        diagonalized[p][p] = app - t * apq
        diagonalized[q][q] = aqq + t * apq
        diagonalized[p][q] = diagonalized[q][p] = 0.0

        for i in range(size):
            if i in (p, q):
                continue
            aip = diagonalized[i][p]
            aiq = diagonalized[i][q]
            diagonalized[i][p] = diagonalized[p][i] = cosine * aip - sine * aiq
            diagonalized[i][q] = diagonalized[q][i] = sine * aip + cosine * aiq

        for i in range(size):
            vip = eigenvectors[i][p]
            viq = eigenvectors[i][q]
            eigenvectors[i][p] = cosine * vip - sine * viq
            eigenvectors[i][q] = sine * vip + cosine * viq

    return [diagonalized[i][i] for i in range(size)], eigenvectors, iteration + 1

# Выполняет QR-разложение методом Грама — Шмидта.
def qr_decompose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    q_matrix = zeros(rows, columns)
    r_matrix = zeros(columns, columns)

    for j in range(columns):
        vector = [matrix[i][j] for i in range(rows)]

        for i in range(j): # ограничиваем, делая верхнетреугольную матрицу (r)
            r_matrix[i][j] = sum(
                q_matrix[k][i] * matrix[k][j] for k in range(rows)
            ) # скалярные произведения векторов q_i и a_j
            for k in range(rows): # вычитаем скаляр (r), умноженный на вектор (q)
                vector[k] -= r_matrix[i][j] * q_matrix[k][i] 

        r_matrix[j][j] = norm(vector) # по факту и есть скалярное произведение вектора самого на себя
        if r_matrix[j][j] < 1e-14: # значит линейно зависим от предыдущих векторов
            continue

        for k in range(rows):
            q_matrix[k][j] = vector[k] / r_matrix[j][j] # превращаем в единичные векторы

    return q_matrix, r_matrix


# Находит собственные значения квадратной матрицы QR-итерациями.
def qr_eigenvalues(matrix, eps = 1e-10, max_iter = 1000):
    current = [list(row) for row in matrix]
    size = len(current)

    for iteration in range(max_iter):
        q_matrix, r_matrix = qr_decompose(current)
        current = np.matmul(r_matrix, q_matrix).tolist()

        # Вещественный QR-алгоритм сходится к квазитреугольной форме:
        # на диагонали остаются блоки 1 x 1 или 2 x 2. Поэтому элементы
        # первой поддиагонали могут не обращаться в ноль для комплексной пары.
        far_below_diagonal = [
            abs(current[i][j])
            for i in range(size)
            for j in range(i - 1)
        ]
        has_overlapping_blocks = any(
            abs(current[i][i - 1]) >= eps
            and abs(current[i + 1][i]) >= eps
            for i in range(1, size - 1)
        )
        if (
            all(value < eps for value in far_below_diagonal)
            and not has_overlapping_blocks
        ): break

    values = []
    index = 0
    while index < size:
        if index + 1 < size and abs(current[index + 1][index]) >= eps:
            first = current[index][index]
            upper = current[index][index + 1]
            lower = current[index + 1][index]
            second = current[index + 1][index + 1]
            trace = first + second
            determinant = first * second - upper * lower
            # Для блока 2 x 2 решаем характеристическое уравнение:
            # lambda^2 - trace * lambda + determinant = 0.
            discriminant = cmath.sqrt(trace * trace - 4 * determinant)
            values.extend(((trace + discriminant) / 2, (trace - discriminant) / 2))
            index += 2
        else:
            values.append(current[index][index])
            index += 1

    return values, iteration + 1
