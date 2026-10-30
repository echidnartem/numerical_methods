"""Загрузка входных данных лабораторных работ из JSON."""

import json
from pathlib import Path


# Читает один тест или набор тестов вместе с общими параметрами.
def load_data(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"Файл с входными данными не найден: {path}")

    try:
        data = json.loads(file_path.read_text(encoding = "utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Ошибка в JSON-файле {path}: {error}") from error

    if not isinstance(data, dict):
        raise SystemExit("В корне JSON-файла должен быть объект с данными")

    return data
