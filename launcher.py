"""Консольное меню для запуска лабораторных работ с данными из JSON."""

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent

LAB_TITLES = {
    1: "Линейная алгебра",
    2: "Нелинейные уравнения",
    3: "Интерполяция и приближение",
}

TASK_TITLES = {
    (1, 1): "LU-разложение",
    (1, 2): "Метод прогонки",
    (1, 3): "Простые итерации и метод Зейделя",
    (1, 4): "Метод вращений Якоби",
    (1, 5): "QR-разложение",
    (2, 1): "Нелинейное уравнение",
    (2, 2): "Система нелинейных уравнений",
    (3, 1): "Многочлены Лагранжа и Ньютона",
    (3, 2): "Кубический сплайн",
    (3, 3): "Метод наименьших квадратов",
}


# Находит задания, для которых есть программа и оба файла данных.
def discover_labs():
    catalog = {}

    for lab_dir in ROOT.glob("lab[0-9]*"):
        lab_suffix = lab_dir.name[3:]
        if not lab_dir.is_dir() or not lab_suffix.isdigit():
            continue

        lab_number = int(lab_suffix)
        tasks = {}

        for script in lab_dir.glob(f"{lab_number}.*.py"):
            parts = script.stem.split(".")
            if len(parts) != 2 or not parts[1].isdigit():
                continue

            task_number = int(parts[1])
            single = lab_dir / "data" / f"lab{task_number}.json"
            examples = lab_dir / "data" / f"lab{task_number}_examples.json"
            if single.is_file() and examples.is_file():
                tasks[task_number] = (script, single, examples)

        if tasks:
            catalog[lab_number] = dict(sorted(tasks.items()))

    return dict(sorted(catalog.items()))


# Печатает варианты выбора и возвращает их номер; ноль означает возврат.
def choose(title, options, back_label):
    while True:
        print(f"\n{title}")
        for number, label in options.items():
            print(f"  {number}. {label}")
        print(f"  0. {back_label}")

        answer = input("Выбор: ").strip()
        if answer.isdigit():
            number = int(answer)
            if number == 0 or number in options:
                return number

        print("Введите номер из списка.")


# Запускает выбранное задание, передавая ему путь к JSON-файлу.
def run_task(script, data_file):
    print(f"\nЗапуск: python {script.relative_to(ROOT)} {data_file.relative_to(ROOT)}\n", flush = True)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), str(data_file)],
        cwd = ROOT,
        check = False,
    )
    if result.returncode != 0:
        print(f"\nПрограмма завершилась с ошибкой: код {result.returncode}")


# Последовательно предлагает лабораторную, задание и файл данных.
def main():
    catalog = discover_labs()
    if not catalog:
        raise SystemExit("Не найдены задания с программами и JSON-файлами.")

    while True:
        lab_options = {
            number: f"Лабораторная № {number} — {LAB_TITLES.get(number, 'Без названия')}"
            for number in catalog
        }
        lab_number = choose("Лабораторные работы", lab_options, "Выход")
        if lab_number == 0:
            return

        while True:
            task_options = {
                number: f"{lab_number}.{number} — {TASK_TITLES.get((lab_number, number), 'Задание')}"
                for number in catalog[lab_number]
            }
            task_number = choose(f"Лабораторная № {lab_number}", task_options, "К лабораторным")
            if task_number == 0:
                break

            script, single, examples = catalog[lab_number][task_number]
            data_options = {
                1: f"Один пример ({single.name})",
                2: f"Набор тестов ({examples.name})",
            }

            while True:
                choice = choose(
                    f"Задание {lab_number}.{task_number}", data_options, "К заданиям"
                )
                if choice == 0:
                    break

                data_file = single if choice == 1 else examples
                run_task(script, data_file)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nВыход.")
