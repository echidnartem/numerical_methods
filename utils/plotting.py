"""Общее сохранение графиков лабораторных работ."""

from io import BytesIO


# Сохраняет PNG через буфер, не открывая путь из Matplotlib/Pillow в режиме w+b.
def save_png(figure, path, dpi = 150):
    buffer = BytesIO()
    figure.savefig(buffer, format = "png", dpi = dpi)
    path.write_bytes(buffer.getvalue())
