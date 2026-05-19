# Часть 1 — обработка облаков точек

Задания по сэмплингу, визуализации, Semantic3D и PointNet.

## Структура

| Каталог | Содержание |
|---------|------------|
| `src/module_a` | Сэмплинг, сегментация, региональные фильтры |
| `src/module_b` | Цвета и скалярные поля |
| `src/module_c` | Semantic3D (заметки и скрипты) |
| `src/module_d` | PointNet для ModelNet |
| `src/module_e` | PointNet для семантической сегментации |

## Быстрый старт

1. Установить [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. В этой папке (`part_1`):

```sh
uv sync
cd src/module_d
uv run train_pointnet.py
```

Параметры — в `config.yaml` / `settings.yaml` / `conf.yaml` внутри модулей.

## Зависимости

`pyproject.toml`, `uv.lock`, `.python-version` — в этой же папке.
