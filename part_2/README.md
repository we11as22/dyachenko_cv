# Часть 2

| Задание | Папка | Описание |
|---------|-------|----------|
| **1** | [`task_1/`](task_1/) | Сравнение моделей сегментации — см. [README](task_1/README.md) и [отчёт](task_1/REPORT.md) |
| **2** | [`task_2/`](task_2/) | Адаптивная реконструкция поверхностей — см. [README](task_2/README.md) и [отчёт](task_2/REPORT.md) |
| **3** | [`task_3/`](task_3/) | Геометрический анализ сегментов — см. [README](task_3/README.md) и [отчёт](task_3/REPORT.md) |

Окружение для задания 1: `pip` + [`task_1/requirements.txt`](task_1/requirements.txt).  
Окружение для задания 2: `pip` + [`task_2/requirements.txt`](task_2/requirements.txt).  
Окружение для задания 3: `pip` + [`task_3/requirements.txt`](task_3/requirements.txt).

Датасет: `data/` (500 PLY, ~123 MB) — общий для `task_1`, `task_2`, `task_3`, …

| Задание | Статус | Ключевые артефакты |
|---------|--------|-------------------|
| **3** | ✓ полный прогон | 4489 сегментов, `task_3/metrics/`, `task_3/figures/` |

## Визуализация датасета

| Путь | Назначение |
|------|------------|
| [`scripts/visualize_point_clouds.py`](scripts/visualize_point_clouds.py) | скрипт визуализации PLY из `data/` |
| [`data_vis/`](data_vis/) | PNG-превью нескольких облаков для сдачи |

```sh
cd part_2
source ~/envs/.venv/bin/activate
python scripts/visualize_point_clouds.py --max-files 6
```
