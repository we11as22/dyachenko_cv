# 3D Coursework Sandbox

Набор черновиков по обработке облаков точек и 3D-геометрии.

## Структура
- `src/module_a` — базовые операции с облаками (сэмплинг, сегментация, фильтры).
- `src/module_b` — цвета и скалярные поля.
- `src/module_c` — заметки по Semantic3D.
- `src/module_d` — мой пайплайн PointNet для ModelNet.
- `src/module_e` — PointNet для семантической сегментации облаков точек.

## Быстрый старт
1. Установить [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. В корне:
   ```sh
   uv sync
   ```
3. Пример запуска обучения в module_d:
   ```sh
   cd src/module_d
   uv run train_pointnet.py
   ```

Параметры можно править в соответствующих `settings.yaml` внутри модулей.
