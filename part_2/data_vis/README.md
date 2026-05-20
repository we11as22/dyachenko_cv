# Визуализации датасета

Небольшая выборка PNG-превью облаков точек из `../data/`.

Каждая картинка показывает:
- слева: облако, раскрашенное по `scalar_Label`;
- справа: геометрию без цветовых меток.

Сгенерировать заново:

```sh
cd part_2
source ~/envs/.venv/bin/activate
python scripts/visualize_point_clouds.py --max-files 6
```

Полный датасет `data/` в git не коммитится.
