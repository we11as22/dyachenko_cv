# Часть 2 — сравнение моделей сегментации

Семантическая сегментация облаков точек (датасет клапанов, PLY, 10 классов).

## Содержимое

| Путь | Назначение |
|------|------------|
| [`REPORT.md`](REPORT.md) | **Отчёт** — сравнение экспериментов (OA, mIoU, F1, IoU по классам) |
| `scripts/` | Скрипты обучения: PointNet++, DGCNN, Point Transformer, KPConv |
| `task_*/metrics/` | Результаты прогонов (`results.csv`) |
| `../data/` | PLY-файлы (общие для части 2, в git не коммитятся) |
| `requirements.txt` | Зависимости части 2 |

## Запуск

```sh
cd part_2/task_1
pip install -r requirements.txt
# положить *.ply в ../data/
python scripts/train_pointnet2.py
python scripts/train_dgcnn.py
python scripts/train_point_transformer.py
python scripts/train_kpconv.py
```

Чекпоинты пишутся в `task_*/checkpoints/` (в git не попадают).  
Итоговое сравнение моделей — в [`REPORT.md`](REPORT.md).
