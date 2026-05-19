# 3D Coursework — Dyachenko

Университетские задания по 3D и облакам точек. Две независимые части:

| Часть | Папка | Описание |
|-------|-------|----------|
| **1** | [`part_1/`](part_1/) | Модули a–e: сэмплинг, цвета, Semantic3D, PointNet — см. [README](part_1/README.md) |
| **2** | [`part_2/`](part_2/) | Сравнение архитектур сегментации — см. [README](part_2/README.md) и [отчёт](part_2/REPORT.md) |

У каждой части своё окружение (`part_1`: uv, `part_2`: pip + `requirements.txt`).

## Что не в git

- `part_2/data/` — датасет PLY
- `**/checkpoints/`, `*.pt`, `*.pth`, тяжёлые `outputs/`
- виртуальные окружения и кэши (см. `.gitignore`)
