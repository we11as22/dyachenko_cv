# 3D Coursework — Dyachenko

Университетские задания по 3D и облакам точек. Две независимые части:

| Часть | Папка | Описание |
|-------|-------|----------|
| **1** | [`part_1/`](part_1/) | Модули a–e: сэмплинг, цвета, Semantic3D, PointNet — см. [README](part_1/README.md) |
| **2** | [`part_2/`](part_2/) | Задания по сегментации и реконструкции — см. [README](part_2/README.md) |

У каждой части своё окружение (`part_1`: uv, `part_2/task_1` и `part_2/task_2`: pip + `requirements.txt`).

## Что не в git

- `part_2/data/` — датасет PLY (общий для заданий части 2)
- `**/checkpoints/`, `*.pt`, `*.pth`, тяжёлые `outputs/`
- виртуальные окружения и кэши (см. `.gitignore`)
