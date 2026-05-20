# Задание 3 — геометрический анализ сегментов

Детальный анализ геометрических характеристик сегментов по `scalar_Label` из `../data`.

## Реализованные этапы

| Раздел ТЗ | Что считается |
|-----------|----------------|
| **6.1 Плотность точек** | среднее расстояние до соседа, равномерность, доли разреженных/сгущённых областей, класс `low/medium/high` |
| **6.2 Форма распределения** | PCA: linearity, planarity, sphericity, anisotropy, число доминирующих осей, класс `linear/planar/volumetric/anisotropic` |
| **6.3 Структура поверхности** | кривизна, вариация нормалей, доля резких изменений, класс `smooth/weakly_curved/strongly_curved/complex` |
| **6.4 Согласованность нормалей** | нормали всех точек, средняя согласованность, резкие области, класс `high/medium/low` |
| **6.5 Топологическая связность** | компоненты связности, изолированные кластеры, среднее число соседей, целостность `intact/mostly_intact/fragmented` |
| **6.6 Итог анализа** | текстовое комплексное описание сегмента в CSV и на визуализации |

## Структура

| Путь | Назначение |
|------|------------|
| `scripts/run_geometric_analysis.py` | запуск анализа |
| `src/geometric_analysis/pipeline.py` | полный код |
| `metrics/segments_analysis.csv` | метрики и интерпретации по каждому сегменту |
| `metrics/summary.json` | сводка по классам |
| `visualizations/` | PNG по каждому сегменту; в git коммитятся примеры для `valve_0001_*` и `valve_0002_*` |
| `figures/` | компактная выборка PNG для сдачи |
| `REPORT.md` | отчёт |

## Запуск

```sh
cd part_2/task_3
source ~/envs/.venv/bin/activate
pip install -r requirements.txt
python scripts/run_geometric_analysis.py
```

Быстрый тест:

```sh
python scripts/run_geometric_analysis.py --max-files 2
```

Без тяжёлых PNG:

```sh
python scripts/run_geometric_analysis.py --no-visualizations
```

## Визуализации

Для каждого сегмента сохраняется 4-panel PNG:

1. локальная плотность точек;
2. форма и PCA-оси;
3. согласованность нормалей;
4. компоненты связности.

Внизу — текстовая интерпретация по пунктам 6.1–6.6.

## Параметры

- `--data-dir` — `part_2/data`;
- `--max-files N` — обработать только N PLY;
- `--min-segment-points N` — отбрасывать мелкие сегменты;
- `--max-points-per-segment N` — ограничение точек сегмента;
- `--no-visualizations` — не сохранять все PNG;
- `--no-figures` — не копировать выборку в `figures/`.
