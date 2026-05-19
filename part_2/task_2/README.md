# Задание 2 — адаптивная реконструкция поверхностей

Программный комплекс строит полигональные модели инженерных объектов по облакам точек ТЛО из `../data`.

Pipeline:

1. загрузка всех `*.ply`;
2. проверка данных и обработка ошибок;
3. удаление статистических выбросов;
4. опциональный voxel downsampling;
5. сегментация по `scalar_Label`;
6. удаление мелких сегментов;
7. геометрический анализ сегментов;
8. классификация сегмента;
9. автоматический выбор метода реконструкции;
10. реконструкция поверхности;
11. сборка модели по сегментам;
12. расчёт метрик качества.

## Структура

| Путь | Назначение |
|------|------------|
| `scripts/run_reconstruction.py` | основной запуск |
| `src/surface_reconstruction/pipeline.py` | полный код pipeline |
| `../data/` | исходные PLY-файлы |
| `outputs/` | реконструированные mesh-модели (`*.ply`) |
| `visualizations/` | PNG-визуализации сегментов |
| `metrics/segments_metrics.csv` | метрики по каждому сегменту |
| `metrics/summary.json` | сводка по запуску |
| `REPORT.md` | отчёт по алгоритму и результатам |

## Установка

```sh
cd part_2/task_2
source ~/envs/.venv/bin/activate
pip install -r requirements.txt
```

В твоём окружении `~/envs/.venv` уже есть нужные библиотеки (`numpy`, `open3d`, `scikit-learn`, `matplotlib`, `tqdm`). Если запускаешь на другой машине, установи зависимости через `pip install -r requirements.txt`.

## Быстрый тест

```sh
cd part_2/task_2
source ~/envs/.venv/bin/activate
python scripts/run_reconstruction.py --max-files 1 --max-points-per-segment 1500
```

## Полный запуск

```sh
cd part_2/task_2
source ~/envs/.venv/bin/activate
python scripts/run_reconstruction.py
```

Для ускорения можно включить downsampling:

```sh
python scripts/run_reconstruction.py --voxel-size 1.0 --max-points-per-segment 2500
```

Без дополнительных аргументов скрипт работает последовательно. `--workers` включает параллельную обработку по PLY-файлам. Для WSL обычно разумно начать с `--workers 4`; если компьютер свободен и хватает памяти, можно поднять до `6` или `8`.

Параллельный полный запуск:

```sh
python scripts/run_reconstruction.py --workers 4
```

Прогресс обновляется после завершения каждого PLY-файла. Чтобы было понятно, что скрипт не завис, по умолчанию дополнительно печатаются строки `[start]` и `[done]` для файлов в работе. Если нужен только компактный tqdm-бар, добавь `--quiet-files`.

По умолчанию каждый PLY обрабатывается в отдельном дочернем процессе. Это защищает полный прогон от падений Open3D/Poisson уровня `Segmentation fault`: если один файл упадёт внутри Open3D, общий запуск продолжит обработку остальных файлов и запишет проблему в метрики. Метрики `metrics/segments_metrics.csv` и `metrics/summary.json` обновляются после каждого файла, поэтому частичный результат не теряется при остановке.

Самый быстрый полный прогон для получения метрик:

```sh
python scripts/run_reconstruction.py --workers 4 --no-visualizations --max-points-per-segment 1500
```

Визуализации можно построить отдельно на небольшой выборке:

```sh
python scripts/run_reconstruction.py --max-files 10 --workers 4 --output-dir outputs_preview --metrics-dir metrics_preview --visualization-dir visualizations_preview
```

## Сравнение подходов

Адаптивный запуск автоматически выбирает метод для каждого сегмента. Для сравнения с базовыми подходами можно принудительно запустить один метод на тех же данных:

```sh
python scripts/run_reconstruction.py --force-method poisson --output-dir outputs_poisson --metrics-dir metrics_poisson --visualization-dir visualizations_poisson
python scripts/run_reconstruction.py --force-method alpha_shape --output-dir outputs_alpha --metrics-dir metrics_alpha --visualization-dir visualizations_alpha
python scripts/run_reconstruction.py --force-method ball_pivoting --output-dir outputs_bpa --metrics-dir metrics_bpa --visualization-dir visualizations_bpa
```

Потом сравниваются `mean_cloud_to_mesh`, `p95_cloud_to_mesh`, `accuracy_ratio`, число ошибок реконструкции и связность mesh. Адаптивный подход должен давать более устойчивый результат, потому что не применяет один алгоритм ко всем типам геометрии.

## Основные параметры

- `--data-dir` — папка с `*.ply`, по умолчанию `part_2/data`;
- `--max-files N` — обработать только первые `N` файлов для отладки;
- `--min-segment-points N` — удалить сегменты меньше `N` точек;
- `--voxel-size X` — voxel downsampling в исходных единицах;
- `--max-points-per-segment N` — ограничение точек сегмента для скорости;
- `--poisson-depth N` — глубина Poisson-реконструкции;
- `--workers N` — число параллельных обработчиков по файлам;
- `--force-method poisson|alpha_shape|ball_pivoting` — принудительно использовать один базовый метод для сравнения;
- `--no-visualizations` — не сохранять PNG;
- `--no-segment-meshes` — не сохранять mesh каждого сегмента, оставить сборки.
- `--quiet-files` — скрыть строки `[start]` / `[done]` по файлам.
- `--no-isolate-files` — отключить безопасную изоляцию файлов и запускать Open3D в основном процессе.

## Логика выбора метода

- `plane` → Alpha Shapes;
- `tube` → Ball Pivoting;
- `sphere` → Poisson Surface Reconstruction;
- `complex` → выбор по согласованности нормалей и связности.

## Результаты

После запуска появятся:

- `outputs/<имя_облака>/label_XX_<method>.ply` — mesh сегмента;
- `outputs/<имя_облака>/<имя_облака>_merged.ply` — собранная модель файла;
- `visualizations/<имя_облака>/label_XX.png` — сравнение облака и mesh;
- `metrics/segments_metrics.csv` — признаки, класс, метод и метрики качества;
- `metrics/summary.json` — глобальная сводка.
