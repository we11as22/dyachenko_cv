# Визуализации для сдачи

Компактная выборка PNG из полного анализа **4489** сегментов (500 PLY).

Каждая картинка — 4 панели:

1. локальная плотность точек;
2. форма и PCA-оси;
3. согласованность нормалей;
4. компоненты связности.

Внизу — текстовая интерпретация по пунктам 6.1–6.6.

| Файл | Тип |
|------|-----|
| `linear_valve_0001_lidar_classes_label_00.png` | линейная форма |
| `planar_valve_0001_lidar_classes_label_05.png` | плоская форма |
| `volumetric_valve_0001_lidar_classes_label_09.png` | объёмная форма |
| `smooth_valve_0001_lidar_classes_label_03.png` | гладкая поверхность |
| `complex_valve_0224_lidar_classes_label_00.png` | сложная поверхность |
| `fragmented_valve_0002_lidar_classes_label_09.png` | фрагментированный сегмент |

Полная папка `visualizations/` (~4489 PNG) не коммитится целиком. В git включены примеры для `valve_0001_lidar_classes` и `valve_0002_lidar_classes` (по 8 PNG).
