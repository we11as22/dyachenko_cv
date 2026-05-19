from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_point_cloud(path: Path, max_rows=None) -> np.ndarray:
    """Читает текстовый файл с облаком точек Semantic3D.

    Каждая строка содержит координаты и атрибуты точек, разделённые пробелами.
    Возвращает массив формы (N, M), где N — число точек, M — число столбцов.
    """

    try:
        arr = np.genfromtxt(path, max_rows=None)
    except Exception as e:
        print(f"loadtxt failed with {e}")
        exit(1)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_labels(path: Path, max_rows=None) -> np.ndarray:
    """Читает файл с метками классов для каждой точки.
    
    Ожидается одна метка (целое число) на строку, соответствующая точке
    из облака. Возвращает одномерный массив меток.
    """
    try:
        labels = np.genfromtxt(path, dtype=int, max_rows=max_rows)
    except Exception as e:
        print(f"loadtxt failed with {e}")
        exit(1)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    return labels


def normalize_coordinates(
    coords: np.ndarray, method: str = "center_scale"
) -> np.ndarray:
    """Приводит координаты к единому масштабу для стабильности обучения.

    Доступные стратегии нормализации:
    - 'center_scale' : центрирование относительно среднего и масштабирование по максимуму модуля
    - 'minmax' : линейное масштабирование всех значений в диапазон [0, 1]
    - 'zscore' : стандартизация (вычитание среднего, деление на стандартное отклонение)
    """
    coords = coords.astype(np.float32)
    if method == "center_scale":
        c = coords - coords.mean(axis=0)
        s = np.max(np.abs(c))
        if s == 0:
            return c
        return c / s
    elif method == "minmax":
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (coords - mins) / denom
    elif method == "zscore":
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        std[std == 0] = 1.0
        return (coords - mean) / std
    else:
        raise ValueError(f"Unknown coords normalization method: {method}")


def normalize_intensity(intensity: np.ndarray, method: str = "max") -> np.ndarray:
    """Нормализует значения интенсивности лидара.

    Методы нормализации:
    - 'max' : деление всех значений на максимальное
    - 'minmax' : линейное преобразование в интервал [0, 1]
    - 'zscore' : стандартизация с нулевым средним и единичной дисперсией
    """
    intensity = intensity.astype(np.float32)
    if method == "max":
        m = intensity.max()
        if m == 0:
            return intensity
        return intensity / m
    elif method == "minmax":
        lo = intensity.min()
        hi = intensity.max()
        denom = hi - lo
        if denom == 0:
            return intensity - lo
        return (intensity - lo) / denom
    elif method == "zscore":
        mean = intensity.mean()
        std = intensity.std()
        if std == 0:
            return intensity - mean
        return (intensity - mean) / std
    else:
        raise ValueError(f"Unknown intensity normalization method: {method}")


def visualize_labels_distribution(labels: np.ndarray, out_dir: Path) -> None:
    """Строит гистограмму частоты встречаемости каждого класса.
    
    Позволяет оценить сбалансированность датасета и выявить доминирующие классы.
    """
    hist_path = out_dir / "labels_hist.png"
    plt.figure()
    plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 2) - 0.5)
    plt.xlabel("label")
    plt.ylabel("count")
    plt.title("Labels distribution")
    plt.tight_layout()
    plt.savefig(hist_path)
