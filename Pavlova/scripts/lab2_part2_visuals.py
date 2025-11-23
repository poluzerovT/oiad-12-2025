from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 17
CANDIDATE_PATHS = [
    Path("datasets/students_simple.csv"),
    Path("./datasets/students_simple.csv"),
    Path("students_simple.csv"),
]
OUTPUT_DIR = Path("outputs") / "lab2_part2_visuals"
DPI = 160

plt.rcParams["font.family"] = "DejaVu Sans"


def freedman_diaconis_bins(values: np.ndarray, max_bins: int = 100) -> int:
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 5
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return min(max_bins, max(5, int(math.sqrt(n))))
    h = 2 * iqr * n ** (-1 / 3)
    if h <= 0:
        return min(max_bins, max(5, int(math.sqrt(n))))
    bins = int(math.ceil((x.max() - x.min()) / h))
    return min(max_bins, max(5, bins))


def plot_hist(data: np.ndarray, label: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    bins = freedman_diaconis_bins(data)
    plt.figure(figsize=(7, 4))
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.8)
    plt.title(f"Гистограмма: {label}")
    plt.xlabel(label)
    plt.ylabel("Частота")
    plt.grid(True, linestyle="--", alpha=0.35)
    fname = outdir / f"hist_{label}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()
    print(f"[saved] {fname}")


def plot_scatter(x: np.ndarray, y: np.ndarray, xlab: str, ylab: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    def sample_std(a): return float(np.std(a, ddof=1))
    def sample_cov(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        return float(np.dot(ac, bc) / (len(a) - 1))
    def pearson(a, b):
        sa, sb = sample_std(a), sample_std(b)
        return sample_cov(a, b) / (sa * sb)

    rho_s = pd.Series(x).rank(method="average").corr(
        pd.Series(y).rank(method="average")
    )
    r_p = pearson(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.85)
    plt.title(f"Scatter plot: {xlab} vs {ylab}\n"
              f"Pearson r={r_p:.3f}, Spearman ρ={rho_s:.3f}")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.35)
    fname = outdir / f"scatter_{xlab}_vs_{ylab}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()
    print(f"[saved] {fname}")


def main():
    csv_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Не найден 'students_simple.csv' в папке datasets/.")

    i1 = N % 5
    i2 = (N ** 2) % 5 + 5

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if max(i1, i2) >= len(cols):
        raise IndexError(f"В датасете {len(cols)} столбцов, а требуются индексы {i1} и {i2}.")

    col_x, col_y = cols[i1], cols[i2]
    xy = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    x = xy[col_x].to_numpy(dtype=float)
    y = xy[col_y].to_numpy(dtype=float)

    print("=== ЛР-2: пункт 2 — Визуализация ===")
    print(f"N = {N}")
    print(f"Файл: {csv_path}")
    print(f"Выбранные столбцы: i1={i1} → '{col_x}',  i2={i2} → '{col_y}'")
    print(f"n = {len(xy)}\n")

    plot_hist(x, col_x, OUTPUT_DIR)
    plot_hist(y, col_y, OUTPUT_DIR)

    plot_scatter(x, y, col_x, col_y, OUTPUT_DIR)

    print("\nГотово. Изображения сохранены в:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()