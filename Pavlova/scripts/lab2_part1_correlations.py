import math
from pathlib import Path
import numpy as np
import pandas as pd

def sample_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1))


def sample_cov(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - x.mean()
    y_c = y - y.mean()
    return float(np.dot(x_c, y_c) / (len(x) - 1))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    sx = sample_std(x)
    sy = sample_std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sample_cov(x, y) / (sx * sy)


def fisher_ci_for_pearson(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return float("nan"), float("nan")
    r = min(0.999999, max(-0.999999, float(r)))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = 1.959963984540054
    lo = float(np.tanh(z - zcrit * se))
    hi = float(np.tanh(z + zcrit * se))
    return lo, hi


def fechner_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    sx = np.sign(x - x.mean())
    sy = np.sign(y - y.mean())
    same = int(np.sum(sx == sy))
    diff = int(np.sum(sx != sy))
    n = len(x)
    return (same - diff) / n, same, diff


def ranks(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average").to_numpy()


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = ranks(x)
    ry = ranks(y)
    return pearson_r(rx, ry)


def kendall_tau_lab(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    n = len(x)
    D = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (x[i] < x[j]) != (y[i] < y[j]):
                D += 1
    tau = 1.0 - 4.0 * D / (n * (n - 1))
    return tau, D


def main():
    N = 17
    candidate_paths = [
        Path("datasets/students_simple.csv"),
        Path("./datasets/students_simple.csv"),
        Path("students_simple.csv"),
        Path("/mnt/data/students_simple.csv"),
    ]

    csv_path = next((p for p in candidate_paths if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Не найден 'students_simple.csv'. "
            "Положите файл в ./datasets/ или рядом со скриптом."
        )

    i1 = N % 5
    i2 = (N ** 2) % 5 + 5

    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = list(df.columns)

    if max(i1, i2) >= len(cols):
        raise IndexError(
            f"В датасете {len(cols)} столбцов, а требуются индексы {i1} и {i2}."
        )

    col_x = cols[i1]
    col_y = cols[i2]

    xy = df[[col_x, col_y]].dropna()
    x = xy[col_x].to_numpy(dtype=float)
    y = xy[col_y].to_numpy(dtype=float)
    n = len(xy)

    print(f"N = {N}")
    print(f"Выбранные столбцы")
    print(f"  i1 = N % 5 = {i1}  → '{col_x}'")
    print(f"  i2 = (N**2 % 5) + 5 = {i2} → '{col_y}'")
    print(f"Размер выборки после dropna: n = {n}\n")

    k_fechner, same_signs, diff_signs = fechner_corr(x, y)
    r_pearson = pearson_r(x, y)
    ci_lo, ci_hi = fisher_ci_for_pearson(r_pearson, n, alpha=0.05)
    rho_s = spearman_rho(x, y)
    tau_k, D = kendall_tau_lab(x, y)

    summary = pd.DataFrame(
        {
            "metric": [
                "Fechner K",
                "Pearson r",
                "Spearman ρ",
                "Kendall τ (lab formula)",
            ],
            "value": [k_fechner, r_pearson, rho_s, tau_k],
            "extra": [
                f"same_signs={same_signs}, diff_signs={diff_signs}, n={n}",
                f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}], n={n}",
                f"n={n} (average ranks for ties)",
                f"discordant pairs D={D}, n={n}",
            ],
        }
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()