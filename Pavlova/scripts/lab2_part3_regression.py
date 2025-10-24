from __future__ import annotations
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
    Path("/mnt/data/students_simple.csv"),
]
OUTDIR = Path("outputs") / "lab2_part3_regression"
DPI = 160
plt.rcParams["font.family"] = "DejaVu Sans"


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")


def fit_ols(X: np.ndarray, y: np.ndarray):
    n = len(y)
    X_design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    return beta, y_hat, r2_score(y, y_hat)


def plot_scatter_with_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_curve: np.ndarray,
    title: str,
    xlab: str,
    ylab: str,
    outpath: Path,
):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.85)
    plt.plot(x_grid, y_curve)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def main():
    csv_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Не найден 'students_simple.csv'. Положите файл в ./datasets/ или рядом со скриптом."
        )

    i1 = N % 5
    i2 = (N ** 2) % 5 + 5

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if max(i1, i2) >= len(cols):
        raise IndexError(f"В датасете {len(cols)} столбцов, требуются индексы {i1} и {i2}.")
    col_x, col_y = cols[i1], cols[i2]

    xy = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    x = xy[col_x].to_numpy(dtype=float)
    y = xy[col_y].to_numpy(dtype=float)
    n = len(x)
    x_grid = np.linspace(x.min(), x.max(), 300)

    print(f"Файл: {csv_path}")
    print(f"N = {N}")
    print(f"X: i1={i1} → '{col_x}',  Y: i2={i2} → '{col_y}',  n = {n}\n")

    rows = []

    beta_lin, yhat_lin, r2_lin = fit_ols(x.reshape(-1, 1), y)
    w0_lin, w1_lin = beta_lin.tolist()
    y_grid_lin = w0_lin + w1_lin * x_grid
    plot_scatter_with_curve(
        x, y, x_grid, y_grid_lin,
        title=f"Линейная: y = w1·x + w0\nw0={w0_lin:.3f}, w1={w1_lin:.6f}, R²={r2_lin:.3f}",
        xlab=col_x, ylab=col_y,
        outpath=OUTDIR / "linear_fit.png",
    )
    rows.append(["Linear", {"w0": w0_lin, "w1": w1_lin}, r2_lin])

    X_quad = np.column_stack([x, x ** 2])
    beta_quad, yhat_quad, r2_quad = fit_ols(X_quad, y)
    w0_q, w1_q, w2_q = beta_quad.tolist()
    y_grid_quad = w0_q + w1_q * x_grid + w2_q * (x_grid ** 2)
    plot_scatter_with_curve(
        x, y, x_grid, y_grid_quad,
        title=(
            "Квадратичная: y = w2·x² + w1·x + w0\n"
            f"w0={w0_q:.3f}, w1={w1_q:.6f}, w2={w2_q:.8f}, R²={r2_quad:.3f}"
        ),
        xlab=col_x, ylab=col_y,
        outpath=OUTDIR / "quadratic_fit.png",
    )
    rows.append(["Quadratic", {"w0": w0_q, "w1": w1_q, "w2": w2_q}, r2_quad])

    if np.any(x == 0):
        raise ValueError("Гиперболическая модель требует x ≠ 0.")
    z = 1.0 / x
    beta_hyp, yhat_hyp, r2_hyp = fit_ols(z.reshape(-1, 1), y)
    w0_h, w1_h = beta_hyp.tolist()
    y_grid_hyp = w0_h + w1_h * (1.0 / x_grid)
    plot_scatter_with_curve(
        x, y, x_grid, y_grid_hyp,
        title=f"Гиперболическая: y = w1/x + w0\nw0={w0_h:.3f}, w1={w1_h:.3f}, R²={r2_hyp:.3f}",
        xlab=col_x, ylab=col_y,
        outpath=OUTDIR / "hyperbolic_fit.png",
    )
    rows.append(["Hyperbolic", {"w0": w0_h, "w1_over_x": w1_h}, r2_hyp])

    if np.any(y <= 0):
        raise ValueError("Экспоненциальная модель требует y > 0.")
    lny = np.log(y)
    beta_exp, lny_hat, r2_exp_on_log = fit_ols(x.reshape(-1, 1), lny)
    ln_w0, ln_w1 = beta_exp.tolist()
    w0_exp = float(np.exp(ln_w0))
    w1_exp = float(np.exp(ln_w1))
    y_grid_exp = w0_exp * (w1_exp ** x_grid)
    yhat_exp = w0_exp * (w1_exp ** x)
    r2_exp = r2_score(y, yhat_exp)
    plot_scatter_with_curve(
        x, y, x_grid, y_grid_exp,
        title=f"Экспоненциальная: y = w0·w1^x\nw0={w0_exp:.3f}, w1={w1_exp:.6f}, R²={r2_exp:.3f}",
        xlab=col_x, ylab=col_y,
        outpath=OUTDIR / "exponential_fit.png",
    )
    rows.append(["Exponential", {"w0": w0_exp, "w1": w1_exp, "ln_w0": ln_w0, "ln_w1": ln_w1}, r2_exp])

    summary = pd.DataFrame(
        {
            "model": [r[0] for r in rows],
            "coefficients": [r[1] for r in rows],
            "R2_on_y": [r[2] for r in rows],
        }
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nSaved to:", OUTDIR.resolve())


if __name__ == "__main__":
    main()