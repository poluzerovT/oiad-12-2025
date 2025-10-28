from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mpmath as mp

N = 17
CANDIDATE_PATHS = [
    Path("datasets/students_simple.csv"),
    Path("./datasets/students_simple.csv"),
    Path("students_simple.csv"),
    Path("/mnt/data/students_simple.csv"),
]

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

def fit_ols(X: np.ndarray, y: np.ndarray):
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    return beta, yhat, r2_score(y, yhat)

def f_test_from_r2(R2: float, n: int, k: int, alpha: float = 0.05):
    df1, df2 = k - 1, n - k
    F = (R2 / (1.0 - R2)) * (df2 / df1)
    z = (df1 * F) / (df1 * F + df2)
    a, b = df1 / 2.0, df2 / 2.0
    cdf = float(mp.betainc(a, b, 0, z, regularized=True))
    p = 1.0 - cdf
    def cdf_from_x(x):
        zz = (df1 * x) / (df1 * x + df2)
        return float(mp.betainc(a, b, 0, zz, regularized=True))
    lo, hi = 0.0, 1e6
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if cdf_from_x(mid) < 1 - alpha else (lo, mid)
    Fcrit = 0.5 * (lo + hi)
    return F, df1, df2, p, Fcrit, (F > Fcrit)

def main():
    csv_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Не найден students_simple.csv")
    df = pd.read_csv(csv_path)
    i1 = N % 5
    i2 = (N**2 % 5) + 5
    col_x, col_y = df.columns[i1], df.columns[i2]
    xy = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    x = xy[col_x].to_numpy(float)
    y = xy[col_y].to_numpy(float)
    n = len(x)

    _, yhat_lin, R2_lin = fit_ols(x.reshape(-1,1), y)
    X_quad = np.column_stack([x, x**2])
    _, yhat_quad, R2_quad = fit_ols(X_quad, y)
    z = 1.0 / x
    _, yhat_hyp, R2_hyp = fit_ols(z.reshape(-1,1), y)
    lny = np.log(y)
    beta_exp, _, _ = fit_ols(x.reshape(-1,1), lny)
    ln_w0, ln_w1 = beta_exp.tolist()
    w0, w1 = float(np.exp(ln_w0)), float(np.exp(ln_w1))
    yhat_exp = w0 * (w1 ** x)
    R2_exp = r2_score(y, yhat_exp)

    models = {
        "Linear": (R2_lin, 2),
        "Quadratic": (R2_quad, 3),
        "Hyperbolic": (R2_hyp, 2),
        "Exponential": (R2_exp, 2),
    }
    best = max(models.items(), key=lambda kv: kv[1][0])[0]
    worst = min(models.items(), key=lambda kv: kv[1][0])[0]

    rows = []
    for name in (best, worst):
        R2, k = models[name]
        F, df1, df2, p, Fcrit, reject = f_test_from_r2(R2, n, k, 0.05)
        rows.append({
            "model": name, "R2": R2, "n": n, "k": k,
            "df1": df1, "df2": df2, "F_stat": F, "F_crit@0.05": Fcrit,
            "p_value": p, "reject_H0": bool(reject),
            "H0": "все наклоны = 0 (регрессия не значима)"
        })

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()