from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "insurance_train.csv"
TEST_PATH  = DATA_DIR / "insurance_test.csv"

TARGET = "charges"
BASE_REGION = "region_southwest"

def preprocess_for_model(df: pd.DataFrame, base_region: str = BASE_REGION) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    df["sex"]    = df["sex"].map({"female": 0, "male": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    dummies = pd.get_dummies(df["region"], prefix="region", drop_first=False)
    if base_region not in dummies.columns:
        base_region = dummies.columns[0]
    dummies = dummies.drop(columns=[base_region])
    df = pd.concat([df.drop(columns=["region"]), dummies], axis=1)
    return df, base_region

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0

def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float):
    Xcf = add_intercept(X)
    R = np.eye(Xcf.shape[1]); R[0, 0] = 0.0
    XtX = Xcf.T @ Xcf + alpha * R
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    return XtX_inv @ Xcf.T @ y

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma

def soft_threshold(z, gamma):
    if z > gamma: return z - gamma
    if z < -gamma: return z + gamma
    return 0.0

def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, lam: float,
                             max_iter=5000, tol=1e-6):
    n, p = X.shape
    Xs, mu, sigma = standardize_fit(X)
    y_mean = y.mean()
    yc = y - y_mean
    w = np.zeros(p)
    Xj_sq = np.sum(Xs ** 2, axis=0) / n
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            r = yc + Xs[:, j] * w[j] - Xs @ w
            rho_j = float(np.dot(Xs[:, j], r) / n)
            w[j] = soft_threshold(rho_j, lam) / (Xj_sq[j] if Xj_sq[j] != 0 else 1.0)
        if np.max(np.abs(w - w_old)) < tol:
            break
    w_unstd = w / sigma
    w0_unstd = y_mean - np.sum(mu * w_unstd)
    return np.concatenate([[w0_unstd], w_unstd])

def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return np.array_split(idx, k)

def cv_score_ridge_closed_form(X, y, alphas, k=5, seed=0):
    folds = kfold_indices(len(y), k, seed)
    cv = []
    for a in alphas:
        mses = []
        for i in range(k):
            te = folds[i]; tr = np.hstack([folds[j] for j in range(k) if j != i])
            w = ridge_closed_form(X[tr], y[tr], a)
            mses.append(mse(y[te], add_intercept(X[te]) @ w))
        cv.append(np.mean(mses))
    return np.array(cv)

def cv_score_lasso_cd(X, y, lambdas, k=5, seed=0):
    folds = kfold_indices(len(y), k, seed)
    cv = []
    for lam in lambdas:
        mses = []
        for i in range(k):
            te = folds[i]; tr = np.hstack([folds[j] for j in range(k) if j != i])
            w = lasso_coordinate_descent(X[tr], y[tr], lam)
            mses.append(mse(y[te], add_intercept(X[te]) @ w))
        cv.append(np.mean(mses))
    return np.array(cv)

def main():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    train_m, base_region = preprocess_for_model(train, BASE_REGION)
    test_m,  _           = preprocess_for_model(test,  base_region)

    features = [c for c in train_m.columns if c != TARGET]
    Xtr = train_m[features].to_numpy().astype(float); ytr = train_m[TARGET].to_numpy().astype(float)
    Xte = test_m[features].to_numpy().astype(float);  yte = test_m[TARGET].to_numpy().astype(float)

    results = []

    yhat_const = np.full_like(yte, ytr.mean(), dtype=float)
    results.append({"model":"constant_mean","params":"y_hat = mean(train)",
                    "test_MSE": mse(yte,yhat_const),
                    "test_MAE": mae(yte,yhat_const),
                    "test_R2":  r2(yte,yhat_const)})

    Xtr_cf = add_intercept(Xtr); Xte_cf = add_intercept(Xte)
    XtX = Xtr_cf.T @ Xtr_cf
    try: XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError: XtX_inv = np.linalg.pinv(XtX)
    w_ols = XtX_inv @ Xtr_cf.T @ ytr
    yhat_ols = Xte_cf @ w_ols
    results.append({"model":"ols_closed_form","params":"—",
                    "test_MSE": mse(yte,yhat_ols),
                    "test_MAE": mae(yte,yhat_ols),
                    "test_R2":  r2(yte,yhat_ols)})

    ridge_alphas = np.logspace(-3, 4, 20)
    cv_ridge = cv_score_ridge_closed_form(Xtr, ytr, ridge_alphas, k=5, seed=0)
    alpha_best = float(ridge_alphas[np.argmin(cv_ridge)])
    w_ridge = ridge_closed_form(Xtr, ytr, alpha_best)
    yhat_ridge = Xte_cf @ w_ridge
    results.append({"model":"ridge_closed_form", "params":f"alpha={alpha_best:.4g}",
                    "test_MSE": mse(yte,yhat_ridge),
                    "test_MAE": mae(yte,yhat_ridge),
                    "test_R2":  r2(yte,yhat_ridge)})

    lasso_lambdas = np.logspace(-3, 3, 20)
    cv_lasso = cv_score_lasso_cd(Xtr, ytr, lasso_lambdas, k=5, seed=0)
    lambda_best = float(lasso_lambdas[np.argmin(cv_lasso)])
    w_lasso = lasso_coordinate_descent(Xtr, ytr, lambda_best)
    yhat_lasso = Xte_cf @ w_lasso
    results.append({"model":"lasso_coordinate_descent","params":f"lambda={lambda_best:.4g}",
                    "test_MSE": mse(yte,yhat_lasso),
                    "test_MAE": mae(yte,yhat_lasso),
                    "test_R2":  r2(yte,yhat_lasso)})

    summary = pd.DataFrame(results).sort_values("test_MSE").reset_index(drop=True)
    summary.to_csv(OUT_DIR / "lab3_task4_test_mse_comparison.csv", index=False)

    plt.figure(figsize=(8,5))
    plt.bar(summary["model"], summary["test_MSE"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Test MSE")
    plt.title("Task 4: Test MSE comparison")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lab3_task4_test_mse_bar.png", dpi=150)
    plt.close()

    print("Результаты в:", OUT_DIR)

if __name__ == "__main__":
    main()