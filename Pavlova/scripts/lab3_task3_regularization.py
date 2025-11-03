from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "insurance_train.csv"
TEST_PATH  = DATA_DIR / "insurance_test.csv"

TARGET = "charges"
BASE_REGION = "region_southwest"

def preprocess_for_model(df: pd.DataFrame, base_region: str = BASE_REGION):
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

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma

def standardize_apply(X: np.ndarray, mu, sigma):
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

def mse(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def save_metrics(prefix, y_tr, yhat_tr, y_te, yhat_te):
    dfm = pd.DataFrame({
        "split": ["train", "test"],
        "MSE": [mse(y_tr, yhat_tr), mse(y_te, yhat_te)],
        "MAE": [mae(y_tr, yhat_tr), mae(y_te, yhat_te)],
        "R2":  [r2(y_tr, yhat_tr),  r2(y_te, yhat_te)],
    })
    dfm.to_csv(OUT_DIR / f"lab3_task3_metrics_{prefix}.csv", index=False)
    return dfm

def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float):
    Xcf = add_intercept(X)
    n_feat = Xcf.shape[1]
    R = np.eye(n_feat)
    R[0, 0] = 0.0
    XtX = Xcf.T @ Xcf + alpha * R
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    w = XtX_inv @ Xcf.T @ y
    return w

def ridge_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float,
                           lr=0.05, max_iter=3000, tol=1e-7, seed=0):

    Xs, mu, sigma = standardize_fit(X)
    Xg = add_intercept(Xs)
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.01, size=Xg.shape[1])
    R = np.zeros_like(w)
    R[1:] = 1.0

    history = []
    prev = None
    for it in range(max_iter):
        yhat = Xg @ w
        grad = 2.0 * (Xg.T @ (yhat - y)) / len(y) + 2.0 * alpha * np.r_[0.0, w[1:]]
        w -= lr * grad
        loss = mse(y, yhat) + alpha * float(np.sum((w[1:]) ** 2))
        history.append(loss)
        if prev is not None and abs(prev - loss) < tol:
            break
        prev = loss

    w0_std, w_rest_std = w[0], w[1:]
    w_rest_unstd = w_rest_std / sigma
    w0_unstd = w0_std - np.sum(w_rest_std * (mu / sigma))
    w_unstd = np.concatenate([[w0_unstd], w_rest_unstd])
    return w_unstd, history

def soft_threshold(z, gamma):
    if z > gamma:   return z - gamma
    if z < -gamma:  return z + gamma
    return 0.0

def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, lam: float,
                             max_iter=5000, tol=1e-6):
    n, p = X.shape
    Xs, mu, sigma = standardize_fit(X)
    y_mean = y.mean()
    yc = y - y_mean

    w = np.zeros(p)
    Xj_sq = np.sum(Xs ** 2, axis=0) / n

    for it in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            r = yc + Xs[:, j] * w[j] - Xs @ w
            rho_j = np.dot(Xs[:, j], r) / n
            w[j] = soft_threshold(rho_j, lam) / (Xj_sq[j] if Xj_sq[j] != 0 else 1.0)
        if np.max(np.abs(w - w_old)) < tol:
            break

    w_unstd = w / sigma
    w0_unstd = y_mean - np.sum((mu * w_unstd))
    w_full = np.concatenate([[w0_unstd], w_unstd])
    return w_full

def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def cv_score_ridge_closed_form(X, y, alphas, k=5, seed=0):
    folds = kfold_indices(len(y), k, seed)
    cv = []
    for a in alphas:
        mses = []
        for i in range(k):
            te = folds[i]; tr = np.hstack([folds[j] for j in range(k) if j != i])
            w = ridge_closed_form(X[tr], y[tr], a)
            yhat = add_intercept(X[te]) @ w
            mses.append(mse(y[te], yhat))
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
            yhat = add_intercept(X[te]) @ w
            mses.append(mse(y[te], yhat))
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

    ridge_alphas  = np.logspace(-3, 4, 20)   # [1e-3 .. 1e4]
    lasso_lambdas = np.logspace(-3, 3, 20)   # [1e-3 .. 1e3]

    cv_ridge = cv_score_ridge_closed_form(Xtr, ytr, ridge_alphas, k=5, seed=0)
    best_alpha = float(ridge_alphas[np.argmin(cv_ridge)])

    w_ridge_cf = ridge_closed_form(Xtr, ytr, best_alpha)
    yhat_tr_cf = add_intercept(Xtr) @ w_ridge_cf
    yhat_te_cf = add_intercept(Xte) @ w_ridge_cf
    pd.DataFrame({"feature": ["intercept"] + features, "weight": w_ridge_cf}) \
      .to_csv(OUT_DIR / "lab3_task3_ridge_weights_closed_form.csv", index=False)
    save_metrics("ridge_closed_form", ytr, yhat_tr_cf, yte, yhat_te_cf)

    w_ridge_gd, history_ridge = ridge_gradient_descent(Xtr, ytr, best_alpha, lr=0.05, max_iter=3000)
    yhat_tr_gd = add_intercept(Xtr) @ w_ridge_gd
    yhat_te_gd = add_intercept(Xte) @ w_ridge_gd
    pd.DataFrame({"feature": ["intercept"] + features, "weight": w_ridge_gd}) \
      .to_csv(OUT_DIR / "lab3_task3_ridge_weights_gradient.csv", index=False)
    save_metrics("ridge_gradient", ytr, yhat_tr_gd, yte, yhat_te_gd)

    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(history_ridge)+1), history_ridge)
    plt.xlabel("Iteration"); plt.ylabel("Objective = MSE + alpha * L2")
    plt.title(f"Ridge (GD) — loss curve (alpha={best_alpha:.4g})")
    plt.tight_layout(); plt.savefig(OUT_DIR / "lab3_task3_ridge_loss_curve.png", dpi=150); plt.close()

    cv_lasso = cv_score_lasso_cd(Xtr, ytr, lasso_lambdas, k=5, seed=0)
    best_lambda = float(lasso_lambdas[np.argmin(cv_lasso)])

    w_lasso = lasso_coordinate_descent(Xtr, ytr, best_lambda)
    yhat_tr_l1 = add_intercept(Xtr) @ w_lasso
    yhat_te_l1 = add_intercept(Xte) @ w_lasso
    pd.DataFrame({"feature": ["intercept"] + features, "weight": w_lasso}) \
      .to_csv(OUT_DIR / "lab3_task3_lasso_weights_coordinate_descent.csv", index=False)
    save_metrics("lasso_coordinate_descent", ytr, yhat_tr_l1, yte, yhat_te_l1)

    coeffs_ridge = []
    for a in ridge_alphas:
        w = ridge_closed_form(Xtr, ytr, a)
        coeffs_ridge.append(w[1:])
    coeffs_ridge = np.vstack(coeffs_ridge)
    plt.figure(figsize=(8,5))
    for j, name in enumerate(features):
        plt.plot(np.log10(ridge_alphas), coeffs_ridge[:, j], label=name)
    plt.xlabel("log10(alpha)"); plt.ylabel("Coefficient")
    plt.title("Ridge — coefficient paths")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_DIR / "lab3_task3_ridge_coef_paths.png", dpi=150); plt.close()

    coeffs_lasso = []
    for lam in lasso_lambdas:
        w = lasso_coordinate_descent(Xtr, ytr, lam)
        coeffs_lasso.append(w[1:])
    coeffs_lasso = np.vstack(coeffs_lasso)
    plt.figure(figsize=(8,5))
    for j, name in enumerate(features):
        plt.plot(np.log10(lasso_lambdas), coeffs_lasso[:, j], label=name)
    plt.xlabel("log10(lambda)"); plt.ylabel("Coefficient")
    plt.title("LASSO — coefficient paths")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_DIR / "lab3_task3_lasso_coef_paths.png", dpi=150); plt.close()

    pd.DataFrame({"alpha": ridge_alphas, "cv_mse": cv_ridge}).to_csv(
        OUT_DIR / "lab3_task3_ridge_cv.csv", index=False
    )
    pd.DataFrame({"lambda": lasso_lambdas, "cv_mse": cv_lasso}).to_csv(
        OUT_DIR / "lab3_task3_lasso_cv.csv", index=False
    )
    plt.figure(figsize=(7,5))
    plt.plot(np.log10(ridge_alphas), cv_ridge, marker="o")
    plt.xlabel("log10(alpha)"); plt.ylabel("CV MSE")
    plt.title("Ridge — CV curve")
    plt.tight_layout(); plt.savefig(OUT_DIR / "lab3_task3_ridge_cv_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(7,5))
    plt.plot(np.log10(lasso_lambdas), cv_lasso, marker="o")
    plt.xlabel("log10(lambda)"); plt.ylabel("CV MSE")
    plt.title("LASSO — CV curve")
    plt.tight_layout(); plt.savefig(OUT_DIR / "lab3_task3_lasso_cv_curve.png", dpi=150); plt.close()

    print("Результаты в:", OUT_DIR)

if __name__ == "__main__":
    main()