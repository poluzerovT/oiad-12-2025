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
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0

def save_metrics(prefix, y_tr, yhat_tr, y_te, yhat_te):
    dfm = pd.DataFrame({
        "split": ["train", "test"],
        "MSE": [mse(y_tr, yhat_tr), mse(y_te, yhat_te)],
        "MAE": [mae(y_tr, yhat_tr), mae(y_te, yhat_te)],
        "R2":  [r2(y_tr, yhat_tr),  r2(y_te, yhat_te)],
    })
    dfm.to_csv(OUT_DIR / f"lab3_task2_metrics_{prefix}.csv", index=False)
    return dfm

def parity_plot(y_true, y_pred, path, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual charges"); plt.ylabel("Predicted charges")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train_m, base_region = preprocess_for_model(train, BASE_REGION)
test_m,  _           = preprocess_for_model(test,  base_region)

features = [c for c in train_m.columns if c != TARGET]
Xtr = train_m[features].to_numpy().astype(float); ytr = train_m[TARGET].to_numpy().astype(float)
Xte = test_m[features].to_numpy().astype(float);  yte = test_m[TARGET].to_numpy().astype(float)

Xtr_cf = add_intercept(Xtr)
Xte_cf = add_intercept(Xte)

XtX = Xtr_cf.T @ Xtr_cf
try:
    XtX_inv = np.linalg.inv(XtX)
except np.linalg.LinAlgError:
    XtX_inv = np.linalg.pinv(XtX)

w_cf = XtX_inv @ Xtr_cf.T @ ytr

yhat_tr_cf = Xtr_cf @ w_cf
yhat_te_cf = Xte_cf @ w_cf

pd.DataFrame({"feature": ["intercept"] + features, "weight": w_cf}) \
  .to_csv(OUT_DIR / "lab3_task2_weights_closed_form.csv", index=False)

save_metrics("closed_form", ytr, yhat_tr_cf, yte, yhat_te_cf)
parity_plot(ytr, yhat_tr_cf, OUT_DIR/"lab3_task2_parity_closed_form_train.png", "Parity (Closed-form) — Train")
parity_plot(yte, yhat_te_cf, OUT_DIR/"lab3_task2_parity_closed_form_test.png",  "Parity (Closed-form) — Test")

Xtr_s, mu, sigma = standardize_fit(Xtr)
Xte_s = standardize_apply(Xte, mu, sigma)

Xtr_gd = add_intercept(Xtr_s)
Xte_gd = add_intercept(Xte_s)

rng = np.random.default_rng(0)
w = rng.normal(scale=0.01, size=Xtr_gd.shape[1])

lr = 0.05
max_iter = 2000
tol = 1e-7
history = []
prev_loss = None
for it in range(max_iter):
    yhat = Xtr_gd @ w
    grad = 2.0 * (Xtr_gd.T @ (yhat - ytr)) / len(ytr)
    w -= lr * grad
    loss = mse(ytr, yhat)
    history.append(loss)
    if prev_loss is not None and abs(prev_loss - loss) < tol:
        break
    prev_loss = loss

yhat_tr_gd = Xtr_gd @ w
yhat_te_gd = Xte_gd @ w

pd.DataFrame({"feature": ["intercept_std"] + features, "weight": w}) \
  .to_csv(OUT_DIR / "lab3_task2_weights_gradient_descent_std.csv", index=False)

w0_std, w_rest_std = w[0], w[1:]
w_rest_unstd = w_rest_std / sigma
w0_unstd = w0_std - np.sum(w_rest_std * (mu / sigma))
w_unstd = np.concatenate([[w0_unstd], w_rest_unstd])
pd.DataFrame({"feature": ["intercept"] + features, "weight": w_unstd}) \
  .to_csv(OUT_DIR / "lab3_task2_weights_gradient_descent_unstd.csv", index=False)

save_metrics("gradient_descent", ytr, yhat_tr_gd, yte, yhat_te_gd)

plt.figure(figsize=(7,5))
plt.plot(range(1, len(history)+1), history)
plt.xlabel("Iteration"); plt.ylabel("MSE (train)")
plt.title("Gradient Descent — Loss curve")
plt.tight_layout(); plt.savefig(OUT_DIR/"lab3_task2_loss_curve.png", dpi=150); plt.close()

print("Результаты и графики — в", OUT_DIR)