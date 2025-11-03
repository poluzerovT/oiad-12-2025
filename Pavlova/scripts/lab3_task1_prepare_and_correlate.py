from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
OUT_DIR = ROOT / "outputs"
BOX_DIR = OUT_DIR / "lab3_task1_boxplots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOX_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "insurance_train.csv"

df = pd.read_csv(TRAIN_PATH)

missing_abs = df.isna().sum().rename("missing_count")
missing_pct = (df.isna().mean() * 100).round(2).rename("missing_%")
missing_report = pd.concat([missing_abs, missing_pct], axis=1)
missing_report.to_csv(OUT_DIR / "lab3_task1_missing_report.csv", index=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
iqr_rows = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((df[col] < low) | (df[col] > high)).sum()
    iqr_rows.append({
        "feature": col,
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "low_fence": low, "high_fence": high,
        "outliers_count": int(outliers),
        "outliers_%": round(100 * outliers / len(df), 2)
    })
    plt.figure(figsize=(5, 4))
    plt.boxplot(df[col].dropna(), vert=True, showmeans=True)
    plt.title(f"Boxplot (IQR) — {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(BOX_DIR / f"box_{col}.png", dpi=150)
    plt.close()

iqr_report = pd.DataFrame(iqr_rows)
iqr_report.to_csv(OUT_DIR / "lab3_task1_outliers_iqr_report.csv", index=False)

df_prep = df.copy()

if "sex" in df_prep.columns:
    df_prep["sex"] = df_prep["sex"].map({"female": 0, "male": 1}).astype("Int64")
if "smoker" in df_prep.columns:
    df_prep["smoker"] = df_prep["smoker"].map({"no": 0, "yes": 1}).astype("Int64")

if "region" in df_prep.columns:
    region_dummies = pd.get_dummies(df_prep["region"], prefix="region", drop_first=False)
    df_prep = pd.concat([df_prep.drop(columns=["region"]), region_dummies], axis=1)

df_prep.to_csv(OUT_DIR / "lab3_task1_prepared_dataset.csv", index=False)

corr = df_prep.corr(numeric_only=True, method="pearson")
corr.to_csv(OUT_DIR / "lab3_task1_corr_matrix.csv", index=True)

if "charges" in df_prep.columns:
    target_corr = corr["charges"].sort_values(ascending=False)
    target_corr.to_csv(OUT_DIR / "lab3_task1_corr_with_charges.csv", header=["pearson_r"])

plt.figure(figsize=(8, 6))
im = plt.imshow(corr, interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation heatmap (Pearson)")
plt.tight_layout()
plt.savefig(OUT_DIR / "lab3_task1_corr_heatmap.png", dpi=150)
plt.close()

print("Отчёты и графики сохранены в:", OUT_DIR)