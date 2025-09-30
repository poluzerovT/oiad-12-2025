import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def main():
    N = 8

    cols = [
        "Daily_Usage_Hours",
        "Sleep_Hours",
        "Exercise_Hours",
        "Screen_Time_Before_Bed",
        "Time_on_Social_Media",
        "Time_on_Gaming",
        "Time_on_Education",
    ]
    selected_col = cols[N % len(cols)]

    data_path = "datasets/teen_phone_addiction_dataset.csv"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if selected_col not in df.columns:
        raise ValueError(
            f"В датасете нет столбца '{selected_col}'. Доступные столбцы: {list(df.columns)}"
        )

    s = pd.to_numeric(df[selected_col], errors="coerce").dropna()
    n = len(s)

    mean_val = s.mean()
    var_sample = s.var(ddof=1)
    var_population = s.var(ddof=0)
    modes = s.mode()
    mode_repr = ", ".join(map(str, modes.tolist())) if len(modes) else "—"
    median_val = s.median()
    q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
    iqr = q75 - q25
    skew_val = s.skew()
    kurt_excess = s.kurt()

    stats_df = pd.DataFrame({
        "Metric": [
            "Column",
            "Count (n)",
            "Mean",
            "Variance (sample, ddof=1)",
            "Variance (population, ddof=0)",
            "Mode(s)",
            "Median",
            "Quantile 0.25",
            "Quantile 0.5 (Median)",
            "Quantile 0.75",
            "Interquartile Range (IQR)",
            "Skewness",
            "Excess Kurtosis",
            "Min",
            "Max",
        ],
        "Value": [
            selected_col,
            n,
            mean_val,
            var_sample,
            var_population,
            mode_repr,
            median_val,
            q25,
            q50,
            q75,
            iqr,
            skew_val,
            kurt_excess,
            s.min(),
            s.max(),
        ]
    })

    stats_csv_path = os.path.join(out_dir, f"part_I_stats_{selected_col}.csv")
    stats_df.to_csv(stats_csv_path, index=False)

    plt.figure()
    plt.hist(s, bins="auto")
    plt.title(f"Histogram of {selected_col}")
    plt.xlabel(selected_col)
    plt.ylabel("Frequency")
    hist_path = os.path.join(out_dir, f"hist_{selected_col}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    x_sorted = np.sort(s.values)
    y = np.arange(1, n + 1) / n

    plt.figure()
    plt.step(x_sorted, y, where="post")
    plt.title(f"Empirical CDF of {selected_col}")
    plt.xlabel(selected_col)
    plt.ylabel("F(x)")
    ecdf_path = os.path.join(out_dir, f"ecdf_{selected_col}.png")
    plt.savefig(ecdf_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Пункт I для столбца: {selected_col} (N={N})")
    print(f"Файлы сохранены в: {os.path.abspath(out_dir)}")
    print(f" - CSV: {stats_csv_path}")
    print(f" - Histogram: {hist_path}")
    print(f" - ECDF: {ecdf_path}")

    printable = stats_df.copy()
    printable["Value"] = printable["Value"].map(fmt_value)
    print("\n=== Descriptive statistics ===")
    print(printable.to_string(index=False))

    print("\n=== Quick summary ===")
    print(f"Count: {n}")
    print(f"Mean: {mean_val:.4f}, Median: {median_val:.4f}")
    print(f"Q25: {q25:.4f}, Q50: {q50:.4f}, Q75: {q75:.4f}, IQR: {iqr:.4f}")
    print(f"Variance (sample): {var_sample:.4f}, Variance (pop): {var_population:.4f}")
    print(f"Skewness: {skew_val:.4f}, Excess Kurtosis: {kurt_excess:.4f}")
    print(f"Mode(s): {mode_repr}")
    print(f"Min: {s.min():.4f}, Max: {s.max():.4f}\n")


if __name__ == "__main__":
    main()
