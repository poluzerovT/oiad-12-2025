import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def grade_sort_key(v: str):
    if pd.isna(v):
        return (1, "")  # пусть NaN идут первыми
    s = str(v)
    m = re.search(r'\d+', s)
    if m:
        return (0, int(m.group()))
    return (1, s.lower())


def main():
    # --- выбор столбца по N ---
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
    selected_col = cols[N % len(cols)]  # 8 % 7 = 1 -> 'Sleep_Hours'

    data_path = "datasets/teen_phone_addiction_dataset.csv"
    out_dir = "outputs/part4"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if selected_col not in df.columns:
        raise ValueError(f"Нет столбца '{selected_col}'. Доступные: {list(df.columns)}")
    if "School_Grade" not in df.columns:
        raise ValueError("В датасете нет столбца 'School_Grade'.")

    gdf = df[["School_Grade", selected_col]].copy()
    gdf[selected_col] = pd.to_numeric(gdf[selected_col], errors="coerce")
    gdf = gdf.dropna(subset=["School_Grade", selected_col])

    groups = []
    for level, sub in gdf.groupby("School_Grade"):
        arr = sub[selected_col].to_numpy()
        n = arr.size
        if n == 0:
            continue
        mean = float(arr.mean())
        var_sample = float(arr.var(ddof=1))   # выборочная дисперсия
        var_pop = float(arr.var(ddof=0))      # "генеральная"
        groups.append({
            "School_Grade": level,
            "n": n,
            "mean": mean,
            "var_sample": var_sample,
            "var_population": var_pop,
        })

    if not groups:
        raise ValueError("После очистки данные по группам пусты.")

    groups_sorted = sorted(groups, key=lambda d: grade_sort_key(d["School_Grade"]))

    stats_df = pd.DataFrame(groups_sorted)
    stats_csv = os.path.join(out_dir, f"group_stats_{selected_col}_by_School_Grade.csv")
    stats_df.to_csv(stats_csv, index=False)

    pooled = gdf[selected_col].to_numpy()
    bin_edges = np.histogram_bin_edges(pooled, bins="auto")

    plt.figure(figsize=(9, 6))

    labels = []
    for grp in groups_sorted:
        level = grp["School_Grade"]
        arr = gdf.loc[gdf["School_Grade"] == level, selected_col].to_numpy()
        plt.hist(arr, bins=bin_edges, density=True, histtype="step", linewidth=2, label=str(level), alpha=0.9)
        labels.append(level)

    plt.title(f"Гистограммы по группам School_Grade — {selected_col}")
    plt.xlabel(selected_col)
    plt.ylabel("Плотность")
    plt.legend(title="School_Grade", ncol=2, frameon=False)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"hist_by_grade_{selected_col}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Пункт IV для столбца: {selected_col} (N={N})")
    print(f"Таблица со статистиками по группам: {stats_csv}")
    print(f"Гистограммы (общий график):       {fig_path}")
    print("\n=== Grouped means & variances ===")
    show = stats_df.copy()
    show["mean"] = show["mean"].map(lambda x: f"{x:.4f}")
    show["var_sample"] = show["var_sample"].map(lambda x: f"{x:.4f}")
    show["var_population"] = show["var_population"].map(lambda x: f"{x:.4f}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
