import math
import os
import pandas as pd
import matplotlib.pyplot as plt

def to_float_list(series):
    vals = []
    for v in series:
        try:
            x = float(v)
            if not (math.isnan(x) or math.isinf(x)):
                vals.append(x)
        except Exception:
            pass
    return vals

def mean_manual(a):
    s = 0.0
    for x in a:
        s += x
    return s / len(a)

def central_moment_sums(a, mu):
    s2 = s3 = s4 = 0.0
    for x in a:
        d = x - mu
        d2 = d * d
        s2 += d2
        s3 += d2 * d
        s4 += d2 * d2
    return s2, s3, s4

def variance_population_from_s2sum(n, s2sum):
    return s2sum / n

def variance_sample_from_s2sum(n, s2sum):
    return s2sum / (n - 1)

def mode_manual(a, rounding=6):
    freq = {}
    for x in a:
        key = round(x, rounding)
        freq[key] = freq.get(key, 0) + 1
    maxc = -1
    modes = []
    for k, c in freq.items():
        if c > maxc:
            maxc = c
            modes = [k]
        elif c == maxc:
            modes.append(k)
    modes.sort()
    return modes  # может быть несколько

def sorted_copy(a):
    b = list(a)
    b.sort()
    return b

def quantile_linear(sorted_a, p):
    n = len(sorted_a)
    if n == 0:
        return math.nan
    if p <= 0:
        return sorted_a[0]
    if p >= 1:
        return sorted_a[-1]
    h = (n - 1) * p
    low = int(math.floor(h))
    high = int(math.ceil(h))
    if low == high:
        return sorted_a[low]
    w_high = h - low
    w_low = 1.0 - w_high
    return sorted_a[low] * w_low + sorted_a[high] * w_high

def iqr_from_sorted(sorted_a):
    q25 = quantile_linear(sorted_a, 0.25)
    q75 = quantile_linear(sorted_a, 0.75)
    return q25, q75, (q75 - q25)

def skewness_unbiased(a, mu, s2sum):
    n = len(a)
    s2 = variance_sample_from_s2sum(n, s2sum)
    s = math.sqrt(s2)
    # ∑(x-μ)^3
    s3sum = 0.0
    for x in a:
        d = x - mu
        s3sum += d * d * d
    return (n / ((n - 1) * (n - 2))) * (s3sum / (s ** 3))

def kurtosis_excess_unbiased(a, mu, s2sum):
    n = len(a)
    s2 = variance_sample_from_s2sum(n, s2sum)
    s4 = s2 * s2
    # ∑(x-μ)^4
    s4sum = 0.0
    for x in a:
        d = x - mu
        d2 = d * d
        s4sum += d2 * d2
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * (s4sum / s4)
    term2 = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return term1 - term2

def min_max_manual(a):
    mn = mx = a[0]
    for x in a[1:]:
        if x < mn: mn = x
        if x > mx: mx = x
    return mn, mx

def freedman_diaconis_bin_width(sorted_a):
    n = len(sorted_a)
    q1, q3, iqr = iqr_from_sorted(sorted_a)
    if iqr <= 0:
        return None
    return 2.0 * iqr * (n ** (-1.0 / 3.0))

def histogram_manual(a_sorted, h=None, min_edge=None, max_edge=None):
    n = len(a_sorted)
    mn = a_sorted[0]
    mx = a_sorted[-1]
    if min_edge is None: min_edge = mn
    if max_edge is None: max_edge = mx
    if h is None:
        h = freedman_diaconis_bin_width(a_sorted)
    if (h is None) or (h <= 0):
        # fallback: правило Стерджеса
        k = int(1 + math.log2(n))
        h = (max_edge - min_edge) / max(k, 1)
    # количество бинов
    bins = max(1, int(math.ceil((max_edge - min_edge) / h)))
    edges = [min_edge + i * h for i in range(bins + 1)]
    counts = [0] * bins
    # считаем частоты
    j = 0
    for x in a_sorted:
        # индекс бина
        if x >= edges[-1]:
            idx = bins - 1
        else:
            idx = int((x - min_edge) // h)
            if idx < 0: idx = 0
            if idx >= bins: idx = bins - 1
        counts[idx] += 1
    return edges, counts


def main():
    N = 8
    cols = [
        "Daily_Usage_Hours","Sleep_Hours","Exercise_Hours",
        "Screen_Time_Before_Bed","Time_on_Social_Media",
        "Time_on_Gaming","Time_on_Education",
    ]
    selected_col = cols[N % len(cols)]  # 8 % 7 = 1 -> Sleep_Hours

    data_path = "datasets/teen_phone_addiction_dataset.csv"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if selected_col not in df.columns:
        raise ValueError(f"Нет столбца '{selected_col}'. Доступные: {list(df.columns)}")

    a = to_float_list(df[selected_col].values)
    n = len(a)
    if n == 0:
        raise ValueError("После очистки нет данных")

    mu = mean_manual(a)
    s2sum, s3sum, s4sum = central_moment_sums(a, mu)  # sum (x-μ)^k
    var_pop = variance_population_from_s2sum(n, s2sum)
    var_sam = variance_sample_from_s2sum(n, s2sum)

    modes = mode_manual(a)  # список мод
    a_sorted = sorted_copy(a)

    # квантили и IQR (через линейную интерполяцию)
    q25 = quantile_linear(a_sorted, 0.25)
    q50 = quantile_linear(a_sorted, 0.5)   # медиана
    q75 = quantile_linear(a_sorted, 0.75)
    iqr = q75 - q25

    # асимметрия и эксцесс (bias-corrected)
    skew = skewness_unbiased(a, mu, s2sum)
    kurt_excess = kurtosis_excess_unbiased(a, mu, s2sum)

    mn, mx = min_max_manual(a)

    def fmt(x): return f"{x:.4f}" if isinstance(x, float) else str(x)
    print(f"[OK] Пункт I (ручной расчет) для столбца: {selected_col} (N={N})")
    print("\n=== Descriptive statistics (manual) ===")
    rows = [
        ("Column", selected_col),
        ("Count (n)", n),
        ("Mean", mu),
        ("Variance (sample, ddof=1)", var_sam),
        ("Variance (population, ddof=0)", var_pop),
        ("Mode(s)", ", ".join(map(str, modes)) if modes else "—"),
        ("Median (Q0.5)", q50),
        ("Quantile 0.25", q25),
        ("Quantile 0.75", q75),
        ("Interquartile Range (IQR)", iqr),
        ("Skewness (unbiased)", skew),
        ("Excess Kurtosis (unbiased)", kurt_excess),
        ("Min", mn),
        ("Max", mx),
    ]
    for k, v in rows:
        print(f"{k:>32}: {fmt(v)}")

    edges, counts = histogram_manual(a_sorted)
    # рисуем как столбиковую диаграмму по границам бинов
    plt.figure()
    # ширина каждого столбца = разница соседних границ
    widths = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
    plt.bar(edges[:-1], counts, width=widths, align="edge")
    plt.title(f"Histogram of {selected_col} (manual bins)")
    plt.xlabel(selected_col)
    plt.ylabel("Frequency")
    hist_path = os.path.join(out_dir, f"hist_manual_{selected_col}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    y = [ (i+1) / n for i in range(n) ]
    plt.figure()
    plt.step(a_sorted, y, where="post")
    plt.title(f"Empirical CDF of {selected_col} (manual)")
    plt.xlabel(selected_col)
    plt.ylabel("F(x)")
    ecdf_path = os.path.join(out_dir, f"ecdf_manual_{selected_col}.png")
    plt.savefig(ecdf_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nФайлы графиков:")
    print(f" - Histogram (manual): {hist_path}")
    print(f" - ECDF (manual):     {ecdf_path}")

if __name__ == "__main__":
    main()