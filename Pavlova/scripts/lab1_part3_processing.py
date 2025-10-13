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
    for x in a: s += x
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

def var_sample_from_s2sum(n, s2sum):
    return s2sum / (n - 1)

def min_max_manual(a):
    mn = mx = a[0]
    for x in a[1:]:
        if x < mn: mn = x
        if x > mx: mx = x
    return mn, mx

def sorted_copy(a):
    b = list(a); b.sort(); return b

def quantile_linear(sorted_a, p):
    n = len(sorted_a)
    if p <= 0: return sorted_a[0]
    if p >= 1: return sorted_a[-1]
    h = (n - 1) * p
    lo = int(h)
    hi = int(math.ceil(h))
    if lo == hi: return sorted_a[lo]
    w = h - lo
    return sorted_a[lo]*(1-w) + sorted_a[hi]*w

def iqr_from_sorted(sorted_a):
    q25 = quantile_linear(sorted_a, 0.25)
    q75 = quantile_linear(sorted_a, 0.75)
    return q25, q75, (q75 - q25)

def mode_manual(a, rounding=6):
    freq = {}
    for x in a:
        k = round(x, rounding)
        freq[k] = freq.get(k, 0) + 1
    best = max(freq.values())
    modes = sorted([k for k, c in freq.items() if c == best])
    return modes

# Асимметрия/эксцесс (несмещённые оценки)
def skew_kurt_manual(a, mu):
    n = len(a)
    s2sum, s3sum, s4sum = 0.0, 0.0, 0.0
    for x in a:
        d = x - mu; d2 = d*d
        s2sum += d2
        s3sum += d2 * d
        s4sum += d2 * d2
    s2 = s2sum / (n - 1)
    s = math.sqrt(s2)
    g1 = (n / ((n - 1) * (n - 2))) * (s3sum / (s ** 3))
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (s4sum / (s ** 4)) \
         - 3.0 * (((n - 1) ** 2) / ((n - 2) * (n - 3)))
    return g1, g2


def norm_cdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_ppf(p):
    # Акклам (устойчивый inverse нормали)
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return float('inf')
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1-0.02425
    if p < pl:
        q = math.sqrt(-2.0*math.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p > ph:
        q = math.sqrt(-2.0*math.log(1.0 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    else:
        q = p - 0.5; r = q*q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    # одно уточнение Ньютоном
    e = norm_cdf(x) - p
    pdf = (1.0 / math.sqrt(2.0*math.pi)) * math.exp(-0.5*x*x)
    x -= e / max(pdf, 1e-15)
    return x

def gammainc_regularized_P(a, x, eps=1e-12, itmax=1000):
    if x < 0 or a <= 0: return float('nan')
    if x == 0.0: return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a; s = 1.0 / a; d = s
        for _ in range(itmax):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s)*eps: break
        return s * math.exp(-x + a*math.log(x) - gln)
    else:
        b = x + 1.0 - a; c = 1.0/1e-30; d = 1.0/b; h = d
        for i in range(1, itmax+1):
            an = -i * (i - a)
            b += 2.0
            d = an*d + b; d = 1.0 / (d if abs(d)>1e-30 else 1e-30)
            c = b + an / (c if abs(c)>1e-30 else 1e-30)
            delta = d * c; h *= delta
            if abs(delta - 1.0) < eps: break
        Q = math.exp(-x + a*math.log(x) - gln) * h
        return 1.0 - Q

def chi2_sf(x, k):
    return 1.0 - gammainc_regularized_P(0.5*k, 0.5*x)

def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def freedman_diaconis_bin_width(sorted_a):
    q1, q3, iqr = iqr_from_sorted(sorted_a)
    if iqr <= 0: return None
    n = len(sorted_a)
    return 2.0 * iqr * (n ** (-1.0/3.0))

def histogram_manual(sorted_a):
    n = len(sorted_a)
    mn, mx = sorted_a[0], sorted_a[-1]
    h = freedman_diaconis_bin_width(sorted_a)
    if not h or h <= 0:
        k = max(1, int(1 + math.log2(n)))
        h = (mx - mn) / k if k>0 else (mx - mn) or 1.0
    bins = max(1, int(math.ceil((mx - mn) / h)))
    edges = [mn + i*h for i in range(bins+1)]
    counts = [0]*bins
    for x in sorted_a:
        if x >= edges[-1]:
            idx = bins-1
        else:
            idx = int((x - mn) // h)
            idx = min(max(idx, 0), bins-1)
        counts[idx] += 1
    return edges, counts

def plot_hist_ecdf(a, out_prefix, title, xlabel):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    srt = sorted_copy(a)

    # гистограмма
    edges, counts = histogram_manual(srt)
    widths = [edges[i+1]-edges[i] for i in range(len(edges)-1)]
    plt.figure()
    plt.bar(edges[:-1], counts, width=widths, align="edge")
    plt.title(f"Histogram — {title}"); plt.xlabel(xlabel); plt.ylabel("Frequency")
    hist_path = out_prefix + "_hist.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight"); plt.close()

    # ECDF
    n = len(srt)
    y = [(i+1)/n for i in range(n)]
    plt.figure()
    plt.step(srt, y, where="post")
    plt.title(f"ECDF — {title}"); plt.xlabel(xlabel); plt.ylabel("F(x)")
    ecdf_path = out_prefix + "_ecdf.png"
    plt.savefig(ecdf_path, dpi=150, bbox_inches="tight"); plt.close()

    return hist_path, ecdf_path

def qqplot_normal(a, mu, sigma, out_path, title):
    n = len(a)
    srt = sorted_copy(a)
    probs = [(i+0.5)/n for i in range(n)]
    theo = [mu + sigma*norm_ppf(p) for p in probs]
    plt.figure()
    plt.scatter(theo, srt, s=8)
    # опорная линия по квартилям
    i1, i3 = int(0.25*n), int(0.75*n)-1
    xq = [theo[i1], theo[i3]]; yq = [srt[i1], srt[i3]]
    b = (yq[1]-yq[0]) / (xq[1]-xq[0] + 1e-12); a0 = yq[0] - b*xq[0]
    xline = [min(theo), max(theo)]; yline = [a0 + b*x for x in xline]
    plt.plot(xline, yline, linewidth=2)
    plt.title(title); plt.xlabel("Theoretical quantiles (Normal)"); plt.ylabel("Sample quantiles")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def chi_square_normal_test(a):
    n = len(a)
    mu = mean_manual(a)
    s2sum, _, _ = central_moment_sums(a, mu)
    s2 = var_sample_from_s2sum(n, s2sum)
    sigma = math.sqrt(s2)

    sturges = int(round(1 + math.log2(n)))
    k = max(6, sturges)
    if n / k < 5: k = max(3, int(n//5))

    ps = [i/k for i in range(0, k+1)]
    edges = [mu + sigma*norm_ppf(p) if 0<p<1 else (-float('inf') if p<=0 else float('inf')) for p in ps]

    observed = [0]*k
    for x in a:
        lo, hi = 0, k
        while lo < hi:
            mid = (lo+hi)//2
            if x >= edges[mid+1]:
                lo = mid+1
            else:
                hi = mid
        idx = lo if lo<k else k-1
        observed[idx] += 1

    expected = [n/k]*k
    chi2 = sum((Oi-Ei)*(Oi-Ei)/Ei for Oi, Ei in zip(observed, expected))
    df = max(1, k-1-2)
    pval = chi2_sf(chi2, df)

    return {"n":n, "mu":mu, "sigma":sigma, "k":k, "df":df,
            "chi2":chi2, "p_value":pval, "edges":edges,
            "observed":observed, "expected":expected}

# ------------------------ Преобразования данных ------------------------

def zscore_transform(a):
    mu = mean_manual(a)
    s2sum, _, _ = central_moment_sums(a, mu)
    s = math.sqrt(var_sample_from_s2sum(len(a), s2sum))
    return [(x - mu)/s for x in a]

def tukey_bounds(a):
    srt = sorted_copy(a)
    q1, q3, iqr = iqr_from_sorted(srt)
    low = q1 - 1.5*iqr
    high = q3 + 1.5*iqr
    return low, high

def winsorize_tukey(a):
    low, high = tukey_bounds(a)
    return [ low if x<low else (high if x>high else x) for x in a ]

def trim_tukey(a):
    low, high = tukey_bounds(a)
    return [x for x in a if (low <= x <= high)]

def boxcox_transform(a, lam):
    # для lam!=0: (x^lam - 1)/lam; lam==0: ln x  (x>0)
    return [ (x**lam - 1.0)/lam if abs(lam) > 1e-12 else math.log(x) for x in a ]

def choose_boxcox_lambda(a):
    # подбираем λ, минимизируя skew^2 + excess^2
    # значения x > 0 гарантированы (Sleep_Hours >= 3)
    best = (None, float('inf'))
    for lam_i in [ -2.0 + i*0.05 for i in range(int((2.0 - (-2.0))/0.05)+1) ]:
        y = boxcox_transform(a, lam_i)
        mu = mean_manual(y)
        skew, excess = skew_kurt_manual(y, mu)
        score = skew*skew + excess*excess
        if score < best[1]:
            best = (lam_i, score)
    return best[0]


def run_all_for_variant(a, variant_name, out_root, xlabel):
    os.makedirs(out_root, exist_ok=True)

    # Пункт I — описательная статистика
    n = len(a)
    mu = mean_manual(a)
    s2sum, s3sum, s4sum = central_moment_sums(a, mu)
    var_sam = var_sample_from_s2sum(n, s2sum)
    srt = sorted_copy(a)
    q25 = quantile_linear(srt, 0.25)
    q50 = quantile_linear(srt, 0.5)
    q75 = quantile_linear(srt, 0.75)
    iqr = q75 - q25
    modes = mode_manual(a)
    skew, excess = skew_kurt_manual(a, mu)
    mn, mx = min_max_manual(a)

    # сохраняем CSV со статистикой
    stats_path = os.path.join(out_root, f"{variant_name}_stats.csv")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")
        rows = [
            ("Count (n)", n), ("Mean", mu),
            ("Variance (sample, ddof=1)", var_sam),
            ("Mode(s)", "; ".join(map(str, modes))),
            ("Median", q50), ("Quantile 0.25", q25), ("Quantile 0.75", q75),
            ("Interquartile Range (IQR)", iqr),
            ("Skewness", skew), ("Excess Kurtosis", excess),
            ("Min", mn), ("Max", mx)
        ]
        for k, v in rows:
            f.write(f"{k},{v}\n")

    # Графики Пункта I
    prefix = os.path.join(out_root, f"{variant_name}")
    hist_path, ecdf_path = plot_hist_ecdf(a, prefix, f"{variant_name}", xlabel)

    # Пункт II — χ² + skew/kurt + Q-Q
    chi2_res = chi_square_normal_test(a)
    chi2_bins_csv = os.path.join(out_root, f"{variant_name}_chi2_bins.csv")
    with open(chi2_bins_csv, "w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,observed,expected\n")
        for i in range(chi2_res["k"]):
            f.write(f"{chi2_res['edges'][i]},{chi2_res['edges'][i+1]},{chi2_res['observed'][i]},{chi2_res['expected'][i]}\n")
    # график бинов
    plt.figure()
    xs = list(range(1, chi2_res["k"]+1))
    plt.bar(xs, chi2_res["observed"], label="Observed")
    plt.plot(xs, chi2_res["expected"], linewidth=2, label="Expected (n/k)")
    plt.title(f"Chi-square bins — {variant_name} (k={chi2_res['k']}, df={chi2_res['df']})")
    plt.xlabel("Bin"); plt.ylabel("Count"); plt.legend()
    chi2_plot = os.path.join(out_root, f"{variant_name}_chi2_bins.png")
    plt.savefig(chi2_plot, dpi=150, bbox_inches="tight"); plt.close()

    # z-тесты асимметрии/эксцесса
    z_skew  = skew   / math.sqrt(6.0 / n)
    z_kurt  = excess / math.sqrt(24.0 / n)
    p_skew  = two_sided_p_from_z(z_skew)
    p_kurt  = two_sided_p_from_z(z_kurt)

    # Q–Q
    qq_path = os.path.join(out_root, f"{variant_name}_qq.png")
    sigma = math.sqrt(var_sam)
    qqplot_normal(a, mu, sigma, qq_path, f"Q–Q — {variant_name}")

    # печать краткой сводки
    def fmt(x): return f"{x:.6f}" if isinstance(x, float) else str(x)
    print(f"\n[{variant_name}] n={n}")
    print(f"  mean={fmt(mu)}, var(sample)={fmt(var_sam)}, skew={fmt(skew)}, excess={fmt(excess)}")
    print(f"  χ²: k={chi2_res['k']}, df={chi2_res['df']}, value={fmt(chi2_res['chi2'])}, p={fmt(chi2_res['p_value'])}")
    print(f"  Z_skew={fmt(z_skew)}, p={fmt(p_skew)}  |  Z_kurt={fmt(z_kurt)}, p={fmt(p_kurt)}")
    print(f"  Files: {stats_path}, {hist_path}, {ecdf_path}, {chi2_bins_csv}, {chi2_plot}, {qq_path}")

    return {
        "variant": variant_name,
        "p_chi2": chi2_res["p_value"],
        "p_skew": p_skew,
        "p_kurt": p_kurt,
        "skew": skew,
        "excess": excess
    }

# ------------------------ MAIN ------------------------

def main():
    # 1) выбор признака по правилу N%7
    N = 8
    cols = [
        "Daily_Usage_Hours","Sleep_Hours","Exercise_Hours",
        "Screen_Time_Before_Bed","Time_on_Social_Media",
        "Time_on_Gaming","Time_on_Education",
    ]
    selected_col = cols[N % len(cols)]  # 8%7=1 -> Sleep_Hours

    data_path = "datasets/teen_phone_addiction_dataset.csv"
    out_dir = "outputs/part3"
    os.makedirs(out_dir, exist_ok=True)

    # 2) читаем столбец и приводим к числам
    df = pd.read_csv(data_path)
    if selected_col not in df.columns:
        raise ValueError(f"Нет столбца '{selected_col}'. Доступные: {list(df.columns)}")
    x = to_float_list(df[selected_col].values)
    if len(x) == 0:
        raise ValueError("После очистки нет данных")

    # 3) формируем варианты преобразований
    variants = []

    # raw
    variants.append(("raw", list(x), selected_col))

    # z-score
    variants.append(("zscore", zscore_transform(x), f"z({selected_col})"))

    # winsorization by Tukey
    variants.append(("winsor15", winsorize_tukey(x), f"winsor({selected_col})"))

    # trimming by Tukey
    x_trim = trim_tukey(x)
    variants.append(("trim15", x_trim, f"trim({selected_col})"))

    # Box-Cox + стандартизация
    lam = choose_boxcox_lambda(x)
    y_bc = boxcox_transform(x, lam)
    y_bc_z = zscore_transform(y_bc)
    variants.append(("boxcox_best", y_bc_z, f"boxcox(λ={lam:.2f})→z"))

    print(f"[INFO] Выбрана колонка: {selected_col}")
    print(f"[INFO] Box–Cox best λ ≈ {lam:.2f}")
    print("[INFO] Готовим отчёты для вариантов:", ", ".join(v[0] for v in variants))

    # 4) прогоняем Пункты I+II для всех вариантов
    summary = []
    for name, arr, label in variants:
        out_root = os.path.join(out_dir, name)
        summary.append(run_all_for_variant(arr, name, out_root, label))

    # 5) финальная сводка p-value
    print("\n=== Summary of normality p-values (higher is better) ===")
    print(f"{'variant':<12} {'p_chi2':>12} {'p_skew':>12} {'p_kurt':>12} {'skew':>10} {'excess':>10}")
    for r in summary:
        print(f"{r['variant']:<12} {r['p_chi2']:>12.6f} {r['p_skew']:>12.6f} {r['p_kurt']:>12.6f} {r['skew']:>10.4f} {r['excess']:>10.4f}")

    print("\n[OK] Пункт III выполнен. Смотри папку outputs/part3/* для всех графиков и CSV.")

if __name__ == "__main__":
    main()
