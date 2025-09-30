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

def var_sample_manual(a, mu):
    s2sum = 0.0
    for x in a:
        d = x - mu
        s2sum += d * d
    return s2sum / (len(a) - 1)

def skew_kurt_manual(a, mu):
    n = len(a)
    s2sum = s3sum = s4sum = 0.0
    for x in a:
        d = x - mu
        d2 = d * d
        s2sum += d2
        s3sum += d2 * d
        s4sum += d2 * d2
    s2 = s2sum / (n - 1)
    s = math.sqrt(s2)
    g1 = (n / ((n - 1) * (n - 2))) * (s3sum / (s ** 3))
    g2 = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (s4sum / (s ** 4)) \
         - 3.0 * (((n - 1) ** 2) / ((n - 2) * (n - 3)))
    return g1, g2

def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_ppf(p):
    if p <= 0.0:
        return -float('inf')
    if p >= 1.0:
        return float('inf')

    # коэффициенты рациональных аппроксимаций
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    # одно уточнение Ньютоном (стабильно, т.к. старт уже хороший)
    e = norm_cdf(x) - p
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
    x -= e / max(pdf, 1e-15)
    return x

def gammainc_regularized_P(a, x, eps=1e-12, itmax=1000):
    if x < 0 or a <= 0:
        return float('nan')
    if x == 0.0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # ряд
        ap = a
        sum_ = 1.0 / a
        delt = sum_
        for _ in range(itmax):
            ap += 1.0
            delt *= x / ap
            sum_ += delt
            if abs(delt) < abs(sum_) * eps:
                break
        return sum_ * math.exp(-x + a * math.log(x) - gln)
    else:
        # непрерывные дроби для Q, затем P = 1 - Q
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, itmax + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        Q = math.exp(-x + a * math.log(x) - gln) * h
        return 1.0 - Q

def chi2_sf(x, k):
    a = 0.5 * k
    return 1.0 - gammainc_regularized_P(a, 0.5 * x)

def two_sided_p_from_z(z):
    p_one = 1.0 - norm_cdf(abs(z))
    return 2.0 * p_one


def chi_square_normal_test(a):
    n = len(a)
    mu = mean_manual(a)
    s2 = var_sample_manual(a, mu)
    sigma = math.sqrt(s2)

    sturges = int(round(1 + math.log2(n)))
    k = max(6, sturges)
    if n / k < 5:
        k = max(3, int(n // 5))

    ps = [i / k for i in range(0, k + 1)]
    edges = [mu + sigma * norm_ppf(p) if 0 < p < 1 else
             (-float('inf') if p <= 0 else float('inf')) for p in ps]

    # наблюдённые частоты
    observed = [0] * k
    for x in a:
        # бинарный поиск бина
        lo, hi = 0, k
        while lo < hi:
            mid = (lo + hi) // 2
            if x >= edges[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        idx = lo if lo < k else k - 1
        observed[idx] += 1

    expected = [n / k] * k  # при равновероятностных бинах

    chi2 = 0.0
    for Oi, Ei in zip(observed, expected):
        chi2 += (Oi - Ei) * (Oi - Ei) / Ei
    df = k - 1 - 2
    if df < 1:
        df = 1
    p_value = chi2_sf(chi2, df)

    return {
        "n": n, "mu": mu, "sigma": sigma,
        "k": k, "df": df, "chi2": chi2, "p_value": p_value,
        "edges": edges, "observed": observed, "expected": expected
    }


def qqplot_normal(a, mu, sigma, out_path, title):
    n = len(a)
    a_sorted = sorted(a)
    probs = [(i + 0.5) / n for i in range(n)]
    theo = [mu + sigma * norm_ppf(p) for p in probs]

    plt.figure()
    plt.scatter(theo, a_sorted, s=8)
    q1_idx, q3_idx = int(0.25*n), int(0.75*n)-1
    xq = [theo[q1_idx], theo[q3_idx]]
    yq = [a_sorted[q1_idx], a_sorted[q3_idx]]
    b = (yq[1] - yq[0]) / (xq[1] - xq[0] + 1e-12)
    a0 = yq[0] - b * xq[0]
    x_line = [min(theo), max(theo)]
    y_fit = [a0 + b * x for x in x_line]
    plt.plot(x_line, y_fit, linewidth=2)

    plt.title(title)
    plt.xlabel("Theoretical quantiles (Normal)")
    plt.ylabel("Sample quantiles")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


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
    if n < 10:
        raise ValueError("Слишком мало данных для нормальных тестов")

    mu = mean_manual(a)
    s2 = var_sample_manual(a, mu)
    sigma = math.sqrt(s2)

    # 1) χ²-тест
    res = chi_square_normal_test(a)
    chi2_txt = os.path.join(out_dir, f"chi2_bins_{selected_col}.csv")
    with open(chi2_txt, "w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,observed,expected\n")
        for i in range(res["k"]):
            left = res["edges"][i]
            right = res["edges"][i+1]
            f.write(f"{left},{right},{res['observed'][i]},{res['expected'][i]}\n")

    # график бинов (Observed vs Expected)
    plt.figure()
    xs = list(range(1, res["k"] + 1))
    plt.bar(xs, res["observed"], label="Observed")
    plt.plot(xs, res["expected"], linewidth=2, label="Expected (n/k)")
    plt.title(f"Chi-square bins for {selected_col} (k={res['k']}, df={res['df']})")
    plt.xlabel("Bin")
    plt.ylabel("Count")
    plt.legend()
    chi2_plot = os.path.join(out_dir, f"chi2_bins_plot_{selected_col}.png")
    plt.savefig(chi2_plot, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Тесты асимметрии/эксцесса (асимптотика)
    skew, excess = skew_kurt_manual(a, mu)
    z_skew = skew / math.sqrt(6.0 / n)
    z_kurt = excess / math.sqrt(24.0 / n)
    p_skew = two_sided_p_from_z(z_skew)
    p_kurt = two_sided_p_from_z(z_kurt)

    # 3) Q–Q plot
    qq_path = os.path.join(out_dir, f"qqplot_{selected_col}.png")
    qqplot_normal(a, mu, sigma, qq_path, f"Q–Q plot: {selected_col} vs Normal(μ,σ²)")

    def fmt(x): return f"{x:.6f}" if isinstance(x, float) else str(x)
    print(f"[OK] Пункт II для столбца: {selected_col} (N={N})")
    print("\n-- Chi-square test (manual) --")
    print(f"n={res['n']}, k={res['k']}, df={res['df']}")
    print(f"chi2 = {fmt(res['chi2'])},  p-value = {fmt(res['p_value'])}")
    print(f"Бины и частоты: {chi2_txt}")
    print(f"График бинов:   {chi2_plot}")

    print("\n-- Skewness & Kurtosis tests (asymptotic) --")
    print(f"skew = {fmt(skew)},  Z_skew = {fmt(z_skew)},  p = {fmt(p_skew)}")
    print(f"excess kurtosis = {fmt(excess)},  Z_kurt = {fmt(z_kurt)},  p = {fmt(p_kurt)}")
    alpha = 0.05
    for name, p in [("χ²", res['p_value']), ("Skew", p_skew), ("Kurt", p_kurt)]:
        verdict = "reject normality" if p < alpha else "do NOT reject normality"
        print(f"  {name}: at α={alpha} -> {verdict}")

    print(f"\nQ–Q plot: {qq_path}")

if __name__ == "__main__":
    main()
