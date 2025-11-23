import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from typing import Dict

class NormalityTester:
    
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data, ddof=1)
        self.results = {}
    
    def chi_square_test(self, alpha: float = 0.05) -> Dict:
        k = int(1 + 3.322 * math.log10(self.n))
        
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        interval_width = (max_val - min_val) / k
        
        intervals = []
        for i in range(k):
            start = min_val + i * interval_width
            end = min_val + (i + 1) * interval_width
            intervals.append((start, end))
        
        observed_freq = np.zeros(k)
        for value in self.data:
            for i, (start, end) in enumerate(intervals):
                if start <= value < end or (i == k-1 and value == end):
                    observed_freq[i] += 1
                    break
        
        expected_freq = np.zeros(k)
        cumulative_prob = 0
        
        for i, (start, end) in enumerate(intervals):
            z_start = (start - self.mean) / self.std
            z_end = (end - self.mean) / self.std
            
            if i == 0:
                prob = self._normal_cdf(z_end) - self._normal_cdf(-10)
            elif i == k-1:
                prob = self._normal_cdf(10) - self._normal_cdf(z_start)
            else:
                prob = self._normal_cdf(z_end) - self._normal_cdf(z_start)
            
            expected_freq[i] = prob * self.n
            cumulative_prob += prob
        
        valid_intervals = expected_freq >= 5
        
        chi_square_stat = 0
        for i in range(k):
            if expected_freq[i] > 0:
                component = (observed_freq[i] - expected_freq[i])**2 / expected_freq[i]
                chi_square_stat += component
        
        df = k - 3
        critical_value = self._chi_square_quantile(1 - alpha, df)
        p_value = 1 - self._chi_square_cdf(chi_square_stat, df)
        reject_h0 = chi_square_stat > critical_value
        
        result = {
            'chi_square': chi_square_stat,
            'df': df,
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'intervals': intervals,
            'observed_freq': observed_freq,
            'expected_freq': expected_freq
        }
        
        self.results['chi_square'] = result
        return result
    
    def skewness_kurtosis_test(self) -> Dict:
        third_moment = np.mean((self.data - self.mean) ** 3)
        sigma_3 = self.std ** 3
        skewness = third_moment / sigma_3
        
        fourth_moment = np.mean((self.data - self.mean) ** 4)
        sigma_4 = self.std ** 4
        kurtosis_pearson = fourth_moment / sigma_4
        excess = kurtosis_pearson - 3
        
        se_skewness = math.sqrt(6 * self.n * (self.n - 1) / ((self.n - 2) * (self.n + 1) * (self.n + 3)))
        se_kurtosis = math.sqrt(24 * self.n * (self.n - 1)**2 / ((self.n - 3) * (self.n - 2) * (self.n + 3) * (self.n + 5)))
        
        z_skewness = skewness / se_skewness
        z_kurtosis = excess / se_kurtosis
        
        skewness_normal = abs(skewness) < 0.5
        excess_normal = abs(excess) < 1
        z_skewness_normal = abs(z_skewness) < 1.96
        z_kurtosis_normal = abs(z_kurtosis) < 1.96
        
        overall_normal = skewness_normal and excess_normal and z_skewness_normal and z_kurtosis_normal
        
        result = {
            'skewness': skewness,
            'excess': excess,
            'se_skewness': se_skewness,
            'se_kurtosis': se_kurtosis,
            'z_skewness': z_skewness,
            'z_kurtosis': z_kurtosis,
            'skewness_normal': skewness_normal,
            'excess_normal': excess_normal,
            'z_skewness_normal': z_skewness_normal,
            'z_kurtosis_normal': z_kurtosis_normal,
            'overall_normal': overall_normal
        }
        
        self.results['skewness_kurtosis'] = result
        return result
    
    def plot_qq(self) -> plt.Figure:
        sorted_data = np.sort(self.data)
        
        theoretical_quantiles = []
        
        for i in range(self.n):
            p = (i + 1 - 0.5) / self.n
            z = self._normal_quantile(p)
            theoretical_quantiles.append(z)
        
        theoretical_quantiles = np.array(theoretical_quantiles)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.scatter(theoretical_quantiles, sorted_data, alpha=0.7, color='blue')
        
        min_theoretical = np.min(theoretical_quantiles)
        max_theoretical = np.max(theoretical_quantiles)
        line_x = np.array([min_theoretical, max_theoretical])
        line_y = self.mean + self.std * line_x
        ax1.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2, label='Идеальная нормальность')
        
        ax1.set_xlabel('Теоретические квантили N(0,1)')
        ax1.set_ylabel('Эмпирические квантили')
        ax1.set_title('Q-Q Plot для проверки нормальности')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(self.data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        x_range = np.linspace(np.min(self.data), np.max(self.data), 100)
        normal_pdf = stats.norm.pdf(x_range, self.mean, self.std)
        ax2.plot(x_range, normal_pdf, 'r-', linewidth=2, label='Нормальное распределение')
        
        ax2.set_xlabel('Значения')
        ax2.set_ylabel('Плотность вероятности')
        ax2.set_title('Гистограмма с нормальной кривой')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        correlation = np.corrcoef(theoretical_quantiles, sorted_data)[0, 1]
        
        if correlation > 0.975:
            assessment = "отличное соответствие нормальному распределению"
        elif correlation > 0.95:
            assessment = "хорошее соответствие нормальному распределению" 
        elif correlation > 0.90:
            assessment = "удовлетворительное соответствие нормальному распределению"
        else:
            assessment = "плохое соответствие нормальному распределению"
        
        self.results['qq_plot'] = {
            'correlation': correlation,
            'assessment': assessment
        }
        
        return fig
    
    def comprehensive_normality_test(self, alpha: float = 0.05) -> Dict:
        chi2_result = self.chi_square_test(alpha)
        skew_kurt_result = self.skewness_kurtosis_test()
        qq_fig = self.plot_qq()
        
        chi2_decision = "Нормальность ОТВЕРГАЕТСЯ" if chi2_result['reject_h0'] else "Нормальность НЕ ОТВЕРГАЕТСЯ"
        skew_kurt_decision = "НОРМАЛЬНО" if skew_kurt_result['overall_normal'] else "НЕ НОРМАЛЬНО"
        qq_decision = self.results['qq_plot']['assessment']
        
        normal_methods = []
        if not chi2_result['reject_h0']:
            normal_methods.append("Хи-квадрат")
        if skew_kurt_result['overall_normal']:
            normal_methods.append("асимметрия и эксцесс")
        if self.results['qq_plot']['correlation'] > 0.95:
            normal_methods.append("Q-Q plot")
        
        if len(normal_methods) >= 2:
            overall_decision = "Распределение можно считать НОРМАЛЬНЫМ"
        else:
            overall_decision = "Распределение СИЛЬНО ОТЛИЧАЕТСЯ от нормального"
        
        comprehensive_result = {
            'chi_square': chi2_result,
            'skewness_kurtosis': skew_kurt_result,
            'qq_plot': self.results['qq_plot'],
            'overall_decision': overall_decision,
            'normal_methods': normal_methods
        }
        
        self.results['comprehensive'] = comprehensive_result
        return comprehensive_result
    
    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_quantile(self, p: float) -> float:
        if p == 0:
            return -10
        if p == 1:
            return 10
        
        sign = -1 if p < 0.5 else 1
        p = p if p < 0.5 else 1 - p
        
        t = math.sqrt(-2 * math.log(p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3)
        
        return sign * z
    
    def _chi_square_quantile(self, p: float, df: int) -> float:
        if df <= 0:
            return 0
        
        mu = df
        sigma = math.sqrt(2 * df)
        
        z = self._normal_quantile(p)
        x = mu * (1 - 2/(9*mu) + z * math.sqrt(2/(9*mu)))**3
        
        return max(0, x)
    
    def _chi_square_cdf(self, x: float, df: int) -> float:
        if x <= 0:
            return 0
        
        z = ( (x/df)**(1/3) - 1 + 2/(9*df) ) / math.sqrt(2/(9*df))
        
        return self._normal_cdf(z)

def test_real_data():
    try:
        data = pd.read_csv('lab1/teen_phone_addiction_dataset.csv')
        sleep_hours = data['Sleep_Hours'].dropna()
        
        tester = NormalityTester(sleep_hours)
        results = tester.comprehensive_normality_test()
        
        plt.show()
        
        return tester, results
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None

if __name__ == "__main__":
    tester, results = test_real_data()