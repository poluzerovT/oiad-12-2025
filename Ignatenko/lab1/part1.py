import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class StatisticsCalculator:
    
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
        self.results = {}
        self.formulas = {}
        
    def calculate_all_statistics(self):
        self._calculate_mean()
        self._calculate_variance()
        self._calculate_mode()
        self._calculate_median()
        self._calculate_quantiles()
        self._calculate_skewness()
        self._calculate_excess()
        self._calculate_iqr()
        self._calculate_additional_stats()

        return self.results, self.formulas
    
    def _calculate_mean(self):
        mean_value = np.sum(self.data) / self.n
        self.results['mean'] = mean_value
    
    def _calculate_variance(self):
        mean = self.results['mean']
        squared_deviations = np.sum((self.data - mean) ** 2)
        variance_value = squared_deviations / (self.n - 1)
        std_dev = math.sqrt(variance_value)
        
        self.results['variance'] = variance_value
        self.results['std_dev'] = std_dev
    
    def _calculate_mode(self):
        values, counts = np.unique(self.data, return_counts=True)
        max_count = np.max(counts)
        mode_values = values[counts == max_count]
        
        self.results['mode'] = mode_values
        self.results['mode_count'] = max_count
    
    def _calculate_median(self):
        sorted_data = np.sort(self.data)
        
        if self.n % 2 == 1:
            median_index = self.n // 2
            median_value = sorted_data[median_index]
        else:
            median_index1 = self.n // 2 - 1
            median_index2 = self.n // 2
            median_value = (sorted_data[median_index1] + sorted_data[median_index2]) / 2
        
        self.results['median'] = median_value
    
    def _calculate_quantiles(self):
        sorted_data = np.sort(self.data)
        quantiles = {}
        
        for q in [0.25, 0.5, 0.75]:
            pos = q * (self.n - 1)
            lower_idx = int(np.floor(pos))
            upper_idx = int(np.ceil(pos))
            
            if lower_idx == upper_idx:
                quantile_val = sorted_data[lower_idx]
            else:
                weight = pos - lower_idx
                quantile_val = (1 - weight) * sorted_data[lower_idx] + weight * sorted_data[upper_idx]
            
            quantiles[f'q_{int(q*100)}'] = quantile_val
        
        self.results.update(quantiles)
    
    def _calculate_excess(self):
        mean = self.results['mean']
        std_dev = self.results['std_dev']
        fourth_moment = np.mean((self.data - mean) ** 4)
        sigma_4 = std_dev ** 4
        kurtosis_pearson = fourth_moment / sigma_4
        excess_value = kurtosis_pearson - 3
        
        self.results['excess'] = excess_value

    def _calculate_skewness(self):
        mean = self.results['mean']
        std_dev = self.results['std_dev']
        third_moment = np.mean((self.data - mean) ** 3)
        sigma_3 = std_dev ** 3
        skewness_value = third_moment / sigma_3
        
        self.results['skewness'] = skewness_value
    
    def _calculate_iqr(self):
        q25 = self.results['q_25']
        q75 = self.results['q_75']
        iqr_value = q75 - q25
        
        self.results['iqr'] = iqr_value
    
    def _calculate_additional_stats(self):
        self.results['min'] = np.min(self.data)
        self.results['max'] = np.max(self.data)
        self.results['range'] = self.results['max'] - self.results['min']
        self.results['sum'] = np.sum(self.data)
        
        if self.results['mean'] != 0:
            self.results['cv'] = (self.results['std_dev'] / self.results['mean']) * 100
        else:
            self.results['cv'] = 0

    def print_statistics_report(self):
        print("статистические")
        print(f"Объем выборки n {self.n}")
        print(f"Сумма всех значений {self.results['sum']:.4f}")
        
        print("\nМеры центральной тенденции")
        print(f"Среднее {self.results['mean']:.4f}")
        print(f"Медиана {self.results['median']:.4f}")
        print(f"Мода {list(self.results['mode'])}")
        
        print("\nМеры разброса")
        print(f"Дисперсия {self.results['variance']:.4f}")
        print(f"Стандартное отклонение {self.results['std_dev']:.4f}")
        print(f"Размах {self.results['range']:.4f}")
        print(f"Интерквартильный размах {self.results['iqr']:.4f}")
        print(f"Коэффициент вариации {self.results['cv']:.2f}%")
        
        print("\nКвантили")
        print(f"Q25 {self.results['q_25']:.4f}")
        print(f"Q50 {self.results['q_50']:.4f}")
        print(f"Q75 {self.results['q_75']:.4f}")
        
        print("\nФорма распределения")
        print(f"Асимметрия {self.results['skewness']:.4f}")
        print(f"Эксцесс {self.results['excess']:.4f}")
        
        print("\nЭкстремальные значения")
        print(f"Минимум {self.results['min']:.4f}")
        print(f"Максимум {self.results['max']:.4f}")

    def plot_histogram(self):
        plt.figure(figsize=(12, 6))
        
        plt.hist(self.data, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        plt.axvline(self.results['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее {self.results["mean"]:.2f}')
        plt.axvline(self.results['median'], color='green', linestyle='--', linewidth=2,
                   label=f'Медиана {self.results["median"]:.2f}')
        
        plt.xlabel('Часы сна')
        plt.ylabel('Плотность вероятности')
        plt.title('Гистограмма распределения часов сна')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_empirical_cdf(self):
        sorted_data = np.sort(self.data)
        n = len(sorted_data)
        
        y_values = np.arange(1, n + 1) / n
        
        plt.figure(figsize=(12, 6))
        plt.step(sorted_data, y_values, where='post', linewidth=2, color='blue')
        
        # линии квантилей
        quantiles = {
            'Q1': self.results['q_25'],
            'Медиана': self.results['q_50'], 
            'Q3': self.results['q_75']
        }
        
        colors = ['red', 'green', 'orange']
        
        for (label, quantile_value), color in zip(quantiles.items(), colors):
            plt.axvline(quantile_value, color=color, linestyle='--', alpha=0.7, linewidth=2, 
                       label=f'{label} {quantile_value:.2f} ч')
        
        plt.xlabel('Часы сна')
        plt.ylabel('Вероятность')
        plt.title('Эмпирическая функция распределения часов сна')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

def calculate_complete_statistics(data):
    calculator = StatisticsCalculator(data)
    return calculator.calculate_all_statistics()

if __name__ == "__main__":
    print("Загрузка")
    try:
        data = pd.read_csv('lab1/teen_phone_addiction_dataset.csv')
        sleep_hours = data['Sleep_Hours'].dropna()
        print(f"Количество записей о часах сна {len(sleep_hours)}")
        
        print("\nРасчет статистических характеристик")
        
        calculator = StatisticsCalculator(sleep_hours)
        stats, formulas = calculator.calculate_all_statistics()
        
        print("\nдополнительная информация")
        
        print(f"\nОписательная статистика pandas")
        print(sleep_hours.describe())
        
        unique_values = sleep_hours.unique()
        print(f"\nКоличество уникальных значений {len(unique_values)}")
        print(f"Уникальные значения {np.sort(unique_values)}")
        
        Q1 = stats['q_25']
        Q3 = stats['q_75']
        IQR = stats['iqr']
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = sleep_hours[(sleep_hours < lower_bound) | (sleep_hours > upper_bound)]
        print(f"\n Выбросы")
        print(f"Q1 25% {Q1:.2f}")
        print(f"Q3 75% {Q3:.2f}")
        print(f"IQR {IQR:.2f}")
        print(f"Нижняя граница {lower_bound:.2f}")
        print(f"Верхняя граница {upper_bound:.2f}")
        print(f"Количество выбросов {len(outliers)}")
        if len(outliers) > 0:
            print(f"Выбросы {outliers.values}")

        calculator.print_statistics_report()
        calculator.plot_histogram()
        calculator.plot_empirical_cdf()
        
    except Exception as e:
        print(f"ОШИБКА {e}")