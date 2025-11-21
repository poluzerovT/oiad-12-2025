import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from typing import Dict
from part1 import StatisticsCalculator
from part2 import NormalityTester

class DataNormalizer:
    def __init__(self, data):
        self.original_data = np.array(data)
        self.data = np.array(data)
        self.transformations = {}
    
    def remove_outliers_iqr(self):
        """
        УДАЛЕНИЕ ВЫБРОСОВ МЕТОДОМ IQR
        Формулы:
        Q1 = 25-й перцентиль, Q3 = 75-й перцентиль
        IQR = Q3 - Q1
        Нижняя граница = Q1 - 1.5 * IQR
        Верхняя граница = Q3 + 1.5 * IQR
        """
        Q1 = np.percentile(self.original_data, 25)
        Q3 = np.percentile(self.original_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_data = self.original_data[
            (self.original_data >= lower_bound) & 
            (self.original_data <= upper_bound)
        ]
        
        self.transformations['iqr_filter'] = {
            'method': 'IQR',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'removed_count': len(self.original_data) - len(filtered_data)
        }
        
        self.data = filtered_data
        return self
    
    def truncate_outliers(self, method='std', n_std=3):
        """
        УСЕЧЕНИЕ ВЫБРОСОВ
        Методы:
        - 'std': границы = mean ± n_std * std
        - 'percentile': границы = 1-й и 99-й перцентили
        """
        if method == 'std':
            mean = np.mean(self.original_data)
            std = np.std(self.original_data)
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
        else: 
            lower_bound = np.percentile(self.original_data, 1)
            upper_bound = np.percentile(self.original_data, 99)
        
        truncated_data = np.clip(self.original_data, lower_bound, upper_bound)
        
        self.transformations['truncation'] = {
            'method': method,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        self.data = truncated_data
        return self
    
    def standardize(self):
        """
        СТАНДАРТИЗАЦИЯ ДАННЫХ
        Формула: z = (x - μ) / σ
        """
        mean = np.mean(self.data)
        std = np.std(self.data)
        standardized_data = (self.data - mean) / std
        
        self.transformations['standardization'] = {
            'mean': mean,
            'std': std
        }
        
        self.data = standardized_data
        return self
    
    def normalize_minmax(self):
        """
        НОРМИРОВКА ДАННЫХ
        Формула: x_scaled = (x - min) / (max - min)
        """
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        normalized_data = (self.data - min_val) / (max_val - min_val)
        
        self.transformations['normalization'] = {
            'min': min_val,
            'max': max_val
        }
        
        self.data = normalized_data
        return self
    
    def log_transform(self):
        """
        ЛОГАРИФМИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ
        Формула: y = ln(x + c)
        """
        if np.min(self.data) <= 0:
            shifted_data = self.data - np.min(self.data) + 1
        else:
            shifted_data = self.data
        
        log_data = np.log(shifted_data)
        
        self.transformations['log'] = {
            'shift': np.min(self.data) <= 0
        }
        
        self.data = log_data
        return self
    
    def get_transformed_data(self):
        return self.data

def analyze_normalization_methods():
    try:
        data = pd.read_csv('lab1/teen_phone_addiction_dataset.csv')
        sleep_hours = data['Sleep_Hours'].dropna()
        
        print("ИСХОДНЫЕ ДАННЫЕ")
        print(f"Объем выборки: {len(sleep_hours)}")
        print(f"Среднее: {sleep_hours.mean():.2f}")
        print(f"Стандартное отклонение: {sleep_hours.std():.2f}")
        print(f"Асимметрия: {stats.skew(sleep_hours):.2f}")
        print(f"Эксцесс: {stats.kurtosis(sleep_hours):.2f}")
        
        methods = [
            ("Исходные данные", lambda x: x),
            ("Удаление выбросов (IQR)", lambda x: DataNormalizer(x).remove_outliers_iqr().get_transformed_data()),
            ("Усечение выбросов", lambda x: DataNormalizer(x).truncate_outliers().get_transformed_data()),
            ("Логарифмирование", lambda x: DataNormalizer(x).log_transform().get_transformed_data()),
            ("Логарифмирование + стандартизация", lambda x: DataNormalizer(x).log_transform().standardize().get_transformed_data()),
        ]
        
        results = []
        
        for method_name, transform_func in methods:
            print(f"АНАЛИЗ МЕТОДА: {method_name}")
            
            transformed_data = transform_func(sleep_hours)
            
            # I часть: Статистические характеристики 
            print("\nI. СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
            stats_calc = StatisticsCalculator(transformed_data)
            stats_results = stats_calc.calculate_all_statistics()
            
            print("\nГРАФИКИ С УКАЗАНИЕМ МЕТОДА ПРЕОБРАЗОВАНИЯ:")
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(transformed_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Значения')
            plt.ylabel('Частота')
            plt.title(f'Гистограмма распределения\nМетод: {method_name}')
            plt.grid(True, alpha=0.3)
            
            # Эмпирическая функция распределения 
            plt.subplot(1, 2, 2)
            sorted_data = np.sort(transformed_data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, y, marker='.', linestyle='-', linewidth=1, markersize=2)
            plt.xlabel('Значения')
            plt.ylabel('Вероятность')
            plt.title(f'Эмпирическая функция распределения\nМетод: {method_name}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            stats_calc.print_statistics_report()
            
            # II часть: Проверка нормальности
            print("\nII. ПРОВЕРКА НОРМАЛЬНОСТИ:")
            normality_tester = NormalityTester(transformed_data)
            comprehensive_result = normality_tester.comprehensive_normality_test()
            
            # результаты проверки нормальности
            normal_tests_count = len(comprehensive_result['normal_methods'])
            is_normal = normal_tests_count >= 2
            
            print(f"Результаты проверки нормальности для {method_name}:")
            print(f"Хи-квадрат: p-value = {comprehensive_result['chi_square']['p_value']:.4f}")
            print(f"Асимметрия: {comprehensive_result['skewness_kurtosis']['skewness']:.4f}")
            print(f"Эксцесс: {comprehensive_result['skewness_kurtosis']['excess']:.4f}")
            print(f"Q-Q корреляция: {comprehensive_result['qq_plot']['correlation']:.4f}")
            print(f"Методов нормальности: {normal_tests_count}/3")
            print(f"Общее решение: {comprehensive_result['overall_decision']}")
            print(f"НОРМАЛЬНОСТЬ: {'ДА' if is_normal else 'НЕТ'}")
            
            # Q-Q plot с указанием метода нормализации
            plt.figure(figsize=(10, 6))
            stats.probplot(transformed_data, dist="norm", plot=plt)
            plt.title(f'Q-Q plot для проверки нормальности\nМетод преобразования: {method_name}\n'
                     f'Корреляция: {comprehensive_result["qq_plot"]["correlation"]:.4f}')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            results.append({
                'method': method_name,
                'sample_size': len(transformed_data),
                'is_normal': is_normal,
                'normal_tests_count': normal_tests_count,
                'skewness': comprehensive_result['skewness_kurtosis']['skewness'],
                'excess': comprehensive_result['skewness_kurtosis']['excess'],
                'chi_square_p': comprehensive_result['chi_square']['p_value'],
                'qq_correlation': comprehensive_result['qq_plot']['correlation']
            })
        
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПРИВЕДЕНИЯ К НОРМАЛЬНОМУ РАСПРЕДЕЛЕНИЮ")
        
        successful_methods = [r for r in results if r['is_normal']]
        
        if successful_methods:
            print("УДАЛОСЬ ПРИВЕСТИ К НОРМАЛЬНОМУ РАСПРЕДЕЛЕНИЮ:")
            for method in successful_methods:
                print(f"- {method['method']}:")
                print(f"  Тестов нормальности: {method['normal_tests_count']}/3")
                print(f"  Асимметрия: {method['skewness']:.3f}")
                print(f"  Эксцесс: {method['excess']:.3f}")
                print(f"  p-value (χ²): {method['chi_square_p']:.4f}")
                print(f"  Q-Q корреляция: {method['qq_correlation']:.4f}")
        else:
            print("НЕ УДАЛОСЬ ПОЛНОСТЬЮ ПРИВЕСТИ К НОРМАЛЬНОМУ РАСПРЕДЕЛЕНИЮ")
            best_method = max(results, key=lambda x: x['normal_tests_count'])
            print(f"Наилучший результат: {best_method['method']} ({best_method['normal_tests_count']}/3 тестов)")
        
        print(f"\nАНАЛИЗ ЭФФЕКТИВНОСТИ МЕТОДОВ:")
        for result in results:
            status = "УСПЕХ" if result['is_normal'] else "НЕУДАЧА"
            print(f"{result['method']}: {status} ({result['normal_tests_count']}/3 тестов)")
        
        plt.figure(figsize=(14, 8))
        
        # Сравнение асимметрии и эксцесса
        plt.subplot(2, 2, 1)
        methods_names = [r['method'] for r in results]
        skewness_vals = [r['skewness'] for r in results]
        excess_vals = [r['excess'] for r in results]
        
        x_pos = np.arange(len(methods_names))
        width = 0.35
        
        plt.bar(x_pos - width/2, skewness_vals, width, label='Асимметрия', alpha=0.7)
        plt.bar(x_pos + width/2, excess_vals, width, label='Эксцесс', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Нормальное распределение')
        plt.xlabel('Методы преобразования')
        plt.ylabel('Значение')
        plt.title('Сравнение асимметрии и эксцесса\nпо методам преобразования')
        plt.xticks(x_pos, methods_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # p-value критерия хи-квадрат
        plt.subplot(2, 2, 2)
        chi_square_p = [r['chi_square_p'] for r in results]
        colors = ['green' if p > 0.05 else 'red' for p in chi_square_p]
        
        bars = plt.bar(methods_names, chi_square_p, color=colors, alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle='--', label='Уровень значимости 0.05')
        plt.xlabel('Методы преобразования')
        plt.ylabel('p-value')
        plt.title('p-value критерия хи-квадрат\n(зеленый > 0.05 - нормальность)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Q-Q корреляция
        plt.subplot(2, 2, 3)
        qq_corr = [r['qq_correlation'] for r in results]
        colors_qq = ['green' if corr > 0.95 else 'orange' for corr in qq_corr]
        
        plt.bar(methods_names, qq_corr, color=colors_qq, alpha=0.7)
        plt.axhline(y=0.95, color='red', linestyle='--', label='Порог 0.95')
        plt.xlabel('Методы преобразования')
        plt.ylabel('Корреляция')
        plt.title('Q-Q plot корреляция\n(зеленый > 0.95 - нормальность)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        normal_tests = [r['normal_tests_count'] for r in results]
        colors_tests = ['green' if count >= 2 else 'red' for count in normal_tests]
        
        plt.bar(methods_names, normal_tests, color=colors_tests, alpha=0.7)
        plt.axhline(y=2, color='blue', linestyle='--', label='Порог нормальности (2/3)')
        plt.xlabel('Методы преобразования')
        plt.ylabel('Количество тестов')
        plt.title('Количество пройденных тестов нормальности\n(зеленый ≥ 2 - распределение нормально)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")
        return None

if __name__ == "__main__":
    results = analyze_normalization_methods()