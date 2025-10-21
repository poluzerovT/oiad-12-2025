import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from part1 import StatisticsCalculator

data = pd.read_csv('lab1/teen_phone_addiction_dataset.csv')
sleep_hours = data['Sleep_Hours'].dropna()

class GroupAnalyzer:
    def __init__(self, data):
        self.data = data
        self.groups = data['School_Grade'].unique()
    
    def calculate_group_statistics(self):
        group_stats = {}
        
        for grade in self.groups:
            # это массив с часами сна только для текущего класса
            grade_data = self.data[self.data['School_Grade'] == grade]['Sleep_Hours'].dropna()
            
            if len(grade_data) > 0:
                calculator = StatisticsCalculator(grade_data)
                stats, formulas = calculator.calculate_all_statistics()
                group_stats[grade] = {
                    'stats': stats,           # все рассчитанные статистики
                    'data': grade_data,       # исходные данные группы
                    'n': len(grade_data)      # размер группы
                }
                
                print(f"\n{grade}:")
                print(f"  n = {len(grade_data)}")  
                print(f"  Среднее = {stats['mean']:.3f}") 
                print(f"  Дисперсия = {stats['variance']:.3f}")
                print(f"  Ст. отклонение = {stats['std_dev']:.3f}")
                print(f"  Медиана = {stats['median']:.3f}")
                print(f"  Асимметрия = {stats['skewness']:.3f}")
        
        return group_stats
    
    def plot_group_histograms(self, group_stats):
        n_groups = len(group_stats)
        
        cols = 2
        rows = (n_groups + 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, (grade, group_info) in enumerate(group_stats.items(), 1):
            plt.subplot(rows, cols, i) # Создает i-й подграфик в сетке rows×cols
            stats = group_info['stats']
            grade_data = group_info['data'] # часы сна
            
            plt.hist(grade_data, bins=10, alpha=0.7, color=f'C{i-1}', edgecolor='black', density=True)
            plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                       label=f'Среднее: {stats["mean"]:.2f}')
            plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2,
                       label=f'Медиана: {stats["median"]:.2f}')
            
            plt.title(f'{grade} класс\n(n={len(grade_data)}, μ={stats["mean"]:.2f}, σ={stats["std_dev"]:.2f})')
            plt.xlabel('Часы сна')
            plt.ylabel('Плотность') # # Для каждого бина (столбца):
                            # высота_бина = (количество_наблюдений_в_бине) / (общее_количество × ширина_бина)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison_chart(self, group_stats):
        grades = list(group_stats.keys())
        means = [info['stats']['mean'] for info in group_stats.values()]
        stds = [info['stats']['std_dev'] for info in group_stats.values()]
        counts = [len(info['data']) for info in group_stats.values()]
        
        plt.figure(figsize=(12, 6))
        
        y_pos = np.arange(len(grades))
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'wheat', 'plum', 'lightcyan']
        bar_colors = colors[:len(grades)]
        
        bars = plt.bar(y_pos, means, yerr=stds, capsize=5, alpha=0.7, color=bar_colors)
        
        plt.xlabel('Школьный класс')
        plt.ylabel('Среднее время сна (часы)')
        plt.title('Сравнение среднего времени сна по школьным классам')
        plt.xticks(y_pos, [f'{grade}\n(n={count})' for grade, count in zip(grades, counts)])
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.05, f'{mean:.2f}±{std:.2f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_overlaid_histograms(self, group_stats):
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (grade, group_info) in enumerate(group_stats.items()):
            grade_data = group_info['data']
            stats = group_info['stats']
            
            plt.hist(grade_data, bins=10, alpha=0.5, color=colors[i % len(colors)], 
                    edgecolor='black', density=True, 
                    label=f'{grade} (n={len(grade_data)}, μ={stats["mean"]:.2f})')
        
        plt.xlabel('Часы сна')
        plt.ylabel('Плотность')
        plt.title('Наложенные гистограммы распределения часов сна по классам')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_variability(self, group_stats):
        
        for grade, group_info in group_stats.items():
            stats_results = group_info['stats']
            
            # коэффициент вариации (σ - стандартное отклонение / μ - среднее значение) 100%
            if stats_results['mean'] != 0:
                cv = (stats_results['std_dev'] / stats_results['mean']) * 100
            else:
                cv = 0
                
            variability_type = "высокая" if cv > 30 else "умеренная" if cv > 15 else "низкая"
            
            print(f"{grade}:")
            print(f"  Коэффициент вариации: {cv:.1f}% ({variability_type} изменчивость)")
    
    # Преобразует строки типа "9th" в числа 9 (удаляет суффиксы)
    def _grade_to_number(self, grade_str):
        return int(grade_str.replace('th', '').replace('rd', '').replace('nd', '').replace('st', ''))


analyzer = GroupAnalyzer(data)


group_stats = analyzer.calculate_group_statistics()
analyzer.plot_group_histograms(group_stats)
analyzer.plot_comparison_chart(group_stats)
analyzer.plot_overlaid_histograms(group_stats)


analyzer.analyze_variability(group_stats)
