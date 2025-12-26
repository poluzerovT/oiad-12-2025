# 1. Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Загрузка данных
df = pd.read_csv("famcs_students.csv")  # ⚠ Файл должен быть в той же папке!

# 3. Выбор бинарных колонок
binary_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
               'glasses', 'anime', 'study_form', 'literature']

# по условию: N = 5 → y = binary_cols[5]
y_col = binary_cols[5]
print(f"Целевая переменная y = {y_col}")

# 4. Целевая переменная
y = df[y_col]

# 5. Выбор признаков X — минимум 5, здесь выбрал 7
feature_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
                'glasses', 'study_form', 'literature']
X = df[feature_cols]

# 6. Кодирование категориальных признаков
X = pd.get_dummies(X, drop_first=True)
print("Форма X после кодирования:", X.shape)

# 7. Разделение выборки: train (70%), val (15%), test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
# 0.1765 * 0.85 ≈ 0.15

print("Размеры выборок:")
print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# 8. ***ТРВИАЛЬНЫЙ КЛАССИФИКАТОР***
# всегда выдаёт наиболее частый класс

most_common_class = y_train.value_counts().idxmax()

def trivial_predict(X):
    return [most_common_class] * len(X)

# Предсказания
y_pred_train = trivial_predict(X_train)
y_pred_val = trivial_predict(X_val)
y_pred_test = trivial_predict(X_test)

# 9. Оценка качества
acc_train = accuracy_score(y_train, y_pred_train)
acc_val = accuracy_score(y_val, y_pred_val)
acc_test = accuracy_score(y_test, y_pred_test)

print("\n=== ТРИВИАЛЬНЫЙ КЛАССИФИКАТОР ===")
print("Most frequent class:", most_common_class)
print(f"Train accuracy: {acc_train:.4f}")
print(f"Validation accuracy: {acc_val:.4f}")
print(f"Test accuracy: {acc_test:.4f}")
