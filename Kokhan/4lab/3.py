import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# === 1. Загружаем данные === #
df = pd.read_csv("famcs_students.csv")

binary_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
               'glasses', 'anime', 'study_form', 'literature']
y_col = binary_cols[5]  # anime
y = df[y_col]

# Признаки (минимум 5 — выполняем требование)
feature_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
                'glasses', 'study_form', 'literature']

X = pd.get_dummies(df[feature_cols], drop_first=True)
X = X.astype(int)

# === 2. Split (70 / 15 / 15) === #
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)


# === 3. Реализация метода K-Nearest Neighbors === #
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict_one(self, x):
        # расстояние до всех train объектов
        distances = np.sum((self.X_train - x) ** 2, axis=1)  # без sqrt — быстрее

        # сортировка
        k_idx = distances.argsort()[:self.k]

        # голоса классов
        nearest_labels = self.y_train[k_idx]
        # какой класс чаще — тот и предсказываем
        values, counts = np.unique(nearest_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return [self.predict_one(np.array(x)) for x in X.values]  # список предсказаний


# === 4. Подбор оптимального k === #
best_k = None
best_acc = 0

for k in range(1, 21):  # тестируем k от 1 до 20
    model = KNN(k=k)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)

    print(f"k = {k}, Validation Accuracy = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_k = k

print("\nЛучший k =", best_k)
print("Accuracy на валидации =", best_acc)

# === 5. Оценка на TEST выборке === #
final_model = KNN(k=best_k)
final_model.fit(X_train, y_train)

y_pred_test = final_model.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)

print("\n=== KNN ИТОГОВАЯ ОЦЕНКА ===")
print(f"Accuracy on test = {acc_test:.4f}")
print(f"(k = {best_k})")
