import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# === 1. Загрузка данных === #
df = pd.read_csv("famcs_students.csv")

binary_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
               'glasses', 'anime', 'study_form', 'literature']
y_col = binary_cols[5]  # anime
y = df[y_col]

# --- Преобразуем y в числовой формат 0/1 ---
y = y.astype(str)
mapping = {
    '1': 1, '0': 0,
    'True': 1, 'False': 0,
    'yes': 1, 'no': 0,
    'Да': 1, 'Нет': 0,
    'да': 1, 'нет': 0,
}
y = y.map(mapping)
y = y.fillna(0).astype(int)  # все неопознанные значения → 0

# === 2. Выбор признаков === #
feature_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
                'glasses', 'study_form', 'literature']
X = pd.get_dummies(df[feature_cols], drop_first=True).astype(float)

# === 3. Разделение на train / val / test === #
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# === 4. Логистическая регрессия своими руками === #
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=3000):
        self.lr = lr
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iter):
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)

            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# === 5. Обучение модели === #
model = LogisticRegressionScratch(lr=0.01, n_iter=3000)
model.fit(X_train, y_train)

# === 6. Подбор порога на валидационной выборке === #
probs_val = model.predict_proba(X_val)
thresholds = np.linspace(0, 1, 100)
precisions, recalls, f1_scores = [], [], []

for t in thresholds:
    preds = (probs_val >= t).astype(int)
    precisions.append(precision_score(y_val, preds, zero_division=0))
    recalls.append(recall_score(y_val, preds, zero_division=0))
    f1_scores.append(f1_score(y_val, preds, zero_division=0))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("=== ЛУЧШИЙ ПОРОГ ПО F1 ===")
print("Порог:", best_threshold)
print("F1:", f1_scores[best_idx])

# === 7. График Precision / Recall / F1 vs Threshold === #
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1_scores, label="F1-score", linestyle="--")
plt.axvline(best_threshold, linestyle=':', color='red', label=f"Best threshold = {best_threshold:.2f}")
plt.xlabel("Порог классификации")
plt.ylabel("Значение метрики")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# === 8. Итог на TEST === #
y_pred_test = model.predict(X_test, threshold=best_threshold)
print("\n=== ОЦЕНКА НА TEST ВЫБОРКЕ ===")
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1:", f1_score(y_test, y_pred_test))
