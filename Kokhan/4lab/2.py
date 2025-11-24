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
feature_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
                'glasses', 'study_form', 'literature']

X = pd.get_dummies(df[feature_cols], drop_first=True)

# === 2. Разделяем выборки === #
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15,
                                                  random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765,
                                                  random_state=42, stratify=y_temp)


# === 3. Реализация НАИВНОГО БАЙЕСОВСКОГО КЛАССИФИКАТОРА === #
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}  # P(y)
        self.cond_probs = {}  # P(x|y)

        for cls in self.classes:
            X_c = X[y == cls]  # объекты конкретного класса
            self.priors[cls] = len(X_c) / len(X)  # P(y)

            probs = {}
            for col in X.columns:
                # Laplace smoothing
                p = (X_c[col].sum() + 1) / (len(X_c) + 2)
                probs[col] = p
            self.cond_probs[cls] = probs

    def predict(self, X):
        predictions = []
        for idx in range(len(X)):
            sample = X.iloc[idx]
            class_probs = {}

            for cls in self.classes:
                prob = np.log(self.priors[cls])  # log(P(y))

                for col in X.columns:
                    p = self.cond_probs[cls][col]  # P(x|y)
                    if sample[col] == 1:
                        prob += np.log(p)
                    else:
                        prob += np.log(1 - p)

                class_probs[cls] = prob

            predictions.append(max(class_probs, key=class_probs.get))
        return predictions


# === 4. Обучение модели === #
model = NaiveBayes()
model.fit(X_train, y_train)

# === 5. Предсказания === #
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# === 6. Оценка качества === #
print("\n=== НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР ===")
print("Train accuracy:", accuracy_score(y_train, y_pred_train))
print("Val accuracy:", accuracy_score(y_val, y_pred_val))
print("Test accuracy:", accuracy_score(y_test, y_pred_test))
