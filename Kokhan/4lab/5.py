import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from collections import OrderedDict

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("famcs_students.csv")

binary_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
               'glasses', 'anime', 'study_form', 'literature']
y_col = binary_cols[5]  # anime
y = df[y_col].astype(str)

# --- Приводим y к числовому формату 0/1 ---
mapping = {'1':1, '0':0, 'True':1, 'False':0, 'yes':1, 'no':0, 'Да':1, 'Нет':0, 'да':1, 'нет':0}
y = y.map(mapping).fillna(0).astype(int)

# Признаки
feature_cols = ['ss', 'interest', 'weekend_study', 'bad_sleep',
                'glasses', 'study_form', 'literature']
X = pd.get_dummies(df[feature_cols], drop_first=True).astype(float)

# =========================
# 2. Разделение данных
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# =========================
# 3. Тривиальный классификатор
# =========================
most_common_class = y_train.value_counts().idxmax()
def trivial_predict(X):
    return [most_common_class]*len(X)
y_pred_trivial = trivial_predict(X_test)

# =========================
# 4. Наивный байес
# =========================
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.cond_probs = {}
        for cls in self.classes:
            X_c = X[y==cls]
            self.priors[cls] = len(X_c)/len(X)
            probs = {}
            for col in X.columns:
                probs[col] = (X_c[col].sum() + 1)/(len(X_c) + 2)
            self.cond_probs[cls] = probs
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            sample = X.iloc[i]
            class_probs = {}
            for cls in self.classes:
                prob = np.log(self.priors[cls])
                for col in X.columns:
                    p = self.cond_probs[cls][col]
                    prob += np.log(p) if sample[col]==1 else np.log(1-p)
                class_probs[cls] = prob
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

nb_model = NaiveBayes()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# =========================
# 5. KNN
# =========================
class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    def predict_one(self, x):
        distances = np.sum((self.X_train - x)**2, axis=1)
        k_idx = distances.argsort()[:self.k]
        nearest_labels = self.y_train[k_idx]
        vals, counts = np.unique(nearest_labels, return_counts=True)
        return vals[np.argmax(counts)]
    def predict(self, X):
        return [self.predict_one(np.array(x)) for x in X.values]

# Подбор оптимального k
best_k = 1
best_acc = 0
for k in range(1, 21):
    model_knn = KNN(k=k)
    model_knn.fit(X_train, y_train)
    y_pred_val = model_knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    if acc>best_acc:
        best_acc = acc
        best_k = k

knn_model = KNN(k=best_k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# =========================
# 6. Логистическая регрессия
# =========================
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=3000):
        self.lr = lr
        self.n_iter = n_iter
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iter):
            linear = np.dot(X,self.w)+self.b
            y_pred = self.sigmoid(linear)
            dw = np.dot(X.T,(y_pred-y))/len(y)
            db = np.sum(y_pred-y)/len(y)
            self.w -= self.lr*dw
            self.b -= self.lr*db
    def predict_proba(self,X):
        return self.sigmoid(np.dot(X,self.w)+self.b)
    def predict(self,X,threshold=0.5):
        return (self.predict_proba(X)>=threshold).astype(int)

log_model = LogisticRegressionScratch(lr=0.01,n_iter=3000)
log_model.fit(X_train,y_train)

# Подбор порога по F1 на валидации
probs_val = log_model.predict_proba(X_val)
thresholds = np.linspace(0,1,100)
f1_scores = []
for t in thresholds:
    preds = (probs_val>=t).astype(int)
    f1_scores.append(f1_score(y_val,preds))
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_log = log_model.predict(X_test,threshold=best_threshold)

# =========================
# 7. Оценка качества и матрицы ошибок
# =========================
models = OrderedDict([
    ("Trivial", y_pred_trivial),
    ("NaiveBayes", y_pred_nb),
    ("KNN", y_pred_knn),
    ("LogisticRegression", y_pred_log)
])

for name, y_pred in models.items():
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,zero_division=0)
    rec = recall_score(y_test,y_pred,zero_division=0)
    try:
        roc = roc_auc_score(y_test,y_pred)
    except:
        roc = float('nan')
    cm = confusion_matrix(y_test,y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
