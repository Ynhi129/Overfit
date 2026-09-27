# ==========================================
# 1. IMPORT THƯ VIỆN
# ==========================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 2. ĐỌC DỮ LIỆU
# ==========================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Kích thước dữ liệu:", X.shape)
print("Số lượng mẫu:", len(X))
print("Số lượng đặc trưng:", X.shape[1])


# ==========================================
# 3. CHIA DỮ LIỆU TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nSố mẫu train:", len(X_train))
print("Số mẫu test:", len(X_test))


# ==========================================
# 4. CHUẨN HÓA DỮ LIỆU
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. XÂY DỰNG MÔ HÌNH PERCEPTRON
# ==========================================

model = Perceptron(
    max_iter=1000,
    eta0=0.01,
    random_state=42,
    tol=1e-3
)


# ==========================================
# 6. HUẤN LUYỆN
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. DỰ ĐOÁN
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. TÍNH CÁC ĐỘ ĐO
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


# ==========================================
# 9. IN KẾT QUẢ
# ==========================================

print("\n========== KẾT QUẢ ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))