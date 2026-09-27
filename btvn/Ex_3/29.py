import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None

    # Hàm huấn luyện
    def fit(self, X, y):
        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(X.shape[1])

        # Lặp qua số epoch
        for epoch in range(self.epochs):
            for i in range(len(X)):

                # Tính w^T*x
                wTx = np.dot(self.w, X[i])

                # Dự đoán nhãn
                if wTx >= 0:
                    y_du_doan = 1
                else:
                    y_du_doan = -1

                # Nếu dự đoán sai thì cập nhật trọng số
                if y_du_doan != y[i]:
                    self.w = self.w + self.learning_rate * y[i] * X[i]

        return self

    # Hàm dự đoán
    def predict(self, X):
        wTx = np.dot(X, self.w)

        # Chuyển kết quả thành -1 hoặc 1
        return np.where(wTx >= 0, 1, -1)


# =========================
# DỮ LIỆU HUẤN LUYỆN
# =========================

# Dữ liệu đã thêm bias
X = np.array([
    [1, 2, 3],
    [1, 3, 4],
    [1, -2, -1],
    [1, -3, -2]
])

# Nhãn
y = np.array([1, 1, -1, -1])


# =========================
# HUẤN LUYỆN MÔ HÌNH
# =========================

model = Perceptron(learning_rate=0.1, epochs=10)

model.fit(X, y)

print("Trọng số sau khi huấn luyện:")
print(model.w)


# =========================
# DỰ ĐOÁN DỮ LIỆU MỚI
# =========================

X_moi = np.array([
    [1, 4, 5],
    [1, -4, -3]
])

y_du_doan = model.predict(X_moi)

print("Nhãn dự đoán:")
print(y_du_doan)