# Khai báo w, x và nhãn thực tế y
w = [-2, 1, 0]
x = [2, 3, 1]
y = 1

# Tính w^T x
wTx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

print("Giá trị w^T x ban đầu =", wTx)

# Xác định nhãn dự đoán
if wTx >= 0:
    y_du_doan = 1
else:
    y_du_doan = -1

print("Nhãn dự đoán =", y_du_doan)

# Kiểm tra phân lớp
if y_du_doan != y:
    print("Mẫu bị phân lớp sai.")

    # Cập nhật Perceptron
    w[0] = w[0] + y * x[0]
    w[1] = w[1] + y * x[1]
    w[2] = w[2] + y * x[2]

    print("Vector w sau cập nhật =", w)

    # Tính lại w^T x
    wTx_moi = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

    print("Giá trị w^T x sau cập nhật =", wTx_moi)

else:
    print("Mẫu được phân lớp đúng.")