# Khai báo vector w và x
w = [1, 2, -10]
x = [3, 4, 1]

# Tính w^T x
wTx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

print("w^T x =", wTx)

# Xác định nhãn dự đoán
if wTx >= 0:
    y_du_doan = 1
else:
    y_du_doan = -1

print("Nhãn dự đoán =", y_du_doan)

# Nhãn thực tế
y_thuc_te = -1

# Kiểm tra phân lớp đúng hay sai
if y_du_doan == y_thuc_te:
    print("Điểm dữ liệu được phân lớp đúng.")
else:
    print("Điểm dữ liệu bị phân lớp sai.")