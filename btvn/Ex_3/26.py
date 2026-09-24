# Hàm số f(x) = x^2 - 4x + 5
def f(x):
    return x**2 - 4*x + 5

# Đạo hàm f'(x) = 2x - 4
def df(x):
    return 2*x - 4

# Giá trị ban đầu
x = 5
eta = 0.2

# In giá trị ban đầu
print("Bước 0:")
print("x =", x)
print("f(x) =", f(x))

# Gradient Descent 4 bước
for i in range(1, 5):
    # Công thức cập nhật
    x = x - eta * df(x)

    print(f"\nBước {i}:")
    print("x =", x)
    print("f'(x) =", df(x))
    print("f(x) =", f(x))