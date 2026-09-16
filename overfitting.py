# ============================================================
# DỰ BÁO GIÁ NHÀ - HOUSE PRICES KAGGLE
# THỰC NGHIỆM OVERFITTING VÀ CÁC KỸ THUẬT GIẢM OVERFITTING
# ============================================================


# ============================================================
# 1. IMPORT THƯ VIỆN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Kích thước train:", train.shape)
print("Kích thước test :", test.shape)


# ============================================================
# 3. XEM 5 DÒNG ĐẦU
# ============================================================

print("\n===== 5 DÒNG ĐẦU TRAIN =====")
print(train.head())


# ============================================================
# 4. THÔNG TIN DỮ LIỆU
# ============================================================

print("\n===== THÔNG TIN DỮ LIỆU =====")
train.info()


# ============================================================
# 5. THỐNG KÊ DỮ LIỆU
# ============================================================

print("\n===== THỐNG KÊ SALEPRICE =====")
print(train["SalePrice"].describe())


# ============================================================
# 6. KIỂM TRA GIÁ TRỊ THIẾU
# ============================================================

print("\n===== CÁC CỘT BỊ THIẾU DỮ LIỆU =====")

missing = train.isnull().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)


# ============================================================
# 7. BIỂU ĐỒ PHÂN PHỐI GIÁ NHÀ
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    train["SalePrice"],
    bins=30
)

plt.xlabel("SalePrice")
plt.ylabel("Frequency")

plt.title("Phân phối giá nhà")

plt.show()


# ============================================================
# 8. BIỂU ĐỒ DIỆN TÍCH VÀ GIÁ NHÀ
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    train["GrLivArea"],
    train["SalePrice"]
)

plt.xlabel("Diện tích sử dụng - GrLivArea")
plt.ylabel("Giá nhà - SalePrice")

plt.title("Quan hệ giữa diện tích và giá nhà")

plt.show()


# ============================================================
# 9. CHỌN CÁC ĐẶC TRƯNG
# ============================================================
#
# Chọn các biến số quan trọng để:
# - dễ tiền xử lý
# - dễ giải thích
# - tập trung vào thí nghiệm Overfitting
#
# ============================================================

features = [
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GarageCars",
    "GarageArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF"
]


# ============================================================
# 10. TẠO X VÀ y
# ============================================================

X = train[features]

y = train["SalePrice"]


print("\nSố lượng đặc trưng:", len(features))


# ============================================================
# 11. XỬ LÝ GIÁ TRỊ THIẾU
# ============================================================

X = X.fillna(X.median())


# ============================================================
# 12. CHIA TRAIN / VALIDATION
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n===== KÍCH THƯỚC DỮ LIỆU =====")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)


# ============================================================
# 13. HÀM ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def evaluate_model(model, X_train, y_train, X_val, y_val):

    # Dự đoán tập train
    train_pred = model.predict(X_train)

    # Dự đoán tập validation
    val_pred = model.predict(X_val)

    # R2
    train_r2 = r2_score(
        y_train,
        train_pred
    )

    val_r2 = r2_score(
        y_val,
        val_pred
    )

    # RMSE
    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    # MAE
    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )

    # Khoảng cách R2
    r2_gap = train_r2 - val_r2

    return {
        "Train R2": train_r2,
        "Validation R2": val_r2,
        "R2 Gap": r2_gap,
        "Train RMSE": train_rmse,
        "Validation RMSE": val_rmse,
        "Train MAE": train_mae,
        "Validation MAE": val_mae
    }


# ============================================================
# 14. MÔ HÌNH 1 - CỐ TÌNH TẠO OVERFITTING
# ============================================================
#
# max_depth=None
# min_samples_split=2
# min_samples_leaf=1
#
# Cho phép cây phát triển rất sâu.
#
# ============================================================

model_overfit = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# Huấn luyện
model_overfit.fit(
    X_train,
    y_train
)


# Đánh giá
result_overfit = evaluate_model(
    model_overfit,
    X_train,
    y_train,
    X_val,
    y_val
)


print("\n======================================")
print("MÔ HÌNH 1 - DECISION TREE OVERFITTING")
print("======================================")

for key, value in result_overfit.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# 15. KỸ THUẬT 1 - GIỚI HẠN max_depth
# ============================================================

depths = [
    2,
    4,
    6,
    8,
    10,
    12,
    15,
    20,
    None
]

results_depth = []


for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    result["Model"] = "Max Depth"
    result["Parameter"] = str(depth)

    results_depth.append(result)


df_depth = pd.DataFrame(results_depth)


print("\n======================================")
print("KỸ THUẬT 1 - MAX_DEPTH")
print("======================================")

print(
    df_depth[
        [
            "Parameter",
            "Train R2",
            "Validation R2",
            "R2 Gap",
            "Train RMSE",
            "Validation RMSE"
        ]
    ]
)


# ============================================================
# 16. BIỂU ĐỒ MAX_DEPTH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df_depth["Parameter"],
    df_depth["Train R2"],
    marker="o",
    label="Train R²"
)

plt.plot(
    df_depth["Parameter"],
    df_depth["Validation R2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("max_depth")
plt.ylabel("R²")

plt.title("Ảnh hưởng của max_depth đến Overfitting")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 17. KỸ THUẬT 2 - min_samples_split
# ============================================================

split_values = [
    2,
    5,
    10,
    20,
    30,
    50
]

results_split = []


for value in split_values:

    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=value,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    result["Model"] = "Min Samples Split"
    result["Parameter"] = value

    results_split.append(result)


df_split = pd.DataFrame(results_split)


print("\n======================================")
print("KỸ THUẬT 2 - MIN_SAMPLES_SPLIT")
print("======================================")

print(
    df_split[
        [
            "Parameter",
            "Train R2",
            "Validation R2",
            "R2 Gap",
            "Train RMSE",
            "Validation RMSE"
        ]
    ]
)


# ============================================================
# 18. BIỂU ĐỒ min_samples_split
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df_split["Parameter"],
    df_split["Train R2"],
    marker="o",
    label="Train R²"
)

plt.plot(
    df_split["Parameter"],
    df_split["Validation R2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("min_samples_split")
plt.ylabel("R²")

plt.title("Ảnh hưởng của min_samples_split")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 19. KỸ THUẬT 3 - min_samples_leaf
# ============================================================

leaf_values = [
    1,
    2,
    5,
    10,
    20
]

results_leaf = []


for value in leaf_values:

    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=value,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    result["Model"] = "Min Samples Leaf"
    result["Parameter"] = value

    results_leaf.append(result)


df_leaf = pd.DataFrame(results_leaf)


print("\n======================================")
print("KỸ THUẬT 3 - MIN_SAMPLES_LEAF")
print("======================================")

print(
    df_leaf[
        [
            "Parameter",
            "Train R2",
            "Validation R2",
            "R2 Gap",
            "Train RMSE",
            "Validation RMSE"
        ]
    ]
)


# ============================================================
# 20. BIỂU ĐỒ min_samples_leaf
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df_leaf["Parameter"],
    df_leaf["Train R2"],
    marker="o",
    label="Train R²"
)

plt.plot(
    df_leaf["Parameter"],
    df_leaf["Validation R2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("min_samples_leaf")
plt.ylabel("R²")

plt.title("Ảnh hưởng của min_samples_leaf")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 21. KỸ THUẬT 4 - PRUNING
# ============================================================
#
# Sử dụng ccp_alpha
#
# ============================================================

alphas = [
    0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05
]

results_pruning = []


for alpha in alphas:

    model = DecisionTreeRegressor(
        ccp_alpha=alpha,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    result["Model"] = "Pruning"
    result["Parameter"] = alpha

    results_pruning.append(result)


df_pruning = pd.DataFrame(
    results_pruning
)


print("\n======================================")
print("KỸ THUẬT 4 - PRUNING")
print("======================================")

print(
    df_pruning[
        [
            "Parameter",
            "Train R2",
            "Validation R2",
            "R2 Gap",
            "Train RMSE",
            "Validation RMSE"
        ]
    ]
)


# ============================================================
# 22. BIỂU ĐỒ PRUNING
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df_pruning["Parameter"],
    df_pruning["Train R2"],
    marker="o",
    label="Train R²"
)

plt.plot(
    df_pruning["Parameter"],
    df_pruning["Validation R2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("ccp_alpha")
plt.ylabel("R²")

plt.title("Ảnh hưởng của Pruning")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 23. KỸ THUẬT 5 - CROSS VALIDATION
# ============================================================
#
# Sử dụng 5-Fold Cross Validation
#
# ============================================================

model_cv = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


cv_scores = cross_val_score(
    model_cv,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)


print("\n======================================")
print("KỸ THUẬT 5 - CROSS VALIDATION")
print("======================================")

print("R² từng Fold:")

for i, score in enumerate(cv_scores, start=1):

    print(
        f"Fold {i}: {score:.4f}"
    )


print(
    f"\nR² trung bình: {cv_scores.mean():.4f}"
)

print(
    f"Độ lệch chuẩn: {cv_scores.std():.4f}"
)


# ============================================================
# 24. KỸ THUẬT 6 - RANDOM FOREST
# ============================================================

model_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42
)


model_rf.fit(
    X_train,
    y_train
)


result_rf = evaluate_model(
    model_rf,
    X_train,
    y_train,
    X_val,
    y_val
)


print("\n======================================")
print("KỸ THUẬT 6 - RANDOM FOREST")
print("======================================")

for key, value in result_rf.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# 25. TẠO BẢNG SO SÁNH TẤT CẢ MÔ HÌNH
# ============================================================

all_results = []


# ------------------------------------------------------------
# Decision Tree Overfit
# ------------------------------------------------------------

overfit_result = result_overfit.copy()

overfit_result["Model"] = "Decision Tree Overfit"

all_results.append(
    overfit_result
)


# ------------------------------------------------------------
# Decision Tree max_depth
# ------------------------------------------------------------

best_depth_row = df_depth.loc[
    df_depth["Validation R2"].idxmax()
]

all_results.append(
    best_depth_row.to_dict()
)


# ------------------------------------------------------------
# min_samples_split
# ------------------------------------------------------------

best_split_row = df_split.loc[
    df_split["Validation R2"].idxmax()
]

all_results.append(
    best_split_row.to_dict()
)


# ------------------------------------------------------------
# min_samples_leaf
# ------------------------------------------------------------

best_leaf_row = df_leaf.loc[
    df_leaf["Validation R2"].idxmax()
]

all_results.append(
    best_leaf_row.to_dict()
)


# ------------------------------------------------------------
# Pruning
# ------------------------------------------------------------

best_pruning_row = df_pruning.loc[
    df_pruning["Validation R2"].idxmax()
]

all_results.append(
    best_pruning_row.to_dict()
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf_result = result_rf.copy()

rf_result["Model"] = "Random Forest"

all_results.append(
    rf_result
)


# ============================================================
# 26. BẢNG KẾT QUẢ
# ============================================================

comparison = pd.DataFrame(
    all_results
)


print("\n======================================")
print("SO SÁNH CÁC MÔ HÌNH")
print("======================================")

print(
    comparison[
        [
            "Model",
            "Train R2",
            "Validation R2",
            "R2 Gap",
            "Train RMSE",
            "Validation RMSE",
            "Train MAE",
            "Validation MAE"
        ]
    ].round(4)
)


# ============================================================
# 27. BIỂU ĐỒ SO SÁNH R²
# ============================================================

plt.figure(figsize=(12, 6))

plt.bar(
    comparison["Model"],
    comparison["Validation R2"]
)

plt.ylabel("Validation R²")

plt.title(
    "So sánh Validation R² giữa các mô hình"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 1)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 28. BIỂU ĐỒ SO SÁNH RMSE
# ============================================================

plt.figure(figsize=(12, 6))

plt.bar(
    comparison["Model"],
    comparison["Validation RMSE"]
)

plt.ylabel("Validation RMSE")

plt.title(
    "So sánh Validation RMSE giữa các mô hình"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 29. BIỂU ĐỒ TRAIN VS VALIDATION
# ============================================================

x = np.arange(
    len(comparison)
)

width = 0.35


plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    comparison["Train R2"],
    width,
    label="Train R²"
)

plt.bar(
    x + width / 2,
    comparison["Validation R2"],
    width,
    label="Validation R²"
)


plt.xticks(
    x,
    comparison["Model"],
    rotation=30,
    ha="right"
)

plt.ylabel("R²")

plt.title(
    "So sánh Train R² và Validation R²"
)

plt.legend()

plt.grid(
    axis="y"
)

plt.show()


# ============================================================
# 30. TÌM MÔ HÌNH CÓ VALIDATION R² CAO NHẤT
# ============================================================

best_model_row = comparison.loc[
    comparison["Validation R2"].idxmax()
]


print("\n======================================")
print("MÔ HÌNH CÓ VALIDATION R² CAO NHẤT")
print("======================================")

print(
    best_model_row
)


# ============================================================
# 31. CROSS VALIDATION CHO RANDOM FOREST
# ============================================================

rf_cv_scores = cross_val_score(
    model_rf,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)


print("\n======================================")
print("RANDOM FOREST - CROSS VALIDATION")
print("======================================")

for i, score in enumerate(
    rf_cv_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )


print(
    f"\nR² trung bình: "
    f"{rf_cv_scores.mean():.4f}"
)

print(
    f"Độ lệch chuẩn: "
    f"{rf_cv_scores.std():.4f}"
)


# ============================================================
# 32. TRAIN MÔ HÌNH CUỐI CÙNG TRÊN TOÀN BỘ TRAIN
# ============================================================
#
# Ở đây sử dụng Random Forest.
#
# Sau khi nghiên cứu và lựa chọn mô hình,
# ta train lại trên toàn bộ train.csv.
#
# ============================================================


X_full = train[features]

y_full = train["SalePrice"]


# Xử lý missing
X_full = X_full.fillna(
    X_full.median()
)


# Chuẩn bị test
X_test = test[features]

X_test = X_test.fillna(
    X_full.median()
)


# ============================================================
# 33. TẠO MÔ HÌNH CUỐI
# ============================================================

final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42
)


# ============================================================
# 34. TRAIN TRÊN TOÀN BỘ TRAIN.CSV
# ============================================================

final_model.fit(
    X_full,
    y_full
)


# ============================================================
# 35. DỰ BÁO TEST.CSV
# ============================================================

test_prediction = final_model.predict(
    X_test
)


print("\n======================================")
print("DỰ BÁO TEST")
print("======================================")

print(
    test_prediction[:10]
)


# ============================================================
# 36. TẠO FILE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_prediction
})


# ============================================================
# 37. LƯU FILE
# ============================================================

submission.to_csv(
    "submission.csv",
    index=False
)


print("\n======================================")
print("ĐÃ TẠO FILE submission.csv")
print("======================================")

print(
    submission.head(10)
)