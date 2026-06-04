import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = r"D:\HUST-MI2\2025.2\Đồ án 1-Thầy Đoàn Duy Trung\Data\superstore_cleaned.csv"

encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
df = None

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Đọc dữ liệu thành công với encoding: {enc}")
        break
    except Exception:
        continue

if df is None:
    raise ValueError("Không đọc được file CSV. Hãy kiểm tra lại đường dẫn hoặc encoding.")


df.columns = df.columns.str.strip()

print("\n========== THÔNG TIN DỮ LIỆU ==========")
print("Kích thước dữ liệu:", df.shape)

print("\nTên cột:")
print(df.columns.tolist())

print("\n5 dòng đầu:")
print(df.head())

print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

print("\nSố dòng trùng lặp:", df.duplicated().sum())


for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "Order Date" in df.columns:
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Day"] = df["Order Date"].dt.day

if "Ship Date" in df.columns:
    df["Ship_Year"] = df["Ship Date"].dt.year
    df["Ship_Month"] = df["Ship Date"].dt.month
    df["Ship_Day"] = df["Ship Date"].dt.day

if "Order Date" in df.columns and "Ship Date" in df.columns:
    df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days


target = "Profit"  

if target not in df.columns:
    raise ValueError(f"Không tìm thấy cột mục tiêu: {target}")


drop_cols = [target]


for col in ["Order Date", "Ship Date"]:
    if col in df.columns:
        drop_cols.append(col)


if target == "Profit" and "Sales" in df.columns:
    drop_cols.append("Sales")


if target == "Sales" and "Profit" in df.columns:
    drop_cols.append("Profit")


id_like_cols = [
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name",
    "Postal Code"
]
for col in id_like_cols:
    if col in df.columns:
        drop_cols.append(col)

drop_cols = list(set(drop_cols))

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target]

print("\n========== BIẾN ĐẦU VÀO ==========")
print("Số cột đầu vào:", X.shape[1])
print("Danh sách cột đầu vào:")
print(X.columns.tolist())


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nBiến số:")
print(numeric_features)

print("\nBiến phân loại:")
print(categorical_features)


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nKích thước tập train:", X_train.shape)
print("Kích thước tập test :", X_test.shape)


rf_model.fit(X_train, y_train)


y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)


def evaluate_model(y_true, y_pred, name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n========== {name.upper()} ==========")
    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    return mae, mse, rmse, r2

train_metrics = evaluate_model(y_train, y_train_pred, "Train")
test_metrics = evaluate_model(y_test, y_test_pred, "Test")


result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_test_pred,
    "Residual": y_test.values - y_test_pred
})

print("\n10 dòng đầu của bảng kết quả:")
print(result.head(10))


feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 biến quan trọng nhất:")
print(importance_df.head(15))


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("Random Forest Regressor: Actual vs Predicted")
plt.grid(True)
plt.show()


errors = y_test - y_test_pred

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30, edgecolor="black")
plt.xlabel("Sai số")
plt.ylabel("Tần suất")
plt.title("Phân bố sai số của Random Forest Regressor")
plt.grid(True)
plt.show()


top_n = 10
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
plt.xlabel("Mức độ quan trọng")
plt.ylabel("Biến")
plt.title(f"Top {top_n} biến quan trọng nhất trong Random Forest")
plt.grid(True)
plt.show()


result.to_csv("random_forest_predictions.csv", index=False, encoding="utf-8-sig")
importance_df.to_csv("random_forest_feature_importance.csv", index=False, encoding="utf-8-sig")

metrics_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "MAE": [train_metrics[0], test_metrics[0]],
    "MSE": [train_metrics[1], test_metrics[1]],
    "RMSE": [train_metrics[2], test_metrics[2]],
    "R2": [train_metrics[3], test_metrics[3]]
})
metrics_df.to_csv("random_forest_metrics.csv", index=False, encoding="utf-8-sig")

print("\nĐã lưu các file kết quả:")
print("- random_forest_predictions.csv")
print("- random_forest_feature_importance.csv")
print("- random_forest_metrics.csv")