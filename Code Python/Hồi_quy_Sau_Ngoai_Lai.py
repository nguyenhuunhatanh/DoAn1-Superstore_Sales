import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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

print("\n========== THÔNG TIN DỮ LIỆU ==========")
print("Kích thước dữ liệu:", df.shape)
print("\nTên cột:")
print(df.columns.tolist())

print("\n5 dòng đầu:")
print(df.head())

print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

print("\nSố dòng trùng lặp:", df.duplicated().sum())

df.columns = df.columns.str.strip()

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

id_like_cols = ["Row ID", "Order ID", "Customer ID", "Customer Name", "Product ID", "Product Name"]
for col in id_like_cols:
    if col in df.columns:
        drop_cols.append(col)

drop_cols = list(set(drop_cols))

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target]

print("\n========== BIẾN ĐẦU VÀO BAN ĐẦU ==========")
print("Số cột đầu vào:", X.shape[1])
print(X.columns.tolist())



numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nBiến số:")
print(numeric_features)

print("\nBiến phân loại:")
print(categorical_features)



def cap_outliers_iqr(dataframe, columns):
    df_capped = dataframe.copy()
    cap_info = []

    for col in columns:
        if col not in df_capped.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df_capped[col]):
            continue

        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        before_outliers = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()

        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

        after_outliers = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()

        cap_info.append({
            "Column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outliers_Before": int(before_outliers),
            "Outliers_After": int(after_outliers)
        })

    cap_info_df = pd.DataFrame(cap_info)
    return df_capped, cap_info_df



X_capped, cap_info_df = cap_outliers_iqr(X, numeric_features)

print("\n========== THÔNG TIN XỬ LÝ OUTLIER ==========")
print(cap_info_df)

# Có thể xử lý outlier cho cả biến mục tiêu nếu muốn
# Thường nên thử cả 2 cách để so sánh
y_df = pd.DataFrame({target: y})
y_capped_df, y_cap_info = cap_outliers_iqr(y_df, [target])
y_capped = y_capped_df[target]

print("\n========== XỬ LÝ OUTLIER CHO BIẾN MỤC TIÊU ==========")
print(y_cap_info)

X_train, X_test, y_train, y_test = train_test_split(
    X_capped, y_capped,
    test_size=0.2,
    random_state=42
)

print("\nKích thước tập train:", X_train.shape)
print("Kích thước tập test :", X_test.shape)


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



model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


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

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nTop 15 hệ số dương lớn nhất:")
print(coef_df.head(15))

print("\nTop 15 hệ số âm lớn nhất:")
print(coef_df.tail(15))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("So sánh giá trị thực tế và giá trị dự đoán sau xử lý outlier")
plt.grid(True)
plt.show()

errors = y_test - y_test_pred

plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30, edgecolor="black")
plt.xlabel("Sai số")
plt.ylabel("Tần suất")
plt.title("Phân bố sai số của mô hình sau xử lý outlier")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
stats.probplot(errors, dist="norm", plot=plt)
plt.title("QQ Plot của phần dư sau xử lý outlier")
plt.grid(True)
plt.show()

if len(numeric_features) > 0:
    col_plot = numeric_features[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(X[col_plot].dropna())
    plt.title(f"Trước xử lý outlier\n{col_plot}")

    plt.subplot(1, 2, 2)
    plt.boxplot(X_capped[col_plot].dropna())
    plt.title(f"Sau xử lý outlier\n{col_plot}")

    plt.tight_layout()
    plt.show()


result.to_csv("linear_regression_outlier_predictions.csv", index=False, encoding="utf-8-sig")
coef_df.to_csv("linear_regression_outlier_coefficients.csv", index=False, encoding="utf-8-sig")
cap_info_df.to_csv("outlier_capping_info_X.csv", index=False, encoding="utf-8-sig")
y_cap_info.to_csv("outlier_capping_info_y.csv", index=False, encoding="utf-8-sig")

metrics_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "MAE": [train_metrics[0], test_metrics[0]],
    "MSE": [train_metrics[1], test_metrics[1]],
    "RMSE": [train_metrics[2], test_metrics[2]],
    "R2": [train_metrics[3], test_metrics[3]]
})
metrics_df.to_csv("linear_regression_outlier_metrics.csv", index=False, encoding="utf-8-sig")

print("\nĐã lưu các file kết quả:")
print("- linear_regression_outlier_predictions.csv")
print("- linear_regression_outlier_coefficients.csv")
print("- outlier_capping_info_X.csv")
print("- outlier_capping_info_y.csv")
print("- linear_regression_outlier_metrics.csv")