import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

file_path = r"D:\HUST-MI2\2025.2\Đồ án 1-Thầy Đoàn Duy Trung\Data\superstore_cleaned.csv"
output_dir = r"D:\HUST-MI2\2025.2\Đồ án 1-Thầy Đoàn Duy Trung\Model_Output\KMeans_Sau_Ngoai_Lai"
os.makedirs(output_dir, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.4f}".format)

def save_fig(filename):
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu hình: {path}")

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

        outliers_before = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()

        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

        outliers_after = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()

        cap_info.append({
            "Column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outliers_Before": int(outliers_before),
            "Outliers_After": int(outliers_after)
        })

    return df_capped, pd.DataFrame(cap_info)

def build_cluster_description(cluster_summary):
    cluster_description = {}

    for _, row in cluster_summary.iterrows():
        cluster = int(row["Cluster"])
        sales = row["Avg_Total_Sales"]
        profit = row["Avg_Total_Profit"]
        margin = row["Avg_Profit_Margin"]

        if sales >= cluster_summary["Avg_Total_Sales"].quantile(0.66) and profit >= cluster_summary["Avg_Total_Profit"].quantile(0.66):
            desc = "Nhóm khách hàng giá trị cao"
        elif profit < 0 or margin < 0:
            desc = "Nhóm khách hàng lợi nhuận thấp hoặc gây lỗ"
        elif sales <= cluster_summary["Avg_Total_Sales"].quantile(0.33):
            desc = "Nhóm khách hàng giá trị thấp"
        else:
            desc = "Nhóm khách hàng giá trị trung bình"

        cluster_description[cluster] = desc

    return cluster_description


# 1. ĐỌC DỮ LIỆU
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
print("Số dòng trùng lặp:", df.duplicated().sum())
print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())


# 2. XỬ LÝ CỘT NGÀY VÀ THỜI GIAN GIAO HÀNG

for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "Order Date" in df.columns and "Ship Date" in df.columns:
    df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

if "Shipping_Days" not in df.columns:
    df["Shipping_Days"] = 0


# 3. TỔNG HỢP DỮ LIỆU THEO KHÁCH HÀNG

required_cols = [
    "Customer ID", "Customer Name", "Segment", "Sales", "Profit",
    "Quantity", "Discount", "Order ID", "Shipping_Days"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Thiếu các cột cần thiết cho K-Means: {missing_cols}")

customer_df = (
    df.groupby(["Customer ID", "Customer Name", "Segment"], as_index=False)
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Avg_Discount=("Discount", "mean"),
        Avg_Shipping_Days=("Shipping_Days", "mean"),
        Order_Count=("Order ID", "nunique")
    )
)

customer_df["Sales_Per_Order"] = customer_df["Total_Sales"] / customer_df["Order_Count"]
customer_df["Profit_Per_Order"] = customer_df["Total_Profit"] / customer_df["Order_Count"]
customer_df["Profit_Margin"] = customer_df["Total_Profit"] / customer_df["Total_Sales"]

customer_df = customer_df.replace([np.inf, -np.inf], np.nan)
customer_df = customer_df.dropna()

print("\n========== DỮ LIỆU KHÁCH HÀNG SAU TỔNG HỢP BAN ĐẦU ==========")
print("Kích thước dữ liệu khách hàng:", customer_df.shape)
print(customer_df.head())


# 4. CHỌN BIẾN PHÂN CỤM

cluster_features = [
    "Total_Sales",
    "Total_Profit",
    "Total_Quantity",
    "Avg_Discount",
    "Avg_Shipping_Days",
    "Order_Count",
    "Sales_Per_Order",
    "Profit_Per_Order",
    "Profit_Margin"
]

print("\nCác biến dùng để phân cụm:")
print(cluster_features)


# 5. XỬ LÝ NGOẠI LAI BẰNG IQR CAPPING

customer_features_capped, cap_info = cap_outliers_iqr(
    customer_df[cluster_features],
    cluster_features
)

customer_df[cluster_features] = customer_features_capped[cluster_features]

print("\n========== THÔNG TIN XỬ LÝ NGOẠI LAI CHO DỮ LIỆU KHÁCH HÀNG ==========")
print(cap_info)

cap_info.to_csv(os.path.join(output_dir, "kmeans_after_outlier_capping_info_customer_features.csv"), index=False, encoding="utf-8-sig")

X = customer_df[cluster_features].copy()


# 6. CHUẨN HÓA DỮ LIỆU

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. CHỌN SỐ CỤM BẰNG ELBOW METHOD VÀ SILHOUETTE SCORE

k_values = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)

evaluation_df = pd.DataFrame({
    "K": list(k_values),
    "WCSS_Inertia": inertias,
    "Silhouette_Score": silhouette_scores
})

print("\n========== ĐÁNH GIÁ SỐ CỤM ==========")
print(evaluation_df)

best_k = int(evaluation_df.sort_values("Silhouette_Score", ascending=False).iloc[0]["K"])

print("\nSố cụm gợi ý theo Silhouette Score:", best_k)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method - KMeans sau xử lý ngoại lai")
plt.grid(True)
save_fig("kmeans_after_elbow_method.png")

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - KMeans sau xử lý ngoại lai")
plt.grid(True)
save_fig("kmeans_after_silhouette_score.png")


# 8. HUẤN LUYỆN K-MEANS VỚI SỐ CỤM ĐƯỢC CHỌN

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
customer_df["Cluster"] = kmeans_final.fit_predict(X_scaled)


# 9. TÓM TẮT ĐẶC ĐIỂM TỪNG CỤM

cluster_summary = (
    customer_df.groupby("Cluster")
    .agg(
        Customer_Count=("Customer ID", "count"),
        Avg_Total_Sales=("Total_Sales", "mean"),
        Avg_Total_Profit=("Total_Profit", "mean"),
        Avg_Total_Quantity=("Total_Quantity", "mean"),
        Avg_Discount=("Avg_Discount", "mean"),
        Avg_Order_Count=("Order_Count", "mean"),
        Avg_Sales_Per_Order=("Sales_Per_Order", "mean"),
        Avg_Profit_Per_Order=("Profit_Per_Order", "mean"),
        Avg_Profit_Margin=("Profit_Margin", "mean")
    )
    .reset_index()
)

cluster_description = build_cluster_description(cluster_summary)
customer_df["Cluster_Description"] = customer_df["Cluster"].map(cluster_description)
cluster_summary["Cluster_Description"] = cluster_summary["Cluster"].map(cluster_description)

print("\n========== TÓM TẮT ĐẶC ĐIỂM TỪNG CỤM ==========")
print(cluster_summary)

print("\n========== 20 KHÁCH HÀNG ĐẦU TIÊN SAU PHÂN CỤM ==========")
print(customer_df[["Customer ID", "Customer Name", "Segment", "Cluster", "Cluster_Description"] + cluster_features].head(20))


# 10. BIỂU ĐỒ PHÂN CỤM

plt.figure(figsize=(8, 6))
plt.scatter(customer_df["Total_Sales"], customer_df["Total_Profit"], c=customer_df["Cluster"], alpha=0.7)
plt.xlabel("Tổng doanh thu của khách hàng")
plt.ylabel("Tổng lợi nhuận của khách hàng")
plt.title("Phân cụm khách hàng theo doanh thu và lợi nhuận - sau xử lý ngoại lai")
plt.grid(True)
save_fig("kmeans_after_customer_clusters_sales_profit.png")

plt.figure(figsize=(8, 5))
plt.bar(cluster_summary["Cluster"].astype(str), cluster_summary["Customer_Count"])
plt.xlabel("Cụm khách hàng")
plt.ylabel("Số lượng khách hàng")
plt.title("Số lượng khách hàng trong từng cụm - sau xử lý ngoại lai")
plt.grid(True)
save_fig("kmeans_after_customer_count_by_cluster.png")

x = np.arange(len(cluster_summary["Cluster"]))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, cluster_summary["Avg_Total_Sales"], width, label="Doanh thu trung bình")
plt.bar(x + width / 2, cluster_summary["Avg_Total_Profit"], width, label="Lợi nhuận trung bình")
plt.xlabel("Cụm khách hàng")
plt.ylabel("Giá trị trung bình")
plt.title("Doanh thu và lợi nhuận trung bình theo cụm - sau xử lý ngoại lai")
plt.xticks(x, cluster_summary["Cluster"].astype(str))
plt.legend()
plt.grid(True)
save_fig("kmeans_after_avg_sales_profit_by_cluster.png")


# 11. LƯU KẾT QUẢ

customer_df.to_csv(os.path.join(output_dir, "kmeans_after_customer_clusters.csv"), index=False, encoding="utf-8-sig")
cluster_summary.to_csv(os.path.join(output_dir, "kmeans_after_cluster_summary.csv"), index=False, encoding="utf-8-sig")
evaluation_df.to_csv(os.path.join(output_dir, "kmeans_after_evaluation_k.csv"), index=False, encoding="utf-8-sig")

best_row = evaluation_df[evaluation_df["K"] == best_k].iloc[0]

metrics_df = pd.DataFrame({
    "Model": ["KMeans"],
    "Case": ["Sau xử lý ngoại lai"],
    "Best_K": [best_k],
    "Best_K_WCSS_Inertia": [best_row["WCSS_Inertia"]],
    "Best_K_Silhouette_Score": [best_row["Silhouette_Score"]]
})

metrics_df.to_csv(os.path.join(output_dir, "kmeans_after_metrics.csv"), index=False, encoding="utf-8-sig")

print("\nĐã lưu các file kết quả vào thư mục:")
print(output_dir)
