import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"D:\HUST-MI2\2025.2\Đồ án 1-Thầy Đoàn Duy Trung\Data\superstore_cleaned.csv"
output_dir = r"D:\HUST-MI2\2025.2\Đồ án 1-Thầy Đoàn Duy Trung\EDA_Output"


os.makedirs(output_dir, exist_ok=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)

def save_fig(filename):
    """
    Lưu hình vào thư mục output_dir với độ phân giải cao.
    """
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu hình: {path}")

def print_title(title):
    """
    In tiêu đề phần cho dễ theo dõi output.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def format_number(x):
    """
    Định dạng số theo kiểu dễ đọc.
    """
    return f"{x:,.2f}"

# 1. ĐỌC DỮ LIỆU
print_title("1. ĐỌC DỮ LIỆU")
df = pd.read_csv(file_path, encoding="utf-8-sig")
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
print("Đọc dữ liệu thành công.")
print("Kích thước dữ liệu:", df.shape)
print("\n5 dòng đầu của dữ liệu:")
print(df.head())
print("\nDanh sách các cột dữ liệu:")
print(df.columns.tolist())
# 2. KIỂM TRA CẤU TRÚC VÀ CHẤT LƯỢNG DỮ LIỆU
print_title("2. KIỂM TRA CẤU TRÚC VÀ CHẤT LƯỢNG DỮ LIỆU")
print("\nThông tin dữ liệu:")
df.info()
print("\nSố giá trị thiếu theo từng cột:")
print(df.isnull().sum())
print("\nTổng số giá trị thiếu trong toàn bộ dữ liệu:")
print(df.isnull().sum().sum())
print("\nSố dòng dữ liệu trùng lặp:")
print(df.duplicated().sum())
print("\nSố dòng lỗi ngày đặt hàng sau khi chuyển datetime:")
print(df["Order Date"].isna().sum())
print("\nSố dòng lỗi ngày giao hàng sau khi chuyển datetime:")
print(df["Ship Date"].isna().sum())
print("\nKhoảng thời gian dữ liệu:")
print("Ngày đặt hàng sớm nhất:", df["Order Date"].min())
print("Ngày đặt hàng muộn nhất:", df["Order Date"].max())

quality_summary = pd.DataFrame({
    "Chỉ tiêu": ["Số dòng", "Số cột", "Giá trị thiếu", "Dòng trùng lặp"],
    "Giá trị": [df.shape[0], df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()]
})
plt.figure(figsize=(8, 5))
plt.bar(quality_summary["Chỉ tiêu"], quality_summary["Giá trị"])
plt.title("Tổng quan chất lượng dữ liệu")
plt.xlabel("Chỉ tiêu")
plt.ylabel("Giá trị")
save_fig("hinh_3_1_tong_quan_chat_luong_du_lieu.png")
# 3. THỐNG KÊ MÔ TẢ TỔNG QUAN
print_title("3. THỐNG KÊ MÔ TẢ TỔNG QUAN")
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
total_products = df["Product ID"].nunique()
negative_profit_rows = (df["Profit"] < 0).sum()
negative_profit_total = df.loc[df["Profit"] < 0, "Profit"].sum()
print("===== CÁC CHỈ TIÊU TỔNG QUAN =====")
print(f"Tổng doanh thu              : {format_number(total_sales)}")
print(f"Tổng lợi nhuận              : {format_number(total_profit)}")
print(f"Tổng số lượng bán           : {total_quantity:,}")
print(f"Tổng số đơn hàng            : {total_orders:,}")
print(f"Tổng số khách hàng          : {total_customers:,}")
print(f"Tổng số sản phẩm            : {total_products:,}")
print(f"Số dòng có lợi nhuận âm     : {negative_profit_rows:,}")
print(f"Tổng lỗ từ các dòng bị âm   : {format_number(negative_profit_total)}")
print("\nThống kê mô tả các biến định lượng chính:")
numeric_cols = ["Sales", "Profit", "Quantity", "Discount", "Shipping Days"]
print(df[numeric_cols].describe())
print("\nNHẬN XÉT:")
print(
    "Bộ dữ liệu cung cấp các biến định lượng quan trọng như doanh thu, lợi nhuận, "
    "số lượng bán, chiết khấu và thời gian giao hàng. Các chỉ tiêu tổng quan cho phép "
    "đánh giá quy mô kinh doanh, trong khi thống kê mô tả giúp nhận diện mức độ phân tán "
    "và sự khác biệt giữa các giao dịch."
)
# 4. CÂU HỎI 1: DOANH THU THAY ĐỔI NHƯ THẾ NÀO THEO THỜI GIAN?
print_title("4. CÂU HỎI 1: DOANH THU THAY ĐỔI NHƯ THẾ NÀO THEO THỜI GIAN?")
sales_profit_by_year = (
    df.groupby("Year", as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Year")
)

sales_by_month = (
    df.groupby(["Year", "Month"], as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values(["Year", "Month"])
)

sales_by_month["Year-Month"] = (
    sales_by_month["Year"].astype(str) + "-" +
    sales_by_month["Month"].astype(str).str.zfill(2)
)

print("\n===== DOANH THU, LỢI NHUẬN THEO NĂM =====")
print(sales_profit_by_year)

print("\n===== DOANH THU, LỢI NHUẬN THEO THÁNG =====")
print(sales_by_month)

# Vẽ doanh thu và lợi nhuận theo năm
plt.figure(figsize=(10, 5))
plt.plot(sales_profit_by_year["Year"], sales_profit_by_year["Sales"], marker="o", label="Doanh thu")
plt.plot(sales_profit_by_year["Year"], sales_profit_by_year["Profit"], marker="o", label="Lợi nhuận")
plt.title("Doanh thu và lợi nhuận theo năm")
plt.xlabel("Năm")
plt.ylabel("Giá trị")
plt.xticks(sales_profit_by_year["Year"])
plt.legend()
save_fig("hinh_3_2_doanh_thu_loi_nhuan_theo_nam.png")

# Vẽ doanh thu theo tháng
plt.figure(figsize=(14, 5))
plt.plot(sales_by_month["Year-Month"], sales_by_month["Sales"], marker="o")
plt.title("Xu hướng doanh thu theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu")
plt.xticks(rotation=45, ha="right")
save_fig("hinh_3_3_xu_huong_doanh_thu_theo_thang.png")

best_year = sales_profit_by_year.sort_values("Sales", ascending=False).iloc[0]
print("\nNHẬN XÉT:")
print(
    f"Năm có doanh thu cao nhất là {int(best_year['Year'])} "
    f"với doanh thu {format_number(best_year['Sales'])}. "
    "Xu hướng theo thời gian giúp đánh giá sự biến động của doanh thu và lợi nhuận "
    "qua các năm, từ đó hỗ trợ nhận diện giai đoạn kinh doanh hiệu quả."
)

# 5. CÂU HỎI 2: KHU VỰC NÀO ĐÓNG GÓP DOANH THU VÀ LỢI NHUẬN CAO NHẤT?
print_title("5. CÂU HỎI 2: KHU VỰC NÀO ĐÓNG GÓP DOANH THU VÀ LỢI NHUẬN CAO NHẤT?")
region_summary = (
    df.groupby("Region", as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Customers=("Customer ID", "nunique")
    )
    .sort_values(by="Sales", ascending=False)
)

state_summary = (
    df.groupby("State/Province", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values(by="Sales", ascending=False)
)

print("\n===== DOANH THU VÀ LỢI NHUẬN THEO KHU VỰC =====")
print(region_summary)

print("\n===== TOP 10 TỈNH/BANG THEO DOANH THU =====")
print(state_summary.head(10))

print("\n===== TOP 10 TỈNH/BANG THEO LỢI NHUẬN =====")
print(state_summary.sort_values("Profit", ascending=False).head(10))

print("\n===== TOP 10 TỈNH/BANG CÓ LỢI NHUẬN THẤP NHẤT =====")
print(state_summary.sort_values("Profit", ascending=True).head(10))

# Vẽ doanh thu theo khu vực
plt.figure(figsize=(8, 5))
plt.bar(region_summary["Region"], region_summary["Sales"])
plt.title("Doanh thu theo khu vực")
plt.xlabel("Khu vực")
plt.ylabel("Doanh thu")
save_fig("hinh_3_4_doanh_thu_theo_khu_vuc.png")

# Vẽ lợi nhuận theo khu vực
plt.figure(figsize=(8, 5))
plt.bar(region_summary["Region"], region_summary["Profit"])
plt.title("Lợi nhuận theo khu vực")
plt.xlabel("Khu vực")
plt.ylabel("Lợi nhuận")
save_fig("hinh_3_5_loi_nhuan_theo_khu_vuc.png")

best_region_sales = region_summary.sort_values("Sales", ascending=False).iloc[0]
best_region_profit = region_summary.sort_values("Profit", ascending=False).iloc[0]

print("\nNHẬN XÉT:")
print(
    f"Khu vực có doanh thu cao nhất là {best_region_sales['Region']} "
    f"với doanh thu {format_number(best_region_sales['Sales'])}. "
    f"Khu vực có lợi nhuận cao nhất là {best_region_profit['Region']} "
    f"với lợi nhuận {format_number(best_region_profit['Profit'])}. "
    "Việc so sánh đồng thời doanh thu và lợi nhuận giúp đánh giá khu vực không chỉ theo quy mô bán hàng "
    "mà còn theo hiệu quả sinh lời."
)



# 6. CÂU HỎI 3: DANH MỤC SẢN PHẨM NÀO MANG LẠI DOANH THU VÀ LỢI NHUẬN TỐT NHẤT?
print_title("6. CÂU HỎI 3: DANH MỤC SẢN PHẨM NÀO MANG LẠI DOANH THU VÀ LỢI NHUẬN TỐT NHẤT?")

category_summary = (
    df.groupby("Category", as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values(by="Sales", ascending=False)
)

subcategory_summary = (
    df.groupby("Sub-Category", as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .sort_values(by="Sales", ascending=False)
)

top_profit_subcategory = (
    subcategory_summary
    .sort_values(by="Profit", ascending=False)
    .head(10)
)

bottom_profit_subcategory = (
    subcategory_summary
    .sort_values(by="Profit", ascending=True)
    .head(10)
)

top_products_sales = (
    df.groupby("Product Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values(by="Sales", ascending=False)
    .head(10)
)

top_products_profit = (
    df.groupby("Product Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values(by="Profit", ascending=False)
    .head(10)
)

bottom_products_profit = (
    df.groupby("Product Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values(by="Profit", ascending=True)
    .head(10)
)

print("\n===== DOANH THU, LỢI NHUẬN THEO DANH MỤC =====")
print(category_summary)

print("\n===== DOANH THU, LỢI NHUẬN THEO TIỂU DANH MỤC =====")
print(subcategory_summary)

print("\n===== TOP 10 TIỂU DANH MỤC CÓ LỢI NHUẬN CAO NHẤT =====")
print(top_profit_subcategory)

print("\n===== TOP 10 TIỂU DANH MỤC CÓ LỢI NHUẬN THẤP NHẤT =====")
print(bottom_profit_subcategory)

print("\n===== TOP 10 SẢN PHẨM THEO DOANH THU =====")
print(top_products_sales)

print("\n===== TOP 10 SẢN PHẨM THEO LỢI NHUẬN =====")
print(top_products_profit)

print("\n===== TOP 10 SẢN PHẨM GÂY LỖ NHIỀU NHẤT =====")
print(bottom_products_profit)

# Vẽ doanh thu theo danh mục
plt.figure(figsize=(8, 5))
plt.bar(category_summary["Category"], category_summary["Sales"])
plt.title("Doanh thu theo danh mục sản phẩm")
plt.xlabel("Danh mục")
plt.ylabel("Doanh thu")
save_fig("hinh_3_6_doanh_thu_theo_danh_muc.png")

# Vẽ lợi nhuận theo danh mục
category_profit = category_summary.sort_values("Profit", ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(category_profit["Category"], category_profit["Profit"])
plt.title("Lợi nhuận theo danh mục sản phẩm")
plt.xlabel("Danh mục")
plt.ylabel("Lợi nhuận")
save_fig("hinh_3_7_loi_nhuan_theo_danh_muc.png")

# Vẽ top tiểu danh mục lợi nhuận cao
plt.figure(figsize=(10, 6))
plt.barh(top_profit_subcategory["Sub-Category"], top_profit_subcategory["Profit"])
plt.title("Top 10 tiểu danh mục có lợi nhuận cao nhất")
plt.xlabel("Lợi nhuận")
plt.ylabel("Tiểu danh mục")
plt.gca().invert_yaxis()
save_fig("hinh_3_8_top_10_tieu_danh_muc_loi_nhuan_cao.png")

# Vẽ tiểu danh mục lợi nhuận thấp nhất
plt.figure(figsize=(10, 6))
plt.barh(bottom_profit_subcategory["Sub-Category"], bottom_profit_subcategory["Profit"])
plt.title("Top 10 tiểu danh mục có lợi nhuận thấp nhất")
plt.xlabel("Lợi nhuận")
plt.ylabel("Tiểu danh mục")
plt.gca().invert_yaxis()
save_fig("hinh_3_9_top_10_tieu_danh_muc_loi_nhuan_thap.png")

best_category_sales = category_summary.sort_values("Sales", ascending=False).iloc[0]
best_category_profit = category_summary.sort_values("Profit", ascending=False).iloc[0]
worst_subcategory_profit = bottom_profit_subcategory.iloc[0]

print("\nNHẬN XÉT:")
print(
    f"Danh mục có doanh thu cao nhất là {best_category_sales['Category']} "
    f"với doanh thu {format_number(best_category_sales['Sales'])}. "
    f"Danh mục có lợi nhuận cao nhất là {best_category_profit['Category']} "
    f"với lợi nhuận {format_number(best_category_profit['Profit'])}. "
    f"Tiểu danh mục có lợi nhuận thấp nhất là {worst_subcategory_profit['Sub-Category']} "
    f"với lợi nhuận {format_number(worst_subcategory_profit['Profit'])}. "
    "Điều này cho thấy cần xem xét đồng thời doanh thu và lợi nhuận, vì nhóm có doanh thu cao chưa chắc đã là nhóm có hiệu quả sinh lời tốt nhất."
)



# 7. CÂU HỎI 4: NHÓM KHÁCH HÀNG NÀO CÓ XU HƯỚNG TIÊU DÙNG CAO HƠN?
print_title("7. CÂU HỎI 4: NHÓM KHÁCH HÀNG NÀO CÓ XU HƯỚNG TIÊU DÙNG CAO HƠN?")

segment_summary = (
    df.groupby("Segment", as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Customers=("Customer ID", "nunique")
    )
    .sort_values(by="Sales", ascending=False)
)

# Bổ sung chỉ tiêu trung bình để phân tích sâu hơn
segment_summary["Sales_per_Order"] = segment_summary["Sales"] / segment_summary["Orders"]
segment_summary["Profit_per_Order"] = segment_summary["Profit"] / segment_summary["Orders"]
segment_summary["Sales_per_Customer"] = segment_summary["Sales"] / segment_summary["Customers"]
segment_summary["Profit_Margin"] = segment_summary["Profit"] / segment_summary["Sales"]

top_customers_sales = (
    df.groupby("Customer Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
    .sort_values(by="Sales", ascending=False)
    .head(10)
)

top_customers_profit = (
    df.groupby("Customer Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
    .sort_values(by="Profit", ascending=False)
    .head(10)
)

print("\n===== DOANH THU, LỢI NHUẬN THEO PHÂN KHÚC KHÁCH HÀNG =====")
print(segment_summary)

print("\n===== TOP 10 KHÁCH HÀNG THEO DOANH THU =====")
print(top_customers_sales)

print("\n===== TOP 10 KHÁCH HÀNG THEO LỢI NHUẬN =====")
print(top_customers_profit)

# Vẽ doanh thu theo phân khúc
plt.figure(figsize=(8, 5))
plt.bar(segment_summary["Segment"], segment_summary["Sales"])
plt.title("Doanh thu theo phân khúc khách hàng")
plt.xlabel("Phân khúc khách hàng")
plt.ylabel("Doanh thu")
save_fig("hinh_3_10_doanh_thu_theo_phan_khuc_khach_hang.png")

# Vẽ lợi nhuận theo phân khúc
plt.figure(figsize=(8, 5))
plt.bar(segment_summary["Segment"], segment_summary["Profit"])
plt.title("Lợi nhuận theo phân khúc khách hàng")
plt.xlabel("Phân khúc khách hàng")
plt.ylabel("Lợi nhuận")
save_fig("hinh_3_11_loi_nhuan_theo_phan_khuc_khach_hang.png")

# Vẽ top khách hàng theo doanh thu
plt.figure(figsize=(10, 6))
plt.barh(top_customers_sales["Customer Name"], top_customers_sales["Sales"])
plt.title("Top 10 khách hàng có doanh thu cao nhất")
plt.xlabel("Doanh thu")
plt.ylabel("Khách hàng")
plt.gca().invert_yaxis()
save_fig("hinh_3_12_top_10_khach_hang_theo_doanh_thu.png")

best_segment = segment_summary.sort_values("Sales", ascending=False).iloc[0]

print("\nNHẬN XÉT:")
print(
    f"Phân khúc khách hàng có xu hướng tiêu dùng cao nhất xét theo doanh thu là "
    f"{best_segment['Segment']} với doanh thu {format_number(best_segment['Sales'])}. "
    "Ngoài tổng doanh thu, các chỉ tiêu như doanh thu trung bình trên mỗi đơn hàng, "
    "doanh thu trung bình trên mỗi khách hàng và biên lợi nhuận cũng cần được xem xét "
    "để đánh giá toàn diện giá trị của từng phân khúc khách hàng."
)

# 8. CÂU HỎI 5: MỨC CHIẾT KHẤU ẢNH HƯỞNG NHƯ THẾ NÀO ĐẾN LỢI NHUẬN?
print_title("8. CÂU HỎI 5: MỨC CHIẾT KHẤU ẢNH HƯỞNG NHƯ THẾ NÀO ĐẾN LỢI NHUẬN?")

# Phân tích theo từng mức chiết khấu gốc
discount_profit = (
    df.groupby("Discount", as_index=False)
    .agg(
        Avg_Profit=("Profit", "mean"),
        Total_Profit=("Profit", "sum"),
        Sales=("Sales", "sum"),
        Rows=("Profit", "count")
    )
    .sort_values(by="Discount")
)

corr = df[["Discount", "Profit", "Sales", "Quantity"]].corr()

print("\n===== LỢI NHUẬN THEO TỪNG MỨC CHIẾT KHẤU =====")
print(discount_profit)

print("\n===== MA TRẬN TƯƠNG QUAN GIỮA DISCOUNT, PROFIT, SALES, QUANTITY =====")
print(corr)

# Phân nhóm chiết khấu để dễ trình bày trong báo cáo
bins = [-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%"]

df["Discount Group"] = pd.cut(df["Discount"], bins=bins, labels=labels)

discount_group_summary = (
    df.groupby("Discount Group", observed=False, as_index=False)
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Avg_Profit=("Profit", "mean"),
        Orders=("Order ID", "nunique"),
        Rows=("Profit", "count")
    )
)

print("\n===== DOANH THU VÀ LỢI NHUẬN THEO NHÓM CHIẾT KHẤU =====")
print(discount_group_summary)

# Scatter chiết khấu và lợi nhuận
plt.figure(figsize=(8, 5))
plt.scatter(df["Discount"], df["Profit"], alpha=0.5)
plt.title("Mối quan hệ giữa chiết khấu và lợi nhuận")
plt.xlabel("Chiết khấu")
plt.ylabel("Lợi nhuận")
save_fig("hinh_3_13_moi_quan_he_chiet_khau_loi_nhuan.png")

# Lợi nhuận trung bình theo từng mức chiết khấu
plt.figure(figsize=(8, 5))
plt.plot(discount_profit["Discount"], discount_profit["Avg_Profit"], marker="o")
plt.title("Lợi nhuận trung bình theo mức chiết khấu")
plt.xlabel("Chiết khấu")
plt.ylabel("Lợi nhuận trung bình")
save_fig("hinh_3_14_loi_nhuan_trung_binh_theo_chiet_khau.png")

# Tổng lợi nhuận theo nhóm chiết khấu
plt.figure(figsize=(10, 5))
plt.bar(discount_group_summary["Discount Group"].astype(str), discount_group_summary["Profit"])
plt.title("Tổng lợi nhuận theo nhóm chiết khấu")
plt.xlabel("Nhóm chiết khấu")
plt.ylabel("Tổng lợi nhuận")
plt.xticks(rotation=30, ha="right")
save_fig("hinh_3_15_tong_loi_nhuan_theo_nhom_chiet_khau.png")

discount_corr = corr.loc["Discount", "Profit"]

print("\nNHẬN XÉT:")
print(
    f"Hệ số tương quan giữa chiết khấu và lợi nhuận là {discount_corr:.4f}. "
    "Khi phân tích theo nhóm chiết khấu, có thể quan sát rõ hơn xu hướng lợi nhuận thay đổi "
    "khi mức giảm giá tăng lên. Nếu các nhóm chiết khấu cao có lợi nhuận âm, điều đó cho thấy "
    "chính sách giảm giá cần được kiểm soát để tránh làm giảm hiệu quả sinh lời."
)

# 9. PHÂN TÍCH THÊM: THỜI GIAN GIAO HÀNG
print_title("9. PHÂN TÍCH THÊM: THỜI GIAN GIAO HÀNG")

shipping_summary = df["Shipping Days"].describe()

shipmode_summary = (
    df.groupby("Ship Mode", as_index=False)
    .agg(
        Avg_Shipping_Days=("Shipping Days", "mean"),
        Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    )
    .sort_values("Avg_Shipping_Days")
)

print("\n===== THỐNG KÊ THỜI GIAN GIAO HÀNG =====")
print(shipping_summary)

print("\n===== THỜI GIAN GIAO HÀNG THEO HÌNH THỨC VẬN CHUYỂN =====")
print(shipmode_summary)

# Vẽ phân bố thời gian giao hàng
plt.figure(figsize=(8, 5))
plt.hist(df["Shipping Days"].dropna(), bins=12, edgecolor="black")
plt.title("Phân bố thời gian giao hàng")
plt.xlabel("Số ngày giao hàng")
plt.ylabel("Tần suất")
save_fig("hinh_3_16_phan_bo_thoi_gian_giao_hang.png")

# Vẽ thời gian giao hàng trung bình theo Ship Mode
plt.figure(figsize=(9, 5))
plt.bar(shipmode_summary["Ship Mode"], shipmode_summary["Avg_Shipping_Days"])
plt.title("Thời gian giao hàng trung bình theo hình thức vận chuyển")
plt.xlabel("Hình thức vận chuyển")
plt.ylabel("Số ngày giao hàng trung bình")
plt.xticks(rotation=20, ha="right")
save_fig("hinh_3_17_thoi_gian_giao_hang_theo_ship_mode.png")

print("\nNHẬN XÉT:")
print(
    "Phân tích thời gian giao hàng giúp bổ sung góc nhìn về vận hành bán lẻ. "
    "Chỉ tiêu này không trực tiếp phản ánh doanh thu nhưng có thể ảnh hưởng đến trải nghiệm khách hàng "
    "và hiệu quả dịch vụ."
)

