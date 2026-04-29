import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce RFM Dashboard", layout="wide")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data.csv")
    return df

df = load_data()

# ======================
# DATA CLEANING
# ======================
if df.empty:
    st.error("Dataset kosong")
    st.stop()

# pastikan kolom penting ada
required_cols = [
    "order_purchase_timestamp",
    "customer_unique_id",
    "payment_value",
    "product_category_name_english"
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom {col} tidak ditemukan")
        st.stop()

# datetime
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"], errors="coerce"
)

# kategori aman (hindari float vs string error)
df["product_category_name_english"] = (
    df["product_category_name_english"]
    .astype(str)
    .replace("nan", None)
)

# ======================
# SIDEBAR FILTER
# ======================
st.sidebar.header("Filter")

years = sorted(df["order_purchase_timestamp"].dt.year.dropna().unique())
selected_year = st.sidebar.selectbox("Pilih Tahun", years)

all_cats = sorted(df["product_category_name_english"].dropna().unique())
selected_cats = st.sidebar.multiselect("Kategori Produk", all_cats)

# filter data
df_filtered = df[df["order_purchase_timestamp"].dt.year == selected_year]

if selected_cats:
    df_filtered = df_filtered[
        df_filtered["product_category_name_english"].isin(selected_cats)
    ]

if df_filtered.empty:
    st.warning("Data kosong setelah filter")
    st.stop()

# ======================
# KPI
# ======================
st.title("📊 E-Commerce RFM Dashboard")

total_revenue = df_filtered["payment_value"].sum()
total_orders = df_filtered.shape[0]
total_customers = df_filtered["customer_unique_id"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
col2.metric("Total Orders", total_orders)
col3.metric("Total Customers", total_customers)

# ======================
# TREND BULANAN
# ======================
st.subheader("Trend Revenue Bulanan")

df_filtered["month"] = df_filtered["order_purchase_timestamp"].dt.to_period("M").astype(str)

monthly = df_filtered.groupby("month")["payment_value"].sum().reset_index()

fig, ax = plt.subplots()
ax.plot(monthly["month"], monthly["payment_value"])
ax.set_xticks(range(len(monthly["month"])))
ax.set_xticklabels(monthly["month"], rotation=45)
ax.set_title("Revenue per Month")

st.pyplot(fig)

# ======================
# TOP KATEGORI
# ======================
st.subheader("Top Kategori Produk")

top_cat = (
    df_filtered.groupby("product_category_name_english")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

fig2, ax2 = plt.subplots()
top_cat.plot(kind="bar", ax=ax2)
ax2.set_title("Top 5 Category")

st.pyplot(fig2)

# ======================
# RFM ANALYSIS
# ======================
st.subheader("RFM Analysis")

snapshot_date = df_filtered["order_purchase_timestamp"].max()

rfm = df_filtered.groupby("customer_unique_id").agg({
    "order_purchase_timestamp": lambda x: (snapshot_date - x.max()).days,
    "customer_unique_id": "count",
    "payment_value": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# scoring
rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

rfm["RFM_score"] = (
    rfm["R_score"].astype(str) +
    rfm["F_score"].astype(str) +
    rfm["M_score"].astype(str)
)

# ======================
# HEATMAP RFM
# ======================
rfm_pivot = rfm.pivot_table(
    index="R_score",
    columns="F_score",
    values="Monetary",
    aggfunc="mean"
)

fig3, ax3 = plt.subplots()
sns.heatmap(rfm_pivot, annot=True, fmt=".0f", ax=ax3)

st.pyplot(fig3)

# ======================
# INSIGHT OTOMATIS
# ======================
st.subheader("Insight")

top_cat3 = top_cat.head(3)

if len(top_cat3) > 0:
    text = "💡 Top kategori:\n"
    for i in range(len(top_cat3)):
        text += f"- {top_cat3.index[i]} (R$ {top_cat3.values[i]:,.0f})\n"
    st.markdown(text)
else:
    st.write("Tidak ada data kategori")
