import streamlit as st
import pandas as pd
import os
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛒 E-Commerce Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "main_data.csv")
    df = pd.read_csv(path)

    # ===== WAJIB: feature engineering =====
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    df["year"] = df["order_purchase_timestamp"].dt.year
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Filter")

years = sorted(df["year"].dropna().unique())
selected_years = st.sidebar.multiselect("Tahun", years, default=years)

categories = sorted(df["product_category_name_english"].dropna().unique())
selected_cats = st.sidebar.multiselect("Kategori", categories)

# ================= FILTER =================
filtered_df = df.copy()

if selected_years:
    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]

if selected_cats:
    filtered_df = filtered_df[
        filtered_df["product_category_name_english"].isin(selected_cats)
    ]

# ===== GUARD =====
if filtered_df.empty:
    st.warning("Data kosong setelah filter")
    st.stop()

# ================= KPI =================
st.subheader("📊 KPI")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"R$ {filtered_df['payment_value'].sum():,.0f}")
col2.metric("Total Orders", filtered_df["order_id"].nunique())
col3.metric("Total Customers", filtered_df["customer_unique_id"].nunique())

# ================= TABS =================
tab1, tab2 = st.tabs(["Revenue", "RFM"])

# ================= TAB 1 =================
with tab1:
    st.subheader("Top Kategori")

    top_cat = (
        filtered_df.groupby("product_category_name_english")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    if top_cat.empty:
        st.warning("Tidak ada data kategori")
    else:
        fig = px.bar(
            x=top_cat.values,
            y=top_cat.index,
            orientation="h",
            title="Top 10 Kategori"
        )
        st.plotly_chart(fig, use_container_width=True)

    # SAFE INDEXING
    names = top_cat.index.tolist()
    top1 = names[0] if len(names) > 0 else "-"
    top2 = names[1] if len(names) > 1 else "-"
    top3 = names[2] if len(names) > 2 else "-"

    st.info(f"Top kategori: {top1}, {top2}, {top3}")

    # ===== TREND =====
    st.subheader("Trend Bulanan")

    monthly = (
        filtered_df.groupby("year_month")["payment_value"]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )

    fig2 = px.line(monthly, x="year_month", y="payment_value", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# ================= RFM =================
def compute_rfm(df):
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (ref - x.max()).days,
        "order_id": "nunique",
        "payment_value": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # FIX qcut
    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1], duplicates="drop")
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5], duplicates="drop")
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5], duplicates="drop")

    def segment(row):
        r, f = int(row["R"]), int(row["F"])
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal"
        elif r <= 2:
            return "At Risk"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(segment, axis=1)

    return rfm

# ================= TAB 2 =================
with tab2:
    st.subheader("RFM Segmentation")

    rfm = compute_rfm(filtered_df)

    seg = rfm["Segment"].value_counts()

    fig3 = px.bar(x=seg.index, y=seg.values, title="Distribusi Segmen")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(rfm.head())
