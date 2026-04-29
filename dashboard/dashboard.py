import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ── CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="Olist Dashboard",
    layout="wide"
)

# ── LOAD DATA ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv", parse_dates=["order_purchase_timestamp"])

    df["year"] = df["order_purchase_timestamp"].dt.year
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    return df

@st.cache_data
def compute_rfm(df):
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp", lambda x: (ref - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("payment_value", "sum")
    ).reset_index()

    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])

    def segment(row):
        r,f,m = int(row["R"]), int(row["F"]), int(row["M"])
        if r>=4 and f>=4 and m>=4: return "Champions"
        elif r>=3 and f>=3: return "Loyal"
        elif r>=4: return "New"
        elif r<=2 and f>=3: return "At Risk"
        else: return "Others"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm


df = load_data()

# ── SIDEBAR ────────────────────────────────────
st.sidebar.title("Filter")

years = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect("Tahun", years, default=years)

categories = sorted(df["product_category_name_english"].dropna().unique())
selected_cat = st.sidebar.multiselect("Kategori", categories)

# ── FILTER ─────────────────────────────────────
df = df[df["year"].isin(selected_years)]

if selected_cat:
    df = df[df["product_category_name_english"].isin(selected_cat)]

# ── HEADER ─────────────────────────────────────
st.title("E-Commerce Dashboard (Olist)")

# ── KPI ────────────────────────────────────────
total_rev = df["payment_value"].sum()
total_orders = df["order_id"].nunique()
avg_order = df.groupby("order_id")["payment_value"].sum().mean()
total_cust = df["customer_unique_id"].nunique()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Revenue", f"R$ {total_rev/1e6:.2f}M")
c2.metric("Orders", f"{total_orders:,}")
c3.metric("Avg Order", f"R$ {avg_order:,.0f}")
c4.metric("Customers", f"{total_cust:,}")

st.divider()

# ── TABS ───────────────────────────────────────
tab1, tab2 = st.tabs(["Revenue Analysis", "RFM Segmentation"])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    st.subheader("Revenue per Kategori")

    top_n = st.slider("Top N", 5, 20, 10)

    top_cat = (
        df.groupby("product_category_name_english")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    # BAR
    fig_bar = px.bar(
        top_cat,
        x="payment_value",
        y="product_category_name_english",
        orientation="h",
        text_auto=".2s"
    )

    fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))

    st.plotly_chart(fig_bar, use_container_width=True)

    # LINE
    monthly = (
        df.groupby("year_month")["payment_value"]
        .sum()
        .reset_index()
    )

    fig_line = px.line(
        monthly,
        x="year_month",
        y="payment_value",
        markers=True
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # INSIGHT DINAMIS
    top1 = top_cat.iloc[0]
    peak = monthly.loc[monthly["payment_value"].idxmax()]

    st.info(f"""
    Top kategori: **{top1['product_category_name_english']}**  
    Revenue: **R$ {top1['payment_value']/1e6:.2f}M**

    Peak revenue terjadi pada **{peak['year_month']}**
    """)

# =========================================================
# TAB 2
# =========================================================
with tab2:

    st.subheader("RFM Segmentation")

    rfm = compute_rfm(df)

    seg = rfm["Segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Count"]

    fig_seg = px.bar(seg, x="Segment", y="Count", color="Segment")
    st.plotly_chart(fig_seg, use_container_width=True)

    # Monetary
    monet = rfm.groupby("Segment")["Monetary"].mean().reset_index()

    fig_m = px.bar(monet, x="Segment", y="Monetary", color="Segment")
    st.plotly_chart(fig_m, use_container_width=True)

    # Insight
    biggest = seg.iloc[0]
    st.info(f"""
    Segmen terbesar adalah **{biggest['Segment']}**
    dengan jumlah **{biggest['Count']} pelanggan**.

    Ini menunjukkan distribusi customer belum optimal.
    """)
