import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Olist Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #457b9d;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #6c757d; font-weight: 500; }
    .metric-card h2 { margin: 4px 0 0 0; font-size: 26px; color: #212529; font-weight: 700; }
    .metric-card small { color: #6c757d; font-size: 11px; }
    .section-title {
        font-size: 18px; font-weight: 700; color: #212529;
        border-bottom: 2px solid #457b9d; padding-bottom: 6px; margin-bottom: 16px;
    }
    .insight-box {
        background: #e8f4f8; border-left: 4px solid #457b9d;
        border-radius: 6px; padding: 12px 16px; margin-top: 12px;
        font-size: 14px; color: #2c3e50;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")

COLOR_H    = "#82b7c6"
COLOR_MAIN = "#d3d3d3"
COLOR_RED  = "#e63946"
COLOR_BLUE = "#457b9d"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    candidates = [
        os.path.join(os.path.dirname(__file__), "main_data.csv"),
        "main_data.csv",
        "dashboard/main_data.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])
            # Kolom turunan — buat jika belum ada
            if "year" not in df.columns:
                df["year"] = df["order_purchase_timestamp"].dt.year
            if "month_year" not in df.columns:
                df["month_year"] = (
                    df["order_purchase_timestamp"].dt.to_period("M").astype(str)
                )
            if "purchase_hour" not in df.columns:
                df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
            if "shipping_ratio" not in df.columns and "freight_value" in df.columns and "price" in df.columns:
                df["shipping_ratio"] = (df["freight_value"] / df["price"]) * 100
            if "delivery_accuracy" not in df.columns:
                for c in ["order_estimated_delivery_date", "order_delivered_customer_date"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                if all(c in df.columns for c in
                       ["order_estimated_delivery_date", "order_delivered_customer_date"]):
                    df["delivery_accuracy"] = (
                        df["order_estimated_delivery_date"]
                        - df["order_delivered_customer_date"]
                    ).dt.days
            df.sort_values("order_purchase_timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    return None


@st.cache_data
def compute_rfm(df_hash):
    # df_hash diterima sebagai df (hash dilakukan streamlit secara internal)
    df = df_hash
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        Recency   = ("order_purchase_timestamp", lambda x: (ref - x.max()).days),
        Frequency = ("order_id",                 "nunique"),
        Monetary  = ("payment_value",             "sum"),
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"],  q=5, labels=[1,2,3,4,5])

    def segment(row):
        r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        elif r >= 3 and f >= 3:           return "Loyal Customers"
        elif r >= 4 and f <= 2:           return "New Customers"
        elif r <= 2 and f >= 3:           return "At Risk"
        elif r <= 2 and f <= 2:           return "Lost"
        else:                             return "Potential Loyalists"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    return rfm


# ── LOAD ──────────────────────────────────────────────────────────────────────
df_raw = load_data()

if df_raw is None:
    st.error(
        "⚠️ File `main_data.csv` tidak ditemukan. "
        "Pastikan file berada satu folder dengan `dashboard.py`."
    )
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Olist Dashboard")
    st.markdown("---")
    st.markdown("## 🔍 Filter Data")

    years = sorted(df_raw["year"].dropna().astype(int).unique())
    sel_years = st.multiselect(
        "Tahun", options=years,
        default=[y for y in years if y in [2017, 2018]],
        help="Pilih tahun yang ingin ditampilkan"
    )

    all_cats = sorted(df_raw["product_category_name_english"].dropna().unique())
    sel_cats = st.multiselect(
        "Kategori Produk (opsional)", options=all_cats, default=[],
        placeholder="Semua kategori",
        help="Kosongkan untuk menampilkan semua kategori"
    )

    st.markdown("---")
    st.markdown("""
**Sumber Data:**  
Brazilian E-Commerce Public Dataset by Olist

**Periode Data:**  
Sep 2016 – Okt 2018

**Dibuat oleh:**  
Elsa Ika Rahmani
    """)

# ── APPLY FILTER ──────────────────────────────────────────────────────────────
if not sel_years:
    sel_years = years

df = df_raw[df_raw["year"].isin(sel_years)].copy()
if sel_cats:
    df = df[df["product_category_name_english"].isin(sel_cats)]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🛒 E-Commerce Olist — Analisis Data Dashboard")
st.markdown(
    f"Menampilkan data **{df['order_purchase_timestamp'].min().strftime('%b %Y')}** "
    f"s/d **{df['order_purchase_timestamp'].max().strftime('%b %Y')}** · "
    f"Filter aktif: {', '.join(map(str, sel_years))} · "
    f"{'Semua kategori' if not sel_cats else f'{len(sel_cats)} kategori dipilih'}"
)
st.markdown("---")

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
total_rev     = df["payment_value"].sum()
total_orders  = df["order_id"].nunique()
avg_order_val = df.groupby("order_id")["payment_value"].sum().mean()
total_cust    = df["customer_unique_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <h3>💰 Total Revenue</h3>
        <h2>R$ {total_rev/1e6:.2f}M</h2>
        <small>Semua transaksi delivered</small>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <h3>🛍️ Total Orders</h3>
        <h2>{total_orders:,}</h2>
        <small>Order unik (status: delivered)</small>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <h3>🧾 Avg Order Value</h3>
        <h2>R$ {avg_order_val:,.0f}</h2>
        <small>Rata-rata per transaksi</small>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <h3>👤 Total Pelanggan</h3>
        <h2>{total_cust:,}</h2>
        <small>Pelanggan unik</small>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📦 Pertanyaan 1: Revenue per Kategori",
    "👥 Pertanyaan 2: Segmentasi RFM",
    "🔍 Eksplorasi Data",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — REVENUE PER KATEGORI & TREN BULANAN
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<p class="section-title">Analisis Revenue per Kategori Produk & Tren Bulanan</p>',
        unsafe_allow_html=True
    )

    _, col_slider = st.columns([1, 3])
    with col_slider:
        top_n = st.slider(
            "Tampilkan Top N kategori", min_value=5, max_value=20, value=10, step=1
        )

    top_cat = (
        df.groupby("product_category_name_english")["payment_value"]
        .sum().sort_values(ascending=False).head(top_n)
    )

    monthly = (
        df.groupby("month_year")["payment_value"]
        .sum().reset_index()
        .sort_values("month_year")
        .reset_index(drop=True)
    )

    period_str = (
        f"{sel_years[0]}–{sel_years[-1]}" if len(sel_years) > 1 else str(sel_years[0])
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Revenue Produk E-Commerce Olist | Periode: {period_str}",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Bar chart — top kategori
    ax1 = axes[0]
    bar_colors = [COLOR_RED if i == 0 else COLOR_BLUE for i in range(len(top_cat))]
    ax1.barh(
        top_cat.index[::-1], top_cat.values[::-1],
        color=bar_colors[::-1], edgecolor="white"
    )
    ax1.set_xlabel("Total Revenue (BRL)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Kategori Produk", fontsize=10, labelpad=8)
    ax1.set_title(
        f"Top {top_n} Kategori berdasarkan Revenue\n({period_str})",
        fontsize=11, fontweight="bold"
    )
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M")
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    for i, val in enumerate(top_cat.values[::-1]):
        ax1.text(
            val + top_cat.max() * 0.01, i,
            f"R$ {val/1e6:.2f}M", va="center", fontsize=7.5, color="#333"
        )
    top_patch   = mpatches.Patch(color=COLOR_RED,  label="Revenue tertinggi")
    other_patch = mpatches.Patch(color=COLOR_BLUE, label="Kategori lainnya")
    ax1.legend(handles=[top_patch, other_patch], fontsize=8, loc="lower right")
    ax1.text(
        0.01, -0.13,
        f"n = {df['order_id'].nunique():,} orders | "
        f"{df['product_category_name_english'].nunique()} kategori",
        transform=ax1.transAxes, fontsize=7.5, color="gray", style="italic"
    )

    # Line chart — tren bulanan
    ax2 = axes[1]
    x = range(len(monthly))
    ax2.plot(
        x, monthly["payment_value"], color=COLOR_BLUE,
        linewidth=2.5, marker="o", markersize=5
    )
    ax2.fill_between(x, monthly["payment_value"], alpha=0.12, color=COLOR_BLUE)
    if len(monthly) > 1:
        peak_idx = monthly["payment_value"].idxmax()
        ax2.annotate(
            f"Puncak: {monthly['month_year'].iloc[peak_idx]}",
            xy=(peak_idx, monthly["payment_value"].iloc[peak_idx]),
            xytext=(max(0, peak_idx - 3),
                    monthly["payment_value"].iloc[peak_idx] * 0.9),
            fontsize=8, color=COLOR_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=1.2)
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(monthly["month_year"], rotation=45, ha="right", fontsize=7.5)
    ax2.set_xlabel("Periode (Tahun-Bulan)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Total Revenue (BRL)", fontsize=10, labelpad=8)
    ax2.set_title(f"Tren Revenue Bulanan ({period_str})", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M")
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    peak_month = monthly.loc[monthly["payment_value"].idxmax(), "month_year"]
    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b>
    Kategori <b>{top_cat.index[0]}</b> menghasilkan revenue tertinggi sebesar
    <b>R$ {top_cat.values[0]/1e6:.2f}M</b>, diikuti <b>{top_cat.index[1]}</b> dan
    <b>{top_cat.index[2]}</b>.
    Tren bulanan menunjukkan pertumbuhan konsisten pada periode {period_str},
    dengan puncak revenue pada <b>{peak_month}</b>.
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Lihat Tabel Top Kategori"):
        tbl = top_cat.reset_index()
        tbl.columns = ["Kategori Produk", "Total Revenue (R$)"]
        tbl["Total Revenue (R$)"] = tbl["Total Revenue (R$)"].map(
            lambda x: f"R$ {x:,.0f}"
        )
        tbl.index = tbl.index + 1
        st.dataframe(tbl, use_container_width=True)

    with st.expander("📋 Lihat Tabel Tren Bulanan"):
        m_tbl = monthly.copy()
        m_tbl.columns = ["Periode", "Revenue (R$)"]
        m_tbl["Revenue (R$)"] = m_tbl["Revenue (R$)"].map(lambda x: f"R$ {x:,.0f}")
        m_tbl.index = m_tbl.index + 1
        st.dataframe(m_tbl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="section-title">Segmentasi Pelanggan Berdasarkan RFM Analysis</p>',
        unsafe_allow_html=True
    )

    rfm = compute_rfm(df)
    total_cust_rfm = len(rfm)

    SEG_ORDER  = ["Champions", "Loyal Customers", "Potential Loyalists",
                  "New Customers", "At Risk", "Lost"]
    SEG_COLORS = ["#2a9d8f", "#57cc99", "#a8dadc", "#457b9d", "#e9c46a", "#e76f51"]

    seg_count    = rfm["Segment"].value_counts().reindex(SEG_ORDER).fillna(0)
    avg_monetary = rfm.groupby("Segment")["Monetary"].mean().reindex(SEG_ORDER).fillna(0)

    # Distribusi + Avg Monetary
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        f"Segmentasi Pelanggan RFM | Total: {total_cust_rfm:,} pelanggan unik",
        fontsize=13, fontweight="bold", y=1.01
    )

    ax1 = axes2[0]
    bars1 = ax1.bar(
        SEG_ORDER, seg_count.values, color=SEG_COLORS, edgecolor="white", linewidth=0.8
    )
    ax1.set_title("Distribusi Jumlah Pelanggan per Segmen", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Segmen Pelanggan", fontsize=10, labelpad=8)
    ax1.set_ylabel("Jumlah Pelanggan", fontsize=10, labelpad=8)
    ax1.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K")
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, seg_count.values):
        pct = val / total_cust_rfm * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
            f"{int(val):,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333"
        )
    legend_els = [
        mpatches.Patch(facecolor=c, label=s) for s, c in zip(SEG_ORDER, SEG_COLORS)
    ]
    ax1.legend(
        handles=legend_els, fontsize=7.5, title="Segmen",
        title_fontsize=8, framealpha=0.7, loc="upper right"
    )

    ax2 = axes2[1]
    bars2 = ax2.bar(
        SEG_ORDER, avg_monetary.values, color=SEG_COLORS, edgecolor="white", linewidth=0.8
    )
    ax2.set_title(
        "Rata-rata Total Belanja (Monetary) per Segmen", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Segmen Pelanggan", fontsize=10, labelpad=8)
    ax2.set_ylabel("Rata-rata Total Belanja per Pelanggan (BRL)", fontsize=9, labelpad=8)
    ax2.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}")
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, avg_monetary.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"R$ {val:,.0f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#333"
        )

    fig2.text(
        0.5, -0.04,
        "Metodologi: Scoring RFM skala 1–5 menggunakan quintile. "
        "Champions = R≥4,F≥4,M≥4 | Loyal = R≥3,F≥3 | New = R≥4,F≤2 | "
        "At Risk = R≤2,F≥3 | Lost = R≤2,F≤2 | Lainnya = Potential Loyalists",
        ha="center", fontsize=7.5, color="gray", style="italic"
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Heatmap
    st.markdown("#### 🔥 RFM Heatmap — Rata-rata Monetary per Kombinasi R & F Score")
    rfm_pivot = rfm.pivot_table(
        index="R_Score", columns="F_Score", values="Monetary", aggfunc="mean"
    ).sort_index(ascending=False)

    annot_labels = rfm_pivot.map(
        lambda x: f"R${x:,.0f}" if not pd.isna(x) else "n/a"
    )

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        rfm_pivot, annot=annot_labels, fmt="", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax3,
        cbar_kws={"label": "Rata-rata Total Belanja (BRL)", "shrink": 0.8}
    )
    ax3.set_title(
        "Rata-rata Monetary berdasarkan R_Score vs F_Score\n"
        "Skala 1–5: semakin tinggi = semakin baik",
        fontsize=12, fontweight="bold", pad=12
    )
    ax3.set_xlabel(
        "F_Score — Frekuensi Pembelian\n(1=Paling jarang → 5=Paling sering)",
        fontsize=10, labelpad=10
    )
    ax3.set_ylabel(
        "R_Score — Recency (Kebaruan)\n(1=Paling lama → 5=Paling baru)",
        fontsize=10, labelpad=10
    )
    ax3.add_patch(plt.Rectangle((4, 0), 1, 1, fill=False, edgecolor=COLOR_RED, lw=3))
    ax3.text(
        4.5, 0.5, "Champions", ha="center", va="center",
        fontsize=8, color=COLOR_RED, fontweight="bold"
    )
    ax3.text(
        0, -0.18,
        f"n = {total_cust_rfm:,} pelanggan unik | "
        "Sel n/a = kombinasi skor tidak tersedia dalam data",
        transform=ax3.transAxes, fontsize=8, color="gray", style="italic"
    )
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Insight dinamis
    top_seg     = seg_count.idxmax()
    top_seg_pct = seg_count.max() / total_cust_rfm * 100
    champ_count = int(seg_count.get("Champions", 0))
    champ_pct   = champ_count / total_cust_rfm * 100
    champ_rev   = float(avg_monetary.get("Champions", 0))

    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b>
    Segmen terbesar adalah <b>{top_seg}</b> ({top_seg_pct:.1f}% dari total pelanggan),
    menunjukkan tingkat repeat purchase yang rendah.
    Segmen <b>Champions</b> hanya berjumlah <b>{champ_count:,} pelanggan ({champ_pct:.1f}%)</b>,
    namun memiliki rata-rata total belanja tertinggi sebesar <b>R$ {champ_rev:,.0f}</b> per pelanggan.
    Fokus retensi pada segmen At Risk dan Lost dapat memulihkan revenue yang berpotensi hilang.
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Lihat Tabel Ringkasan RFM per Segmen"):
        rfm_tbl = rfm.groupby("Segment").agg(
            Jumlah_Pelanggan = ("customer_unique_id", "count"),
            Avg_Recency_Hari = ("Recency",   "mean"),
            Avg_Frequency    = ("Frequency", "mean"),
            Avg_Monetary_BRL = ("Monetary",  "mean"),
        ).round(1).reindex(SEG_ORDER)
        rfm_tbl["% dari Total"] = (
            rfm_tbl["Jumlah_Pelanggan"] / total_cust_rfm * 100
        ).round(1)
        rfm_tbl["Avg_Monetary_BRL"] = rfm_tbl["Avg_Monetary_BRL"].map(
            lambda x: f"R$ {x:,.0f}"
        )
        st.dataframe(rfm_tbl, use_container_width=True)

    with st.expander("🔍 Filter Detail Pelanggan per Segmen"):
        sel_seg = st.selectbox("Pilih Segmen", SEG_ORDER)
        seg_df  = (
            rfm[rfm["Segment"] == sel_seg][
                ["customer_unique_id", "Recency", "Frequency", "Monetary", "Segment"]
            ]
            .sort_values("Monetary", ascending=False)
            .head(50)
        )
        seg_df["Monetary"] = seg_df["Monetary"].map(lambda x: f"R$ {x:,.2f}")
        st.dataframe(seg_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EKSPLORASI DATA
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-title">Eksplorasi Data Pendukung</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Metode Pembayaran
    with col1:
        st.markdown("**💳 Metode Pembayaran Populer**")
        pay_counts = df["payment_type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_H if i == 0 else COLOR_MAIN for i in range(len(pay_counts))]
        bars = ax.bar(pay_counts.index, pay_counts.values, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
        ax.set_title("Metode Pembayaran Populer", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tipe Pembayaran", fontsize=9)
        ax.set_ylabel("Jumlah Transaksi", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
        💡 Dominasi kartu kredit menunjukkan ketergantungan pada sistem kredit.
        Tawarkan promo cicilan 0% untuk meningkatkan nilai keranjang belanja.
        </div>""", unsafe_allow_html=True)

    # Aktivitas per Jam
    with col2:
        st.markdown("**⏰ Aktivitas Pembelian per Jam**")
        hour_counts = df["purchase_hour"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hour_counts.index, hour_counts.values,
                marker="o", color=COLOR_H, linewidth=2)
        ax.fill_between(hour_counts.index, hour_counts.values, color=COLOR_H, alpha=0.1)
        ax.set_title("Aktivitas Pembelian per Jam", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Jam (00–23)", fontsize=9)
        ax.set_ylabel("Jumlah Transaksi", fontsize=9)
        ax.set_xticks(range(0, 24))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
        💡 Pembelian memuncak di jam 10.00–22.00. Jadwalkan flash sale pada pukul 10.00
        dan 16.00 saat niat belanja paling tinggi.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # Top Kota
    with col3:
        st.markdown("**🏙️ Top 10 Kota dengan Transaksi Tertinggi**")
        top_cities = df["customer_city"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_H if i == 0 else COLOR_MAIN for i in range(10)]
        bars = ax.barh(
            top_cities.index[::-1], top_cities.values[::-1],
            color=colors[::-1], edgecolor="white"
        )
        ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
        ax.set_title("Top 10 Kota Konsumen", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Jumlah Transaksi", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
        💡 Sao Paulo mendominasi pasar. Fokuskan inventaris dan hub logistik di area ini
        untuk memangkas biaya pengiriman.
        </div>""", unsafe_allow_html=True)

    # Matriks Korelasi
    with col4:
        st.markdown("**📊 Matriks Korelasi Variabel Kunci**")
        corr_cols = [
            c for c in ["price", "freight_value", "product_weight_g", "review_score"]
            if c in df.columns
        ]
        df_corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            df_corr, annot=True, cmap="RdBu", center=0, fmt=".2f",
            cbar=False, ax=ax, linewidths=0.5
        )
        ax.set_title(
            "Korelasi Finansial & Fisik", loc="left", fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""<div class="insight-box">
        💡 Korelasi kuat (0.61) antara berat produk dan ongkir. Optimasi logistik
        untuk barang berat dapat menekan biaya secara signifikan.
        </div>""", unsafe_allow_html=True)

    # Preview data
    st.markdown("---")
    st.markdown("### 📄 Preview Data Utama")
    n_rows = st.slider(
        "Jumlah baris yang ditampilkan", 5, 50, 10, key="preview_slider"
    )
    st.dataframe(df.head(n_rows), use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:13px; padding:10px 0">
    <b>E-Commerce Olist Dashboard Analysis with RFM</b> · Proyek Analisis Data Dicoding ·
    Sumber: <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank">
    Kaggle — Brazilian E-Commerce Public Dataset by Olist</a>
</div>
""", unsafe_allow_html=True)
