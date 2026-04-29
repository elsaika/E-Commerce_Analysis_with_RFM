import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

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

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "main_data.csv"), parse_dates=["order_purchase_timestamp"])
    df["year"]       = df["order_purchase_timestamp"].dt.year
    df["month"]      = df["order_purchase_timestamp"].dt.month
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df

@st.cache_data
def compute_rfm(df):
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        Recency   = ("order_purchase_timestamp", lambda x: (ref - x.max()).days),
        Frequency = ("order_id",                 "nunique"),
        Monetary  = ("payment_value",             "sum")
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

df_raw = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Olist_logo.png/320px-Olist_logo.png",
             use_container_width=True)
    st.markdown("## 🔍 Filter Data")

    years = sorted(df_raw["year"].unique())
    sel_years = st.multiselect(
        "Tahun", options=years, default=[2017, 2018],
        help="Pilih tahun yang ingin ditampilkan"
    )

    all_cats = sorted(df_raw["product_category_name_english"].dropna().unique())
    sel_cats = st.multiselect(
        "Kategori Produk (opsional)",
        options=all_cats,
        default=[],
        placeholder="Semua kategori",
        help="Kosongkan untuk menampilkan semua kategori"
    )

    st.markdown("---")
    st.markdown("""
    **📊 Sumber Data**
    Brazilian E-Commerce Public Dataset by Olist (Kaggle)

    **📅 Periode Data**
    Sep 2016 – Okt 2018

    **👤 Dibuat oleh**
    [Nama Mahasiswa]
    """)

# ── FILTER DATA ───────────────────────────────────────────────────────────────
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
total_rev      = df["payment_value"].sum()
total_orders   = df["order_id"].nunique()
avg_order_val  = df.groupby("order_id")["payment_value"].sum().mean()
total_cust     = df["customer_unique_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <h3>💰 Total Revenue</h3>
        <h2>R$ {total_rev/1e6:.2f}M</h2>
        <small>Semua transaksi delivered</small>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <h3>📦 Total Orders</h3>
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
        <h3>👥 Total Pelanggan</h3>
        <h2>{total_cust:,}</h2>
        <small>Pelanggan unik</small>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📦 Pertanyaan 1 — Revenue per Kategori",
    "👥 Pertanyaan 2 — Segmentasi RFM"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — REVENUE PER KATEGORI & TREN BULANAN
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Analisis Revenue per Kategori Produk & Tren Bulanan</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        top_n = st.slider("Tampilkan Top N kategori", min_value=5, max_value=20, value=10, step=1)

    # ── Data prep ────────────────────────────────────────────────────────────
    top_cat = (
        df.groupby("product_category_name_english")["payment_value"]
        .sum().sort_values(ascending=False).head(top_n)
    )

    monthly = (
        df.groupby("year_month")["payment_value"]
        .sum().reset_index()
        .sort_values("year_month")
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    period_str = f"{sel_years[0]}–{sel_years[-1]}" if len(sel_years) > 1 else str(sel_years[0])
    fig.suptitle(
        f"Revenue Produk E-Commerce Olist | Periode: {period_str}",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Bar chart
    ax1 = axes[0]
    bar_colors = ["#e63946" if i == 0 else "#457b9d" for i in range(len(top_cat))]
    ax1.barh(top_cat.index[::-1], top_cat.values[::-1], color=bar_colors[::-1], edgecolor="white")
    ax1.set_xlabel("Total Revenue (BRL)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Kategori Produk", fontsize=10, labelpad=8)
    ax1.set_title(f"Top {top_n} Kategori berdasarkan Revenue\n({period_str})",
                  fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    # Label nilai
    for i, (val, idx) in enumerate(zip(top_cat.values[::-1], top_cat.index[::-1])):
        ax1.text(val + top_cat.max() * 0.01, i, f"R$ {val/1e6:.2f}M",
                 va="center", fontsize=7.5, color="#333")
    top_patch = mpatches.Patch(color="#e63946", label="Revenue tertinggi")
    other_patch = mpatches.Patch(color="#457b9d", label="Kategori lainnya")
    ax1.legend(handles=[top_patch, other_patch], fontsize=8, loc="lower right")
    ax1.text(0.01, -0.13,
             f"n = {df['order_id'].nunique():,} orders | {df['product_category_name_english'].nunique()} kategori",
             transform=ax1.transAxes, fontsize=7.5, color="gray", style="italic")

    # Line chart
    ax2 = axes[1]
    x = range(len(monthly))
    ax2.plot(x, monthly["payment_value"], color="#457b9d",
             linewidth=2.5, marker="o", markersize=5)
    ax2.fill_between(x, monthly["payment_value"], alpha=0.12, color="#457b9d")
    if len(monthly) > 1:
        peak_idx = monthly["payment_value"].idxmax()
        ax2.annotate(
            f"Puncak: {monthly['year_month'].iloc[peak_idx]}",
            xy=(peak_idx, monthly["payment_value"].iloc[peak_idx]),
            xytext=(max(0, peak_idx - 3), monthly["payment_value"].iloc[peak_idx] * 0.9),
            fontsize=8, color="#e63946", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.2)
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=7.5)
    ax2.set_xlabel("Periode (Tahun-Bulan)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Total Revenue (BRL)", fontsize=10, labelpad=8)
    ax2.set_title(f"Tren Revenue Bulanan ({period_str})", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b>
    Kategori <b>{top_cat.index[0]}</b> menghasilkan revenue tertinggi sebesar
    <b>R$ {top_cat.values[0]/1e6:.2f}M</b>, diikuti <b>{top_cat.index[1]}</b> dan
    <b>{top_cat.index[2]}</b>.
    Tren bulanan menunjukkan pertumbuhan konsisten pada periode {period_str},
    dengan puncak revenue pada <b>{monthly.loc[monthly['payment_value'].idxmax(), 'year_month']}</b>.
    </div>""", unsafe_allow_html=True)

    # Tabel ringkasan
    with st.expander("📋 Lihat Tabel Top Kategori"):
        tbl = top_cat.reset_index()
        tbl.columns = ["Kategori Produk", "Total Revenue (R$)"]
        tbl["Total Revenue (R$)"] = tbl["Total Revenue (R$)"].map(lambda x: f"R$ {x:,.0f}")
        tbl.index = tbl.index + 1
        st.dataframe(tbl, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Segmentasi Pelanggan Berdasarkan RFM Analysis</p>',
                unsafe_allow_html=True)

    rfm = compute_rfm(df)
    total_cust_rfm = len(rfm)

    SEG_ORDER  = ["Champions", "Loyal Customers", "Potential Loyalists",
                  "New Customers", "At Risk", "Lost"]
    SEG_COLORS = ["#2a9d8f", "#57cc99", "#a8dadc", "#457b9d", "#e9c46a", "#e76f51"]

    seg_count    = rfm["Segment"].value_counts().reindex(SEG_ORDER).fillna(0)
    avg_monetary = rfm.groupby("Segment")["Monetary"].mean().reindex(SEG_ORDER).fillna(0)
    avg_recency  = rfm.groupby("Segment")["Recency"].mean().reindex(SEG_ORDER).fillna(0)

    # ── Distribusi + Monetary ────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        f"Segmentasi Pelanggan RFM | Total: {total_cust_rfm:,} pelanggan unik",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Bar distribusi
    ax1 = axes2[0]
    bars1 = ax1.bar(SEG_ORDER, seg_count.values, color=SEG_COLORS,
                    edgecolor="white", linewidth=0.8)
    ax1.set_title("Distribusi Jumlah Pelanggan per Segmen", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Segmen Pelanggan", fontsize=10, labelpad=8)
    ax1.set_ylabel("Jumlah Pelanggan", fontsize=10, labelpad=8)
    ax1.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, seg_count.values):
        pct = val / total_cust_rfm * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                 f"{int(val):,}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333")
    legend_els = [mpatches.Patch(facecolor=c, label=s)
                  for s, c in zip(SEG_ORDER, SEG_COLORS)]
    ax1.legend(handles=legend_els, fontsize=7.5, title="Segmen",
               title_fontsize=8, framealpha=0.7, loc="upper right")

    # Bar avg monetary
    ax2 = axes2[1]
    bars2 = ax2.bar(SEG_ORDER, avg_monetary.values, color=SEG_COLORS,
                    edgecolor="white", linewidth=0.8)
    ax2.set_title("Rata-rata Total Belanja (Monetary) per Segmen", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Segmen Pelanggan", fontsize=10, labelpad=8)
    ax2.set_ylabel("Rata-rata Total Belanja per Pelanggan (BRL)", fontsize=9, labelpad=8)
    ax2.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, avg_monetary.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"R$ {val:,.0f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#333")

    fig2.text(0.5, -0.04,
              "Metodologi: Scoring RFM skala 1–5 menggunakan quintile. "
              "Champions = R≥4,F≥4,M≥4 | Loyal = R≥3,F≥3 | New = R≥4,F≤2 | "
              "At Risk = R≤2,F≥3 | Lost = R≤2,F≤2 | Lainnya = Potential Loyalists",
              ha="center", fontsize=7.5, color="gray", style="italic")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown("#### 🔥 RFM Heatmap — Rata-rata Monetary per Kombinasi R & F Score")
    rfm_pivot = rfm.pivot_table(
        index="R_Score", columns="F_Score",
        values="Monetary", aggfunc="mean"
    ).sort_index(ascending=False)

    annot_labels = rfm_pivot.applymap(
        lambda x: f"R${x:,.0f}" if not pd.isna(x) else "n/a"
    )

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(rfm_pivot, annot=annot_labels, fmt="",
                cmap="YlOrRd", linewidths=0.5, linecolor="white",
                ax=ax3, cbar_kws={"label": "Rata-rata Total Belanja (BRL)", "shrink": 0.8})
    ax3.set_title(
        "Rata-rata Monetary berdasarkan R_Score vs F_Score\n"
        "Skala 1–5: semakin tinggi = semakin baik",
        fontsize=12, fontweight="bold", pad=12
    )
    ax3.set_xlabel("F_Score — Frekuensi Pembelian\n(1=Paling jarang → 5=Paling sering)",
                   fontsize=10, labelpad=10)
    ax3.set_ylabel("R_Score — Recency (Kebaruan)\n(1=Paling lama → 5=Paling baru)",
                   fontsize=10, labelpad=10)
    ax3.add_patch(plt.Rectangle((4, 0), 1, 1, fill=False,
                                edgecolor="#e63946", lw=3))
    ax3.text(4.5, 0.5, "Champions", ha="center", va="center",
             fontsize=8, color="#e63946", fontweight="bold")
    ax3.text(0, -0.18,
             f"n = {total_cust_rfm:,} pelanggan unik | "
             "Sel n/a = kombinasi skor tidak tersedia dalam data",
             transform=ax3.transAxes, fontsize=8, color="gray", style="italic")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Insight
    top_seg      = seg_count.idxmax()
    top_seg_pct  = seg_count.max() / total_cust_rfm * 100
    champ_count  = int(seg_count.get("Champions", 0))
    champ_pct    = champ_count / total_cust_rfm * 100
    champ_rev    = float(avg_monetary.get("Champions", 0))

    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b>
    Segmen terbesar adalah <b>{top_seg}</b> ({top_seg_pct:.1f}% dari total pelanggan),
    menunjukkan tingkat repeat purchase yang rendah.
    Segmen <b>Champions</b> hanya berjumlah <b>{champ_count:,} pelanggan ({champ_pct:.1f}%)</b>,
    namun memiliki rata-rata total belanja tertinggi sebesar <b>R$ {champ_rev:,.0f}</b> per pelanggan.
    Fokus retensi pada segmen At Risk dan Lost dapat memulihkan revenue yang berpotensi hilang.
    </div>""", unsafe_allow_html=True)

    # Tabel RFM ringkasan
    with st.expander("📋 Lihat Tabel Ringkasan RFM per Segmen"):
        rfm_tbl = rfm.groupby("Segment").agg(
            Jumlah_Pelanggan  = ("customer_unique_id", "count"),
            Avg_Recency_Hari  = ("Recency",   "mean"),
            Avg_Frequency     = ("Frequency", "mean"),
            Avg_Monetary_BRL  = ("Monetary",  "mean")
        ).round(1).reindex(SEG_ORDER)
        rfm_tbl["% dari Total"] = (rfm_tbl["Jumlah_Pelanggan"] / total_cust_rfm * 100).round(1)
        rfm_tbl["Avg_Monetary_BRL"] = rfm_tbl["Avg_Monetary_BRL"].map(lambda x: f"R$ {x:,.0f}")
        st.dataframe(rfm_tbl, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:13px; padding:10px 0">
    📊 <b>E-Commerce Olist Dashboard</b> · Proyek Analisis Data Dicoding ·
    Sumber: <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank">
    Kaggle — Brazilian E-Commerce Public Dataset by Olist</a>
</div>
""", unsafe_allow_html=True)
