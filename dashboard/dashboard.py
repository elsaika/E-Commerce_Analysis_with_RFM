import os, gc
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

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #457b9d;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-card h3 { margin:0; font-size:13px; color:#6c757d; font-weight:500; }
    .metric-card h2 { margin:4px 0 0 0; font-size:26px; color:#212529; font-weight:700; }
    .metric-card small { color:#6c757d; font-size:11px; }
    .section-title {
        font-size:18px; font-weight:700; color:#212529;
        border-bottom:2px solid #457b9d; padding-bottom:6px; margin-bottom:16px;
    }
    .insight-box {
        background:#e8f4f8; border-left:4px solid #457b9d;
        border-radius:6px; padding:12px 16px; margin-top:12px;
        font-size:14px; color:#2c3e50;
    }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams["font.family"] = "sans-serif"
sns.set_style("whitegrid")
COLOR_H    = "#82b7c6"
COLOR_MAIN = "#d3d3d3"
COLOR_RED  = "#e63946"
COLOR_BLUE = "#457b9d"

# ── KOLOM YANG BENAR-BENAR DIPAKAI ───────────────────────────────────────────
KEEP_COLS = [
    "order_id", "order_purchase_timestamp",
    "payment_value", "payment_type",
    "price", "freight_value", "product_weight_g", "review_score",
    "product_category_name_english",
    "customer_unique_id", "customer_city",
    # kolom turunan yang mungkin sudah ada di CSV
    "year", "month_year", "purchase_hour",
]

# ── LOAD DATA — hanya baca kolom yang dipakai, downcast tipe ─────────────────
@st.cache_data(show_spinner="Memuat data…")
def load_data() -> pd.DataFrame:
    candidates = ["main_data.csv", "dashboard/main_data.csv"]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.insert(0, os.path.join(here, "main_data.csv"))
    except NameError:
        pass

    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    # Baca hanya kolom yang ada & dipakai
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use_cols = [c for c in KEEP_COLS if c in header]

    df = pd.read_csv(
        path,
        usecols=use_cols,
        parse_dates=["order_purchase_timestamp"],
        dtype={
            "payment_type": "category",
            "product_category_name_english": "category",
            "customer_city": "category",
            "customer_unique_id": "string",
            "order_id": "string",
        }
    )

    # Kolom turunan ringan
    if "year" not in df.columns:
        df["year"] = df["order_purchase_timestamp"].dt.year.astype("int16")
    if "month_year" not in df.columns:
        df["month_year"] = (
            df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        )
    if "purchase_hour" not in df.columns:
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour.astype("int8")

    # Downcast numerik
    for col in ["price", "freight_value", "product_weight_g",
                "payment_value", "review_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    df.sort_values("order_purchase_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    gc.collect()
    return df


@st.cache_data(show_spinner="Menghitung RFM…")
def compute_rfm(_df: pd.DataFrame, year_filter: tuple) -> pd.DataFrame:
    """
    Terima DataFrame yang sudah difilter.
    Underscore pada _df mencegah streamlit mencoba hash-nya
    (DataFrame besar bisa lambat di-hash).
    """
    sub = _df[_df["year"].isin(year_filter)][
        ["customer_unique_id", "order_id",
         "order_purchase_timestamp", "payment_value"]
    ].copy()

    ref = sub["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = sub.groupby("customer_unique_id").agg(
        Recency   = ("order_purchase_timestamp", lambda x: (ref - x.max()).days),
        Frequency = ("order_id",                 "nunique"),
        Monetary  = ("payment_value",             "sum"),
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]
    )
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1,2,3,4,5])

    def segment(row):
        r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        elif r >= 3 and f >= 3:           return "Loyal Customers"
        elif r >= 4 and f <= 2:           return "New Customers"
        elif r <= 2 and f >= 3:           return "At Risk"
        elif r <= 2 and f <= 2:           return "Lost"
        else:                             return "Potential Loyalists"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    del sub
    gc.collect()
    return rfm


# ── GUARD ────────────────────────────────────────────────────────────────────
df_raw = load_data()
if df_raw is None:
    st.error(
        "⚠️ File `main_data.csv` tidak ditemukan.\n\n"
        "Pastikan file berada **satu folder** dengan `dashboard.py`, lalu push ulang ke GitHub."
    )
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
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
        placeholder="Semua kategori"
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

# ── FILTER ────────────────────────────────────────────────────────────────────
if not sel_years:
    sel_years = years

mask = df_raw["year"].isin(sel_years)
if sel_cats:
    mask &= df_raw["product_category_name_english"].isin(sel_cats)
df = df_raw[mask]   # view, bukan copy — hemat memori

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🛒 E-Commerce Olist — Analisis Data Dashboard")
date_min = df["order_purchase_timestamp"].min().strftime("%b %Y")
date_max = df["order_purchase_timestamp"].max().strftime("%b %Y")
cat_label = "Semua kategori" if not sel_cats else f"{len(sel_cats)} kategori dipilih"
st.markdown(
    f"Menampilkan data **{date_min}** s/d **{date_max}** · "
    f"Filter aktif: {', '.join(map(str, sel_years))} · {cat_label}"
)
st.markdown("---")

# ── KPI ───────────────────────────────────────────────────────────────────────
total_rev     = float(df["payment_value"].sum())
total_orders  = df["order_id"].nunique()
avg_order_val = float(df.groupby("order_id")["payment_value"].sum().mean())
total_cust    = df["customer_unique_id"].nunique()

cols = st.columns(4)
for col, title, val, sub in [
    (cols[0], "💰 Total Revenue",   f"R$ {total_rev/1e6:.2f}M",   "Semua transaksi"),
    (cols[1], "🛍️ Total Orders",    f"{total_orders:,}",           "Order unik"),
    (cols[2], "🧾 Avg Order Value", f"R$ {avg_order_val:,.0f}",    "Rata-rata per transaksi"),
    (cols[3], "👤 Total Pelanggan", f"{total_cust:,}",             "Pelanggan unik"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <h3>{title}</h3><h2>{val}</h2><small>{sub}</small>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📦 Pertanyaan 1: Revenue per Kategori",
    "👥 Pertanyaan 2: Segmentasi RFM",
    "🔍 Eksplorasi Data",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<p class="section-title">Analisis Revenue per Kategori Produk & Tren Bulanan</p>',
        unsafe_allow_html=True
    )
    _, col_sl = st.columns([1, 3])
    with col_sl:
        top_n = st.slider("Tampilkan Top N kategori", 5, 20, 10, step=1)

    # Agregasi kecil — buang df asli segera
    top_cat = (
        df.groupby("product_category_name_english", observed=True)["payment_value"]
        .sum().sort_values(ascending=False).head(top_n)
    )
    monthly = (
        df.groupby("month_year")["payment_value"]
        .sum().reset_index()
        .sort_values("month_year")
        .reset_index(drop=True)
    )
    period_str = f"{sel_years[0]}–{sel_years[-1]}" if len(sel_years) > 1 else str(sel_years[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Revenue E-Commerce Olist | {period_str}",
                 fontsize=13, fontweight="bold", y=1.01)

    # Bar
    bar_colors = [COLOR_RED if i == 0 else COLOR_BLUE for i in range(len(top_cat))]
    ax1.barh(top_cat.index[::-1], top_cat.values[::-1],
             color=bar_colors[::-1], edgecolor="white")
    ax1.set_xlabel("Total Revenue (BRL)", fontsize=10)
    ax1.set_title(f"Top {top_n} Kategori | {period_str}", fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
    for i, v in enumerate(top_cat.values[::-1]):
        ax1.text(v + top_cat.max()*0.01, i, f"R${v/1e6:.2f}M", va="center", fontsize=7.5)
    ax1.legend(handles=[
        mpatches.Patch(color=COLOR_RED,  label="Tertinggi"),
        mpatches.Patch(color=COLOR_BLUE, label="Lainnya"),
    ], fontsize=8, loc="lower right")

    # Line
    x = range(len(monthly))
    ax2.plot(x, monthly["payment_value"], color=COLOR_BLUE, linewidth=2.5, marker="o", markersize=5)
    ax2.fill_between(x, monthly["payment_value"], alpha=0.12, color=COLOR_BLUE)
    if len(monthly) > 1:
        pk = int(monthly["payment_value"].idxmax())
        ax2.annotate(
            f"Puncak: {monthly['month_year'].iloc[pk]}",
            xy=(pk, monthly["payment_value"].iloc[pk]),
            xytext=(max(0, pk-3), monthly["payment_value"].iloc[pk]*0.88),
            fontsize=8, color=COLOR_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=1.2)
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(monthly["month_year"].tolist(), rotation=45, ha="right", fontsize=7.5)
    ax2.set_title(f"Tren Revenue Bulanan ({period_str})", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M"))
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4); ax2.set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig); gc.collect()

    peak_month = monthly.loc[monthly["payment_value"].idxmax(), "month_year"]
    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b> Kategori <b>{top_cat.index[0]}</b> menghasilkan revenue tertinggi
    <b>R$ {top_cat.values[0]/1e6:.2f}M</b>, diikuti <b>{top_cat.index[1]}</b> dan
    <b>{top_cat.index[2]}</b>. Tren menunjukkan pertumbuhan konsisten dengan puncak pada
    <b>{peak_month}</b>.
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Tabel Top Kategori"):
        tbl = top_cat.reset_index()
        tbl.columns = ["Kategori", "Revenue (R$)"]
        tbl["Revenue (R$)"] = tbl["Revenue (R$)"].map(lambda x: f"R$ {x:,.0f}")
        tbl.index += 1
        st.dataframe(tbl, use_container_width=True)

    with st.expander("📋 Tabel Tren Bulanan"):
        m = monthly.copy(); m.columns = ["Periode","Revenue (R$)"]
        m["Revenue (R$)"] = m["Revenue (R$)"].map(lambda x: f"R$ {x:,.0f}")
        m.index += 1
        st.dataframe(m, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="section-title">Segmentasi Pelanggan Berdasarkan RFM Analysis</p>',
        unsafe_allow_html=True
    )

    rfm = compute_rfm(df_raw, tuple(sorted(sel_years)))
    N   = len(rfm)

    SEG_ORDER  = ["Champions","Loyal Customers","Potential Loyalists",
                  "New Customers","At Risk","Lost"]
    SEG_COLORS = ["#2a9d8f","#57cc99","#a8dadc","#457b9d","#e9c46a","#e76f51"]

    seg_count    = rfm["Segment"].value_counts().reindex(SEG_ORDER).fillna(0)
    avg_monetary = rfm.groupby("Segment")["Monetary"].mean().reindex(SEG_ORDER).fillna(0)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    fig2.suptitle(f"Segmentasi RFM | {N:,} pelanggan unik",
                  fontsize=13, fontweight="bold", y=1.01)

    b1 = a1.bar(SEG_ORDER, seg_count.values, color=SEG_COLORS, edgecolor="white")
    a1.set_title("Distribusi Jumlah Pelanggan per Segmen", fontsize=11, fontweight="bold")
    a1.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    a1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    a1.spines[["top","right"]].set_visible(False)
    a1.grid(axis="y", linestyle="--", alpha=0.4); a1.set_axisbelow(True)
    for bar, val in zip(b1, seg_count.values):
        pct = val / N * 100
        a1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150,
                f"{int(val):,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    a1.legend(handles=[mpatches.Patch(facecolor=c, label=s)
                       for s, c in zip(SEG_ORDER, SEG_COLORS)],
              fontsize=7.5, title="Segmen", framealpha=0.7, loc="upper right")

    b2 = a2.bar(SEG_ORDER, avg_monetary.values, color=SEG_COLORS, edgecolor="white")
    a2.set_title("Rata-rata Monetary per Segmen", fontsize=11, fontweight="bold")
    a2.set_xticklabels(SEG_ORDER, rotation=25, ha="right", fontsize=9)
    a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    a2.spines[["top","right"]].set_visible(False)
    a2.grid(axis="y", linestyle="--", alpha=0.4); a2.set_axisbelow(True)
    for bar, val in zip(b2, avg_monetary.values):
        a2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"R${val:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig2.text(0.5, -0.05,
              "Scoring quintile 1–5. Champions=R≥4,F≥4,M≥4 | Loyal=R≥3,F≥3 | "
              "New=R≥4,F≤2 | At Risk=R≤2,F≥3 | Lost=R≤2,F≤2 | else=Potential Loyalists",
              ha="center", fontsize=7.5, color="gray", style="italic")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2); gc.collect()

    # Heatmap
    st.markdown("#### 🔥 RFM Heatmap — Rata-rata Monetary per R × F Score")
    rfm_pivot = rfm.pivot_table(
        index="R_Score", columns="F_Score", values="Monetary", aggfunc="mean"
    ).sort_index(ascending=False)
    annot = rfm_pivot.map(lambda x: f"R${x:,.0f}" if not pd.isna(x) else "n/a")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    sns.heatmap(rfm_pivot, annot=annot, fmt="", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax3,
                cbar_kws={"label":"Avg Monetary (BRL)","shrink":0.8})
    ax3.set_title("Avg Monetary: R_Score × F_Score\n(semakin tinggi = semakin baik)",
                  fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("F_Score (1=jarang → 5=sering)", fontsize=9, labelpad=8)
    ax3.set_ylabel("R_Score (1=lama → 5=baru)", fontsize=9, labelpad=8)
    ax3.add_patch(plt.Rectangle((4, 0), 1, 1, fill=False, edgecolor=COLOR_RED, lw=3))
    ax3.text(4.5, 0.5, "Champions", ha="center", va="center",
             fontsize=8, color=COLOR_RED, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3); gc.collect()

    top_seg     = seg_count.idxmax()
    top_seg_pct = seg_count.max() / N * 100
    champ_n     = int(seg_count.get("Champions", 0))
    champ_pct   = champ_n / N * 100
    champ_rev   = float(avg_monetary.get("Champions", 0))

    st.markdown(f"""<div class="insight-box">
    💡 <b>Insight:</b> Segmen terbesar adalah <b>{top_seg}</b> ({top_seg_pct:.1f}%).
    <b>Champions</b> hanya <b>{champ_n:,} pelanggan ({champ_pct:.1f}%)</b> namun
    avg monetary tertinggi <b>R$ {champ_rev:,.0f}</b>.
    Prioritaskan retensi <b>At Risk</b> dan <b>Lost</b> untuk memulihkan revenue.
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Tabel Ringkasan RFM per Segmen"):
        tbl2 = rfm.groupby("Segment").agg(
            Jumlah   = ("customer_unique_id","count"),
            Recency  = ("Recency","mean"),
            Freq     = ("Frequency","mean"),
            Monetary = ("Monetary","mean"),
        ).round(1).reindex(SEG_ORDER)
        tbl2["% Total"] = (tbl2["Jumlah"] / N * 100).round(1)
        tbl2["Monetary"] = tbl2["Monetary"].map(lambda x: f"R$ {x:,.0f}")
        st.dataframe(tbl2, use_container_width=True)

    with st.expander("🔍 Detail Pelanggan per Segmen"):
        sel_seg = st.selectbox("Pilih Segmen", SEG_ORDER)
        detail  = (rfm[rfm["Segment"]==sel_seg]
                   [["customer_unique_id","Recency","Frequency","Monetary"]]
                   .sort_values("Monetary", ascending=False).head(50).copy())
        detail["Monetary"] = detail["Monetary"].map(lambda x: f"R$ {x:,.2f}")
        st.dataframe(detail, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-title">Eksplorasi Data Pendukung</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💳 Metode Pembayaran**")
        pay = df["payment_type"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [COLOR_H if i==0 else COLOR_MAIN for i in range(len(pay))]
        b = ax.bar(pay.index, pay.values, color=colors, edgecolor="white")
        ax.bar_label(b, fmt="{:,.0f}", padding=3, fontsize=8)
        ax.set_title("Metode Pembayaran", loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig); gc.collect()
        st.markdown("""<div class="insight-box">
        💡 Kartu kredit mendominasi. Promo cicilan 0% dapat meningkatkan AOV.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**⏰ Aktivitas per Jam**")
        hr = df["purchase_hour"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(hr.index, hr.values, marker="o", color=COLOR_H, linewidth=2)
        ax.fill_between(hr.index, hr.values, color=COLOR_H, alpha=0.1)
        ax.set_title("Aktivitas per Jam", loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(range(0, 24)); ax.set_xlabel("Jam (00–23)", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig); gc.collect()
        st.markdown("""<div class="insight-box">
        💡 Puncak jam 10.00–22.00. Jadwalkan flash sale pukul 10.00 & 16.00.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🏙️ Top 10 Kota**")
        cities = df["customer_city"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [COLOR_H if i==0 else COLOR_MAIN for i in range(10)]
        b = ax.barh(cities.index[::-1], cities.values[::-1], color=colors[::-1], edgecolor="white")
        ax.bar_label(b, fmt="{:,.0f}", padding=3, fontsize=8)
        ax.set_title("Top 10 Kota", loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig); gc.collect()
        st.markdown("""<div class="insight-box">
        💡 Sao Paulo mendominasi. Fokuskan hub logistik di sana untuk efisiensi biaya.
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("**📊 Matriks Korelasi**")
        corr_cols = [c for c in ["price","freight_value","product_weight_g","review_score"]
                     if c in df.columns]
        if len(corr_cols) >= 2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap="RdBu", center=0,
                        fmt=".2f", cbar=False, ax=ax, linewidths=0.5)
            ax.set_title("Korelasi Variabel Kunci", loc="left", fontsize=11, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig); gc.collect()
            st.markdown("""<div class="insight-box">
            💡 Korelasi kuat antara berat produk & ongkir. Optimasi packaging dapat menekan biaya.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Preview Data")
    n = st.slider("Jumlah baris", 5, 50, 10, key="preview_slider")
    st.dataframe(df.head(n), use_container_width=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6c757d;font-size:13px;padding:10px 0">
    <b>E-Commerce Olist Dashboard · RFM Analysis</b> · Proyek Analisis Data Dicoding ·
    <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank">
    Dataset: Brazilian E-Commerce by Olist</a>
</div>
""", unsafe_allow_html=True)
