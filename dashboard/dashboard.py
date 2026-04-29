import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Olist Dashboard",
    page_icon="🛒",
    layout="wide",
)

COLOR_HIGHLIGHT = '#82b7c6'
COLOR_MAIN      = '#d3d3d3'
COLOR_RED       = '#e63946'
COLOR_BLUE      = '#457b9d'

SEG_ORDER  = ['Champions', 'Loyal Customers', 'Potential Loyalists',
              'New Customers', 'At Risk', 'Lost']
SEG_COLORS = ['#2a9d8f', '#57cc99', '#a8dadc', '#457b9d', '#e9c46a', '#e76f51']

plt.rcParams['font.family'] = 'sans-serif'
sns.set_style('whitegrid')

# ──────────────────────────────────────────────
# DATA LOADER
# ──────────────────────────────────────────────
@st.cache_data
def load_main_data():
    import os
    # Coba beberapa lokasi file
    candidates = [
        'main_data.csv',
        'dashboard/main_data.csv',
        '../main_data.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
            df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
            return df
    return None


def build_revenue_df(df):
    rev = df.dropna(subset=['product_category_name_english', 'payment_value']).copy()
    rev['year']       = rev['order_purchase_timestamp'].dt.year
    rev['month']      = rev['order_purchase_timestamp'].dt.month
    rev['year_month'] = rev['order_purchase_timestamp'].dt.to_period('M')
    return rev


def build_rfm(df):
    rfm_raw = df[['order_id', 'customer_id', 'customer_unique_id',
                  'order_purchase_timestamp', 'payment_value']].dropna()
    ref = rfm_raw['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = rfm_raw.groupby('customer_unique_id').agg(
        Recency   = ('order_purchase_timestamp', lambda x: (ref - x.max()).days),
        Frequency = ('order_id', 'nunique'),
        Monetary  = ('payment_value', 'sum'),
    ).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'],   q=5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],  q=5, labels=[1,2,3,4,5])

    def segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:   return 'Champions'
        elif r >= 3 and f >= 3:             return 'Loyal Customers'
        elif r >= 4 and f <= 2:             return 'New Customers'
        elif r <= 2 and f >= 3:             return 'At Risk'
        elif r <= 2 and f <= 2:             return 'Lost'
        else:                               return 'Potential Loyalists'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    return rfm


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Olist_logo.svg", width=160)
    st.markdown("---")
    st.markdown("## 🗂️ Navigasi")
    page = st.radio("Pilih Halaman", [
        "📊 Overview",
        "📦 Analisis Revenue",
        "👥 Segmentasi Pelanggan (RFM)",
        "🔍 Eksplorasi Data",
    ])
    st.markdown("---")
    st.caption("**Proyek Analisis Data**\nE-Commerce Public Dataset (Olist)\n\nElsa Ika Rahmani · 2025")


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
main_df = load_main_data()

if main_df is None:
    st.error("⚠️ File `main_data.csv` tidak ditemukan. "
             "Pastikan file tersebut berada satu folder dengan `dashboard.py`.")
    st.stop()

revenue_df = build_revenue_df(main_df)
rfm        = build_rfm(main_df)

# ──────────────────────────────────────────────
# HELPER: filter tahun
# ──────────────────────────────────────────────
available_years = sorted(revenue_df['year'].unique())
year_filter = st.sidebar.multiselect(
    "Filter Tahun", available_years, default=[y for y in available_years if y in [2017, 2018]]
)
if not year_filter:
    year_filter = available_years

rev_filtered = revenue_df[revenue_df['year'].isin(year_filter)]


# ══════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🛒 E-Commerce Olist — Dashboard Analisis")
    st.markdown(
        "Dashboard ini menyajikan hasil analisis dari **Olist E-Commerce Public Dataset** "
        "mencakup analisis revenue produk dan segmentasi pelanggan berbasis RFM."
    )
    st.markdown("---")

    # KPI
    total_revenue  = rev_filtered['payment_value'].sum()
    total_orders   = rev_filtered['order_id'].nunique()
    total_customers= main_df['customer_unique_id'].nunique()
    avg_order_val  = rev_filtered.groupby('order_id')['payment_value'].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",   f"R$ {total_revenue:,.0f}")
    c2.metric("🛍️ Total Orders",    f"{total_orders:,}")
    c3.metric("👤 Pelanggan Unik",  f"{total_customers:,}")
    c4.metric("🧾 Avg Order Value", f"R$ {avg_order_val:,.2f}")

    st.markdown("---")

    # Tren revenue bulanan
    monthly = (
        rev_filtered.groupby('year_month')['payment_value']
        .sum().reset_index()
    )
    monthly['year_month_str'] = monthly['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(monthly)), monthly['payment_value'],
            color=COLOR_BLUE, linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(range(len(monthly)), monthly['payment_value'],
                    alpha=0.15, color=COLOR_BLUE)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['year_month_str'], rotation=45, ha='right', fontsize=8)
    ax.set_title('Tren Revenue Bulanan', loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Total Revenue (R$)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Pertanyaan bisnis
    st.markdown("### 📋 Pertanyaan Bisnis")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Pertanyaan 1**\n\nKategori produk apa yang menghasilkan **total pendapatan tertinggi** "
                "dan bagaimana **tren penjualannya** secara bulanan sepanjang tahun 2017–2018?")
    with col2:
        st.success("**Pertanyaan 2**\n\nBagaimana profil pelanggan berdasarkan **segmentasi RFM** "
                   "(Recency, Frequency, Monetary) dan kelompok mana yang paling bernilai bagi bisnis?")


# ══════════════════════════════════════════════
# PAGE: ANALISIS REVENUE
# ══════════════════════════════════════════════
elif page == "📦 Analisis Revenue":
    st.title("📦 Analisis Revenue Produk")
    st.caption("Menjawab Pertanyaan 1: Kategori produk dengan revenue tertinggi & tren bulanan")
    st.markdown("---")

    top_n = st.slider("Tampilkan Top N Kategori", min_value=5, max_value=20, value=10)

    top_cat = (
        rev_filtered.groupby('product_category_name_english')['payment_value']
        .sum().sort_values(ascending=False).head(top_n)
    )

    monthly = (
        rev_filtered.groupby('year_month')['payment_value']
        .sum().reset_index()
    )
    monthly['year_month_str'] = monthly['year_month'].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analisis Revenue E-Commerce Olist', fontsize=14, fontweight='bold')

    # Plot 1: Top kategori
    ax1 = axes[0]
    colors = [COLOR_RED if i == 0 else COLOR_BLUE for i in range(len(top_cat))]
    bars = ax1.barh(top_cat.index[::-1], top_cat.values[::-1], color=colors[::-1])
    ax1.set_xlabel('Total Revenue (R$)')
    ax1.set_title(f'Top {top_n} Kategori Produk\nberdasarkan Total Revenue', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
    ax1.spines[['top','right']].set_visible(False)
    ax1.bar_label(bars, labels=[f'R${v/1e6:.2f}M' for v in top_cat.values[::-1]],
                  padding=3, fontsize=8)

    # Plot 2: Tren bulanan
    ax2 = axes[1]
    ax2.plot(range(len(monthly)), monthly['payment_value'],
             color=COLOR_BLUE, linewidth=2.5, marker='o', markersize=5)
    ax2.fill_between(range(len(monthly)), monthly['payment_value'],
                     alpha=0.15, color=COLOR_BLUE)
    # Tandai bulan tertinggi
    max_idx = monthly['payment_value'].idxmax()
    ax2.plot(max_idx, monthly['payment_value'][max_idx],
             marker='*', markersize=14, color=COLOR_RED, label='Peak Revenue', zorder=5)
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly['year_month_str'], rotation=45, ha='right', fontsize=8)
    ax2.set_title('Tren Revenue Bulanan', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Revenue (R$)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1e6:.1f}M'))
    ax2.spines[['top','right']].set_visible(False)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Insight
    st.info(
        "**💡 Insight:**\n"
        "- `health_beauty` menjadi kategori dengan revenue tertinggi, diikuti `watches_gifts` dan `bed_bath_table`.\n"
        "- Tren revenue menunjukkan **pertumbuhan konsisten** sepanjang 2017 hingga pertengahan 2018.\n"
        "- Terdapat **lonjakan signifikan** pada November 2017 (Black Friday effect).\n"
        "- Revenue stagnan di Q3 2018, mengindikasikan kejenuhan pasar."
    )

    st.markdown("---")
    st.markdown("### 📊 Statistik Revenue Bulanan")
    stats_df = monthly[['year_month_str', 'payment_value']].rename(
        columns={'year_month_str': 'Periode', 'payment_value': 'Revenue (R$)'}
    )
    stats_df['Revenue (R$)'] = stats_df['Revenue (R$)'].apply(lambda x: f'R$ {x:,.0f}')
    st.dataframe(stats_df, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: RFM SEGMENTASI
# ══════════════════════════════════════════════
elif page == "👥 Segmentasi Pelanggan (RFM)":
    st.title("👥 Segmentasi Pelanggan — RFM Analysis")
    st.caption("Menjawab Pertanyaan 2: Profil pelanggan dan segmen paling bernilai")
    st.markdown("---")

    # Ringkasan segmen
    seg_summary = rfm.groupby('Segment').agg(
        Jumlah   = ('customer_unique_id', 'count'),
        Recency  = ('Recency', 'mean'),
        Freq     = ('Frequency', 'mean'),
        Monetary = ('Monetary', 'mean'),
    ).round(1)
    seg_summary['% Pelanggan'] = (seg_summary['Jumlah'] / len(rfm) * 100).round(1)
    seg_summary = seg_summary.reindex([s for s in SEG_ORDER if s in seg_summary.index])

    seg_count = rfm['Segment'].value_counts().reindex(SEG_ORDER).fillna(0)
    avg_mon   = rfm.groupby('Segment')['Monetary'].mean().reindex(SEG_ORDER).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Segmentasi Pelanggan E-Commerce Olist — RFM Analysis',
                 fontsize=14, fontweight='bold')

    # Plot 1: Distribusi jumlah pelanggan
    ax1 = axes[0]
    bars1 = ax1.bar(SEG_ORDER, seg_count.values, color=SEG_COLORS, edgecolor='white')
    ax1.set_title('Distribusi Jumlah Pelanggan\nper Segmen RFM', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Jumlah Pelanggan')
    ax1.set_xticklabels(SEG_ORDER, rotation=25, ha='right', fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    ax1.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars1, seg_count.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{int(val):,}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Plot 2: Avg Monetary per segmen
    ax2 = axes[1]
    bars2 = ax2.bar(SEG_ORDER, avg_mon.values, color=SEG_COLORS, edgecolor='white')
    ax2.set_title('Rata-rata Nilai Transaksi (Monetary)\nper Segmen RFM', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Monetary (R$)')
    ax2.set_xticklabels(SEG_ORDER, rotation=25, ha='right', fontsize=9)
    ax2.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars2, avg_mon.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'R${val:,.0f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "**💡 Insight:**\n"
        "- Segmen `New Customers` mendominasi secara jumlah → platform berhasil akuisisi, tapi **gagal retensi**.\n"
        "- `Champions` memiliki monetary value tertinggi meski jumlahnya kecil — **revenue driver utama**.\n"
        "- Segmen `At Risk` dan `Lost` perlu program reaktivasi segera."
    )

    st.markdown("---")
    st.markdown("### 🔥 RFM Heatmap — Avg Monetary per R_Score × F_Score")

    rfm_pivot = rfm.pivot_table(
        index='R_Score', columns='F_Score',
        values='Monetary', aggfunc='mean'
    ).sort_index(ascending=False)

    fig2, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(rfm_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Avg Monetary (R$)'})
    ax.set_title('Heatmap: Rata-rata Monetary berdasarkan R_Score × F_Score',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('F_Score (Frequency — semakin tinggi semakin sering)', fontsize=10)
    ax.set_ylabel('R_Score (Recency — semakin tinggi semakin baru)', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.info(
        "**💡 Insight Heatmap:**\n"
        "- Pelanggan dengan R_Score & F_Score tinggi (pojok kanan atas) secara konsisten memiliki Monetary tertinggi.\n"
        "- Zona tersebut adalah target utama untuk program loyalitas eksklusif."
    )

    st.markdown("---")
    st.markdown("### 📋 Ringkasan Segmen Pelanggan")
    st.dataframe(seg_summary.style.format({
        'Recency': '{:.1f} hari',
        'Freq': '{:.2f}x',
        'Monetary': 'R$ {:.0f}',
        '% Pelanggan': '{:.1f}%',
    }), use_container_width=True)

    # Filter segmen
    selected_seg = st.selectbox("🔍 Lihat Detail Pelanggan per Segmen", SEG_ORDER)
    seg_df = rfm[rfm['Segment'] == selected_seg][
        ['customer_unique_id', 'Recency', 'Frequency', 'Monetary', 'Segment']
    ].sort_values('Monetary', ascending=False).head(50)
    seg_df['Monetary'] = seg_df['Monetary'].apply(lambda x: f'R$ {x:,.2f}')
    st.dataframe(seg_df, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: EKSPLORASI DATA
# ══════════════════════════════════════════════
elif page == "🔍 Eksplorasi Data":
    st.title("🔍 Eksplorasi Data")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Metode Pembayaran
    with col1:
        st.markdown("**💳 Metode Pembayaran Populer**")
        pay_counts = main_df['payment_type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_HIGHLIGHT if i == 0 else COLOR_MAIN for i in range(len(pay_counts))]
        bars = ax.bar(pay_counts.index, pay_counts.values, color=colors)
        ax.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=8)
        ax.set_title('Metode Pembayaran', loc='left', fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Aktivitas per Jam
    with col2:
        st.markdown("**⏰ Aktivitas Pembelian per Jam**")
        main_df['purchase_hour'] = main_df['order_purchase_timestamp'].dt.hour
        hour_counts = main_df['purchase_hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hour_counts.index, hour_counts.values, marker='o', color=COLOR_HIGHLIGHT, linewidth=2)
        ax.fill_between(hour_counts.index, hour_counts.values, color=COLOR_HIGHLIGHT, alpha=0.1)
        ax.set_title('Aktivitas per Jam', loc='left', fontsize=11, fontweight='bold')
        ax.set_xlabel('Jam (00-23)')
        ax.set_xticks(range(0, 24))
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # Top kota
    with col3:
        st.markdown("**🏙️ Top 10 Kota Konsumen**")
        top_cities = main_df['customer_city'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COLOR_HIGHLIGHT if i == 0 else COLOR_MAIN for i in range(10)]
        bars = ax.barh(top_cities.index[::-1], top_cities.values[::-1], color=colors[::-1])
        ax.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=8)
        ax.set_title('Top 10 Kota', loc='left', fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Korelasi
    with col4:
        st.markdown("**📊 Matriks Korelasi**")
        df_corr = main_df[['price', 'freight_value', 'product_weight_g', 'review_score']].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_corr, annot=True, cmap='RdBu', center=0, fmt='.2f',
                    cbar=False, ax=ax, linewidths=0.5)
        ax.set_title('Korelasi Variabel Finansial & Fisik', loc='left', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("### 📄 Preview Data Utama")
    n_rows = st.slider("Jumlah baris yang ditampilkan", 5, 50, 10)
    st.dataframe(main_df.head(n_rows), use_container_width=True)
