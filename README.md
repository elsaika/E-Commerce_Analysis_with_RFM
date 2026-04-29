# E-Commerce_Analysis_with_RFM

Dashboard analisis data interaktif menggunakan Streamlit untuk dataset Olist E-Commerce Public Dataset.

## Cara Menjalankan

1. Pastikan `main_data.csv` berada di folder `dashboard/` (sudah digenerate dari notebook).

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan dashboard:
```bash
streamlit run dashboard.py
```

## Struktur File

```
submission/
├── dashboard/
│   ├── main_data.csv       ← export dari notebook
│   └── dashboard.py        ← file ini
├── data/
│   └── ...csv
├── notebook.ipynb
├── requirements.txt
├── README.md
└── url.txt                 ← (jika deploy ke Streamlit Cloud)
```

## Pertanyaan Bisnis

1. **Revenue Analysis**: Kategori produk apa yang menghasilkan total pendapatan tertinggi dan bagaimana tren penjualannya secara bulanan sepanjang tahun 2017–2018?

2. **RFM Segmentation**: Bagaimana profil pelanggan berdasarkan segmentasi RFM dan kelompok mana yang paling bernilai bagi bisnis?

## Fitur Dashboard

- 📊 **Overview**: KPI utama dan tren revenue keseluruhan
- 📦 **Analisis Revenue**: Top kategori produk & tren bulanan dengan filter interaktif
- 👥 **Segmentasi Pelanggan (RFM)**: Distribusi segmen, heatmap RFM, dan detail per segmen
- 🔍 **Eksplorasi Data**: Metode pembayaran, aktivitas per jam, top kota, matriks korelasi
