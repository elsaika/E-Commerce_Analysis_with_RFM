# E-Commerce Olist — Data Analysis Dashboard

Dashboard interaktif hasil analisis dataset Brazilian E-Commerce Public Dataset (Olist).

## 📁 Struktur Direktori

```
submission/
├── dashboard/
│   ├── dashboard.py       # File utama Streamlit
│   └── main_data.csv      # Data gabungan yang sudah di-clean
├── data/
│   ├── orders_dataset.csv
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   ├── customers_dataset.csv
│   ├── products_dataset.csv
│   ├── sellers_dataset.csv
│   └── product_category_name_translation.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

## 🚀 Cara Menjalankan Dashboard (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Masuk ke folder dashboard:
   ```bash
   cd dashboard
   ```

3. Jalankan Streamlit:
   ```bash
   streamlit run dashboard.py
   ```

4. Buka browser di `http://localhost:8501`

## 📊 Pertanyaan Bisnis

1. **Kategori produk apa yang menghasilkan total pendapatan tertinggi** dan bagaimana tren penjualannya secara bulanan sepanjang 2017–2018?

2. **Bagaimana profil pelanggan berdasarkan segmentasi RFM** dan kelompok mana yang paling bernilai bagi bisnis?

## 🔗 Sumber Data

[Brazilian E-Commerce Public Dataset by Olist — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
