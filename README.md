# Android Malware Scanner

Aplikasi deteksi malware Android berbasis analisis statis (*static analysis*) menggunakan **Androguard 4.1.4**, seleksi fitur **Recursive Feature Elimination (RFE)**, dan model klasifikasi **XGBoost** terlatih dari dataset penelitian **MH-100K**.

Proyek ini dikembangkan sebagai bagian dari Skripsi oleh:
- **Nama**: Muhammad Zhainal Firdaus
- **NIM**: 301220041

---

## 🌟 Fitur Utama
1. **Analisis Statis Aman**: Mengekstraksi permissions dan API calls dari file APK Android tanpa perlu menjalankan file APK (bebas risiko eksekusi malware).
2. **Feature Extraction Standar Penelitian**: Mendeteksi lebih dari 24.000 fitur biner (Android permissions & API signatures).
3. **Seleksi Fitur RFE**: Mengurangi dimensi fitur menjadi 4.916 fitur terpilih menggunakan indexing langsung yang efisien.
4. **Klasifikasi Cepat XGBoost**: Menghasilkan probabilitas dan prediksi kategori (*Benign* vs *Malware*) dalam hitungan detik.
5. **Antarmuka Web Streamlit**: Dilengkapi visualisasi metrik performa model, detail fitur aktif, dan laporan ekstraksi.

---

## 📁 Struktur Direktori
```text
Android-Malware-Scanner/
├── app.py                     # Aplikasi antarmuka web Streamlit
├── requirements.txt           # Dependensi Python
├── .gitignore                 # Konfigurasi keamanan file Git
├── models/                    # Model XGBoost, selector RFE, dan metadata
│   ├── feature_names.csv
│   ├── metrics_rfe.json
│   ├── model_metadata.json
│   ├── rfe_selector.joblib
│   ├── selected_features.csv
│   └── xgboost_rfe.json
├── src/                       # Modul inti pemrosesan
│   ├── __init__.py
│   ├── feature_extractor.py   # Ekstraktor izin & API call APK
│   └── model_loader.py        # Pipeline inferensi XGBoost & RFE
└── tests/                     # Unit & integration tests
    ├── test_feature_extraction.py
    ├── test_model_loading.py
    └── test_real_apk_extraction.py
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.10 - 3.12 (disarankan)
- Git

### 2. Clone Repositori
```bash
git clone https://github.com/MuhammaDZhainalFirdaus/MuhammadZhainalFirdaus-301220041-Skripsi.git
cd MuhammadZhainalFirdaus-301220041-Skripsi
```

### 3. Buat & Aktifkan Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi
```bash
streamlit run app.py
```

---

## 🔒 Catatan Keamanan
- File APK yang diuji dan sampel malware **tidak disimpan** di repositori demi mematuhi kebijakan keamanan GitHub.
- Pengujian menggunakan dataset publik dan simulasi statis.
