"""Android Malware Scanner - Web Application (Streamlit).

Aplikasi deteksi malware Android berbasis analisis statis menggunakan Androguard 4.1.4,
seleksi fitur RFE (selected_features.csv), dan model XGBoost terlatih dari dataset penelitian MH-100K.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import src.feature_extractor
from src.feature_extractor import APKFeatureExtractor, inspect_archive
from src.model_loader import DEFAULT_MODELS_DIR, AndroidMalwarePipeline

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Page Configuration
st.set_page_config(
    page_title="Android Malware Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Global Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-card-benign {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1.5px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-card-malware {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-header {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .disclaimer-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 25px;
    }
    .diag-box {
        background: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> AndroidMalwarePipeline:
    """Inisialisasi pipeline model XGBoost dan pemetaan fitur RFE."""
    return AndroidMalwarePipeline(models_dir=DEFAULT_MODELS_DIR, use_rfe_joblib=False)


@st.cache_resource(show_spinner=False)
def get_extractor() -> APKFeatureExtractor:
    """Inisialisasi modul ekstraksi fitur Androguard."""
    importlib.reload(src.feature_extractor)
    return src.feature_extractor.APKFeatureExtractor()


def main() -> None:
    # Sidebar: Model and Research Information
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/AndroidLight.svg",
            width=60,
        )
        st.title("Informasi Model")
        st.markdown(
            """
            Sistem deteksi ini menggunakan model penelitian **XGBoost** terlatih pada dataset **MH-100K** (101.934 aplikasi Android riil).
            
            **Parameter Penelitian:**
            - **Dataset:** MH-100K
            - **Total Fitur Asli:** 24.583
              - 166 Izin (Permissions)
              - 24.417 API Calls
            - **Fitur Terpilih (RFE):** 4.916
            - **Akurasi Model:** 97,49%
            - **F1-Score:** 87,20%
            - **Engine Ekstraksi:** Androguard 4.1.4
            """
        )

        st.divider()
        st.subheader("Pengaturan Analisis")
        threshold = st.slider(
            "Ambang Batas Malware (Threshold)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probabilitas malware di atas nilai ini diklasifikasikan sebagai Terindikasi Malware.",
        )

        st.caption(
            "Model asli menggunakan threshold 0.50 berdasarkan pemisahan biner `binary:logistic`."
        )

    # Main Area Header
    st.markdown(
        '<div class="main-title">🛡️ Android Malware Scanner</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Pemindaian statis APK Android menggunakan ekstraksi izin (Manifest) dan pemanggilan API (Dalvik DEX) berbasis model XGBoost terlatih.</div>',
        unsafe_allow_html=True,
    )

    # Cache initialization
    with st.spinner("Memuat model XGBoost dan skema fitur..."):
        try:
            pipeline = get_pipeline()
            extractor = get_extractor()
        except Exception as exc:
            st.error(f"Gagal memuat artefak model penelitian: {exc}")
            st.stop()

    # Upload Section
    uploaded_file = st.file_uploader(
        "Pilih berkas Android APK (*.apk, *.apkm)",
        type=["apk", "apkm"],
        help="Unggah satu berkas aplikasi Android (.apk atau bundel .apkm) untuk dipindai secara statis.",
    )

    if uploaded_file is None:
        st.info(
            "💡 Silakan pilih atau seret berkas `.apk` / `.apkm` ke area di atas untuk memulai analisis statis."
        )

        st.markdown(
            """
            #### Format yang Didukung:
            1. **APK Standar (*.apk):** Berkas APK mandiri standar yang memuat `AndroidManifest.xml` dan berkas `classes.dex`.
            2. **APK Bundle / APKM (*.apk, *.apkm):** Berkas bundel arsip (seperti unduhan APKMirror) yang memuat `base.apk` di dalamnya akan dideteksi dan diekstrak secara otomatis untuk dianalisis.
            
            #### Cara Kerja Pemindaian:
            1. **Validasi & Integritas:** Memastikan keutuhan byte berkas dan validasi struktur arsip Android.
            2. **Ekstraksi Statis (Androguard 4.1.4):**
               - Membaca izin Android yang diminta pada Manifest.
               - Melacak seluruh instruksi pemanggilan API (`APICall`) pada Dalvik bytecode melalui analisis cross-reference (XREF).
            3. **Pemetaan Fitur:** Membentuk vektor biner 24.583 dimensi sesuai skema MH-100K.
            4. **Seleksi Fitur (RFE):** Mengambil tepat 4.916 fitur terpilih sesuai urutan penelitian.
            5. **Klasifikasi XGBoost:** Menghitung probabilitas kelas Benign dan Malware.
            """
        )
        return

    # File size validation (limit 150 MB for local safety)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 150:
        st.error(
            f"Ukuran berkas ({file_size_mb:.1f} MB) melebihi batas aman pemindaian lokal (150 MB)."
        )
        return

    # Scan Trigger Button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_scan = st.button("🚀 Mulai Pemindaian APK", type="primary", use_container_width=True)

    with col_info:
        st.caption(f"Berkas: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

    if start_scan:
        progress_bar = st.progress(0, text="Menyiapkan berkas APK...")
        status_text = st.empty()

        # Temporary file creation with secure prefix and timestamp
        temp_dir = Path(tempfile.gettempdir())
        temp_apk = temp_dir / f"scan_{int(time.time())}_{uploaded_file.name}"

        try:
            # 1. Pastikan objek unggahan dibaca dari awal (offset 0)
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()

            if len(raw_bytes) != uploaded_file.size:
                raise ValueError(
                    f"Integritas baca unggahan gagal: {len(raw_bytes)} bytes terbaca dari total {uploaded_file.size} bytes."
                )

            # Hitung checksum SHA-256 untuk memastikan byte tidak termodifikasi
            orig_sha256 = hashlib.sha256(raw_bytes).hexdigest()

            # 2. Tulis seluruh byte ke file sementara
            with open(temp_apk, "wb") as f:
                f.write(raw_bytes)
                f.flush()

            # Verifikasi bahwa berkas sementara memiliki ukuran dan hash yang sama persis
            written_size = temp_apk.stat().st_size
            with open(temp_apk, "rb") as f:
                written_sha256 = hashlib.sha256(f.read()).hexdigest()

            if written_size != uploaded_file.size or written_sha256 != orig_sha256:
                raise ValueError(
                    f"Verifikasi penyimpanan sementara gagal: byte terdistorsi saat penulisan ke disk "
                    f"({written_size} bytes vs {uploaded_file.size} bytes)."
                )

            progress_bar.progress(15, text="1/4: Memvalidasi integritas arsip APK...")
            time.sleep(0.1)

            # 3. Jalankan diagnostik arsip
            diag = (
                extractor.inspect_archive(temp_apk)
                if hasattr(extractor, "inspect_archive")
                else inspect_archive(temp_apk)
            )
            if diag["has_base_apk"]:
                status_text.info(f"📦 Terdeteksi format APK Bundle / APKM. Mengekstrak '{diag['base_apk_entry']}' secara otomatis...")
                time.sleep(0.2)

            progress_bar.progress(35, text="2/4: Mengekstrak izin & pemanggilan API Dalvik DEX (Androguard)...")
            status_text.info("Sedang menganalisis bytecode DEX dan cross-reference metode. Mohon tunggu...")

            # 4. Ekstraksi fitur
            t_start = time.time()
            vector, report = extractor.extract_features_from_apk(temp_apk)
            t_extract = time.time() - t_start
            status_text.empty()

            progress_bar.progress(75, text="3/4: Melakukan seleksi fitur RFE (4.916 fitur)...")
            time.sleep(0.1)

            # 5. Seleksi fitur dan inferensi
            selected_vector = pipeline.transform(vector)

            progress_bar.progress(90, text="4/4: Menjalankan inferensi model XGBoost...")
            proba = pipeline.model.predict_proba(selected_vector)
            p_benign = float(proba[0][0])
            p_malware = float(proba[0][1])

            is_malware = p_malware >= threshold
            progress_bar.progress(100, text="Pemindaian Selesai!")
            time.sleep(0.2)
            progress_bar.empty()

            # Display Results
            st.markdown("---")
            if is_malware:
                st.markdown(
                    f"""
                    <div class="status-card-malware">
                        <div class="status-header" style="color: #DC2626;">⚠️ TERINDIKASI MALWARE</div>
                        <div style="font-size: 1.05rem; color: #7F1D1D;">
                            Model XGBoost mendeteksi pola pemanggilan API dan kombinasi izin yang berkarakteristik jahat/malicious.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-card-benign">
                        <div class="status-header" style="color: #059669;">🛡️ TERINDIKASI BENIGN / AMAN</div>
                        <div style="font-size: 1.05rem; color: #064E3B;">
                            Pola izin dan pemanggilan API Dalvik DEX konsisten dengan karakteristik aplikasi normal/benign.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probabilities display
            st.subheader("Probabilitas Klasifikasi")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**Benign (Aplikasi Normal):** `{p_benign * 100:.2f}%`")
                st.progress(p_benign)
            with col_p2:
                st.markdown(f"**Malware (Aplikasi Berbahaya):** `{p_malware * 100:.2f}%`")
                st.progress(p_malware)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metadata and metrics cards
            st.subheader("Ringkasan Metadata dan Fitur APK")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            with m_col1:
                bundle_tag = " (Bundle APKM)" if report["is_bundle"] else ""
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Nama Paket{bundle_tag}</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; word-break: break-all;">
                            {report['package_name']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Izin Cocok Penelitian</div>
                        <div class="metric-value" style="color: #2563EB;">
                            {report['matched_research_permissions']} <span style="font-size: 0.9rem; font-weight: 400; color: #64748B;">/ {report['total_extracted_permissions']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">API Calls Cocok</div>
                        <div class="metric-value" style="color: #7C3AED;">
                            {report['matched_research_api_calls']} <span style="font-size: 0.9rem; font-weight: 400; color: #64748B;">/ {report['total_extracted_api_calls']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m_col4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Fitur Aktif di Model</div>
                        <div class="metric-value" style="color: #059669;">
                            {int(np.sum(selected_vector))} <span style="font-size: 0.9rem; font-weight: 400; color: #64748B;">/ 4.916</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed Feature Breakdown Tabs
            tab_perm, tab_api, tab_meta = st.tabs(
                [
                    f"📋 Daftar Izin ({report['matched_research_permissions']} cocok)",
                    f"🔍 Sampel API Call ({report['matched_research_api_calls']} cocok)",
                    "⚙️ Detail Teknis & Diagnostik",
                ]
            )

            with tab_perm:
                if report["matched_permissions_list"]:
                    st.write("Izin Android yang terdeteksi dan tercatat dalam dataset penelitian:")
                    perm_data = [
                        {"Nama Fitur": p, "Tipe": "Android Permission"}
                        for p in report["matched_permissions_list"]
                    ]
                    st.dataframe(pd.DataFrame(perm_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Tidak ada izin yang cocok dengan 166 izin skema penelitian.")

            with tab_api:
                if report["matched_api_calls_sample"]:
                    st.write(
                        f"Menampilkan 50 sampel dari {report['matched_research_api_calls']} pemanggilan API Dalvik DEX yang cocok dengan skema penelitian:"
                    )
                    api_data = [
                        {"Signature API Call Dalvik": a}
                        for a in report["matched_api_calls_sample"]
                    ]
                    st.dataframe(pd.DataFrame(api_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Tidak ada pemanggilan API yang cocok dengan skema penelitian.")

            with tab_meta:
                meta_table = [
                    {"Parameter": "Nama Berkas Unggahan", "Nilai": report["apk_filename"]},
                    {"Parameter": "Ukuran Berkas", "Nilai": f"{report['apk_size_bytes'] / (1024*1024):.2f} MB ({report['apk_size_bytes']:,} bytes)"},
                    {"Parameter": "SHA-256 Checksum", "Nilai": orig_sha256},
                    {"Parameter": "Tipe Paket", "Nilai": "APK Bundle / APKM (base.apk diekstrak)" if report["is_bundle"] else "Standalone APK"},
                    {"Parameter": "Nama Aplikasi", "Nilai": report["app_name"]},
                    {"Parameter": "Nama Paket", "Nilai": report["package_name"]},
                    {"Parameter": "Target SDK", "Nilai": str(report["target_sdk_version"])},
                    {"Parameter": "Min SDK", "Nilai": str(report["min_sdk_version"])},
                    {"Parameter": "Waktu Ekstraksi Statis", "Nilai": f"{report['extraction_time_seconds']} detik"},
                    {"Parameter": "Total Fitur Aktif (24.583)", "Nilai": f"{report['total_active_features']}"},
                    {"Parameter": "Total Fitur Aktif RFE (4.916)", "Nilai": f"{int(np.sum(selected_vector))}"},
                ]
                st.table(pd.DataFrame(meta_table))

            # Mandatory Security Disclaimer
            st.markdown(
                """
                <div class="disclaimer-box">
                    <strong>Pemberitahuan Keamanan & Batasan Model:</strong><br>
                    1. Hasil analisis ini merupakan estimasi berbasis analisis statis dan model machine learning XGBoost dari dataset penelitian MH-100K.<br>
                    2. Prediksi <em>Benign/Aman</em> bukan merupakan jaminan mutlak bahwa aplikasi 100% bebas dari ancaman zero-day, eksploitasi dinamis, atau payload terenkripsi.<br>
                    3. Berkas APK yang diunggah <strong>hanya dibaca struktur statisnya</strong>, sama sekali <strong>tidak pernah dijalankan atau dipasang</strong> pada sistem operasi host. Berkas sementara segera dibersihkan setelah proses pemindaian selesai.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except ValueError as val_err:
            st.error(f"⚠️ **Validasi Berkas Gagal:** {val_err}")
            # Tampilkan kartu diagnostik jika tersedia
            diag = (
                extractor.inspect_archive(temp_apk)
                if hasattr(extractor, "inspect_archive")
                else inspect_archive(temp_apk)
            )
            if diag["is_valid_zip"]:
                st.markdown(
                    f"""
                    <div class="diag-box">
                        <strong>Diagnostik Arsip:</strong><br>
                        - Ukuran: {diag['file_size_bytes']:,} bytes<br>
                        - Format ZIP Valid: Ya<br>
                        - Memuat AndroidManifest.xml: {'Ya' if diag['has_manifest'] else 'Tidak'}<br>
                        - Memuat base.apk: {'Ya' if diag['has_base_apk'] else 'Tidak'}<br>
                        - Berkas DEX ditemukan: {len(diag['dex_entries'])} ({', '.join(diag['dex_entries'][:5]) or 'Tidak ada'})<br>
                        - Entri relevan: {', '.join(diag['relevant_entries']) or 'Tidak ada'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"❌ **Terjadi kesalahan saat memproses APK:** {exc}")
            logger.exception("Error processing APK in Streamlit app")
        finally:
            # Cleanup temporary file safely
            if temp_apk.exists():
                try:
                    temp_apk.unlink()
                except Exception as clean_err:
                    logger.warning("Gagal menghapus berkas sementara %s: %s", temp_apk, clean_err)


if __name__ == "__main__":
    main()
