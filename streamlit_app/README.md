# Aplikasi Prediksi Kualitas Udara Kalimantan (Streamlit)

Menjalankan:

    pip install -r requirements.txt
    streamlit run app.py

Isi folder (semua harus berada bersama app.py):

- app.py : tampilan dan logika aplikasi (4 tab: Demo, Manual, CSV, Model)
- utils.py : transformasi fitur dan pembuatan fitur dari CSV unggahan
- model_rf_karhutla.joblib, fitur_model.json : model dan daftar fitur
- dataset_karhutla.csv, hasil_perbandingan_model.csv : data demo dan tabel perbandingan model
- .streamlit/config.toml : tema (warna). Folder ini tersembunyi; ubah warna di sini.

Mengubah tampilan:
- Warna dasar: .streamlit/config.toml
- Kartu, tile, dan tab: blok CSS di bagian atas app.py
- Mengganti model atau data: ganti berkasnya lalu jalankan ulang aplikasi

Membuka dari HP di Wi-Fi yang sama: gunakan alamat Network URL yang tampil di terminal.

Unggah CSV: kolom wajib date, province, pm25, n_hotspot (harian, minimal 7 hari berurutan per provinsi).
Contoh berkas dapat diunduh dari tab CSV.

Catatan: model disimpan dengan scikit-learn 1.9.1. Gunakan versi yang sama agar model dapat dimuat tanpa peringatan.
