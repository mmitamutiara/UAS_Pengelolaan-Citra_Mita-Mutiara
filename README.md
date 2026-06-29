# Aplikasi Analisis Ekspresi Wajah Interaktif Berbasis Deep Learning (YOLOv8)

Proyek ini dibuat untuk memenuhi tugas akhir (UAS) Mata Kuliah Pengelolaan Citra Digital. Sistem ini mengintegrasikan peningkatan kualitas citra (*Image Enhancement*), deteksi tepi (*Edge Detection*), dan klasifikasi ekspresi wajah menggunakan Deep Learning secara *end-to-end*.

## 👥 Anggota Tim / Pengembang
* **Nama:** Mita Mutiara

## 📑 Fitur Utama Aplikasi
1. **Enhancement:** Peningkatan kontras gambar menggunakan *Histogram Equalization*.
2. **Edge Detection:** Ekstraksi fitur struktural garis wajah menggunakan *Canny Edge Detection*.
3. **Deep Learning Classification:** Klasifikasi 5 kelas emosi wajah menggunakan model *YOLOv8-Classification* yang telah dilatih.
4. **Interactive UI:** Antarmuka web yang dinamis dan responsif berbasis *Streamlit*.

## 📊 Dataset
Dataset yang digunakan dalam proyek ini adalah **FER-2013 (Facial Expression Recognition 2013)**. Karena keterbatasan ukuran penyimpanan GitHub, dataset dapat diunduh secara resmi melalui tautan berikut:
* **Link Dataset:** [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
* *Petunjuk:* Setelah diunduh, ekstrak dan masukkan folder `train`, `val`, dan `test` ke dalam sebuah folder bernama `dataset/` di direktori proyek ini.

## 🛠️ Cara Menjalankan Aplikasi di Lokal (Laptop)

Ikuti instruksi berikut untuk menjalankan aplikasi ini di komputer Anda:

### 1. Prasyarat (Prerequisites)
Pastikan Anda telah menginstal Python (versi 3.10 atau yang lebih baru). Instal pustaka yang diperlukan dengan menjalankan perintah berikut di terminal/command prompt:
```bash
pip install streamlit opencv-python numpy pillow ultralytics
