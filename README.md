# Deteksi Risiko Korupsi Tender Pengadaan Publik
Sistem ini dirancang untuk memprediksi potensi risiko korupsi dalam tender pengadaan publik di Indonesia, menggunakan pendekatan Machine Learning dengan algoritma Random Forest dan Support Vector Machine (SVM) yang dibangun secara manual.

## Deskripsi Proyek
Korupsi dalam sektor pengadaan publik menjadi isu utama di Indonesia. Proyek ini membangun sistem prediksi berbasis data yang dapat membantu mengidentifikasi tender yang berisiko korupsi berdasarkan fitur-fitur kunci seperti:

- jumlah penawar

- harga penawaran

- jenis prosedur

- durasi pengajuan/keputusan, dan lainnya.

Model prediksi diimplementasikan ke dalam aplikasi web berbasis Streamlit, sehingga dapat diakses dan digunakan oleh pengguna non-teknis.

## Aplikasi dapat diakses di:
https://deteksi-risiko-korupsi-tender-pengadaan.streamlit.app/

## Dataset
Data diambil dari Global Contract-level Public Procurement Dataset (2008–2021), difokuskan pada entri tahun 2021 untuk negara Indonesia. Variabel target adalah CRI (Corruption Risk Index), yang dikonversi menjadi label biner untuk klasifikasi.

Sumber Dataset: https://www.sciencedirect.com/science/article/pii/S2352340924003810?ref=pdf_download&fr=RR-2&rr=930b93990f309b8b

## Tahapan Proyek
1. EDA (Exploratory Data Analysis)

2. Distribusi fitur numerik dan kategorik

3. Korelasi fitur terhadap target (CRI)

4. Analisis distribusi kelas target

5. Pre-Processing

6. Penanganan missing values

7. Deteksi outlier

8. SMOTE untuk menangani imbalance

9. Transformasi label dari nilai CRI

10. Modeling:

    - Random Forest Manual (dengan perhitungan entropy, information gain, dan voting mayoritas)
 
    - SVM Manual (dengan optimasi SMO, α, w, dan b)

12. Evaluasi (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)

## Hasil Evaluasi
Model	Machine Learning:

- Random Forest Manual	94.9%	

- SVM Manual	68.9%	

Model Random Forest menunjukkan performa yang jauh lebih stabil dan akurat dibandingkan SVM.

## Fitur Aplikasi Web
- Input data tender melalui form (jenis prosedur, jumlah penawar, harga, validitas, dll.)

- Output prediksi dari dua model (RF dan SVM)

- Indikator status risiko: "Aman" atau "Terdeteksi Risiko"

- Penjelasan otomatis mengapa tender dianggap berisiko

- Fitur ekspor hasil prediksi ke file Excel

## Tim Pengembang

1. Muhammad Zaky Taj Aldien (065)

2. Putri Manika Rumamaya (091)

3. Sintiya Risla Miftaqul Nikmah (244)
