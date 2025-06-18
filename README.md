🚨 Deteksi Risiko Korupsi Tender Pengadaan Publik
Sistem ini dirancang untuk memprediksi potensi risiko korupsi dalam tender pengadaan publik di Indonesia, menggunakan pendekatan Machine Learning dengan algoritma Random Forest dan Support Vector Machine (SVM) yang dibangun secara manual.

📌 Deskripsi Proyek
Korupsi dalam sektor pengadaan publik menjadi isu utama di Indonesia. Proyek ini membangun sistem prediksi berbasis data yang dapat membantu mengidentifikasi tender yang berisiko korupsi berdasarkan fitur-fitur kunci seperti:

jumlah penawar,

harga penawaran,

jenis prosedur,

durasi pengajuan/keputusan, dan lainnya.

Model prediksi diimplementasikan ke dalam aplikasi web berbasis Streamlit, sehingga dapat diakses dan digunakan oleh pengguna non-teknis.

🔗 Aplikasi dapat diakses di:
https://deteksi-risiko-korupsi-tender-pengadaan.streamlit.app/

📊 Dataset
Data diambil dari Global Contract-level Public Procurement Dataset (2008–2021), difokuskan pada entri tahun 2021 untuk negara Indonesia. Variabel target adalah CRI (Corruption Risk Index), yang dikonversi menjadi label biner untuk klasifikasi.

🔧 Tahapan Proyek
EDA (Exploratory Data Analysis)

Distribusi fitur numerik dan kategorik

Korelasi fitur terhadap target (CRI)

Analisis distribusi kelas target

Pre-Processing

Penanganan missing values

Deteksi outlier

SMOTE untuk menangani imbalance

Transformasi label dari nilai CRI

Modeling

Random Forest Manual (dengan perhitungan entropy, information gain, dan voting mayoritas)

SVM Manual (dengan optimasi SMO, α, w, dan b)

Evaluasi

Accuracy, Precision, Recall, F1-Score

Confusion Matrix

🧪 Hasil Evaluasi
Model	Akurasi	Precision	Recall	F1-score
Random Forest Manual	94.9%	0.99 / 0.92	0.91 / 0.99	0.95
SVM Manual	68.9%	0.67 / 0.71	0.73 / 0.65	0.69

Model Random Forest menunjukkan performa yang jauh lebih stabil dan akurat dibandingkan SVM.

💻 Fitur Aplikasi Web
Input data tender melalui form (jenis prosedur, jumlah penawar, harga, validitas, dll.)

Output prediksi dari dua model (RF dan SVM)

Indikator status risiko: "Aman" atau "Terdeteksi Risiko"

Penjelasan otomatis mengapa tender dianggap berisiko

Fitur ekspor hasil prediksi ke file Excel

👨‍💻 Tim Pengembang
Kelompok 5 - Kelas 2023B, Prodi S1 Sains Data, Universitas Negeri Surabaya:

Muhammad Zaky Taj Aldien (23031554065)

Putri Manika Rumamaya (23031554091)

Sintiya Risla Miftaqul Nikmah (23031554204)
