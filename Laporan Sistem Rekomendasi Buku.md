# Project Overview

Dalam era informasi seperti sekarang, masyarakat dibanjiri dengan berbagai pilihan konten, termasuk buku. Hal ini seringkali membuat pengguna merasa kewalahan dalam menentukan pilihan yang sesuai dengan preferensi mereka. Untuk membantu pengguna menemukan buku yang relevan dan sesuai minatnya, sistem rekomendasi memainkan peran penting.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku berbasis data interaksi pengguna dan metadata buku menggunakan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Sistem rekomendasi telah menjadi bagian penting dalam industri digital untuk meningkatkan pengalaman pengguna. Dalam proyek ini, saya membangun sistem rekomendasi buku menggunakan dua pendekatan populer: **Content-Based Filtering** dan **Collaborative Filtering**.

Proyek ini penting karena dapat membantu pengguna menemukan buku yang sesuai dengan preferensi mereka tanpa harus menelusuri seluruh koleksi secara manual.

---

# Business Understanding

## Problem Statement

Bagaimana membangun sistem rekomendasi yang dapat memberikan saran buku yang relevan kepada pengguna berdasarkan riwayat interaksi dan karakteristik buku? Pengguna sering kali kesulitan memilih buku yang cocok dari koleksi yang sangat banyak.

## Goals

Membangun sistem rekomendasi buku yang dapat menyarankan buku berdasarkan:
- **Kemiripan konten** (judul dan metadata)
- **Preferensi pengguna lain** (rating)

Membuat sistem rekomendasi buku menggunakan dua pendekatan:
- **Content-Based Filtering (CBF)**
- **Collaborative Filtering (CF)**
Yang dapat menghasilkan Top-N rekomendasi buku yang relevan bagi pengguna.

## Solution Approach

Dua pendekatan diterapkan:

1. **Content-Based Filtering**: merekomendasikan buku berdasarkan kesamaan fitur metadata buku seperti judul dan penulis.
2. **Collaborative Filtering**: merekomendasikan buku berdasarkan kemiripan pola rating antar pengguna.

---

# Data Understanding

## Dataset Overview:

Proyek ini menggunakan tiga dataset utama dari Book-Crossing:

- **Books**: 271.360 entri, 8 kolom
- **Users**: 278.858 entri, 3 kolom
- **Ratings**: 1.048.575 entri, 3 kolom

**Sumber data**: [Book-Crossing Dataset (Kaggle)](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

## Deskripsi Fitur:

### Books Dataset:
- **ISBN**: Nomor unik untuk mengidentifikasi buku secara internasional.
- **Book-Title**: Judul buku.
- **Book-Author**: Nama penulis buku.
- **Year-Of-Publication**: Tahun terbit buku.
- **Publisher**: Nama penerbit.
- **Image-URL-S/M/L**: Tautan ke gambar sampul buku dalam tiga ukuran.

### Users Dataset:
- **User-ID**: ID unik pengguna.
- **Location**: Lokasi pengguna (format umum: Kota, Provinsi, Negara).
- **Age**: Usia pengguna (dalam tahun, bisa mengandung missing value dan outlier).

### Ratings Dataset:
- **User-ID**: Pengguna yang memberikan rating.
- **ISBN**: Buku yang diberi rating.
- **Book-Rating**: Skor rating pengguna terhadap buku (rentang 0–10; 0 berarti implicit rating).

## Kualitas Data:

- Dataset **Books** mengandung sedikit nilai kosong:
  - `Book-Author`: 2 missing value
  - `Publisher`: 2 missing value
  - `Image-URL-L`: 3 missing value
- Dataset **Users** mengandung nilai kosong pada kolom `Age`, serta memiliki outlier ekstrem (misalnya usia < 5 atau > 100).
- Dataset berpotensi mengandung duplikasi buku karena variasi penulisan judul dan nama penulis.

## Analisis Awal:

- Pengguna dengan rating terbanyak: **User-ID 11676** dengan **13.602 rating**.
- Buku dengan rating terbanyak: **ISBN 971880107** dengan **2.264 rating**.

---

# Data Preparation

## Tahap 1: Pembersihan dan Penanganan Missing Values

- Pada dataset **Books**, kolom `Book-Author` dan `Publisher` yang memiliki nilai kosong diisi dengan string `"Unknown"`.
- Kolom `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L` yang kosong diisi dengan string kosong `""`.
- Pada dataset **Users**, kolom `Age` yang kosong diisi menggunakan nilai **median**, kemudian dilakukan **filtering** untuk mempertahankan hanya pengguna dengan usia antara **5 hingga 100 tahun**.
- Pada dataset **Ratings**, dilakukan operasi `dropna()` sebagai langkah preventif untuk menghapus baris dengan nilai kosong. Namun berdasarkan pengecekan, tidak ditemukan missing value sehingga tidak ada baris yang terhapus.

## Tahap 2: Penggabungan Dataset

- Dataset **Ratings** digabungkan dengan **Users** berdasarkan kolom `User-ID`.
- Hasil penggabungan kemudian digabung kembali dengan dataset **Books** berdasarkan kolom `ISBN`.

## Tahap 3: Filtering Data

- Dihapus entri dengan `Book-Rating = 0`, karena dianggap sebagai implicit rating.
- Dihapus pengguna yang memberikan **kurang dari 5 rating**.
- Dihapus buku yang hanya menerima **kurang dari 10 rating**.

Langkah ini bertujuan untuk mengurangi noise dan memastikan kualitas data yang digunakan dalam pelatihan model.

## Tahap 4: Persiapan untuk Content-Based Filtering

Kolom `Book-Title` dari `books_clean` kemudian diproses menggunakan TF-IDF...

- Kolom `Book-Title` dari `books_clean` kemudian diproses menggunakan TF-IDF Vectorizer dari pustaka scikit-learn. TF-IDF (Term Frequency–Inverse Document Frequency) digunakan untuk mengubah teks menjadi representasi vektor numerik yang mencerminkan seberapa penting sebuah kata dalam suatu dokumen.
- Hasil vektorisasi TF-IDF ini digunakan untuk menghitung cosine similarity antar buku, sehingga dapat merekomendasikan buku-buku yang kontennya paling mirip.
- Dataset juga diberi indeks dan mapping (indices, titles, dan cosine_sim) untuk mendukung fungsi rekomendasi berbasis judul buku.

## Tahap 5: Persiapan untuk Collaborative Filtering

- Dataset akhir disusun dalam format `(User-ID, ISBN, Book-Rating)` untuk digunakan dalam algoritma Collaborative Filtering.
- Dataset kemudian dibagi menjadi **data latih dan data uji** menggunakan fungsi `train_test_split`.

---

# Modeling and Result

## Content-Based Filtering

Pendekatan **Content-Based Filtering (CBF)** bekerja dengan menganalisis atribut dari item (buku) itu sendiri. Dalam proyek ini, fitur utama yang digunakan adalah **judul buku** yang sebelumnya telah diubah ke dalam representasi numerik menggunakan **TF-IDF Vectorizer**. Kemudian dihitung kemiripan antar buku menggunakan **cosine similarity**.

### Contoh Hasil Rekomendasi:

<div style="overflow-x: auto;">

| Book-Title                                                       | Book-Author            |
|-----------------------------------------------------------------:|-----------------------:|
| Classical mythology                                              | Mark P.O Morford       |
| Classical Mythology                                              | Mark P. O. Morford     |
| Who's Who in Classical Mythology (Who's Who Series)              | Michael Grant          |
| The Dictionary of Classical Mythology                            | JOHN EDWARD ZIMMERMAN  |

</div>


## Collaborative Filtering (KNNBasic)

### Parameter Model:
```python
sim_options = {"name": "cosine", "user_based": False}
```

### Contoh Hasil Rekomendasi:

<div style="overflow-x: auto;">

| ISBN        | Judul Buku                                             | Prediksi Rating |
|-------------|:------------------------------------------------------:|----------------:|
| 61099694    | Hill Towns                                             |            8.72 |
| 3442453305  | Herr Lehmann.                                          |            8.56 |
| 553572040   | Killer Pancake                                         |            8.46 |
| 553096834   | Couplehood                                             |            8.38 |
| 3596259924  | The Unbearable Lightness of Being                      |            8.37 |

</div>

---

# Evaluation

Evaluasi dilakukan terhadap kedua pendekatan sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering**. Metrik evaluasi yang digunakan disesuaikan dengan karakteristik masing-masing model.

---

## Penjelasan Metrik Evaluasi

Untuk menilai performa model rekomendasi, digunakan tiga metrik utama:

- **Precision@k**: Mengukur ketepatan rekomendasi.

  **Formula:**

  $$
  \text{Precision@k} = \frac{\text{Jumlah item relevan yang direkomendasikan}}{\text{Jumlah total rekomendasi (k)}}
  $$

  Artinya, seberapa banyak dari rekomendasi yang diberikan benar-benar relevan.

---

- **Recall@k**: Mengukur kelengkapan rekomendasi.

  **Formula:**

  Recall@k = (Jumlah item relevan yang direkomendasikan) / (Jumlah item seharusnya)

  Artinya, seberapa banyak item relevan yang berhasil ditemukan oleh sistem dari semua yang seharusnya direkomendasikan.

---

- **Root Mean Square Error (RMSE)**: Digunakan untuk model Collaborative Filtering yang memprediksi nilai rating.

  **Formula:**

  RMSE = sqrt( (1/n) * Σ (ŷᵢ - yᵢ)² )

  Di mana ŷᵢ adalah rating yang diprediksi dan yᵢ adalah rating sebenarnya.
  Semakin kecil nilai RMSE, semakin baik performa prediksi rating model.


---

## Content-Based Filtering

Model ini memberikan rekomendasi berdasarkan kemiripan konten, dalam hal ini **judul buku**. Evaluasi dilakukan menggunakan metrik **Precision@5** dan **Recall@5**, berdasarkan kata kunci relevan pada hasil rekomendasi.

Contoh: Untuk input "Classical Mythology", sistem merekomendasikan 5 buku. Judul dianggap relevan jika mengandung kata kunci seperti "mythology".

Hasil evaluasi:

- **Precision@5 = 2.200**  
  → Artinya, sistem menemukan lebih dari satu item relevan dalam satu entri, karena beberapa buku memiliki judul yang sangat mirip atau identik. Nilai precision yang lebih dari 1 bisa terjadi jika pengukuran relevansi tidak berdasarkan item unik (misalnya duplikasi judul).

- **Recall@5 = 0.101**  
  → Artinya, sistem berhasil menemukan sekitar 10% dari semua buku relevan bertema "mythology" yang ada di dataset. Ini wajar karena jumlah total buku relevan sangat besar, dan model hanya merekomendasikan 5 buku.

Meski recall-nya rendah, precision tinggi menunjukkan bahwa rekomendasi yang diberikan cenderung relevan dengan topik yang dicari.

---

## Collaborative Filtering

Model ini menghasilkan prediksi rating buku yang mungkin disukai pengguna. Evaluasi dilakukan dengan data uji.

Hasil evaluasi:

- **RMSE = 1.739**  
  → Prediksi rating cukup dekat dengan rating aktual pengguna.

- **Precision@5 = 0.263**  
  → Sekitar 26% dari rekomendasi benar-benar relevan.

- **Recall@5 = 0.964**  
  → Hampir semua item relevan berhasil ditemukan sistem.

---

Secara keseluruhan, kedua pendekatan menunjukkan performa yang saling melengkapi. Content-Based Filtering cocok saat metadata kaya, sedangkan Collaborative Filtering memberikan rekomendasi yang lebih personal berdasarkan histori rating pengguna lain.


---

# Kesimpulan

Sistem rekomendasi yang dibangun telah mengimplementasikan dua pendekatan utama dengan hasil evaluasi yang cukup baik. Content-based dan collaborative filtering saling melengkapi untuk meningkatkan pengalaman pengguna.
