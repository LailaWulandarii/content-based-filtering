# Laporan Proyek Machine Learning - Laila Wulandari

## Project Overview

Industri fashion digital saat ini menghadapi tantangan besar dalam hal penemuan produk (product discovery). Menurut penelitian [1], platform e-commerce fashion seperti Myntra dapat menampung banyak produk dengan variasi warna, gaya, dan merek yang sangat beragam. Kondisi ini menyebabkan kesulitan bagi konsumen dalam menemukan item yang relevan, yang berdampak pada rendahnya conversion rate (tingkat konversi) dan tingginya angka return produk [2].

Sistem rekomendasi berbasis konten (content-based filtering) menjadi solusi efektif untuk masalah ini. Pendekatan ini bekerja dengan menganalisis fitur intrinsik produk seperti deskripsi tekstual, merek, warna dominan, dan kategori gender.
Seperti ditunjukkan dalam penelitian [3], sistem berbasis konten meningkatkan engagement pengguna hingga 40% untuk platform fashion karena mampu menangani masalah cold-start (produk baru tanpa riwayat interaksi) yang tidak dapat diatasi oleh metode collaborative filtering. 

Referensi:

[1] A. Gupta et al., "Fashion E-Commerce Analytics: A Case Study of Myntra", IEEE Access, vol.9, pp.102345-102358, 2021.

[2] B. Chen, "Challenges in Online Fashion Retail", Proc. IEEE Int. Conf. Data Mining, pp.223-230, 2022.

[3] L. Wang, "Content-Based Filtering for Fashion Recommendation", IEEE Trans. on Consumer Electronics, vol.68(3), 2023.

## Business Understanding

### Problem Statements

- Kompleksitas navigasi produk dalam katalog besar seperti Myntra menyebabkan rendahnya efisiensi pencarian dan kesulitan menemukan item dengan atribut spesifik.
- Ketergantungan collaborative filtering pada riwayat interaksi membuatnya tidak efektif untuk produk baru (cold-start) dan item niche.
### Goals

- Mengembangkan sistem rekomendasi berbasis konten yang menyaring banyak produk menjadi 5 teratas dengan kemiripan di atas 60% menggunakan fitur intrinsik.
- Mengoptimalkan kualitas rekomendasi hingga mencapai Precision@5 dan nDCG@5 lebih dari 60% untuk menunjukkan relevansi dan urutan produk yang baik.
 
### Solution Statement
Untuk mengatasi kesulitan pencarian produk dan masalah cold-start, dikembangkan sistem rekomendasi berbasis konten yang memanfaatkan atribut intrinsik produk untuk menyaring item menjadi 5 rekomendasi paling relevan.

Solusi 1: Cosine Similarity
Menggunakan TF-IDF dari gabungan deskripsi, merek, dan warna produk untuk menghitung kemiripan berdasarkan arah vektor.

Solusi 2: Euclidean Distance
Menggunakan TF-IDF dan menghitung jarak antar produk dalam ruang vektor untuk menentukan kemiripan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Fashion Clothing Products Dataset dari Myntra, yang tersedia secara publik di Kaggle melalui tautan berikut: https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog/data. Dataset ini memiliki total 12.491 entri dengan 8 kolom, yaitu:

### Variabel-variabel pada Fashion Clothing Products Dataset dari Myntra adalah sebagai berikut:

- ProductID : ID unik untuk setiap produk.
- ProductName : Nama lengkap dari produk.
- ProductBrand : Merek atau brand produk.
- Gender : Kategori gender pengguna (Men, Women, Unisex).
- Price (INR) : Harga produk dalam satuan Rupee India.
- NumImages : Jumlah gambar yang tersedia untuk produk.
- Description : Deskripsi lengkap mengenai produk.
- PrimaryColor : Warna utama produk (beberapa data memiliki nilai kosong).

### Exploratory Data Analysis yang dilakukan:
EDA dilakukan untuk memahami struktur dan karakteristik data sebelum membangun model. Berikut langkah-langkah analisis eksploratif yang dilakukan:
- Melihat 5 Data Pertama: Untuk mendapatkan gambaran umum struktur data, dilakukan pemanggilan data.head(), hasilnya menunjukkan bahwa setiap baris mewakili satu produk fashion, dengan informasi lengkap tentang deskripsi, brand, warna, dan harga.
- Mengecek Tipe Data dan Nilai Null: Fungsi data.info() digunakan untuk melihat tipe data dan mendeteksi nilai kosong, hasilnya semua kolom memiliki tipe data sesuai (int64/obj). Kolom PrimaryColor memiliki 894 nilai kosong, sedangkan kolom lainnya lengkap.
- Analisis Missing Value: Kolom PrimaryColor memiliki 894 missing values (sekitar 7,2% dari total data). Nilai kosong ini kemudian diisi dengan "Unknown" selama tahap preprocessing.
- Distribusi Merek Produk (Brand): Brand paling populer divisualisasikan menggunakan grafik batang (bar plot), hasilnya menunjukkan Indian Terrain adalah brand dengan jumlah produk terbanyak sedangkan Brand populer lainnya termasuk Puma, Pepe Jeans, dan Raymond.
- Distribusi Harga Produk: Distribusi harga divisualisasikan menggunakan histogram, hasilnya harga mayoritas produk berada dalam kisaran 500 – 5000 INR, menunjukkan orientasi pasar pada produk mid-range.

## Data Preparation
Dilakukan beberapa teknik data preparation untuk memastikan data siap digunakan dalam pemodelan. Tahap ini bertujuan untuk meningkatkan kualitas data, menghindari bias, dan memastikan model yang dibangun memiliki performa yang baik. Berikut tahapannya:
- Handling Missing Values: Kolom PrimaryColor yang memiliki nilai kosong diisi dengan 'Unknown', dengan tujuan untuk menghindari hilangnya informasi dan menjaga integritas data.
- Text Cleaning:
  
    a. Lowercasing dan Menghapus Karakter Khusus: Semua huruf diubah menjadi huruf kecil dan karakter non-alfabet dihapus agar tidak mengganggu representasi vektor nanti.
  
    b. Penghapusan Stopwords Khusus Fashion
    Selain stopwords bahasa Inggris umum dari NLTK, ditambahkan stopwords khusus seperti:
    'size', 'cm', 'inch', 'pack', 'item', 'made' — karena kata-kata ini sering muncul namun tidak memberi makna dalam konteks konten fashion.

    c. Tokenisasi dan Filtering: Teks dibagi menjadi kata-kata, dan hanya kata-kata dengan panjang lebih dari 2 huruf yang bukan stopwords yang dipertahankan.
    
    d. Stemming: Digunakan Snowball Stemmer untuk menyederhanakan kata menjadi bentuk dasar.

- Ekstraksi Fitur Tambahan (Warna Produk): Dari kolom Description, dilakukan pencarian dan ekstraksi kata kunci warna seperti black, blue, red, dll. Hasil ekstraksi disimpan di kolom extracted_color. Jika tidak ditemukan, diisi dengan string kosong saat penggabungan.
- Peningkatan Bobot Kata Kunci Penting: Beberapa kata kunci penting yang menjadi ciri khas suatu brand atau deskripsi diperkuat dengan pengulangan (boosting) untuk membantu model lebih peka terhadap kemunculan kata-kata yang berpengaruh besar dalam identitas produk. Implementasi dilakukan dalam fungsi boost_keywords()
- Pembentukan Fitur Gabungan (text_features): Beberapa kolom penting digabung menjadi satu fitur teks utama text_features. Langkah ini menciptakan representasi teks yang lebih kaya, mencakup semua aspek konten produk.
- Aplikasi Preprocessing dan Keyword Boosting: dilakukan pembersihan dan boosting pada kolom text_features.
- Membuat objek TF-IDF Vectorizer dengan stopwords bahasa Inggris kemudian mengubah teks fitur menjadi representasi numerik TF-IDF.
## Modeling
Tahap modeling ini membangun sistem rekomendasi produk fashion berbasis konten (content-based filtering) dengan pendekatan pemrosesan teks. Sistem dirancang untuk menyarankan produk-produk yang mirip berdasarkan deskripsi, nama, brand, dan warna, menggunakan representasi teks dan pengukuran kemiripan antar produk.
Terdapat dua pendekatan yang digunakan dalam modeling yaitu Cosine Similarity dan Euclidean Distance. 
1. Cosine Similarity: 
  - Pendekatan Cosine Similarity dengan TF-IDF mengubah teks menjadi vektor bobot numerik, memungkinkan pengukuran kemiripan antar produk berdasarkan sudut vektor mereka.
  - Kelebihan:
    - Menangkap perbedaan halus antara deskripsi produk.
    - Lebih toleran terhadap variasi kata dan sinonim.
    - Memberikan bobot penting pada istilah unik.
  - Kekurangan:
    - Lebih sensitif terhadap preprocessing teks.
    - Sensitif terhadap preprocessing teks—hasil dapat berubah drastis tergantung pada penghapusan stopwords atau stemming.
    - Tidak mempertimbangkan urutan kata, sehingga konteks frasa kompleks bisa kurang akurat.
2. Euclidean Distance:
  - Pendekatan Euclidean Distance mengukur kemiripan antar produk berdasarkan jarak antara vektor teks dalam ruang numerik, di mana nilai yang lebih kecil menunjukkan tingkat kesamaan lebih tinggi.
 - Kelebihan:
    - Sederhana dan cepat dalam perhitungan.
    - Cocok untuk data dengan representasi numerik langsung.
    - Dapat digunakan dengan berbagai representasi fitur, termasuk TF-IDF.
  - Kekurangan:
    - Tidak mempertimbangkan hubungan semantik antara kata-kata.
    - Rentan terhadap skala data—perbedaan kecil dalam nilai dapat memengaruhi hasil.
    - Bisa kurang akurat dalam menangkap nuansa deskripsi produk dibanding Cosine Similarity.

Untuk menghasilkan rekomendasi, dibuat fungsi recommended() yang menerima nama produk, matriks kesamaan, dan jumlah hasil (top_n). Fungsi ini berjalan dengan mencari indeks produk input, mengambil skor kesamaan terhadap semua produk lain, mengurutkan skor dan mengambil top_n teratas (selain dirinya sendiri).
Kemudian dilakukan implementasi untuk produk "Casual shirt". Hasil rekomendasi menunjukkan bahwa Cosine Similarity dan Euclidean Distance menghasilkan daftar produk yang identik, mencerminkan kemiripan deskripsi dalam dataset. Semua rekomendasi berasal dari merek Parx dan kategori Casual Shirt, menandakan sistem bekerja konsisten.

![image](https://github.com/user-attachments/assets/d185e3a0-f096-4203-b2ff-c3c838f9ae4f)

![image](https://github.com/user-attachments/assets/e6c52f53-495a-42b8-a215-ace3ec92dee8)

![image](https://github.com/user-attachments/assets/c069c64c-ec4d-42a7-8483-3f776473ba89)

## Evaluation
Evaluasi dilakukan menggunakan metrik Precision@K dan Normalized Discounted Cumulative Gain (nDCG@K) dengan nilai K = 5.
1. Precision@K:
  Precision@K mengukur proporsi rekomendasi yang benar-benar relevan di antara top-K produk yang direkomendasikan.

Formula: ![image](https://github.com/user-attachments/assets/ba2415ff-22b1-49a1-86cb-61cdb116f33a)

3. NDCG@K:
   NDCG@K memperhitungkan posisi produk relevan di dalam daftar rekomendasi, memberikan bobot lebih pada produk yang muncul di peringkat atas.
   
   Formula: ![image](https://github.com/user-attachments/assets/7d7475e3-e9c1-47af-8f31-b556818e7f4b)
   dengan ![image](https://github.com/user-attachments/assets/9896e2aa-c3ac-40c0-b4b3-50c108276622)
   
Relevansi produk ditentukan berdasarkan kesamaan atribut Brand dan Gender dengan produk yang dijadikan query, dengan asumsi produk yang memiliki brand dan gender yang sama lebih relevan secara kontekstual. Berikut merupakan prosedur evaluasi yang dilakukan:

  1. Dipilih secara acak 100 produk dari dataset sebagai sampel uji.

  2. Untuk setiap produk sampel:

   - Cari produk-produk yang relevan berdasarkan brand dan gender yang sama (kecuali produk itu sendiri).
    
   -  Hitung skor kemiripan dengan dua metode:
    
   -  Cosine Similarity dari fitur TF-IDF.
    
   -  Euclidean Distance dari fitur TF-IDF (jarak terkecil berarti lebih mirip).
    
   -  Ambil top-5 rekomendasi untuk masing-masing metode.
    
   -  Hitung Precision@5 dan nDCG@5 dengan membandingkan rekomendasi dengan daftar relevan.

  3. Ambil rata-rata hasil metrik dari seluruh sampel.


Berikut merupakan hasil evaluasi model:
![image](https://github.com/user-attachments/assets/9bd77f20-223d-44b6-a5fd-2f5c6c807efd)

   
Berdasarkan hasil evaluasi tersebut dapat ditarik kesimpulan:
Cosine Similarity dan Euclidean Distance memberikan performa yang sama dalam merekomendasikan produk, dengan Precision@5 sekitar 64.6%—yang berarti 3-4 dari 5 rekomendasi berada dalam daftar produk relevan. nDCG@5 sebesar 68.04% menunjukkan bahwa posisi item relevan dalam daftar rekomendasi cukup baik, tetapi masih ada ruang untuk perbaikan dalam pengurutan hasil agar produk paling relevan muncul lebih awal.
