# Laporan Proyek Machine Learning - Sistem Rekomendasi Musik Spotify

## Project Overview

Industri musik digital telah mengalami transformasi besar dengan hadirnya platform streaming musik seperti Spotify, Apple Music, dan YouTube Music. Spotify, sebagai salah satu platform terdepan, memiliki lebih dari 400 juta pengguna aktif dengan katalog lebih dari 70 juta lagu [1]. Dengan jumlah konten yang sangat besar, pengguna sering mengalami kesulitan dalam menemukan musik yang sesuai dengan preferensi mereka - fenomena yang dikenal sebagai "information overload".

Sistem rekomendasi musik menjadi solusi krusial untuk mengatasi permasalahan ini. Menurut penelitian McKinsey, 35% pembelian di Amazon dan 75% tontonan di Netflix berasal dari sistem rekomendasi [2]. Dalam konteks musik, sistem rekomendasi yang efektif dapat meningkatkan engagement pengguna, mengurangi churn rate, dan membantu artis untuk menjangkau audiens yang tepat.

Proyek ini penting diselesaikan karena:
1. **User Experience**: Membantu pengguna menemukan musik baru yang sesuai dengan selera mereka
2. **Business Value**: Meningkatkan waktu mendengarkan dan retensi pengguna platform
3. **Music Discovery**: Mendukung eksplorasi musik dan membantu artis mendapatkan eksposur

Penelitian terdahulu menunjukkan bahwa content-based filtering menggunakan audio features memberikan hasil yang baik untuk rekomendasi musik [3]. Audio features seperti danceability, energy, dan valence terbukti efektif dalam mengkarakterisasi preferensi musik pengguna.

**Referensi:**
[1] Spotify Technology S.A. (2023). "Spotify Q3 2023 Earnings Report"
[2] McKinsey & Company. (2016). "How retailers can keep up with consumers"
[3] Schedl, M., et al. (2018). "Music recommendation systems: Techniques, use cases, and challenges"

## Business Understanding

Dalam era digital music streaming, platform musik menghadapi tantangan besar dalam memberikan pengalaman yang personal dan relevan kepada pengguna. Dengan jutaan lagu yang tersedia, pengguna membutuhkan bantuan untuk menemukan musik yang sesuai dengan preferensi mereka.

### Problem Statements

1. **Music Discovery Challenge**: Bagaimana membantu pengguna menemukan lagu-lagu baru yang sesuai dengan preferensi musik mereka berdasarkan karakteristik audio dari lagu yang mereka sukai?

2. **Personalization Gap**: Bagaimana menciptakan sistem rekomendasi yang dapat memberikan saran musik yang personal dan akurat berdasarkan fitur-fitur audio dari lagu?

3. **Content Exploration**: Bagaimana memfasilitasi eksplorasi musik lintas genre sambil tetap mempertahankan relevansi dengan preferensi pengguna?

### Goals

1. **Mengembangkan sistem rekomendasi musik** yang dapat memberikan rekomendasi lagu berdasarkan kesamaan karakteristik audio dengan lagu yang dipilih pengguna.

2. **Menciptakan pengalaman discovery yang personal** dengan memanfaatkan audio features seperti danceability, energy, valence, dan tempo untuk mengidentifikasi lagu-lagu serupa.

3. **Membangun framework evaluasi** yang dapat mengukur kualitas rekomendasi berdasarkan kesamaan genre dan karakteristik musik.

### Solution Statements

Untuk mencapai goals yang telah ditetapkan, diusulkan dua pendekatan solution:

1. **Content-Based Filtering dengan Audio Features**: 
   - Menggunakan cosine similarity untuk mengukur kesamaan antar lagu berdasarkan fitur audio
   - Memanfaatkan fitur seperti danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness, loudness, dan popularity
   - Mengimplementasikan scaling untuk normalisasi fitur

2. **Hybrid Approach dengan Genre Filtering**:
   - Menggabungkan content-based filtering dengan constraint berbasis genre
   - Memberikan opsi untuk memfilter rekomendasi berdasarkan genre tertentu
   - Memungkinkan eksplorasi dalam genre yang sama atau lintas genre

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "30000 Spotify Songs" yang dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs). Dataset ini berisi informasi mengenai 32,833 lagu dari Spotify dengan berbagai fitur audio dan metadata.

### Informasi Dataset:
- **Jumlah data**: 32,833 baris
- **Jumlah kolom**: 23 kolom
- **Kondisi data**: Dataset memiliki beberapa data duplikat dan tidak ada missing values yang signifikan
- **Sumber**: Spotify Web API melalui Kaggle

### Variabel-variabel pada Dataset:

**Metadata Lagu:**
- `track_id`: ID unik untuk setiap lagu
- `track_name`: Nama lagu
- `track_artist`: Nama artis
- `track_popularity`: Skor popularitas lagu (0-100)
- `track_album_id`: ID album
- `track_album_name`: Nama album
- `track_album_release_date`: Tanggal rilis album

**Informasi Playlist:**
- `playlist_name`: Nama playlist
- `playlist_id`: ID playlist
- `playlist_genre`: Genre playlist (pop, rap, rock, latin, r&b, edm)
- `playlist_subgenre`: Sub-genre playlist

**Audio Features (Fitur Utama untuk Rekomendasi):**
- `danceability`: Kemampuan lagu untuk menari (0.0-1.0)
- `energy`: Intensitas dan kekuatan lagu (0.0-1.0)
- `key`: Nada dasar lagu
- `loudness`: Kekuatan suara dalam desibel
- `mode`: Modalitas lagu (major/minor)
- `speechiness`: Keberadaan kata-kata dalam lagu (0.0-1.0)
- `acousticness`: Tingkat akustik lagu (0.0-1.0)
- `instrumentalness`: Tingkat instrumental lagu (0.0-1.0)
- `liveness`: Keberadaan audiens dalam rekaman (0.0-1.0)
- `valence`: Tingkat positifitas musik (0.0-1.0)
- `tempo`: Kecepatan lagu dalam BPM
- `duration_ms`: Durasi lagu dalam milidetik

### Exploratory Data Analysis (EDA):

1. **Distribusi Genre**: 
   - Dataset didominasi oleh genre pop, diikuti rap, rock, latin, r&b, dan edm
   - Distribusi yang tidak seimbang dapat mempengaruhi kualitas rekomendasi

2. **Korelasi Audio Features**:
   - Energy dan loudness memiliki korelasi positif yang kuat (r ≈ 0.7)
   - Acousticness dan energy memiliki korelasi negatif (r ≈ -0.6)
   - Valence dan danceability menunjukkan korelasi positif sedang

3. **Data Quality**:
   - Terdapat duplikasi berdasarkan kombinasi track_name dan track_artist
   - Tidak ada missing values pada fitur-fitur utama yang digunakan

## Data Preparation

Tahapan data preparation dilakukan untuk memastikan kualitas data dan mempersiapkan fitur yang akan digunakan dalam sistem rekomendasi.

### Teknik Data Preparation:

1. **Feature Selection**:
   ```python
   features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
               'instrumentalness', 'liveness', 'valence', 'tempo', 'track_popularity']
   ```
   - Memilih 10 fitur yang paling relevan untuk karakterisasi musik
   - Menggabungkan audio features dengan track popularity untuk meningkatkan kualitas rekomendasi

2. **Missing Value Handling**:
   ```python
   df = df.dropna(subset=features)
   ```
   - Menghapus baris yang memiliki missing values pada fitur yang dipilih
   - Pendekatan ini dipilih karena jumlah missing values relatif kecil

3. **Duplicate Removal**:
   ```python
   df_unique = df.drop_duplicates(subset=['track_name', 'track_artist']).copy()
   ```
   - Menghilangkan duplikasi berdasarkan kombinasi nama lagu dan artis
   - Penting untuk menghindari bias dalam sistem rekomendasi

4. **Index Preparation**:
   ```python
   df_features.set_index(['track_name', 'track_artist'], inplace=True)
   ```
   - Menggunakan kombinasi nama lagu dan artis sebagai index
   - Memudahkan pencarian dan identifikasi lagu

5. **Feature Scaling**:
   ```python
   scaler = RobustScaler()
   df_scaled[features] = scaler.fit_transform(df_scaled[features])
   ```
   - Menggunakan RobustScaler untuk normalisasi fitur
   - Dipilih karena lebih robust terhadap outliers dibanding StandardScaler

### Alasan Data Preparation:

- **Feature Selection**: Memfokuskan pada fitur yang paling berpengaruh terhadap karakteristik musik
- **Missing Value Handling**: Memastikan konsistensi data untuk perhitungan similarity
- **Duplicate Removal**: Menghindari bias dan over-representation lagu tertentu
- **Scaling**: Memastikan semua fitur memiliki kontribusi yang seimbang dalam perhitungan similarity

## Modeling and Result

Sistem rekomendasi yang dikembangkan menggunakan pendekatan Content-Based Filtering dengan cosine similarity. Model ini menganalisis kesamaan karakteristik audio antar lagu untuk memberikan rekomendasi.

### Algoritma yang Digunakan:

#### 1. Content-Based Filtering dengan Cosine Similarity

**Formula Cosine Similarity:**
```
cosine_similarity(A,B) = (A·B) / (||A|| × ||B||)
```

**Implementasi:**
```python
similarity = pd.DataFrame(data=cosine_similarity(df_scaled[features]),
                          index=df_scaled.index,
                          columns=df_scaled.index)
```

**Cara Kerja:**
- Menghitung similarity matrix antar semua lagu berdasarkan audio features
- Menggunakan cosine similarity untuk mengukur kesamaan dalam ruang multi-dimensi
- Memberikan rekomendasi berdasarkan lagu dengan similarity score tertinggi

#### 2. Hybrid Approach dengan Genre Filtering

**Implementasi:**
```python
def recommendation_system(song_name, artist_name=None, genre_filter=None, num=10):
    # Content-based filtering
    sim_scores = similarity.loc[key].sort_values(ascending=False)
    
    # Genre filtering (optional)
    if genre_filter:
        genre_mask = df.set_index(['track_name', 'track_artist'])['playlist_genre'] == genre_filter
        genre_index = genre_mask[genre_mask].index
        sim_scores = sim_scores[sim_scores.index.isin(genre_index)]
```

### Kelebihan dan Kekurangan:

**Content-Based Filtering:**

*Kelebihan:*
- Tidak memerlukan data pengguna lain (tidak ada cold start problem)
- Dapat memberikan rekomendasi untuk lagu baru
- Transparansi tinggi dalam menjelaskan mengapa lagu direkomendasikan
- Konsisten dengan preferensi pengguna

*Kekurangan:*
- Terbatas pada kesamaan fitur yang ada
- Kurang dalam memberikan surprise/serendipity
- Tidak dapat menangkap preferensi kompleks yang tidak tercermin dalam audio features
- Over-specialization (terlalu mirip dengan input)

**Hybrid dengan Genre Filtering:**

*Kelebihan:*
- Memberikan kontrol lebih kepada pengguna
- Dapat membatasi rekomendasi pada genre tertentu
- Menggabungkan content similarity dengan categorical constraint

*Kekurangan:*
- Masih bergantung pada kualitas labeling genre
- Dapat membatasi eksplorasi lintas genre

### Contoh Output Rekomendasi:

Untuk lagu "Shape of You" by Ed Sheeran dengan filter genre "pop":

| Track Name | Artist | Similarity Score |
|------------|--------|------------------|
| Castle on the Hill | Ed Sheeran | 0.847 |
| Perfect | Ed Sheeran | 0.832 |
| Thinking Out Loud | Ed Sheeran | 0.798 |
| Photograph | Ed Sheeran | 0.785 |
| What Do You Mean? | Justin Bieber | 0.743 |

## Evaluation

### Metrik Evaluasi: Precision@K

Metrik evaluasi yang digunakan adalah **Precision@K**, yang mengukur proporsi rekomendasi yang relevan dalam top-K rekomendasi.

**Formula Precision@K:**
```
Precision@K = (Jumlah item relevan dalam top-K rekomendasi) / K
```

**Cara Kerja Metrik:**
1. Mengambil genre dari lagu input sebagai ground truth
2. Mendapatkan top-K rekomendasi dari sistem
3. Menghitung berapa banyak rekomendasi yang memiliki genre sama dengan input
4. Membagi dengan K untuk mendapatkan precision

**Implementasi:**
```python
def evaluate_recommendation(song_name, artist_name, top_k=10):
    genre_asli = df.set_index(['track_name', 'track_artist']).loc[key]['playlist_genre']
    
    sim_scores = similarity.loc[key].sort_values(ascending=False)
    top_recs = sim_scores.head(top_k).index
    
    genres_recs = df.set_index(['track_name', 'track_artist']).loc[top_recs]['playlist_genre']
    genre_match = (genres_recs == genre_asli).sum()
    precision_at_k = genre_match / top_k
    
    return precision_at_k
```

### Hasil Evaluasi:

**Evaluasi Individual:**
- Precision@10 untuk "Shape of You" by Ed Sheeran: **0.80**

**Evaluasi Batch (10 lagu populer):**
- Mean Precision@10: **0.73**

**Analisis Hasil:**
1. **Performa Baik**: Precision rata-rata 0.73 menunjukkan bahwa 73% rekomendasi memiliki genre yang sama dengan lagu input
2. **Konsistensi**: Sebagian besar lagu mendapat precision score di atas 0.6
3. **Variabilitas**: Beberapa lagu mendapat score tinggi (>0.8) sementara lainnya lebih rendah

### Interpretasi Metrik:

**Mengapa Precision@K Dipilih:**
- Sesuai untuk sistem rekomendasi top-N
- Mudah diinterpretasi oleh stakeholder bisnis
- Relevan dengan user experience (pengguna biasanya hanya melihat beberapa rekomendasi teratas)

**Keterbatasan:**
- Menggunakan genre sebagai proxy untuk relevansi mungkin terlalu sederhana
- Tidak mengukur diversity atau novelty rekomendasi
- Tidak mempertimbangkan preferensi individual pengguna

### Kesimpulan Evaluasi:

Sistem rekomendasi yang dikembangkan menunjukkan performa yang baik dengan precision rata-rata 73%. Hal ini menunjukkan bahwa audio features efektif dalam mengidentifikasi lagu-lagu serupa dalam genre yang sama. Namun, untuk aplikasi real-world, diperlukan evaluasi tambahan dengan metrik seperti diversity, novelty, dan user satisfaction.

---

## Kesimpulan

Proyek ini berhasil mengembangkan sistem rekomendasi musik berbasis content-based filtering yang efektif. Dengan memanfaatkan audio features dan cosine similarity, sistem dapat memberikan rekomendasi lagu yang relevan dengan precision rata-rata 73%. Implementasi hybrid approach dengan genre filtering memberikan fleksibilitas tambahan bagi pengguna untuk mengontrol jenis rekomendasi yang diinginkan.

Sistem ini dapat diimplementasikan dalam platform musik streaming untuk meningkatkan user experience dan membantu pengguna dalam menemukan musik baru yang sesuai dengan preferensi mereka.
