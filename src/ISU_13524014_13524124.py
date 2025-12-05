def probstat():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')

    # Load dataset
    df = pd.read_csv('utility.csv')

    print("="*60)
    print("TUGAS BESAR IF2120 - PROBABILITAS DAN STATISTIKA")
    print("="*60)

    # Warna visualisasi
    COLOR_1 = '#2E86AB'
    COLOR_2 = '#A23B72'

    # ======================================================================================
    # SOAL 1: ANALISIS STATISTIK DESKRIPTIF
    # ======================================================================================

    print("\n\n" + "="*60)
    print("SOAL 1: ANALISIS STATISTIK DESKRIPTIF")
    print("="*60)

    # Fungsi manual untuk statistik
    def hitungMean(data):
        return sum(data) / len(data)

    def hitungMedian(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            return sorted_data[n//2]

    def hitungModus(data):
        freq = {}
        for val in data:
            freq[val] = freq.get(val, 0) + 1
        max_freq = max(freq.values())
        modes = [k for k, v in freq.items() if v == max_freq]
        return modes[0] if len(modes) == 1 else modes

    def hitungVariansi(data):
        mean = hitungMean(data)
        return sum((x - mean) ** 2 for x in data) / len(data)

    def hitungStdev(data):
        return hitungVariansi(data) ** 0.5

    def hitungKuartil(data, q):
        sorted_data = sorted(data)
        n = len(sorted_data)
        pos = (q / 4) * (n + 1)
        if pos.is_integer():
            return sorted_data[int(pos) - 1]
        else:
            lower = int(pos) - 1
            upper = lower + 1
            if upper >= n:
                return sorted_data[lower]
            fraction = pos - int(pos)
            return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])

    def hitungSkewness(data):
        n = len(data)
        mean = hitungMean(data)
        std = hitungStdev(data)
        return sum(((x - mean) / std) ** 3 for x in data) / n

    def hitung_kurtosis(data):
        n = len(data)
        mean = hitungMean(data)
        std = hitungStdev(data)
        return sum(((x - mean) / std) ** 4 for x in data) / n - 3

    # Identifikasi kolom
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"\nKolom Numerik: {list(numeric_cols)}")
    print(f"Kolom Kategorikal: {list(categorical_cols)}")

    # Analisis kolom numerik
    for col in numeric_cols:
        print(f"\n{'-'*80}")
        print(f"Kolom: {col} (Numerik)")
        print(f"{'-'*80}")
        
        data = df[col].dropna().tolist()
        
        # Hitung manual
        mean_manual = hitungMean(data)
        median_manual = hitungMedian(data)
        modus_manual = hitungModus(data)
        var_manual = hitungVariansi(data)
        std_manual = hitungStdev(data)
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        q1_manual = hitungKuartil(data, 1)
        q2_manual = hitungKuartil(data, 2)
        q3_manual = hitungKuartil(data, 3)
        iqr_manual = q3_manual - q1_manual
        skew_manual = hitungSkewness(data)
        kurt_manual = hitung_kurtosis(data)
        
        # Hitung library
        data_series = df[col].dropna()
        mean_lib = data_series.mean()
        median_lib = data_series.median()
        modus_lib = data_series.mode()[0] if len(data_series.mode()) > 0 else modus_manual
        var_lib = data_series.var(ddof=0)
        std_lib = data_series.std(ddof=0)
        min_lib = data_series.min()
        max_lib = data_series.max()
        range_lib = max_lib - min_lib
        q1_lib = data_series.quantile(0.25)
        q2_lib = data_series.quantile(0.50)
        q3_lib = data_series.quantile(0.75)
        iqr_lib = q3_lib - q1_lib
        skew_lib = stats.skew(data_series)
        kurt_lib = stats.kurtosis(data_series)
        
        # Tampilkan perbandingan
        print(f"\n{'Statistik':<20} {'Manual':<20} {'Library':<20} {'Selisih':<20}")
        print(f"{'-'*80}")
        print(f"{'Mean':<20} {mean_manual:<20.6f} {mean_lib:<20.6f} {abs(mean_manual-mean_lib):<20.8f}")
        print(f"{'Median':<20} {median_manual:<20.6f} {median_lib:<20.6f} {abs(median_manual-median_lib):<20.8f}")
        print(f"{'Modus':<20} {modus_manual if isinstance(modus_manual, (int, float)) else modus_manual[0]:<20.6f} {modus_lib:<20.6f} {'-':<20}")
        print(f"{'Std Deviasi':<20} {std_manual:<20.6f} {std_lib:<20.6f} {abs(std_manual-std_lib):<20.8f}")
        print(f"{'Variansi':<20} {var_manual:<20.6f} {var_lib:<20.6f} {abs(var_manual-var_lib):<20.8f}")
        print(f"{'Min':<20} {min_val:<20.6f} {min_lib:<20.6f} {abs(min_val-min_lib):<20.8f}")
        print(f"{'Max':<20} {max_val:<20.6f} {max_lib:<20.6f} {abs(max_val-max_lib):<20.8f}")
        print(f"{'Range':<20} {range_val:<20.6f} {range_lib:<20.6f} {abs(range_val-range_lib):<20.8f}")
        print(f"{'Q1':<20} {q1_manual:<20.6f} {q1_lib:<20.6f} {abs(q1_manual-q1_lib):<20.8f}")
        print(f"{'Q2':<20} {q2_manual:<20.6f} {q2_lib:<20.6f} {abs(q2_manual-q2_lib):<20.8f}")
        print(f"{'Q3':<20} {q3_manual:<20.6f} {q3_lib:<20.6f} {abs(q3_manual-q3_lib):<20.8f}")
        print(f"{'IQR':<20} {iqr_manual:<20.6f} {iqr_lib:<20.6f} {abs(iqr_manual-iqr_lib):<20.8f}")
        print(f"{'Skewness':<20} {skew_manual:<20.6f} {skew_lib:<20.6f} {abs(skew_manual-skew_lib):<20.8f}")
        print(f"{'Kurtosis':<20} {kurt_manual:<20.6f} {kurt_lib:<20.6f} {abs(kurt_manual-kurt_lib):<20.8f}")

    # Analisis kolom kategorikal
    for col in categorical_cols:
        print(f"\n{'-'*60}")
        print(f"Kolom: {col} (Kategorikal)")
        print(f"{'-'*60}")
        
        data = df[col].dropna().tolist()
        
        freq = {}
        for val in data:
            freq[val] = freq.get(val, 0) + 1
        
        unique_vals = len(freq)
        total = len(data)
        
        print(f"Unique Values: {unique_vals}")
        print(f"Total Data: {total}")
        print(f"\nFrekuensi dan Persentase:")
        for kategori, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            persen = (count / total) * 100
            print(f"  {kategori}: {count} ({persen:.2f}%)")

    # ======================================================================================
    # SOAL 2: DETEKSI, VISUALISASI, DAN PENANGANAN OUTLIER
    # ======================================================================================

    print("\n\n" + "="*60)
    print("SOAL 2: DETEKSI, VISUALISASI, DAN PENANGANAN OUTLIER")
    print("="*60)

    # Fungsi deteksi outlier
    def deteksiOutlier_iqr(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        q1_pos = 0.25 * (n + 1)
        if q1_pos.is_integer():
            q1 = sorted_data[int(q1_pos) - 1]
        else:
            lower = int(q1_pos) - 1
            upper = lower + 1
            fraction = q1_pos - int(q1_pos)
            q1 = sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
        
        q3_pos = 0.75 * (n + 1)
        if q3_pos.is_integer():
            q3 = sorted_data[int(q3_pos) - 1]
        else:
            lower = int(q3_pos) - 1
            upper = lower + 1
            fraction = q3_pos - int(q3_pos)
            q3 = sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])
        
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        outlier_indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        
        return {
            'q1': q1, 'q3': q3, 'iqr': iqr,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'outliers': outliers, 'outlier_indices': outlier_indices,
            'n_outliers': len(outliers),
            'pct_outliers': (len(outliers) / len(data)) * 100
        }

    outlier_results = {}

    for col in numeric_cols:
        data = df[col].dropna().tolist()
        result = deteksiOutlier_iqr(data)
        outlier_results[col] = result
        
        print(f"\n{col}:")
        print(f"  Q1: {result['q1']:.4f}")
        print(f"  Q3: {result['q3']:.4f}")
        print(f"  IQR: {result['iqr']:.4f}")
        print(f"  Batas Bawah: {result['lower_bound']:.4f}")
        print(f"  Batas Atas: {result['upper_bound']:.4f}")
        print(f"  Jumlah Outlier: {result['n_outliers']} ({result['pct_outliers']:.2f}%)")

    # Visualisasi box plot
    numeric_list = list(numeric_cols)
    groups = [numeric_list[:3], numeric_list[3:6], numeric_list[6:]]

    for idx, group in enumerate(groups, 1):
        if len(group) == 0:
            continue
        
        if len(group) <= 3:
            fig, axes = plt.subplots(1, len(group), figsize=(18, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        fig.suptitle(f'Box Plot - Deteksi Outlier Part {idx}', fontsize=14, fontweight='bold')
        
        if len(group) == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if len(group) > 3 else axes
        
        for i, col in enumerate(group):
            ax = axes[i]
            data = df[col].dropna()
            
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=COLOR_1, alpha=0.6),
                        medianprops=dict(color=COLOR_2, linewidth=2),
                        whiskerprops=dict(color=COLOR_2, linewidth=1.5),
                        capprops=dict(color=COLOR_2, linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor=COLOR_2, markersize=5, alpha=0.5))
            
            result = outlier_results[col]
            ax.set_title(f'{col}\nOutliers: {result["n_outliers"]} ({result["pct_outliers"]:.1f}%)', fontsize=10)
            ax.set_ylabel('Nilai')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print(f"\nMenampilkan box plot part {idx}...")
        plt.show(block=True)

    print("\n\nPenjelasan Penanganan Outlier:")
    print("-"*60)
    print("""
    Metode umum untuk menangani outlier:

    1. Menghapus Outlier
    - Cocok jika outlier adalah error/noise
    - Hanya jika outlier sedikit (<5%)
    
    2. Capping/Winsorizing
    - Ganti outlier dengan batas atas/bawah
    - Tidak kehilangan data
    
    3. Ganti dengan Median
    - Median tidak terpengaruh outlier
    - Cocok untuk data skewed
    
    4. Ganti dengan Mean
    - Cocok untuk data normal
    - Mean mudah terpengaruh outlier
    
    5. Transformasi
    - Log, sqrt, box-cox
    - Mengurangi pengaruh outlier
    """)

    # ======================================================================================
    # SOAL 3: VISUALISASI DISTRIBUSI
    # ======================================================================================

    print("\n\n" + "="*60)
    print("SOAL 3: VISUALISASI DISTRIBUSI")
    print("="*60)

    # Histogram untuk numerik
    histogram_groups = []
    for i in range(0, len(numeric_list), 2):
        histogram_groups.append(numeric_list[i:i+2])

    for idx, group in enumerate(histogram_groups, 1):
        if len(group) == 0:
            continue
        
        n_plots = len(group)
        fig, axes = plt.subplots(1, n_plots, figsize=(16, 6))
        
        fig.suptitle(f'Histogram - Distribusi Data Numerik Part {idx}', fontsize=14, fontweight='bold')
        
        if n_plots == 1:
            axes = [axes]
        
        for i, col in enumerate(group):
            ax = axes[i]
            data = df[col].dropna()
            
            ax.hist(data, bins=30, color=COLOR_1, alpha=0.7, edgecolor=COLOR_2)
            ax.set_title(f'{col}', fontsize=11)
            ax.set_xlabel('Nilai')
            ax.set_ylabel('Frekuensi')
            ax.grid(True, alpha=0.3)
            
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.legend()
        
        plt.tight_layout()
        print(f"\nMenampilkan histogram part {idx}...")
        plt.show(block=True)

    # Bar chart untuk kategorikal
    if len(categorical_cols) > 0:
        n_cat = len(categorical_cols)
        if n_cat <= 3:
            fig, axes = plt.subplots(1, n_cat, figsize=(18, 6))
        else:
            rows = (n_cat + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(18, 6*rows))
        
        fig.suptitle('Bar Chart - Distribusi Data Kategorikal', fontsize=14, fontweight='bold')
        
        if n_cat == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            
            freq = df[col].value_counts()
            ax.bar(range(len(freq)), freq.values, color=COLOR_1, alpha=0.7, edgecolor=COLOR_2)
            ax.set_title(f'{col}', fontsize=11)
            ax.set_xlabel('Kategori')
            ax.set_ylabel('Frekuensi')
            ax.set_xticks(range(len(freq)))
            ax.set_xticklabels(freq.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            for j, val in enumerate(freq.values):
                pct = (val / len(df[col].dropna())) * 100
                ax.text(j, val, f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        print("\nMenampilkan bar chart kategorikal...")
        plt.show(block=True)

    print("\n\nPenjelasan Kondisi Setiap Kolom:")
    print("-"*60)

    for col in numeric_cols:
        data = df[col].dropna()
        mean = data.mean()
        median = data.median()
        skew = stats.skew(data)
        
        print(f"\n{col}:")
        if abs(skew) < 0.5:
            print(f"  - Distribusi cukup simetris (skewness={skew:.2f})")
        elif skew > 0.5:
            print(f"  - Distribusi miring kanan/positively skewed (skewness={skew:.2f})")
            print(f"  - Ada nilai tinggi yang menarik mean ke kanan")
        else:
            print(f"  - Distribusi miring kiri/negatively skewed (skewness={skew:.2f})")
            print(f"  - Ada nilai rendah yang menarik mean ke kiri")
        
        if mean > median:
            print(f"  - Mean ({mean:.2f}) > Median ({median:.2f})")
        elif mean < median:
            print(f"  - Mean ({mean:.2f}) < Median ({median:.2f})")
        else:
            print(f"  - Mean ≈ Median ({mean:.2f})")

    for col in categorical_cols:
        freq = df[col].value_counts()
        dominant = freq.index[0]
        dominant_pct = (freq.values[0] / len(df[col].dropna())) * 100
        
        print(f"\n{col}:")
        print(f"  - Jumlah kategori: {len(freq)}")
        print(f"  - Kategori dominan: {dominant} ({dominant_pct:.1f}%)")
        if dominant_pct > 50:
            print(f"  - Distribusi tidak seimbang, satu kategori sangat dominan")
        else:
            print(f"  - Distribusi cukup seimbang")

    # ======================================================================================
    # SOAL 4: UJI NORMALITAS
    # ======================================================================================

    print("\n\n" + "="*60)
    print("SOAL 4: UJI NORMALITAS")
    print("="*60)

    normality_results = {}

    for col in numeric_cols:
        data = df[col].dropna()
        
        # Shapiro-Wilk test
        if len(data) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Skewness dan Kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        is_normal = False
        if shapiro_p:
            is_normal = shapiro_p > 0.05
        else:
            is_normal = ks_p > 0.05
        
        normality_results[col] = {
            'shapiro_p': shapiro_p,
            'ks_p': ks_p,
            'skew': skew,
            'kurt': kurt,
            'is_normal': is_normal
        }

    # Visualisasi Q-Q Plot
    normality_groups = []
    for i in range(0, len(numeric_list), 2):
        normality_groups.append(numeric_list[i:i+2])

    for idx, group in enumerate(normality_groups, 1):
        if len(group) == 0:
            continue
        
        n_plots = len(group)
        fig = plt.figure(figsize=(16, 6*n_plots))
        fig.suptitle(f'Uji Normalitas: Histogram & Q-Q Plot Part {idx}', fontsize=14, fontweight='bold')
        
        for i, col in enumerate(group):
            data = df[col].dropna()
            
            # Histogram
            ax1 = fig.add_subplot(n_plots, 2, i*2 + 1)
            ax1.hist(data, bins=30, density=True, color=COLOR_1, alpha=0.7, edgecolor=COLOR_2)
            
            mu = data.mean()
            sigma = data.std()
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
            ax1.plot(x, normal_curve, color=COLOR_2, linewidth=2, linestyle='--', label='Normal')
            
            ax1.set_title(f'{col} - Histogram')
            ax1.set_xlabel('Nilai')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q Plot
            ax2 = fig.add_subplot(n_plots, 2, i*2 + 2)
            stats.probplot(data, dist="norm", plot=ax2)
            ax2.get_lines()[0].set_color(COLOR_1)
            ax2.get_lines()[0].set_markersize(4)
            ax2.get_lines()[1].set_color(COLOR_2)
            ax2.get_lines()[1].set_linewidth(2)
            ax2.set_title(f'{col} - Q-Q Plot')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print(f"\nMenampilkan uji normalitas part {idx}...")
        plt.show(block=True)

    print("\n\nHasil Uji Normalitas:")
    print("-"*60)

    for col in numeric_cols:
        result = normality_results[col]
        
        print(f"\n{col}:")
        if result['shapiro_p']:
            print(f"  Shapiro-Wilk p-value: {result['shapiro_p']:.6f}")
        print(f"  Kolmogorov-Smirnov p-value: {result['ks_p']:.6f}")
        print(f"  Skewness: {result['skew']:.4f}")
        print(f"  Kurtosis: {result['kurt']:.4f}")
        
        if result['is_normal']:
            print(f"  Kesimpulan: BERDISTRIBUSI NORMAL (p > 0.05)")
        else:
            print(f"  Kesimpulan: TIDAK BERDISTRIBUSI NORMAL (p < 0.05)")
            
            if abs(result['skew']) < 0.5 and abs(result['kurt']) < 0.5:
                print(f"  Tipe: Mendekati normal tapi tidak sempurna")
            elif result['skew'] > 1:
                print(f"  Tipe: Positively skewed (miring kanan)")
            elif result['skew'] < -1:
                print(f"  Tipe: Negatively skewed (miring kiri)")
            elif result['kurt'] > 1:
                print(f"  Tipe: Leptokurtic (heavy-tailed)")
            elif result['kurt'] < -1:
                print(f"  Tipe: Platykurtic (light-tailed)")
            else:
                print(f"  Tipe: Non-normal dengan karakteristik campuran")

    # ======================================================================================
    # SOAL 5: UJI HIPOTESIS
    # ======================================================================================

    print("\n\n" + "="*60)
    print("SOAL 5: UJI HIPOTESIS")
    print("="*60)
    print("\nTingkat signifikansi α = 0.05")

    # ======================================================================================
    # SOAL 5a: Uji rata-rata penggunaan air per hewan per bulan
    # ======================================================================================

    print("\n\n" + "-"*60)
    print("SOAL 5a: Uji Rata-rata Penggunaan Air per Hewan per Bulan")
    print("-"*60)

    print("\nHipotesis:")
    print("H0: μ >= 3000 liter/bulan per hewan")
    print("H1: μ < 3000 liter/bulan per hewan")
    print("(Uji satu sisi - ekor kiri)")

    # Hitung penggunaan air per hewan per bulan
    df['water_per_animal'] = df['water_liter'] / df['household_size']

    # Statistik deskriptif
    water_per_animal = df['water_per_animal'].dropna()
    n_a = len(water_per_animal)
    mean_a = water_per_animal.mean()
    std_a = water_per_animal.std(ddof=1)
    se_a = std_a / np.sqrt(n_a)

    print(f"\nStatistik Deskriptif:")
    print(f"  n = {n_a}")
    print(f"  Mean = {mean_a:.4f} liter/bulan")
    print(f"  Std Dev = {std_a:.4f}")
    print(f"  Standard Error = {se_a:.4f}")

    # Uji t (one-sample, one-tailed)
    mu_0_a = 3000
    t_stat_a = (mean_a - mu_0_a) / se_a
    df_a = n_a - 1
    p_value_a = stats.t.cdf(t_stat_a, df_a)  # CDF untuk ekor kiri
    t_critical_a = stats.t.ppf(0.05, df_a)

    print(f"\nHasil Uji t:")
    print(f"  t-statistik = {t_stat_a:.4f}")
    print(f"  derajat kebebasan = {df_a}")
    print(f"  p-value = {p_value_a:.6f}")
    print(f"  t-critical (α=0.05) = {t_critical_a:.4f}")

    print(f"\nKeputusan:")
    if p_value_a < 0.05:
        print(f"  Tolak H0 (p-value {p_value_a:.6f} < 0.05)")
        print(f"  Kesimpulan: Rata-rata penggunaan air per hewan per bulan")
        print(f"  KURANG DARI 3000 liter (memenuhi rekomendasi PHH)")
    else:
        print(f"  Gagal tolak H0 (p-value {p_value_a:.6f} >= 0.05)")
        print(f"  Kesimpulan: Tidak cukup bukti bahwa rata-rata penggunaan air")
        print(f"  kurang dari 3000 liter per hewan per bulan")

    # ======================================================================================
    # SOAL 5b: Uji standar deviasi penggunaan listrik
    # ======================================================================================

    print("\n\n" + "-"*60)
    print("SOAL 5b: Uji Standar Deviasi Penggunaan Listrik")
    print("-"*60)

    print("\nHipotesis:")
    print("H0: σ = 300 kWh")
    print("H1: σ ≠ 300 kWh")
    print("(Uji dua sisi)")

    # Statistik deskriptif
    electricity = df['electricity_kwh'].dropna()
    n_b = len(electricity)
    std_b = electricity.std(ddof=1)
    var_b = std_b ** 2

    sigma_0_b = 300
    var_0_b = sigma_0_b ** 2

    print(f"\nStatistik Deskriptif:")
    print(f"  n = {n_b}")
    print(f"  Std Dev (sampel) = {std_b:.4f} kWh")
    print(f"  Variance (sampel) = {var_b:.4f}")
    print(f"  Std Dev (populasi tahun lalu) = {sigma_0_b} kWh")

    # Uji Chi-Square untuk variansi
    chi2_stat_b = (n_b - 1) * var_b / var_0_b
    df_b = n_b - 1
    p_value_b = 2 * min(stats.chi2.cdf(chi2_stat_b, df_b), 
                        1 - stats.chi2.cdf(chi2_stat_b, df_b))
    chi2_critical_lower = stats.chi2.ppf(0.025, df_b)
    chi2_critical_upper = stats.chi2.ppf(0.975, df_b)

    print(f"\nHasil Uji Chi-Square:")
    print(f"  χ² statistik = {chi2_stat_b:.4f}")
    print(f"  v = {df_b}")
    print(f"  p-value = {p_value_b:.6f}")
    print(f"  χ² critical lower (α/2=0.025) = {chi2_critical_lower:.4f}")
    print(f"  χ² critical upper (α/2=0.025) = {chi2_critical_upper:.4f}")

    print(f"\nKeputusan:")
    if p_value_b < 0.05:
        print(f"  Tolak H0 (p-value {p_value_b:.6f} < 0.05)")
        print(f"  Kesimpulan: Standar deviasi penggunaan listrik tahun ini")
        print(f"  BERBEDA dengan tahun lalu (σ ≠ 300 kWh)")
    else:
        print(f"  Gagal tolak H0 (p-value {p_value_b:.6f} >= 0.05)")
        print(f"  Kesimpulan: Tidak cukup bukti bahwa standar deviasi berbeda")
        print(f"  (σ masih sama dengan 300 kWh)")

    # ======================================================================================
    # SOAL 5c: Uji proporsi rumah tangga dengan rating A atau B
    # ======================================================================================

    print("\n\n" + "-"*60)
    print("SOAL 5c: Uji Proporsi Rumah Tangga dengan Rating A atau B")
    print("-"*60)

    # Hitung proporsi A atau B
    energy_rating = df['energy_efficiency_rating'].dropna()
    n_c = len(energy_rating)
    x_c = sum((energy_rating == 'A') | (energy_rating == 'B'))
    p_hat_c = x_c / n_c

    print(f"\nData:")
    print(f"  Total rumah tangga = {n_c}")
    print(f"  Rumah dengan rating A atau B = {x_c}")
    print(f"  Proporsi sampel (p̂) = {p_hat_c:.4f} atau {p_hat_c*100:.2f}%")

    # Uji 1: Proporsi > 50%?
    print(f"\n{'='*60}")
    print("Uji 1: Apakah proporsi > 50%?")
    print(f"{'='*60}")
    print("\nHipotesis:")
    print("H0: p <= 0.50")
    print("H1: p > 0.50")
    print("(Uji satu sisi - ekor kanan)")

    p_0_c1 = 0.50
    se_c1 = np.sqrt(p_0_c1 * (1 - p_0_c1) / n_c)
    z_stat_c1 = (p_hat_c - p_0_c1) / se_c1
    p_value_c1 = 1 - stats.norm.cdf(z_stat_c1)
    z_critical_c1 = stats.norm.ppf(0.95)

    print(f"\nHasil Uji Z:")
    print(f"  z-statistik = {z_stat_c1:.4f}")
    print(f"  p-value = {p_value_c1:.6f}")
    print(f"  z-critical (α=0.05) = {z_critical_c1:.4f}")

    print(f"\nKeputusan:")
    if p_value_c1 < 0.05:
        print(f"  Tolak H0 (p-value {p_value_c1:.6f} < 0.05)")
        print(f"  Kesimpulan: Proporsi rumah tangga dengan rating A atau B")
        print(f"  LEBIH DARI 50% populasi")
    else:
        print(f"  Gagal tolak H0 (p-value {p_value_c1:.6f} >= 0.05)")
        print(f"  Kesimpulan: Tidak cukup bukti bahwa proporsi lebih dari 50%")

    # Uji 2: Proporsi > 60%?
    print(f"\n{'='*80}")
    print("Uji 2: Apakah proporsi > 60%?")
    print(f"{'='*80}")
    print("\nHipotesis:")
    print("H0: p <= 0.60")
    print("H1: p > 0.60")
    print("(Uji satu sisi - ekor kanan)")

    p_0_c2 = 0.60
    se_c2 = np.sqrt(p_0_c2 * (1 - p_0_c2) / n_c)
    z_stat_c2 = (p_hat_c - p_0_c2) / se_c2
    p_value_c2 = 1 - stats.norm.cdf(z_stat_c2)
    z_critical_c2 = stats.norm.ppf(0.95)

    print(f"\nHasil Uji Z:")
    print(f"  z-statistik = {z_stat_c2:.4f}")
    print(f"  p-value = {p_value_c2:.6f}")
    print(f"  z-critical (α=0.05) = {z_critical_c2:.4f}")

    print(f"\nKeputusan:")
    if p_value_c2 < 0.05:
        print(f"  Tolak H0 (p-value {p_value_c2:.6f} < 0.05)")
        print(f"  Kesimpulan: Proporsi rumah tangga dengan rating A atau B")
        print(f"  LEBIH DARI 60% populasi")
    else:
        print(f"  Gagal tolak H0 (p-value {p_value_c2:.6f} >= 0.05)")
        print(f"  Kesimpulan: Tidak cukup bukti bahwa proporsi lebih dari 60%")

    # ======================================================================================
    # SOAL 5d: Uji proporsi data musim panas vs musim dingin
    # ======================================================================================

    print("\n\n" + "-"*60)
    print("SOAL 5d: Uji Proporsi Data Musim Panas vs Musim Dingin")
    print("-"*60)

    print("\nHipotesis:")
    print("H0: p_summer = 0.50 (proporsi sama)")
    print("H1: p_summer ≠ 0.50 (proporsi tidak sama)")
    print("(Uji dua sisi)")

    # Hitung proporsi
    season = df['season'].dropna()
    n_d = len(season)
    x_summer = sum(season == 'Summer')
    x_winter = sum(season == 'Winter')
    p_hat_summer = x_summer / n_d
    p_hat_winter = x_winter / n_d

    print(f"\nData:")
    print(f"  Total data = {n_d}")
    print(f"  Data musim panas (Summer) = {x_summer} ({p_hat_summer*100:.2f}%)")
    print(f"  Data musim dingin (Winter) = {x_winter} ({p_hat_winter*100:.2f}%)")

    # Uji proporsi
    p_0_d = 0.50
    se_d = np.sqrt(p_0_d * (1 - p_0_d) / n_d)
    z_stat_d = (p_hat_summer - p_0_d) / se_d
    p_value_d = 2 * (1 - stats.norm.cdf(abs(z_stat_d)))
    z_critical_d = stats.norm.ppf(0.975)

    print(f"\nHasil Uji Z:")
    print(f"  z-statistik = {z_stat_d:.4f}")
    print(f"  p-value = {p_value_d:.6f}")
    print(f"  z-critical (α/2=0.025) = ±{z_critical_d:.4f}")

    print(f"\nKeputusan:")
    if p_value_d < 0.05:
        print(f"  Tolak H0 (p-value {p_value_d:.6f} < 0.05)")
        print(f"  Kesimpulan: Proporsi data musim panas dan musim dingin")
        print(f"  TIDAK SETARA (tidak 50:50)")
    else:
        print(f"  Gagal tolak H0 (p-value {p_value_d:.6f} >= 0.05)")
        print(f"  Kesimpulan: Proporsi data musim panas dan musim dingin SETARA")
        print(f"  (mendekati 50:50)")

    # ======================================================================================
    # RINGKASAN HASIL UJI HIPOTESIS
    # ======================================================================================

    print("\n\n" + "="*60)
    print("RINGKASAN HASIL UJI HIPOTESIS SOAL 5")
    print("="*60)

    print(f"\n5a. Rata-rata penggunaan air per hewan:")
    print(f"    p-value = {p_value_a:.6f}")
    print(f"    Keputusan: {'TOLAK H0' if p_value_a < 0.05 else 'GAGAL TOLAK H0'}")
    print(f"    Rata-rata < 3000 liter: {'YA' if p_value_a < 0.05 else 'TIDAK TERBUKTI'}")

    print(f"\n5b. Standar deviasi penggunaan listrik:")
    print(f"    p-value = {p_value_b:.6f}")
    print(f"    Keputusan: {'TOLAK H0' if p_value_b < 0.05 else 'GAGAL TOLAK H0'}")
    print(f"    σ masih 300 kWh: {'TIDAK' if p_value_b < 0.05 else 'YA'}")

    print(f"\n5c. Proporsi rumah dengan rating A/B:")
    print(f"    - Uji > 50%:")
    print(f"      p-value = {p_value_c1:.6f}")
    print(f"      Keputusan: {'TOLAK H0' if p_value_c1 < 0.05 else 'GAGAL TOLAK H0'}")
    print(f"      Proporsi > 50%: {'YA' if p_value_c1 < 0.05 else 'TIDAK TERBUKTI'}")
    print(f"    - Uji > 60%:")
    print(f"      p-value = {p_value_c2:.6f}")
    print(f"      Keputusan: {'TOLAK H0' if p_value_c2 < 0.05 else 'GAGAL TOLAK H0'}")
    print(f"      Proporsi > 60%: {'YA' if p_value_c2 < 0.05 else 'TIDAK TERBUKTI'}")

    print(f"\n5d. Proporsi data musim panas vs dingin:")
    print(f"    p-value = {p_value_d:.6f}")
    print(f"    Keputusan: {'TOLAK H0' if p_value_d < 0.05 else 'GAGAL TOLAK H0'}")
    print(f"    Proporsi setara (50:50): {'TIDAK' if p_value_d < 0.05 else 'YA'}")

    print("\n" + "="*60)
    print("SOAL 6: HIPOTESIS 2 SAMPEL")
    print("="*60)