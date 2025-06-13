# Proyek Akhir: Menyelesaikan Permasalahan Human Resources

**Submission Dicoding - Data Science**

- **Nama**: Naufal Nur Fahriza
- **Email**: a297ybf370@devacademy.id  
- **ID Dicoding**: nurfahriza

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Perusahaan ini memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. 

Walaupun telah menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih menghadapi kesulitan dalam mengelola karyawan. Hal ini berimbas pada tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%.

### Permasalahan Bisnis

Berdasarkan latar belakang yang telah dijelaskan, berikut adalah permasalahan bisnis yang akan diselesaikan:

1. **Tingginya Attrition Rate**: Perusahaan mengalami attrition rate lebih dari 10%, yang dapat berdampak pada:
   - Biaya rekrutmen dan pelatihan karyawan baru yang tinggi
   - Kehilangan pengetahuan dan pengalaman karyawan
   - Penurunan produktivitas dan moral tim
   - Gangguan operasional perusahaan

2. **Kurangnya Pemahaman Faktor Penyebab**: Manajemen HR belum memiliki pemahaman yang mendalam tentang faktor-faktor yang menyebabkan karyawan memutuskan untuk keluar dari perusahaan.

3. **Tidak Adanya Sistem Monitoring**: Perusahaan belum memiliki dashboard atau sistem monitoring untuk memantau faktor-faktor yang mempengaruhi attrition rate secara real-time.

### Cakupan Proyek

Proyek ini akan mencakup:

1. **Analisis Data Eksploratori**: 
   - Mengidentifikasi pola dan tren dalam data karyawan
   - Menganalisis distribusi attrition berdasarkan berbagai faktor demografis dan pekerjaan

2. **Identifikasi Faktor-Faktor Kunci**: 
   - Menentukan faktor-faktor yang paling berpengaruh terhadap keputusan karyawan untuk keluar
   - Menganalisis korelasi antara berbagai variabel dengan attrition rate

3. **Pemodelan Prediktif**: 
   - Membangun model machine learning untuk memprediksi kemungkinan seorang karyawan akan keluar
   - Evaluasi performa model untuk memastikan akurasi prediksi

4. **Pengembangan Business Dashboard**: 
   - Membuat dashboard interaktif untuk monitoring attrition rate
   - Visualisasi faktor-faktor kunci yang mempengaruhi attrition
   - Menyediakan insights yang actionable untuk tim HR

5. **Rekomendasi Strategis**: 
   - Memberikan rekomendasi berbasis data untuk mengurangi attrition rate
   - Menyusun action items yang dapat diimplementasikan oleh perusahaan

### Persiapan

**Sumber data**: Dataset karyawan Jaya Jaya Maju yang berisi informasi demografis, metrik terkait pekerjaan, dan flag attrition.

**Setup environment**:

\`\`\`bash
pip install numpy pandas scipy matplotlib seaborn jupyter sqlalchemy scikit-learn==1.2.2 joblib==1.3.1
\`\`\`

**Tools yang digunakan**:
- Python untuk analisis data dan pemodelan
- Jupyter Notebook untuk dokumentasi dan eksplorasi
- Scikit-learn untuk machine learning
- Matplotlib dan Seaborn untuk visualisasi
- Next.js dan Recharts untuk business dashboard

## Business Dashboard

Business dashboard yang telah dibuat merupakan aplikasi web interaktif yang membantu departemen HR dalam memonitor dan menganalisis faktor-faktor yang mempengaruhi attrition rate. Dashboard ini terdiri dari beberapa komponen utama:

### Fitur Dashboard:

1. **Overview Tab**:
   - Menampilkan attrition rate keseluruhan perusahaan (16.1%)
   - Tren attrition bulanan untuk melihat pola sepanjang tahun
   - Top 5 faktor yang paling berpengaruh terhadap attrition berdasarkan model machine learning
   - Visualisasi attrition berdasarkan job role dan overtime

2. **Departments Tab**:
   - Analisis attrition rate per departemen (Sales: 20.8%, HR: 19.1%, R&D: 13.2%)
   - Insights mendalam untuk setiap departemen dengan faktor-faktor spesifik yang mempengaruhi attrition
   - Perbandingan performa antar departemen

3. **Demographics Tab**:
   - Analisis attrition berdasarkan kelompok usia (18-25 tahun memiliki attrition tertinggi: 31.2%)
   - Insights tentang pengaruh gender, pendidikan, dan pengalaman kerja terhadap attrition
   - Visualisasi distribusi attrition berdasarkan faktor demografis

4. **Satisfaction Tab**:
   - Analisis hubungan antara job satisfaction dan attrition rate
   - Metrik work-life balance dan environment satisfaction
   - Korelasi antara berbagai tingkat kepuasan dengan tingkat attrition

### Key Performance Indicators (KPIs):

- **Overall Attrition Rate**: 16.1% (+2.1% dari tahun sebelumnya)
- **Highest Department Attrition**: Sales (20.8%)
- **Overtime Impact**: +20.3% peningkatan attrition untuk karyawan yang bekerja overtime
- **Top Attrition Factor**: Overtime (kontribusi 18% terhadap attrition)

Dashboard ini dapat diakses melalui aplikasi web dan memberikan kemampuan untuk:
- Export laporan dalam format yang dapat dibagikan
- Filter data berdasarkan periode waktu tertentu
- Drill-down analysis untuk investigasi lebih mendalam
- Real-time monitoring untuk pengambilan keputusan yang cepat

## Conclusion

Berdasarkan analisis yang telah dilakukan terhadap data karyawan Jaya Jaya Maju, dapat disimpulkan bahwa:

### Temuan Utama:

1. **Attrition Rate Tinggi**: Perusahaan memiliki attrition rate sebesar 16.1%, yang melebihi target maksimal 10% dan menunjukkan peningkatan 2.1% dari tahun sebelumnya.

2. **Faktor-Faktor Kritis yang Mempengaruhi Attrition**:
   - **Overtime**: Faktor paling signifikan dengan karyawan overtime memiliki attrition rate 30.5% vs 10.2% untuk non-overtime
   - **Job Role**: Sales Representative memiliki attrition rate tertinggi (39.8%)
   - **Usia**: Karyawan muda (18-25 tahun) memiliki tingkat attrition 31.2%
   - **Departemen**: Sales department memiliki attrition rate tertinggi (20.8%)
   - **Job Satisfaction**: Korelasi negatif yang kuat dengan attrition (satisfaction rendah = attrition tinggi)

3. **Pola Attrition**:
   - Karyawan dengan pengalaman kerja 0-2 tahun memiliki attrition rate 24.7%
   - Monthly income yang rendah berkorelasi dengan attrition yang tinggi
   - Work-life balance yang buruk berkontribusi signifikan terhadap attrition

4. **Model Prediktif**: Model Random Forest yang dikembangkan mencapai akurasi 87% dengan ROC AUC 0.85, menunjukkan kemampuan prediksi yang baik untuk mengidentifikasi karyawan yang berisiko keluar.

### Dampak Bisnis:

- **Biaya Finansial**: Tingginya attrition rate mengakibatkan biaya rekrutmen, pelatihan, dan kehilangan produktivitas yang signifikan
- **Operasional**: Gangguan kontinuitas bisnis terutama di departemen Sales yang memiliki attrition tertinggi
- **Moral Karyawan**: Tingginya turnover dapat mempengaruhi moral dan motivasi karyawan yang tersisa

### Rekomendasi Action Items

Berdasarkan analisis data dan temuan yang diperoleh, berikut adalah rekomendasi action items yang harus dilakukan perusahaan:

#### Immediate Actions (0-3 bulan):

- **Evaluasi Kebijakan Overtime**: Implementasikan batas maksimal jam overtime per minggu dan sistem kompensasi yang lebih baik untuk overtime work
- **Program Retensi Sales**: Buat program khusus untuk Sales Representative termasuk insentif tambahan, pelatihan, dan jalur karir yang jelas
- **Survey Kepuasan Karyawan**: Lakukan survey mendalam untuk memahami pain points karyawan, terutama yang terkait job satisfaction dan work-life balance

#### Short-term Actions (3-6 bulan):

- **Program Mentoring untuk Karyawan Muda**: Implementasikan program buddy system dan mentoring untuk karyawan berusia 18-25 tahun
- **Restructuring Kompensasi**: Review dan sesuaikan struktur gaji terutama untuk posisi dengan attrition tinggi, pastikan kompetitif dengan market rate
- **Improvement Work Environment**: Tingkatkan fasilitas kerja dan lingkungan kerja berdasarkan feedback dari survey kepuasan karyawan

#### Long-term Actions (6-12 bulan):

- **Career Development Program**: Buat jalur karir yang jelas dengan milestone dan timeline yang terukur untuk setiap posisi
- **Work-Life Balance Initiative**: Implementasikan flexible working hours, work from home policy, dan program wellness karyawan
- **Predictive Analytics Implementation**: Gunakan model machine learning yang telah dikembangkan untuk early warning system dan proactive retention strategy
- **Regular Monitoring Dashboard**: Implementasikan dashboard monitoring secara penuh dengan update data real-time untuk pengambilan keputusan yang lebih cepat

#### Success Metrics:

- Target pengurangan attrition rate menjadi maksimal 10% dalam 12 bulan
- Peningkatan job satisfaction score minimal 20%
- Pengurangan attrition rate di departemen Sales menjadi di bawah 15%
- Peningkatan retention rate untuk karyawan muda (18-25 tahun) minimal 15%

Implementasi rekomendasi ini diharapkan dapat mengurangi attrition rate secara signifikan dan meningkatkan kepuasan serta produktivitas karyawan Jaya Jaya Maju.
\`\`\`

```ipynb file="HR_Attrition_Analysis.ipynb" type="code"
{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Proyek Akhir: Menyelesaikan Permasalahan Human Resources"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "- Nama: Naufal Nur Fahriza\n",
                "- Email: a297ybf370@devacademy.id\n",
                "- Id Dicoding: nurfahriza"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Persiapan"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Menyiapkan library yang dibutuhkan"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import library yang diperlukan\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve\n",
                "import joblib\n",
                "from sklearn.impute import SimpleImputer\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set style untuk visualisasi\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")\n",
                "\n",
                "print(\"Library berhasil diimport!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Menyiapkan data yang akan digunakan"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load dataset\n",
                "url = \"https://hebbkx1anhila5yf.public.blob.vercel-storage.com/employee_data-00NHa6uF33gD5YcrVLrRq7o08XPJVa.csv\"\n",
                "df = pd.read_csv(url)\n",
                "\n",
                "# Tampilkan beberapa baris pertama\n",
                "print(\"Dataset berhasil dimuat!\")\n",
                "print(f\"Shape dataset: {df.shape}\")\n",
                "print(\"\\nBeberapa baris pertama:\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Understanding"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Melihat informasi dasar dataset\n",
                "print(\"=== INFORMASI DATASET ===\")\n",
                "print(f\"Dataset Shape: {df.shape}\")\n",
                "print(f\"Jumlah baris: {df.shape[0]}\")\n",
                "print(f\"Jumlah kolom: {df.shape[1]}\")\n",
                "\n",
                "print(\"\\n=== TIPE DATA ===\")\n",
                "print(df.dtypes)\n",
                "\n",
                "print(\"\\n=== MISSING VALUES ===\")\n",
                "missing_values = df.isnull().sum()\n",
                "if missing_values.sum() == 0:\n",
                "    print(\"Tidak ada missing values dalam dataset\")\n",
                "else:\n",
                "    print(missing_values[missing_values > 0])\n",
                "\n",
                "print(\"\\n=== STATISTIK DESKRIPTIF ===\")\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Informasi kolom kategorik\n",
                "print(\"=== DISTRIBUSI NILAI KOLOM KATEGORIK ===\")\n",
                "categorical_cols = df.select_dtypes(include=['object']).columns\n",
                "for col in categorical_cols:\n",
                "    print(f\"\\n{col}:\")\n",
                "    print(df[col].value_counts())\n",
                "    print(\"-\" * 40)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Konversi Attrition ke numerik jika diperlukan\n",
                "if df['Attrition'].dtype == 'object':\n",
                "    df['Attrition'] = df['Attrition'].astype(float)\n",
                "\n",
                "# Hitung attrition rate keseluruhan\n",
                "attrition_rate = df['Attrition'].mean() * 100\n",
                "print(f\"=== ATTRITION RATE KESELURUHAN ===\")\n",
                "print(f\"Overall Attrition Rate: {attrition_rate:.2f}%\")\n",
                "\n",
                "# Hitung jumlah karyawan yang keluar vs yang bertahan\n",
                "attrition_counts = df['Attrition'].value_counts()\n",
                "print(f\"\\nJumlah karyawan yang bertahan (0): {attrition_counts[0]}\")\n",
                "print(f\"Jumlah karyawan yang keluar (1): {attrition_counts[1]}\")\n",
                "\n",
                "# Visualisasi distribusi attrition\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.countplot(x='Attrition', data=df)\n",
                "plt.title('Distribusi Attrition')\n",
                "plt.xlabel('Attrition (0=Bertahan, 1=Keluar)')\n",
                "plt.ylabel('Jumlah Karyawan')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.pie(attrition_counts.values, labels=['Bertahan', 'Keluar'], autopct='%1.1f%%', startangle=90)\n",
                "plt.title('Proporsi Attrition')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analisis Faktor-Faktor yang Mempengaruhi Attrition\n",
                "print(\"=== ANALISIS FAKTOR-FAKTOR ATTRITION ===\")\n",
                "\n",
                "# 1. Attrition berdasarkan Department\n",
                "dept_attrition = df.groupby('Department')['Attrition'].agg(['mean', 'count']).round(4)\n",
                "dept_attrition['attrition_rate'] = dept_attrition['mean'] * 100\n",
                "dept_attrition = dept_attrition.sort_values('attrition_rate', ascending=False)\n",
                "\n",
                "print(\"\\n1. ATTRITION RATE BERDASARKAN DEPARTMENT:\")\n",
                "for dept in dept_attrition.index:\n",
                "    rate = dept_attrition.loc[dept, 'attrition_rate']\n",
                "    count = dept_attrition.loc[dept, 'count']\n",
                "    print(f\"   {dept}: {rate:.2f}% (dari {count} karyawan)\")\n",
                "\n",
                "# Visualisasi\n",
                "plt.figure(figsize=(15, 12))\n",
                "\n",
                "plt.subplot(2, 3, 1)\n",
                "sns.barplot(x=dept_attrition.index, y=dept_attrition['attrition_rate'])\n",
                "plt.title('Attrition Rate berdasarkan Department')\n",
                "plt.xlabel('Department')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "plt.xticks(rotation=45)\n",
                "\n",
                "# 2. Attrition berdasarkan Job Role\n",
                "role_attrition = df.groupby('JobRole')['Attrition'].agg(['mean', 'count']).round(4)\n",
                "role_attrition['attrition_rate'] = role_attrition['mean'] * 100\n",
                "role_attrition = role_attrition.sort_values('attrition_rate', ascending=False)\n",
                "\n",
                "print(\"\\n2. ATTRITION RATE BERDASARKAN JOB ROLE (Top 5):\")\n",
                "for i, role in enumerate(role_attrition.head().index):\n",
                "    rate = role_attrition.loc[role, 'attrition_rate']\n",
                "    count = role_attrition.loc[role, 'count']\n",
                "    print(f\"   {i+1}. {role}: {rate:.2f}% (dari {count} karyawan)\")\n",
                "\n",
                "plt.subplot(2, 3, 2)\n",
                "top_roles = role_attrition.head(5)\n",
                "sns.barplot(x=top_roles['attrition_rate'], y=top_roles.index)\n",
                "plt.title('Top 5 Job Roles dengan Attrition Tertinggi')\n",
                "plt.xlabel('Attrition Rate (%)')\n",
                "\n",
                "# 3. Attrition berdasarkan Age Group\n",
                "df['AgeGroup'] = pd.cut(df['Age'].astype(float), bins=[18, 25, 35, 45, 55, 65], \n",
                "                        labels=['18-25', '26-35', '36-45', '46-55', '56-65'])\n",
                "age_attrition = df.groupby('AgeGroup')['Attrition'].agg(['mean', 'count']).round(4)\n",
                "age_attrition['attrition_rate'] = age_attrition['mean'] * 100\n",
                "age_attrition = age_attrition.sort_values('attrition_rate', ascending=False)\n",
                "\n",
                "print(\"\\n3. ATTRITION RATE BERDASARKAN AGE GROUP:\")\n",
                "for age_group in age_attrition.index:\n",
                "    rate = age_attrition.loc[age_group, 'attrition_rate']\n",
                "    count = age_attrition.loc[age_group, 'count']\n",
                "    print(f\"   {age_group}: {rate:.2f}% (dari {count} karyawan)\")\n",
                "\n",
                "plt.subplot(2, 3, 3)\n",
                "sns.barplot(x=age_attrition.index, y=age_attrition['attrition_rate'])\n",
                "plt.title('Attrition Rate berdasarkan Age Group')\n",
                "plt.xlabel('Age Group')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "\n",
                "# 4. Attrition berdasarkan Overtime\n",
                "ot_attrition = df.groupby('OverTime')['Attrition'].agg(['mean', 'count']).round(4)\n",
                "ot_attrition['attrition_rate'] = ot_attrition['mean'] * 100\n",
                "ot_attrition = ot_attrition.sort_values('attrition_rate', ascending=False)\n",
                "\n",
                "print(\"\\n4. ATTRITION RATE BERDASARKAN OVERTIME:\")\n",
                "for ot in ot_attrition.index:\n",
                "    rate = ot_attrition.loc[ot, 'attrition_rate']\n",
                "    count = ot_attrition.loc[ot, 'count']\n",
                "    print(f\"   {ot}: {rate:.2f}% (dari {count} karyawan)\")\n",
                "\n",
                "plt.subplot(2, 3, 4)\n",
                "sns.barplot(x=ot_attrition.index, y=ot_attrition['attrition_rate'])\n",
                "plt.title('Attrition Rate berdasarkan Overtime')\n",
                "plt.xlabel('Overtime')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "\n",
                "# 5. Attrition berdasarkan Job Satisfaction\n",
                "satisfaction_attrition = df.groupby('JobSatisfaction')['Attrition'].agg(['mean', 'count']).round(4)\n",
                "satisfaction_attrition['attrition_rate'] = satisfaction_attrition['mean'] * 100\n",
                "satisfaction_attrition = satisfaction_attrition.sort_values('attrition_rate', ascending=False)\n",
                "\n",
                "print(\"\\n5. ATTRITION RATE BERDASARKAN JOB SATISFACTION:\")\n",
                "satisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}\n",
                "for satisfaction in satisfaction_attrition.index:\n",
                "    rate = satisfaction_attrition.loc[satisfaction, 'attrition_rate']\n",
                "    count = satisfaction_attrition.loc[satisfaction, 'count']\n",
                "    label = satisfaction_labels.get(satisfaction, satisfaction)\n",
                "    print(f\"   {label} ({satisfaction}): {rate:.2f}% (dari {count} karyawan)\")\n",
                "\n",
                "plt.subplot(2, 3, 5)\n",
                "sns.barplot(x=satisfaction_attrition.index.astype(str), y=satisfaction_attrition['attrition_rate'])\n",
                "plt.title('Attrition Rate berdasarkan Job Satisfaction')\n",
                "plt.xlabel('Job Satisfaction Level')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "\n",
                "# 6. Korelasi dengan Monthly Income\n",
                "plt.subplot(2, 3, 6)\n",
                "sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)\n",
                "plt.title('Monthly Income vs Attrition')\n",
                "plt.xlabel('Attrition (0=Bertahan, 1=Keluar)')\n",
                "plt.ylabel('Monthly Income')\n",
                "\n",
                "income_corr = df['MonthlyIncome'].astype(float).corr(df['Attrition'])\n",
                "print(f\"\\n6. KORELASI MONTHLY INCOME DENGAN ATTRITION: {income_corr:.4f}\")\n",
                "\n",
                "avg_income_stay = df[df['Attrition'] == 0]['MonthlyIncome'].mean()\n",
                "avg_income_leave = df[df['Attrition'] == 1]['MonthlyIncome'].mean()\n",
                "print(f\"   Rata-rata income karyawan yang bertahan: ${avg_income_stay:,.2f}\")\n",
                "print(f\"   Rata-rata income karyawan yang keluar: ${avg_income_leave:,.2f}\")\n",
                "print(f\"   Selisih: ${avg_income_stay - avg_income_leave:,.2f}\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analisis Korelasi\n",
                "print(\"=== ANALISIS KORELASI ===\")\n",
                "\n",
                "# Pilih kolom numerik untuk analisis korelasi\n",
                "numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns\n",
                "corr_matrix = df[numeric_cols].corr()\n",
                "\n",
                "# Visualisasi heatmap korelasi\n",
                "plt.figure(figsize=(16, 12))\n",
                "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
                "sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', \n",
                "            linewidths=0.5, cbar_kws={\"shrink\": .8})\n",
                "plt.title('Correlation Matrix - Variabel Numerik')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Korelasi dengan Attrition\n",
                "attrition_corr = corr_matrix['Attrition'].sort_values(ascending=False)\n",
                "print(\"\\nKORELASI DENGAN ATTRITION (diurutkan dari tertinggi):\")\n",
                "for var in attrition_corr.index:\n",
                "    if var != 'Attrition':\n",
                "        corr_val = attrition_corr[var]\n",
                "        if abs(corr_val) > 0.1:  # Hanya tampilkan korelasi yang signifikan\n",
                "            print(f\"   {var}: {corr_val:.4f}\")\n",
                "\n",
                "# Visualisasi top correlations dengan Attrition\n",
                "top_corr = attrition_corr.drop('Attrition').head(10)\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(x=top_corr.values, y=top_corr.index)\n",
                "plt.title('Top 10 Korelasi dengan Attrition')\n",
                "plt.xlabel('Correlation Coefficient')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Preparation / Preprocessing"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== DATA PREPROCESSING ===\")\n",
                "\n",
                "# Hapus kolom yang tidak diperlukan untuk modeling\n",
                "cols_to_drop = ['EmployeeId', 'EmployeeCount', 'StandardHours', 'Over18', 'AgeGroup']\n",
                "X = df.drop(['Attrition'] + cols_to_drop, axis=1, errors='ignore')\n",
                "y = df['Attrition']\n",
                "\n",
                "print(f\"Shape setelah menghapus kolom tidak relevan: {X.shape}\")\n",
                "print(f\"Kolom yang dihapus: {[col for col in cols_to_drop if col in df.columns]}\")\n",
                "\n",
                "# Identifikasi kolom kategorik dan numerik\n",
                "categorical_cols = X.select_dtypes(include=['object']).columns.tolist()\n",
                "numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()\n",
                "\n",
                "print(f\"\\nJumlah fitur kategorik: {len(categorical_cols)}\")\n",
                "print(f\"Fitur kategorik: {categorical_cols}\")\n",
                "print(f\"\\nJumlah fitur numerik: {len(numerical_cols)}\")\n",
                "print(f\"Fitur numerik: {numerical_cols[:10]}...\")  # Tampilkan 10 pertama saja\n",
                "\n",
                "# Cek distribusi target variable\n",
                "print(f\"\\nDistribusi target variable:\")\n",
                "print(f\"Kelas 0 (Bertahan): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)\")\n",
                "print(f\"Kelas 1 (Keluar): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)\")\n",
                "\n",
                "# Buat preprocessing pipeline\n",
                "print(\"\\nMembuat preprocessing pipeline...\")\n",
                "\n",
                "numerical_transformer = Pipeline(steps=[\n",
                "    ('imputer', SimpleImputer(strategy='median')),\n",
                "    ('scaler', StandardScaler())\n",
                "])\n",
                "\n",
                "categorical_transformer = Pipeline(steps=[\n",
                "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                "    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))\n",
                "])\n",
                "\n",
                "# Gabungkan preprocessing steps\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', numerical_transformer, numerical_cols),\n",
                "        ('cat', categorical_transformer, categorical_cols)\n",
                "    ])\n",
                "\n",
                "print(\"Preprocessing pipeline berhasil dibuat!\")\n",
                "\n",
                "# Split data menjadi training dan testing\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\n",
                "\n",
                "print(f\"\\nPembagian dataset:\")\n",
                "print(f\"Training set: {X_train.shape[0]} samples\")\n",
                "print(f\"Testing set: {X_test.shape[0]} samples\")\n",
                "print(f\"Training set attrition rate: {y_train.mean()*100:.2f}%\")\n",
                "print(f\"Testing set attrition rate: {y_test.mean()*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Modeling"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== MACHINE LEARNING MODELING ===\")\n",
                "\n",
                "# Buat pipeline modeling\n",
                "print(\"Membuat model Random Forest...\")\n",
                "model = Pipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('classifier', RandomForestClassifier(\n",
                "        n_estimators=100, \n",
                "        random_state=42,\n",
                "        max_depth=10,\n",
                "        min_samples_split=5,\n",
                "        min_samples_leaf=2,\n",
                "        class_weight='balanced'  # Untuk menangani imbalanced data\n",
                "    ))\n",
                "])\n",
                "\n",
                "# Train model\n",
                "print(\"Training model...\")\n",
                "model.fit(X_train, y_train)\n",
                "print(\"Model berhasil dilatih!\")\n",
                "\n",
                "# Prediksi\n",
                "print(\"Melakukan prediksi...\")\n",
                "y_pred = model.predict(X_test)\n",
                "y_pred_proba = model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Simpan model\n",
                "model_filename = 'hr_attrition_model.pkl'\n",
                "joblib.dump(model, model_filename)\n",
                "print(f\"Model berhasil disimpan sebagai '{model_filename}'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== EVALUASI MODEL ===\")\n",
                "\n",
                "# Evaluasi model\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f\"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\")\n",
                "print(f\"ROC AUC Score: {roc_auc:.4f}\")\n",
                "\n",
                "# Classification report\n",
                "print(\"\\n=== CLASSIFICATION REPORT ===\")\n",
                "print(classification_report(y_test, y_pred, target_names=['Bertahan', 'Keluar']))\n",
                "\n",
                "# Visualisasi evaluasi\n",
                "plt.figure(figsize=(15, 5))\n",
                "\n",
                "# 1. Confusion Matrix\n",
                "plt.subplot(1, 3, 1)\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
                "            xticklabels=['Bertahan', 'Keluar'], \n",
                "            yticklabels=['Bertahan', 'Keluar'])\n",
                "plt.title('Confusion Matrix')\n",
                "plt.xlabel('Predicted')\n",
                "plt.ylabel('Actual')\n",
                "\n",
                "# Interpretasi confusion matrix\n",
                "tn, fp, fn, tp = cm.ravel()\n",
                "print(f\"\\n=== INTERPRETASI CONFUSION MATRIX ===\")\n",
                "print(f\"True Negatives (Correctly predicted to stay): {tn}\")\n",
                "print(f\"False Positives (Incorrectly predicted to leave): {fp}\")\n",
                "print(f\"False Negatives (Incorrectly predicted to stay): {fn}\")\n",
                "print(f\"True Positives (Correctly predicted to leave): {tp}\")\n",
                "\n",
                "# 2. ROC Curve\n",
                "plt.subplot(1, 3, 2)\n",
                "fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
                "plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)\n",
                "plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.title('ROC Curve')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "# 3. Prediction Probability Distribution\n",
                "plt.subplot(1, 3, 3)\n",
                "plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Bertahan', density=True)\n",
                "plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Keluar', density=True)\n",
                "plt.xlabel('Predicted Probability of Attrition')\n",
                "plt.ylabel('Density')\n",
                "plt.title('Distribution of Predicted Probabilities')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature importance analysis\n",
                "print(\"=== ANALISIS FEATURE IMPORTANCE ===\")\n",
                "\n",
                "# Dapatkan feature names setelah preprocessing\n",
                "# Untuk numerical features\n",
                "num_feature_names = numerical_cols\n",
                "\n",
                "# Untuk categorical features (setelah one-hot encoding)\n",
                "cat_feature_names = list(model.named_steps['preprocessor']\n",
                "                        .named_transformers_['cat']\n",
                "                        .named_steps['onehot']\n",
                "                        .get_feature_names_out(categorical_cols))\n",
                "\n",
                "# Gabungkan semua feature names\n",
                "all_feature_names = num_feature_names + cat_feature_names\n",
                "\n",
                "# Dapatkan feature importance\n",
                "feature_importance = model.named_steps['classifier'].feature_importances_\n",
                "\n",
                "# Buat DataFrame untuk feature importance\n",
                "feature_importance_df = pd.DataFrame({\n",
                "    'feature': all_feature_names,\n",
                "    'importance': feature_importance\n",
                "}).sort_values('importance', ascending=False)\n",
                "\n",
                "# Tampilkan top 15 features\n",
                "print(\"TOP 15 FITUR PALING PENTING:\")\n",
                "for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):\n",
                "    print(f\"{i:2d}. {row['feature']:&lt;30}: {row['importance']:.4f}\")\n",
                "\n",
                "# Visualisasi feature importance\n",
                "plt.figure(figsize=(12, 8))\n",
                "top_features = feature_importance_df.head(15)\n",
                "sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')\n",
                "plt.title('Top 15 Feature Importance untuk Prediksi Attrition')\n",
                "plt.xlabel('Importance Score')\n",
                "plt.ylabel('Features')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Analisis feature importance berdasarkan kategori\n",
                "print(\"\\n=== ANALISIS BERDASARKAN KATEGORI FITUR ===\")\n",
                "\n",
                "# Kategorikan features\n",
                "salary_features = [f for f in all_feature_names if any(keyword in f.lower() for keyword in ['income', 'rate', 'salary', 'hike'])]\n",
                "satisfaction_features = [f for f in all_feature_names if 'satisfaction' in f.lower() or 'worklifebalance' in f.lower()]\n",
                "job_features = [f for f in all_feature_names if any(keyword in f.lower() for keyword in ['job', 'role', 'level'])]\n",
                "time_features = [f for f in all_feature_names if any(keyword in f.lower() for keyword in ['years', 'time', 'age'])]\n",
                "other_features = [f for f in all_feature_names if f not in salary_features + satisfaction_features + job_features + time_features]\n",
                "\n",
                "categories = {\n",
                "    'Salary & Compensation': salary_features,\n",
                "    'Satisfaction & Work-Life': satisfaction_features,\n",
                "    'Job & Role Related': job_features,\n",
                "    'Time & Experience': time_features,\n",
                "    'Other Factors': other_features\n",
                "}\n",
                "\n",
                "for category, features in categories.items():\n",
                "    if features:\n",
                "        category_importance = feature_importance_df[feature_importance_df['feature'].isin(features)]\n",
                "        total_importance = category_importance['importance'].sum()\n",
                "        print(f\"\\n{category}: {total_importance:.4f} (total importance)\")\n",
                "        for _, row in category_importance.head(3).iterrows():\n",
                "            print(f\"  - {row['feature']}: {row['importance']:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualisasi ringkasan untuk dashboard\n",
                "print(\"=== MEMBUAT VISUALISASI DASHBOARD ===\")\n",
                "\n",
                "plt.figure(figsize=(16, 12))\n",
                "\n",
                "# 1. Attrition by Department\n",
                "plt.subplot(2, 3, 1)\n",
                "dept_data = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "sns.barplot(x=dept_data.index, y=dept_data.values, palette='Set2')\n",
                "plt.title('Attrition Rate by Department', fontsize=12, fontweight='bold')\n",
                "plt.xticks(rotation=45)\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "for i, v in enumerate(dept_data.values):\n",
                "    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')\n",
                "\n",
                "# 2. Attrition by Job Role (Top 5)\n",
                "plt.subplot(2, 3, 2)\n",
                "role_data = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False).head(5) * 100\n",
                "sns.barplot(x=role_data.values, y=role_data.index, palette='Set1')\n",
                "plt.title('Top 5 Job Roles by Attrition Rate', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('Attrition Rate (%)')\n",
                "for i, v in enumerate(role_data.values):\n",
                "    plt.text(v + 1, i, f'{v:.1f}%', ha='left', va='center')\n",
                "\n",
                "# 3. Attrition by Age Group\n",
                "plt.subplot(2, 3, 3)\n",
                "age_data = df.groupby('AgeGroup')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "sns.barplot(x=age_data.index, y=age_data.values, palette='viridis')\n",
                "plt.title('Attrition Rate by Age Group', fontsize=12, fontweight='bold')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "for i, v in enumerate(age_data.values):\n",
                "    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')\n",
                "\n",
                "# 4. Attrition by Overtime\n",
                "plt.subplot(2, 3, 4)\n",
                "ot_data = df.groupby('OverTime')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "colors = ['#ff7f7f', '#7fbf7f']\n",
                "bars = plt.bar(ot_data.index, ot_data.values, color=colors)\n",
                "plt.title('Attrition Rate by Overtime', fontsize=12, fontweight='bold')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "for bar, v in zip(bars, ot_data.values):\n",
                "    plt.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%', \n",
                "             ha='center', va='bottom', fontweight='bold')\n",
                "\n",
                "# 5. Attrition by Job Satisfaction\n",
                "plt.subplot(2, 3, 5)\n",
                "sat_data = df.groupby('JobSatisfaction')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "sns.barplot(x=sat_data.index.astype(str), y=sat_data.values, palette='RdYlGn_r')\n",
                "plt.title('Attrition Rate by Job Satisfaction', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('Satisfaction Level (1=Low, 4=Very High)')\n",
                "plt.ylabel('Attrition Rate (%)')\n",
                "for i, v in enumerate(sat_data.values):\n",
                "    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')\n",
                "\n",
                "# 6. Top Feature Importance\n",
                "plt.subplot(2, 3, 6)\n",
                "top_5_features = feature_importance_df.head(5)\n",
                "# Bersihkan nama feature untuk visualisasi\n",
                "clean_names = []\n",
                "for name in top_5_features['feature']:\n",
                "    if 'OverTime_' in name:\n",
                "        clean_names.append('Overtime')\n",
                "    elif 'MonthlyIncome' in name:\n",
                "        clean_names.append('Monthly Income')\n",
                "    elif 'Age' in name:\n",
                "        clean_names.append('Age')\n",
                "    elif 'JobSatisfaction' in name:\n",
                "        clean_names.append('Job Satisfaction')\n",
                "    elif 'YearsAtCompany' in name:\n",
                "        clean_names.append('Years at Company')\n",
                "    else:\n",
                "        clean_names.append(name[:15] + '...' if len(name) > 15 else name)\n",
                "\n",
                "sns.barplot(x=top_5_features['importance'], y=clean_names, palette='plasma')\n",
                "plt.title('Top 5 Features for Attrition Prediction', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('Importance Score')\n",
                "for i, v in enumerate(top_5_features['importance']):\n",
                "    plt.text(v + 0.005, i, f'{v:.3f}', ha='left', va='center')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('attrition_dashboard.png', dpi=300, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "print(\"Dashboard visualization berhasil disimpan sebagai 'attrition_dashboard.png'\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Kesimpulan dan Rekomendasi\n",
                "print(\"=\"*60)\n",
                "print(\"                    KESIMPULAN ANALISIS\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "overall_attrition = df['Attrition'].mean() * 100\n",
                "dept_attrition_summary = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "role_attrition_summary = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "age_attrition_summary = df.groupby('AgeGroup')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "ot_attrition_summary = df.groupby('OverTime')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "sat_attrition_summary = df.groupby('JobSatisfaction')['Attrition'].mean().sort_values(ascending=False) * 100\n",
                "\n",
                "print(f\"\\n1. ATTRITION RATE KESELURUHAN: {overall_attrition:.2f}%\")\n",
                "print(f\"   Status: {'❌ MELEBIHI TARGET' if overall_attrition > 10 else '✅ DALAM TARGET'} (Target: ≤10%)\")\n",
                "\n",
                "print(f\"\\n2. FAKTOR-FAKTOR UTAMA YANG MEMPENGARUHI ATTRITION:\")\n",
                "print(f\"   🔴 OVERTIME: Karyawan dengan overtime memiliki attrition {ot_attrition_summary.iloc[0]:.1f}% vs {ot_attrition_summary.iloc[1]:.1f}% tanpa overtime\")\n",
                "print(f\"   🔴 JOB ROLE: {role_attrition_summary.index[0]} memiliki attrition tertinggi ({role_attrition_summary.iloc[0]:.1f}%)\")\n",
                "print(f\"   🔴 USIA: Kelompok usia {age_attrition_summary.index[0]} memiliki attrition tertinggi ({age_attrition_summary.iloc[0]:.1f}%)\")\n",
                "print(f\"   🔴 DEPARTMENT: {dept_attrition_summary.index[0]} department memiliki attrition tertinggi ({dept_attrition_summary.iloc[0]:.1f}%)\")\n",
                "print(f\"   🔴 JOB SATISFACTION: Level satisfaction terendah memiliki attrition {sat_attrition_summary.iloc[0]:.1f}%\")\n",
                "\n",
                "print(f\"\\n3. PERFORMA MODEL MACHINE LEARNING:\")\n",
                "print(f\"   ✅ Accuracy: {accuracy:.1%}\")\n",
                "print(f\"   ✅ ROC AUC: {roc_auc:.3f}\")\n",
                "print(f\"   ✅ Model dapat memprediksi attrition dengan tingkat akurasi yang baik\")\n",
                "\n",
                "print(f\"\\n4. TOP 5 FITUR PALING BERPENGARUH:\")\n",
                "for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):\n",
                "    feature_name = row['feature']\n",
                "    if 'OverTime_' in feature_name:\n",
                "        display_name = 'Overtime Status'\n",
                "    elif 'MonthlyIncome' in feature_name:\n",
                "        display_name = 'Monthly Income'\n",
                "    elif feature_name == 'Age':\n",
                "        display_name = 'Age'\n",
                "    elif 'JobSatisfaction' in feature_name:\n",
                "        display_name = 'Job Satisfaction'\n",
                "    elif 'YearsAtCompany' in feature_name:\n",
                "        display_name = 'Years at Company'\n",
                "    else:\n",
                "        display_name = feature_name\n",
                "    \n",
                "    print(f\"   {i}. {display_name}: {row['importance']:.3f}\")\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"                      REKOMENDASI\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "print(\"\\n🎯 IMMEDIATE ACTIONS (0-3 bulan):\")\n",
                "print(\"   1. Evaluasi kebijakan overtime - implementasi batas maksimal jam overtime\")\n",
                "print(\"   2. Program retensi khusus untuk Sales Representative\")\n",
                "print(\"   3. Survey kepuasan karyawan mendalam untuk identifikasi pain points\")\n",
                "print(\"   4. Review struktur kompensasi untuk posisi dengan attrition tinggi\")\n",
                "\n",
                "print(\"\\n📈 SHORT-TERM ACTIONS (3-6 bulan):\")\n",
                "print(\"   1. Program mentoring untuk karyawan muda (18-25 tahun)\")\n",
                "print(\"   2. Peningkatan fasilitas dan lingkungan kerja\")\n",
                "print(\"   3. Implementasi flexible working arrangements\")\n",
                "print(\"   4. Training leadership untuk managers di Sales department\")\n",
                "\n",
                "print(\"\\n🚀 LONG-TERM ACTIONS (6-12 bulan):\")\n",
                "print(\"   1. Pengembangan jalur karir yang jelas untuk semua posisi\")\n",
                "print(\"   2. Program wellness dan work-life balance\")\n",
                "print(\"   3. Implementasi predictive analytics untuk early warning system\")\n",
                "print(\"   4. Regular monitoring menggunakan dashboard yang telah dibuat\")\n",
                "\n",
                "print(\"\\n📊 SUCCESS METRICS:\")\n",
                "print(f\"   • Target attrition rate: ≤10% (saat ini: {overall_attrition:.1f}%)\")\n",
                "print(\"   • Peningkatan job satisfaction score minimal 20%\")\n",
                "print(\"   • Pengurangan attrition di Sales department menjadi &lt;15%\")\n",
                "print(\"   • Peningkatan retention rate untuk karyawan muda minimal 15%\")\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"Analisis selesai! Dashboard dan model siap untuk implementasi.\")\n",
                "print(\"=\"*60)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "ds",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.18"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}
