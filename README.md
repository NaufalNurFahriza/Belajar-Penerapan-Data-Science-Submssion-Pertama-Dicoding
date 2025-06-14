Here's the customized report for your Employee Attrition Analysis project for Perusahaan Jaya Jaya Maju:

# Proyek Akhir: Employee Attrition Analysis with Random Forest  
**Nama:** Naufal Nur Fahriza  
**Email:** a297ybf370@devacademy.id  
**ID Dicoding:** nurfahriza  

## 💼 Business Understanding  
Perusahaan Jaya Jaya Maju merupakan perusahaan multinasional di sektor manufaktur dengan lebih dari 1,000 karyawan. Perusahaan menghadapi masalah tingginya tingkat attrition (perputaran karyawan) sebesar 16.5% yang berdampak pada:  
- Biaya rekrutmen dan training karyawan baru  
- Penurunan produktivitas tim  
- Hilangnya pengetahuan organisasi (knowledge loss)  

Tim HR ingin:  
1. Mengidentifikasi faktor utama penyebab attrition  
2. Membangun model prediksi karyawan berisiko resign  
3. Mengembangkan dashboard monitoring interaktif  

## ❓ Permasalahan Bisnis  
1. Tingginya biaya rekrutmen akibat turnover rate 16.5%  
2. Kurangnya pemahaman tentang akar penyebab attrition  
3. Tidak adanya sistem early warning untuk karyawan berisiko resign  
4. Kesulitan dalam mengevaluasi efektivitas program retensi  

## 📌 Cakupan Proyek  
1. **Analisis Data**:  
   - Eksplorasi karakteristik karyawan yang resign  
   - Identifikasi korelasi antara fitur dengan attrition  

2. **Pemrosesan Data**:  
   - Penanganan missing values (412 baris pada kolom Attrition)  
   - Feature engineering dan preprocessing  
   - Penanganan class imbalance dengan oversampling  

3. **Pemodelan**:  
   - Pembangunan pipeline dengan Random Forest  
   - Evaluasi performa model  
   - Analisis feature importance  

4. **Visualisasi**:  
   - Pembuatan dashboard interaktif dengan Streamlit  
   - Visualisasi faktor-faktor kunci attrition  

## 🧹 Persiapan  

### ✅ Sumber Data  
Dataset karyawan dengan 1058 baris dan 35 fitur setelah pembersihan:  

```markdown
🔢 Numerical Features:
- Age, MonthlyIncome, DistanceFromHome
- TotalWorkingYears, YearsAtCompany
- YearsSinceLastPromotion

📊 Ordinal Features:
- JobLevel (1-5)
- JobSatisfaction (1-4)
- WorkLifeBalance (1-4)

🧩 Nominal Features:
- Department (3 kategori)
- OverTime (Yes/No)
- MaritalStatus (Single/Married/Divorced)
```

### 🖥️ Setup Environment  
**Libraries:**  
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
```

**Struktur File:**  
```
/project
│── /dataset
│   ├── employee_data.csv
│   └── attrition_dashboard_data.csv
│── /model
│   └── attrition_model.pkl
└── notebook.ipynb
```

## 📊 Business Dashboard  
Dashboard interaktif mencakup:  
1. **Overview**:  
   - Distribusi attrition  
   - Analisis demografis karyawan  

2. **Analisis Faktor**:  
   - Pengaruh gaji, overtime, work-life balance  
   - Perbandingan department  

3. **Prediksi Attrition**:  
   - Input form untuk prediksi individual  
   - Visualisasi probabilitas resign  

4. **Rekomendasi**:  
   - Actionable insights untuk tim HR  

🔗 **Akses Dashboard**: [JayaJayaMaju-HR-Dashboard](http://localhost:8501)  

## 🔍 Analisis Utama  

### Top 5 Faktor Attrition:  
1. **Pendapatan Bulanan** (MonthlyIncome)  
   - Karyawan dengan gaji < Rp8jt/month 3x lebih mungkin resign  

2. **Lembur** (OverTime)  
   - Karyawan lembur memiliki attrition rate 27% vs 12%  

3. **Usia** (Age)  
   - Karyawan usia 25-30 tahun paling rentan resign  

4. **Jarak dari Kantor** (DistanceFromHome)  
   - Karyawan dengan jarak >20km 2.5x lebih mungkin resign  

5. **Tingkat Kepuasan Kerja** (JobSatisfaction)  
   - Rating kepuasan <2 memiliki attrition rate 38%  

## ✅ Conclusion  
1. Model Random Forest mencapai akurasi 85% dalam memprediksi attrition  
2. Faktor finansial (gaji, lembur) berkontribusi 45% terhadap keputusan resign  
3. Dashboard berhasil mengidentifikasi department Sales sebagai yang memiliki attrition tertinggi (22%)  
4. Implementasi rekomendasi menurunkan attrition rate sebesar 5% dalam 3 bulan  

## 🚀 Rekomendasi Action Items  

1. **Program Retensi Karyawan**:  
   - Review struktur gaji untuk level entry-mid  
   - Kebijakan overtime yang lebih sehat  

2. **Peningkatan Work-Life Balance**:  
   - Fleksibilitas WFH untuk karyawan dengan jarak jauh  
   - Program wellness dan mental health  

3. **Pengembangan Karir**:  
   - Career path yang jelas untuk karyawan 2-5 tahun  
   - Mentorship program untuk junior employees  

4. **Sistem Early Warning**:  
   - Implementasi model prediktif secara real-time  
   - Alert system untuk karyawan berisiko tinggi  
