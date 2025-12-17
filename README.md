# 📘 Predicting Student Dropout and Academic Success Using Machine Learning

## 👤 Informasi

* **Nama:** Amro Alfien Syachrian Nadzief (233307004)
* **Repo:** https://github.com/alfienrian/uas-data-science
* **Video:** https://drive.google.com/file/d/18QZL5ZMBYXkRPEGyknZUlo-TO7pnjskj/view?usp=sharing

---

## **1. 🎯 Ringkasan Proyek**

* Menyelesaikan permasalahan prediksi dropout dan keberhasilan akademik mahasiswa
* Melakukan data preparation dan eksplorasi data
* Membangun 3 model: **Baseline**, **Advanced ML**, **Deep Learning**
* Melakukan evaluasi performa dan menentukan model terbaik

---

## **2. 📄 Problem & Goals**

### **Problem Statements**

* Bagaimana memprediksi status akademik mahasiswa (Dropout, Enrolled, Graduate)?
* Model ML apa yang memberikan performa terbaik pada dataset pendidikan?

### **Goals**

* Mengembangkan model prediksi status akademik mahasiswa
* Membandingkan performa model klasik dan deep learning

## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

## **3. 📊 Dataset**

* **Sumber:** UCI Machine Learning Repository – *Predict Students' Dropout and Academic Success*
* **Jumlah Data:** ± 4.400 record
* **Tipe:** Tabular, Multiclass Classification

### **Fitur Utama**

| Fitur                    | Deskripsi                     |
| ------------------------ | ----------------------------- |
| age_at_enrollment        | Usia mahasiswa saat mendaftar |
| gender                   | Jenis kelamin                 |
| admission_grade          | Nilai masuk                   |
| curricular_units_1st_sem | Nilai akademik semester 1     |
| curricular_units_2nd_sem | Nilai akademik semester 2     |
| target                   | Status mahasiswa              |

---

## **4. 🔧 Data Preparation**

* Handling missing values
* Encoding fitur kategorikal
* Feature scaling
* Splitting data (train/test)

---

## **5. 🤖 Modeling**

* **Model 1 – Baseline:** Logistic Regression
* **Model 2 – Advanced ML:** Random Forest
* **Model 3 – Deep Learning:** Neural Network (MLP)

---

## **6. 🧪 Evaluation**

**Metrik:** Accuracy & F1-Score

| Model             | Accuracy | F1-Score         |
| ----------------- | -------- | ---------------- |
| Baseline          | 0.7683   | 0.7531           |
| Random Forest     | 0.7672   | 0.7514           |
| Deep Learning     | 0.7503   | 0.7429           |

---

## **7. 🏁 Kesimpulan**

* Model terbaik: **Deep Learning (MLP)**
* Alasan: mampu menangkap hubungan non-linear
* Insight: performa akademik awal sangat berpengaruh

---

## **8. 🔮 Future Work**

* [ ] Hyperparameter tuning
* [ ] Feature selection
* [ ] Deployment ke web app

---

## **9. 🔁 Reproducibility**

### Environment

Python Version: >= 3.12.10

Buat environment menggunakan venv agar package yang ada di sistem tidak bentrok
```
python -m venv venv
venv\Scripts\activate
```

Install Dependencies:
```
pip install -r requirements.txt
```

### Menjalankan Script Python (Batch Training)

Pastikan dataset berada pada folder `data/` dengan nama `data.csv`

Jalankan script utama:

```
python src/main.py
```

Script akan:
- Meload dataset
- Melakukan preprocessing
- Melatih 3 model (Baseline, Random Forest, Deep Learning)
- Menyimpan model ke folder `models/`

---
