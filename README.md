# 📘 Judul Proyek

**Predicting Student Dropout and Academic Success Using Machine Learning**

## 👤 Informasi

* **Nama:** Amro Alfien Syachrian Nadzief (233307004)
* **Repo:** https://github.com/alfienrian/uas-data-science
* **Video:** [Link Demo / Presentasi]

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

| Model         | Accuracy | Catatan          |
| ------------- | -------- | ---------------- |
| Baseline      | ~0.75    | Model sederhana  |
| Advanced      | ~0.80    | Lebih stabil     |
| Deep Learning | ~0.82    | Performa terbaik |

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

Gunakan environment Python (Colab) + requirements.txt

---
