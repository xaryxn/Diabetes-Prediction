# 🩺 Diabetes Prediction Using Machine Learning

This project implements a **diabetes prediction system** using **Support Vector Machine (SVM)** with an RBF kernel. It uses the **PIMA Indian Diabetes Dataset** to train a binary classification model that predicts whether a person is diabetic based on key health indicators.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Modules](#modules)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Scope](#future-scope)
- [References](#references)

---

## 🔍 Overview

Diabetes is one of the fastest-growing health threats worldwide. Early detection is critical to avoid complications. This project uses **machine learning** to automate diabetes diagnosis using common medical features like glucose level, BMI, and age. The final system is accurate, scalable, and ideal for use in rural or under-resourced healthcare settings.

---

## 📊 Dataset

- **Source:** [PIMA Indian Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features:**  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
  - Outcome (Target variable: 0 = Non-Diabetic, 1 = Diabetic)

---

## 🧰 Tech Stack

- **Language:** Python 3.x  
- **Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `scikit-learn` – Model building, training, evaluation  
  - `matplotlib`, `seaborn` – Data visualization  
  - `joblib` – Model persistence  
  - `pandas-profiling` – Automated EDA  
- **Model Used:** SVM with RBF kernel  
- **Preprocessing:** StandardScaler (mean = 0, std = 1)

---

## 🏗️ Project Architecture
   
diabetes-predictor/
│
├── data/                      # Contains the dataset
│   └── diabetes.csv
│
├── models/                    # Stores the trained SVM model and scaler
│   ├── svm_model.pkl
│   └── scaler.pkl
│
├── src/                       # Source code for each logical module
│   ├── preprocess.py          # Data loading, cleaning, and feature scaling
│   ├── train.py               # Model training and evaluation
│   └── predict.py             # Loads model and scaler for inference
│
├── main.py                    # Entry point for making predictions via CLI
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
