# 📊 Customer Churn Prediction

This project is a professional end-to-end data science pipeline to predict customer churn using the Telco Customer Churn dataset.
It includes all key stages: data cleaning, advanced feature engineering, handling class imbalance, model training with logistic regression and XGBoost, and performance evaluation.

---

## 📁 Dataset
- Source: IBM Sample Dataset (Telco Customer Churn)
- 7,043 customer records with 21 features
- Target: `Churn` (Yes/No)

---

## ⚙️ Technologies Used
- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

---

## 🧹 Data Preprocessing
- Removed `customerID`
- Converted `TotalCharges` to numeric (with imputation)
- Transformed `SeniorCitizen` to categorical
- Created new features: `MonthlyChargeCategory`, `TenureGroup`
- LabelEncoded binary columns
- OneHotEncoded remaining categorical features

---

## 🧠 Models Trained
- Logistic Regression
- XGBoost Classifier

Class imbalance was addressed using SMOTE.

---

## 🎯 Evaluation Metrics
| Model                | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | ~0.80    | ~0.75     | ~0.74  | ~0.74    |
| XGBoost Classifier  | ~0.77    | ~0.56     | ~0.59  | ~0.57    |

---

## 📈 Visuals
- Confusion matrix for both models
- Class distribution before & after SMOTE
- Feature engineering overview

---

## 📂 Folder Structure
```
.
├── Customer_Churn_Prediction_Final.ipynb
├── README.md
└── visuals/
    ├── churn_distribution.png
    ├── logistic_confusion_matrix.png
    └── xgboost_confusion_matrix.png
```

---

## ✅ Conclusion
This project demonstrates a complete and professional churn prediction pipeline, ready for deployment or integration into business tools.

> Made by **Ahmed Shawki**

---
