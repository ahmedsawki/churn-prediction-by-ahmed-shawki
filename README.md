# 📊 Customer Churn Prediction

A complete end-to-end machine learning project to predict customer churn using the **Telco Customer Churn Dataset**. This solution follows best practices in data preprocessing, model building, hyperparameter tuning, and evaluation. The final trained model is saved and ready for deployment.

---

## 🧠 Objectives

- Analyze customer behavior to identify churn risk.
- Build classification models (Logistic Regression, XGBoost).
- Use GridSearchCV to optimize model performance.
- Save the best-performing model for production use.

---

## 📁 Project Structure

```
├── data/                         # Dataset files (CSV)
├── visuals/                      # Charts and visualizations
├── xgboost_model.pkl             # Saved model file
├── Customer_Churn_Prediction.ipynb  # Main notebook
└── README.md                     # Project description
```

---

## 📦 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib / seaborn
- scikit-learn
- xgboost
- joblib

Install using:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Workflow

1. **Data Loading & Cleaning**
   - Handle missing values, convert datatypes, drop irrelevant columns.

2. **Exploratory Data Analysis (EDA)**
   - Visualize class imbalance, correlation, and key feature distributions.

3. **Feature Engineering**
   - Convert categorical variables, normalize numerical data.

4. **Train/Test Split**
   - Stratified sampling to preserve churn ratio.

5. **Model Training**
   - Logistic Regression (baseline)
   - XGBoost Classifier (boosted tree)

6. **Hyperparameter Tuning**
   - Using `GridSearchCV` to optimize:
     - `learning_rate`, `max_depth`, `n_estimators`, `subsample`

7. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix and ROC Curve plotted

8. **Model Saving**
   - Best model saved using `joblib`:
     ```python
     joblib.dump(best_model, 'xgboost_model.pkl')
     ```

---

## 🧪 Best Model Parameters

Output from GridSearchCV:
```python
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}
```

---

## 📈 Results & Visuals

Visualizations are stored in the `visuals/` folder:
- `churn_distribution.png`
- `feature_importance.png`
- `roc_curve.png`
- `logistic_confusion_matrix.png`
- `xgboost_confusion_matrix.png`

---

## 🗂 Dataset

- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Records: ~7000
- Features: Customer demographics, services, contract details

---

## 🚀 Usage

To load and use the saved model:

```python
import joblib
model = joblib.load('xgboost_model.pkl')
preds = model.predict(X_test)
```

---

## 👨‍💻 Author

**Ahmed Shawki**  
[GitHub Profile](https://github.com/ahmedsawki)  
Data Science & AI Enthusiast

---

## 📜 License

This project is licensed under the MIT License.
