# 📊 Customer Churn Prediction

This is a professional end-to-end machine learning project to predict customer churn using the **Telco Customer Churn dataset**.

The project covers the entire pipeline including:
- Data cleaning
- Feature engineering
- Handling class imbalance
- Model training using **Logistic Regression** and **XGBoost**
- Evaluation using confusion matrix, ROC curve, and feature importance
- Saving the best model to `.pkl` format

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/ahmedsawki/churn-prediction-by-ahmed-shawki.git
cd churn-prediction-by-ahmed-shawki
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook
```
Open the file `Customer_Churn_Prediction.ipynb` and run all cells.

---

## 🧠 Using the Trained Model

The best model is saved as:

```
models/xgboost_model.pkl
```

### To load and use it for prediction:

```python
import joblib
model = joblib.load("models/xgboost_model.pkl")
prediction = model.predict(new_data)  # new_data must match training features
```

---

## 📁 Project Structure

```
.
├── Customer_Churn_Prediction.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── xgboost_model.pkl
├── visuals/
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
```

---

## 📌 Credits

Developed by **Ahmed Shawki** as part of a machine learning portfolio project.
