# Customer Churn Prediction 📉

## 🎯 Objective
This project aims to predict customer churn based on various behavioral and service usage data using machine learning models.

## 🧰 Tech Stack
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib & Seaborn
- Joblib

## 🚀 How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

## 📁 Project Structure
```
customer-churn-prediction/
├── README.md
├── Customer_Churn_Prediction.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── xgboost_model.pkl
├── visuals/
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   ├── logistic_confusion_matrix.png
│   ├── roc_curve.png
│   └── xgboost_confusion_matrix.png
├── requirements.txt
```

## 📊 Results
The XGBoost model outperformed other models with the highest accuracy and best AUC score.

## 🔮 Future Work
- Add a web interface to interact with predictions
- Tune hyperparameters further using GridSearchCV
- Evaluate model fairness across different customer segments
