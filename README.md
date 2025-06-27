# 💼 Customer Churn & Salary Prediction using Artificial Neural Networks (ANN)

This project implements **two Streamlit-based web applications** using Artificial Neural Networks (ANNs) to:
1. 🔁 **Classify Customer Churn** (whether a customer will exit the bank).
2. 💰 **Predict Estimated Salary** (regression-based prediction of salary based on profile features).

Both models are built on the same dataset (`Churn_Modelling.csv`) with shared preprocessing steps using Scikit-learn and TensorFlow.

---

## 📁 Folder & File Structure Overview

```
📦 Project Root/

├── Churn_Modelling.csv                           # Original dataset used for both models

├── Churn_Prediction_Using_Classification.ipynb   # Notebook for ANN classification model (churn prediction)

├── Salary_Prediction_Using_Regression.ipynb       # Notebook for ANN regression model (salary prediction)

├── Churn_Prediction_Evaluation.ipynb             # Performance evaluation of the classification model
│
├── classification_app.py                         # Streamlit app for customer churn classification

├── regression_app.py                             # Streamlit app for salary prediction
│
├── Classification_Model.h5                       # Trained ANN model for churn classification

├── Regression_Model.h5                           # Trained ANN model for salary prediction
│
├── Gender_Encoder.pkl                             # LabelEncoder for encoding 'Gender'

├── Geography_OHE.pkl                              # OneHotEncoder for encoding 'Geography'

├── Scaler.pkl                                     # StandardScaler used for scaling input features
│
├── requirements.txt                               # Required packages for deployment on Streamlit Cloud

├── README.md                                      # Project documentation (this file)
```

---

## 🚀 App 1: Customer Churn Classification

- **File:** `classification_app.py`
- **Model File:** `Classification_Model.h5`
- **Goal:** Predict whether a bank customer will churn (exit) or stay.
- **Target Variable:** `Exited` (0 = Stay, 1 = Churn)
- **Metrics Used:** Accuracy, Precision, Recall, F1 Score

### Features Used:
- Geography 🌍
- Gender 👤
- Age 🎂
- Credit Score 💳
- Balance 💰
- Tenure (Years with Bank) 📅
- Number of Products 📦
- Has Credit Card?
- Is Active Member?

---

## 📈 App 2: Salary Prediction (Regression)

- **File:** `regression_app.py`
- **Model File:** `Regression_Model.h5`
- **Goal:** Predict the **EstimatedSalary** of a customer using regression.
- **Target Variable:** `EstimatedSalary` (continuous numeric value)
- **Metric Used:** Mean Absolute Error (MAE)

### Additional Feature:
- Includes `Exited` as a feature for predicting salary.

---

## ⚙️ Preprocessing & Pickle Files

These files are common to both models:
- `Gender_Encoder.pkl`: LabelEncoder trained on gender
- `Geography_OHE.pkl`: OneHotEncoder trained on geography
- `Scaler.pkl`: StandardScaler to normalize all features

They are loaded during app runtime to ensure consistency between training and inference.

---

## 🛠 Requirements

All dependencies are listed in `requirements.txt`:
```
streamlit==1.34.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
tensorboard==2.15.2
tensorflow==2.15.1
```

Use `pip install -r requirements.txt` to install them.

---

## 📦 Deployment

Both apps are compatible with [Streamlit Cloud](https://streamlit.io/cloud). Ensure that:
- All `.pkl` and `.h5` model files are in the project root.
- The correct entry point is used (`classification_app.py` or `regression_app.py`).

---

## 📌 Future Enhancements

- Add visualizations to display customer profile or feature importances.
- Enable multi-customer predictions through batch uploads.
- Use advanced MLOps tools to automate retraining.

---

## 👨‍💻 Author

**Shailesh Gupta** ----> (https://github.com/sg2499)

This project was developed as part of hands-on learning in ANN-based predictive modeling using Streamlit, Scikit-learn, and TensorFlow.

---