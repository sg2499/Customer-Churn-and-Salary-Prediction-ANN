
# Customer Churn & Salary Prediction using ANN

This repository contains two Artificial Neural Network (ANN) based projects built using TensorFlow and deployed using Streamlit. The system performs:

- **Customer Churn Prediction** (Classification Task)
- **Estimated Salary Prediction** (Regression Task)
- **Hyperparameter Tuning** for ANN architectures using GridSearch

## 🔍 Project Overview

1. **Churn Prediction**: Classifies whether a customer will leave the bank.
2. **Salary Prediction**: Predicts the estimated salary of a customer based on various features.
3. **Hyperparameter Tuning**: Optimizes model performance using grid search with different numbers of hidden layers and neurons.

---

## 📁 Folder Structure

```
├── classification_app.py         # Streamlit app for churn classification

├── regression_app.py             # Streamlit app for salary regression

├── Classification_Model.h5       # Trained ANN classification model

├── Regression_Model.h5           # Trained ANN regression model

├── Gender_Encoder.pkl            # LabelEncoder for Gender

├── Geography_OHE.pkl             # OneHotEncoder for Geography

├── Scaler.pkl                    # StandardScaler for preprocessing

├── Churn_Modelling.csv           # Dataset used for training

├── Churn_Prediction_Evaluation.ipynb     # Evaluation of classifier model

├── Churn_Prediction_Using_Regression.ipynb # Regression model training

├── HyperParameter_Tuning_ANN.ipynb       # Grid Search for optimal ANN

├── requirements.txt              # Project dependencies

└── README.md                     # Project documentation
```

---

## 📊 Dataset

- **Name**: `Churn_Modelling.csv`
- **Source**: Contains 10,000 customer records of a bank.
- **Key Columns Used**:
  - CreditScore, Geography, Gender, Age, Tenure, Balance
  - NumOfProducts, HasCrCard, IsActiveMember, Exited (Target)

---

## 🧠 Models

### 1. ANN for Churn Prediction
- **Model Type**: Binary Classifier
- **Output**: Probability of churn (Exited = 0 or 1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Loss Function**: Binary Crossentropy

### 2. ANN for Salary Prediction
- **Model Type**: Regressor
- **Output**: Estimated Salary
- **Activation**: ReLU for hidden layers
- **Loss Function**: Mean Absolute Error (MAE)

---

## ⚙️ Hyperparameter Tuning

File: `HyperParameter_Tuning_ANN.ipynb`  
Uses `KerasClassifier` & `GridSearchCV` to find best combination of:
- Neurons: `[16, 32, 64, 128]`
- Layers: `[1, 2]`
- Epochs: `[50, 100]`

---

## 🖥 How to Run Locally

1. **Create Virtual Environment** (Python 3.10+ recommended)
```bash
conda create -n ann_project python=3.10
conda activate ann_project
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Streamlit App**
```bash
streamlit run classification_app.py
# or
streamlit run regression_app.py
```

---

## 📦 Requirements

See `requirements.txt`. Main libraries:
- streamlit==1.34.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.4.2
- matplotlib==3.8.4
- tensorboard==2.15.2
- tensorflow==2.15.0
- scikeras

---

## 📌 Notes

- Ensure all pickle files (`*.pkl`) and models (`*.h5`) are present in the working directory.
- Compatible with Python 3.10 or below (TensorFlow 2.15.0 ABI issue with Python 3.13).

---

## 📬 Author

This project is developed and maintained by **Shailesh Gupta** -----> (https://github.com/sg2499).
