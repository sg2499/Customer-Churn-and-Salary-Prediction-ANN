# 🧠 Customer Analytics with Artificial Neural Networks (ANN)

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/Customer_ANN_Analytics)
![GitHub stars](https://img.shields.io/github/stars/your-username/Customer_ANN_Analytics?style=social)
![Last Commit](https://img.shields.io/github/last-commit/your-username/Customer_ANN_Analytics)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)

This repository provides a comprehensive solution using **Artificial Neural Networks** (ANNs) to solve two real-world business problems:

- 🔍 **Customer Churn Prediction** (Classification)
- 💼 **Estimated Salary Prediction** (Regression)

Both models are built, trained, and deployed with interactive **Streamlit web apps**, allowing real-time predictions based on user inputs. Additionally, the repository contains a notebook for **hyperparameter tuning using Grid Search** with Keras and Scikit-learn integration.

---

## 📁 Project Folder Structure

```
📦Customer_ANN_Analytics/
├── Churn_Modelling.csv                # Dataset used for both models
├── classification_app.py             # Streamlit app for churn classification
├── regression_app.py                 # Streamlit app for salary regression
├── Classification_Model.h5          # Trained ANN model for classification
├── Regression_Model.h5              # Trained ANN model for regression
├── Gender_Encoder.pkl                # LabelEncoder for 'Gender'
├── Geography_OHE.pkl                 # OneHotEncoder for 'Geography'
├── Scaler.pkl                        # StandardScaler for input features
├── Churn_Prediction_Evaluation.ipynb # Classification model training & evaluation
├── Churn_Prediction_Using_Regression.ipynb # Regression model training notebook
├── HyperParameter_Tuning_ANN.ipynb   # Grid search over neurons/layers/epochs
├── requirements.txt                  # All required Python libraries
├── README.md                         # Project documentation
```

---

## 📊 1. Classification App – Customer Churn Prediction

- **Input:** Credit Score, Gender, Age, Balance, Products, etc.
- **Output:** Churn Probability + Classification (Stay or Exit)
- **Model:** Binary Classification using ANN
- **File:** `classification_app.py`
- **Model File:** `Classification_Model.h5`

---

## 📈 2. Regression App – Estimated Salary Prediction

- **Input:** Customer Demographics + Banking History
- **Output:** Predicted Estimated Salary
- **Model:** ANN Regression
- **File:** `regression_app.py`
- **Model File:** `Regression_Model.h5`

---

## 🧪 3. Hyperparameter Tuning

- Optimize number of neurons, layers, and epochs for the ANN
- Uses `KerasClassifier` from `scikeras` with `GridSearchCV`
- **File:** `HyperParameter_Tuning_ANN.ipynb`

---

## 💾 Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/sg2499/Customer_ANN_Analytics.git
cd Customer_ANN_Analytics
```

### 🐍 Create a Virtual Environment (Recommended)

```bash
conda create -n ann_env python=3.10
conda activate ann_env
```

### 📦 Install All Dependencies

```bash
pip install -r requirements.txt
```

### 🧪 Run the Apps

#### 👉 For Churn Prediction (Classification)

```bash
streamlit run classification_app.py
```

#### 👉 For Salary Prediction (Regression)

```bash
streamlit run regression_app.py
```

---

## 📚 Dataset

The dataset used is `Churn_Modelling.csv` containing customer data from a bank, including:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card and Activity Status
- Estimated Salary
- Exited (Target for classification)

---

## ✅ Requirements

Refer to `requirements.txt`. Major packages used include:

- `tensorflow==2.15.0`
- `streamlit==1.34.0`
- `scikit-learn==1.4.2`
- `pandas`, `numpy`, `matplotlib`
- `scikeras` (for grid search)

---

## 🔍 Author Notes

- Both models use consistent encoders and scalers.
- All pickle files must be in the same directory as the app for successful execution.
- Hyperparameter tuning results may vary based on dataset splits and randomness.

---

## 📬 Contact

For feedback or collaboration, feel free to reach out at [shaileshgupta841@gmail.com] or connect via GitHub.

## 📬 Author

This project is developed and maintained by **Shailesh Gupta** -----> (https://github.com/sg2499).

---

> Built with ❤️ using TensorFlow, Scikit-learn, and Streamlit.
