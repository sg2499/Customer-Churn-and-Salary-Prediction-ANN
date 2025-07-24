# Importing necessary libraries
import streamlit as st                     # For building the web app
import pandas as pd                       # For handling tabular data
import numpy as np                        # For numerical computations
import tensorflow as tf                   # For loading and running the trained ANN model
import pickle as pkl                      # For loading pre-trained encoders and scalers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# -------------------------------
# Load the Trained ANN Model
# -------------------------------
# 'Model.h5' is the saved ANN model trained to predict customer churn
model = tf.keras.models.load_model('Classification_Model.h5', compile = False)

# -------------------------------
# Load Preprocessing Objects
# -------------------------------
# Load the LabelEncoder used for the 'Gender' feature
with open('Gender_Encoder.pkl', 'rb') as file:
    gender_encoder = pkl.load(file)

# Load the OneHotEncoder used for the 'Geography' feature
with open('Geography_OHE.pkl', 'rb') as file:
    geo_ohe = pkl.load(file)

# Load the StandardScaler used to scale numeric input features
with open('Scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)

# -------------------------------
# Streamlit Web App UI
# -------------------------------
st.title('🧠 Customer Churn Prediction')
st.write("Fill in the customer details below to predict the likelihood of churn:")

# User Inputs from the sidebar or main interface
geography = st.selectbox('🌍 Select Geography', geo_ohe.categories_[0])
gender = st.selectbox('👤 Select Gender', gender_encoder.classes_)
age = st.slider('🎂 Age', 18, 100)
balance = st.number_input('🏦 Balance Amount')
credit_score = st.number_input('💳 Credit Score')
estimated_salary = st.number_input('💰 Estimated Salary')
tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('📦 Number Of Products', 1, 5)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

# -------------------------------
# Prepare the Input DataFrame
# -------------------------------
# First, encode 'Gender' using the LabelEncoder
gender_encoded = gender_encoder.transform([gender])[0]

# Create a DataFrame with all numeric and encoded values
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [gender_encoded],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# -------------------------------
# One-Hot Encode 'Geography'
# -------------------------------
# Convert geography into one-hot encoded format using the same encoder used during training
geo_encoded = geo_ohe.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_ohe.get_feature_names_out(['Geography']))

# Merge the encoded geography with the main input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# -------------------------------
# Scale the Input Features
# -------------------------------
# Ensure input matches training data scaling for accurate predictions
# Optional fix depending on how the scaler was trained:
try:
    input_data_scaled = scaler.transform(input_data)
except ValueError:
    input_data_scaled = scaler.transform(input_data.values)


# -------------------------------
# Make the Prediction
# -------------------------------
# Predict churn probability using the trained ANN model
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]  # Get the predicted probability

# -------------------------------
# Display the Result
# -------------------------------
st.subheader("📊 Prediction Result")
st.write(f'**Churn Probability**: `{prediction_prob:.2f}`')

if prediction_prob < 0.5:
    st.success('✅ The customer is likely to **Stay**.')
else:
    st.error('⚠️ The customer is likely to **Churn**.')

# -------------------------------
# End of App
# ---------------------------------