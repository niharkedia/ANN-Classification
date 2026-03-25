import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as f:
    le_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl','rb') as f:
    ohe_geography = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


## streamlit app

st.title('Customer churn Prediction')


geography = st.selectbox('Geography', ohe_geography.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)

credit_score = st.number_input('Credit Score')
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')

EstimatedSalary = st.number_input('Estimated Salary')
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [EstimatedSalary]
})


geo_encoded = ohe_geography.transform(pd.DataFrame({'Geography': [geography]}))
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=ohe_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_encoded_df], axis=1)



input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write("Churn Probability: ",prediction_proba)

if prediction_proba > 0.5:
    st.write('Customer will churn')
else:
    st.write('Customer will not churn')




