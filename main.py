import joblib
import streamlit as st
import pandas as pd


# Page config
st.set_page_config(
    page_title="Distillation column yield prediction"
)

# Page title
st.title('Distillation column yield prediction')
st.image('images/column.jpg')
st.write("\n\n")

st.markdown(
    """
    This app aims to assist in predicting the yield in distillation columns"""
)

# Load the model
model_file = 'model/final_model.joblib'
model = joblib.load(model_file)

# Streamlit interface to input data
flowrate = st.slider(label='Flowrate [m3/s]',min_value=100, max_value=500, value=300, step=1)
temperature = st.slider(label='Temperature [C]',min_value=100, max_value=200, value=130, step=1)
pressure = st.slider(label='Pressure difference [psi]',min_value=-50, max_value=50, value=0, step=1)


# Function to predict the input
def prediction(pressure,flowrate,temperature):
    # Create a df with input data
    df_input = pd.DataFrame({
        'PressureC1_diff': [pressure],
        'FlowC1': [flowrate],
        'Temp1': [temperature]
    })

    prediction = model.predict(df_input)
    return prediction

# Botton to predict
if st.button('Predict'):
    predict = prediction(pressure,flowrate,temperature)
    st.success(predict)