import joblib
import streamlit as st
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Distillation Column Yield Predictor",
    page_icon="🧪",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained model from a joblib file."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
        return None

model = load_model('model/final_model.joblib')

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    st.markdown("""
    Adjust the sliders to match the operational parameters of the distillation column.
    """)

    flowrate = st.slider(
        label='Feed Flowrate (m³/s)',
        min_value=100,
        max_value=500,
        value=300,
        step=1
    )
    st.caption("Represents the volume of the feed mixture entering the column per second. Higher flowrates can impact separation efficiency.")

    temperature = st.slider(
        label='Reboiler Temperature (°C)',
        min_value=100,
        max_value=200,
        value=130,
        step=1
    )
    st.caption("The temperature at the bottom of the column (reboiler). This is crucial for vaporizing the components for separation.")

    pressure = st.slider(
        label='Pressure Difference (psi)',
        min_value=-50,
        max_value=50,
        value=0,
        step=1
    )
    st.caption("The pressure drop across the column. It influences the boiling points of the components and the overall separation process.")

# --- Main Page Content ---
st.title("🧪 Distillation Column Yield Predictor")
#st.image('Figuras/column.jpg', caption='A typical industrial distillation column.')

st.markdown("""
Welcome to the Distillation Yield Predictor! This application uses a machine learning model to forecast the production yield of a chemical product from a distillation column based on key operational parameters.

### What is Distillation?
Distillation is a widely used industrial process to separate liquid mixtures based on differences in their boiling points. By carefully controlling temperature, pressure, and flowrate, we can isolate and purify valuable chemical components.

**This tool can help process engineers and operators to:**
- **Optimize** operating conditions for maximum yield.
- **Predict** the impact of process changes before implementation.
- **Troubleshoot** potential issues by simulating different scenarios.
""")

# --- Explainer Section ---
with st.expander("ℹ️ About the Application"):
    st.markdown("""
    **How does it work?**

    1.  **Input Data:** You provide the key operational parameters (flowrate, temperature, and pressure) using the sliders in the sidebar.
    2.  **Prediction:** The pre-trained machine learning model receives these inputs. It analyzes them based on the patterns learned from historical data to calculate the expected yield.
    3.  **Output:** The app displays the final predicted yield as a percentage, giving you an estimate of the column's efficiency under the specified conditions.

    **Model Details:**

    * **Model Type:** `Regression Model` (likely from the scikit-learn library)
    * **Purpose:** To predict the continuous value of the distillation yield.
    * **Features Used:** Feed Flowrate, Reboiler Temperature, and Pressure Difference.
    """)

st.divider()

# --- Prediction Logic ---
if model is not None:
    if st.button('🚀 Predict Yield', type="primary"):
        df_input = pd.DataFrame({
            'PressureC1_diff': [pressure],
            'FlowC1': [flowrate],
            'Temp1': [temperature]
        })

        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Predicted Result")
            st.success(f"**Predicted Yield:** `{prediction_value[0]:.2f}%`")
            st.info("This value represents the estimated percentage of the desired product that will be recovered from the feed mixture under the specified conditions.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model could not be loaded. Please check the model file path.")
