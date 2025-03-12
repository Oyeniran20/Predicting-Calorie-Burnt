import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide"
)

# Function to load artifacts
def load_components():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('random_forest_model.pkl')
    return preprocessor, model

# Function to preprocess user input
def preprocess_input(input_data, preprocessor):
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    return input_transformed

# Function to make predictions
def predict(model, input_transformed):
    prediction = model.predict(input_transformed)
    return prediction[0] ** 2  # Reverse sqrt transformation

# Streamlit app
def main():
    st.image("calories_banner.jpg", use_container_width=True)
    
    tab1, tab2 = st.tabs(["🔥 Prediction", "ℹ️ About"])
    
    with tab1:
        st.title("🔥 Calories Burnt Predictor")
        st.markdown(
            """
            **Predict the number of calories burnt during exercise**
            based on personal and workout details.
            """
        )
        
        # Load model and preprocessor
        preprocessor, model = load_components()
        
        # Sidebar with input fields
        st.sidebar.header("🔧 Input Features")
        st.sidebar.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("🎂 Age", value=25)
            height = st.number_input("📏 Height (cm)", value=170.0)
            weight = st.number_input("⚖️ Weight (kg)", value=70.0)
            duration = st.number_input("⏳ Duration (mins)", value=30.0)
            heart_rate = st.number_input("💓 Heart Rate", value=80.0)
        
        with col2:
            body_temp = st.number_input("🌡️ Body Temperature (°C)", value=36.5)
            gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
            
        # Calculate BMI and Age Group
        bmi = weight / (height / 100) ** 2
        age_group = "Young" if age < 30 else "Middle-aged" if age < 60 else "Old"
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Heart_Rate': heart_rate,
            'Body_Temp': body_temp,
            'BMI': bmi,
            'Gender': gender,
            'Age_Group': age_group
        }
        
        # Preprocess input data
        input_transformed = preprocess_input(input_data, preprocessor)
        
        if st.button("🚀 Predict Calories Burnt"):
            prediction = predict(model, input_transformed)
            st.success(f"🔥 Predicted Calories Burnt: **{prediction:.2f} kcal**")
        
        if st.button("🔄 Reset Inputs"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with tab2:
        st.write("### ℹ️ About This App")
        st.write("This app predicts calories burnt based on user inputs using machine learning.")
        
        with st.expander("💡 How does this work?"):
            st.write(
                """
                - Enter personal and workout details.
                - Click **Predict** to get an estimate.
                - Model is trained on historical exercise data.
                """
            )

# Run the app
if __name__ == '__main__':
    main()
