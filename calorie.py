import streamlit as st
import pandas as pd
import joblib
import gzip
import shutil
import os

# Set page configuration
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="wide"
)

# Custom Theme Colors
custom_css = """
    <style>
        body {
            color: #333333;
            background-color: #FAFAFA;
        }
        .stButton > button {
            border-radius: 12px;
            background-color: #FF5722;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #E64A19;
        }
        .stSidebar {
            background-color: #F0F2F6;
        }
        .stMarkdown h1, .stMarkdown h2 {
            color: #4CAF50;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Function to decompress files
def decompress_file(compressed_path, output_path):
    """Decompress a .gz file."""
    with gzip.open(compressed_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load model
@st.cache_resource
def load_model():
    compressed_model_path = "random_forest_model.pkl.gz"
    model_path = "random_forest_model.pkl"
    decompress_file(compressed_model_path, model_path)
    return joblib.load(model_path)

# Load preprocessor
@st.cache_resource
def load_preprocessor():
    compressed_preprocessor_path = "preprocessor.pkl.gz"
    preprocessor_path = "preprocessor.pkl"
    decompress_file(compressed_preprocessor_path, preprocessor_path)
    return joblib.load(preprocessor_path)

# Main function to run the app
def main():
    model = load_model()
    preprocessor = load_preprocessor()

    # Sidebar navigation
    with st.sidebar:
        st.image("calories_banner.jpg", use_container_width=True)
        st.header("⚡ Navigation")
        menu = st.radio("Go to:", ["🏋️‍♂️ Prediction", "📖 About"])
        st.markdown("---")
        st.write("Developed by **Your Name**")

    # Main content
    if menu == "🏋️‍♂️ Prediction":
        st.title("🔥 Calorie Burn Predictor")
        st.markdown("**Estimate how many calories you burn based on exercise details.**")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("👤 Age", min_value=10, max_value=100, value=25)
            height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70)
            gender = st.selectbox("🧑‍🤝‍🧑 Gender", ["Male", "Female"])
            age_group = st.selectbox("📅 Age Group", ["Young", "Middle-aged", "Old"])
        with col2:
            duration = st.number_input("⏳ Exercise Duration (minutes)", min_value=1, max_value=300, value=30)
            heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=40, max_value=220, value=120)
            body_temp = st.number_input("🌡️ Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
        
        bmi = weight / ((height / 100) ** 2)
        
        if st.button("🚀 Predict Calories Burned"):
            input_data = pd.DataFrame(
                [[age, height, weight, duration, heart_rate, body_temp, bmi, gender, age_group]],
                columns=["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI", "Gender", "Age_Group"]
            )
            input_data_transformed = preprocessor.transform(input_data)
            prediction = model.predict(input_data_transformed)[0]
            st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")
        
        if st.button("🔄 Reset Inputs"):
            st.session_state.clear()
            st.rerun()
    
    elif menu == "📖 About":
        st.title("📖 About This App")
        st.write("This app predicts the number of calories burned during physical activity using a **machine learning model**.")
        with st.expander("💡 How does this work?"):
            st.write(
                """
                - Enter your personal and workout details.
                - Click the **Predict** button.
                - The model estimates calories burned based on historical data.
                """
            )
        st.markdown("---")
        st.write("🚀 Built with **Streamlit** | 🎯 Model trained using **scikit-learn**")

if __name__ == "__main__":
    main()
