import streamlit as st
import pandas as pd
import joblib
import gzip
import shutil

# Set custom Streamlit theme
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for beautification
def load_custom_css():
    # Custom CSS for a refined light theme
    custom_css = """
        <style>
            /* Set a clean, elegant light background */
            body {
                background-color: #FFF8E7;  /* Soft cream */
                color: #4B3621;  /* Deep brown for readability */
            }

            /* Fixed Title Styling */
            .stMarkdown h1, .stMarkdown h2 {
                color: #DAA520 !important; /* Gold headings */
                font-family: 'Georgia', serif;
                text-shadow: 1px 1px 3px rgba(218, 165, 32, 0.3);
                position: sticky;
                top: 0;
                background-color: #FFF8E7; /* Ensures it's readable */
                padding: 10px 0;
                z-index: 100;
            }

            /* Light Sidebar */
            .stSidebar {
                background-color: #FAF3DD !important;
                border-right: 3px solid #DAA520;
                padding: 20px;
                border-radius: 10px;
            }

            /* Social Media Links with Icons */
            .stSidebar a {
                display: flex;
                align-items: center;
                text-decoration: none;
                font-weight: bold;
                color: #4B3621;
                margin-bottom: 10px;
            }

            .stSidebar a img {
                width: 20px;
                height: 20px;
                margin-right: 10px;
            }

            /* Buttons */
            .stButton > button {
                border-radius: 10px;
                background-color: #DAA520;
                color: #121212;
                font-size: 18px;
                padding: 12px 24px;
                font-weight: bold;
                box-shadow: 0px 4px 10px rgba(218, 165, 32, 0.3);
            }

            .stButton > button:hover {
                background-color: #FFC107;
            }

            /* Input Fields */
            .stTextInput, .stNumberInput, .stSelectbox {
                background-color: #FAF3DD;
                color: #4B3621;
                border-radius: 5px;
            }

            /* Footer */
            .footer {
                position: fixed;
                bottom: 10px;
                width: 100%;
                text-align: center;
                color: #DAA520;
                font-size: 14px;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    
# Function to decompress model files
def decompress_file(compressed_path, output_path):
    with gzip.open(compressed_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load model and preprocessor
@st.cache_resource
def load_model():
    decompress_file("random_forest_model.pkl.gz", "random_forest_model.pkl")
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_preprocessor():
    decompress_file("preprocessor.pkl.gz", "preprocessor.pkl")
    return joblib.load("preprocessor.pkl")

# Define age group criteria
def get_age_group(age):
    if age < 30:
        return "Young"
    elif 30 <= age < 50:
        return "Middle-aged"
    else:
        return "Old"

# Main function
def main():
    load_custom_css()
    model = load_model()
    preprocessor = load_preprocessor()

    st.markdown("<div class='header'>🔥 Calorie Burn Predictor</div>", unsafe_allow_html=True)
    st.sidebar.image("calories_banner.jpg", use_container_width=True)
    st.sidebar.header("⚡ Navigation")
    menu = st.sidebar.radio("Go to:", ["🏋️‍♂️ Prediction", "📖 About"])
    st.sidebar.markdown("---")
    
    # Add social media links with logos
    st.sidebar.write("👤 Developed by **Matthew**")
    st.sidebar.markdown(
        """
        [<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"/> LinkedIn](https://www.linkedin.com/in/oyeniran-matthew/)  
        [<img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="20"/> Twitter](https://x.com/idmathex)  
        [<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="20"/> Email](mailto:oyeniranmatthew@gmail.com)  
        [<img src="https://cdn-icons-png.flaticon.com/512/2111/2111728.png" width="20"/> WhatsApp](https://wa.me/2348106171072)
        """,
        unsafe_allow_html=True
    )

    if menu == "🏋️‍♂️ Prediction":
        st.title("🔥 Predict Your Calorie Burn")
        st.markdown("**Estimate how many calories you burn based on exercise details.**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            age = st.number_input("👤 Age", min_value=10, max_value=100, value=25)
            height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70)
            gender = st.selectbox("🧑‍🤝‍🧑 Gender", ["Male", "Female"])
            age_group = get_age_group(age)
        
        with col2:
            duration = st.number_input("⏳ Exercise Duration (minutes)", min_value=1, max_value=300, value=30)
            heart_rate = st.number_input("❤️ Heart Rate (bpm)", min_value=40, max_value=220, value=120)
            body_temp = st.number_input("🌡️ Body Temperature (°C)", min_value=32.0, max_value=42.0, value=37.0)
        
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
