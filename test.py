import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Oyeniran20/Machine-Learning/refs/heads/main/3.%20Classification%20-/Exercise.csv')
    return df

# Feature Engineering
def feature_engineering(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    age_bins = [0, 30, 60, 80]
    age_labels = ['Young', 'Middle-aged', 'Old']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    df['log_calories'] = np.log1p(df['Calories'])
    df['Calories_sqrt'] = np.sqrt(df['Calories'])
    
    return df

# Load Preprocessor and Model
def load_components():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('random_forest_model.pkl')
    return preprocessor, model

# Evaluate Model
def evaluate_model(preprocessor, model, X_test, y_test):
    # Preprocess the test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_transformed)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance on Test Data:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

# Make Predictions
def make_prediction(preprocessor, model):
    print("\nEnter details for a single prediction:")
    age = float(input("Age: "))
    height = float(input("Height (cm): "))
    weight = float(input("Weight (kg): "))
    duration = float(input("Duration (mins): "))
    heart_rate = float(input("Heart Rate: "))
    body_temp = float(input("Body Temperature (°C): "))
    gender = input("Gender (Male/Female): ").strip().capitalize()
    
    # Calculate BMI and Age Group
    bmi = weight / (height / 100) ** 2
    age_group = 'Young' if age < 30 else 'Middle-aged' if age < 60 else 'Old'

    # Create a DataFrame for the input data
    data = pd.DataFrame([[age, height, weight, duration, heart_rate, body_temp, bmi, gender, age_group]],
                        columns=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Gender', 'Age_Group'])

    # Preprocess the input data
    data_transformed = preprocessor.transform(data)
    
    # Make prediction
    pred = model.predict(data_transformed)
    print(f"\nPredicted Calories Burnt (sqrt-transformed): {pred[0]:.2f}")
    print(f"Approximate Calories Burnt: {pred[0]**2:.2f}")

# Main Function
def main():
    # Load and preprocess data
    df = load_data()
    df = feature_engineering(df)

    # Prepare target and features
    y = df['Calories_sqrt']
    X = df.drop(columns=['User_ID', 'Calories', 'log_calories', 'Calories_sqrt'])

    # Load preprocessor and model
    preprocessor, model = load_components()

    # Evaluate model performance
    evaluate_model(preprocessor, model, X, y)

    # Make a single prediction
    make_prediction(preprocessor, model)

# Entry Point
if __name__ == "__main__":
    main()