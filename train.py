import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Oyeniran20/Machine-Learning/refs/heads/main/3.%20Classification%20-/Exercise.csv')
    print("Columns in the dataset:", df.columns.tolist())
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

# Preprocessing
def get_preprocessor():
    num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    cat_features = ['Gender', 'Age_Group']

    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
    pca_transformer = Pipeline(steps=[('pca', PCA(n_components=2))]) 

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('pca', pca_transformer, ['Duration', 'Heart_Rate', 'Body_Temp'])  
    ])
    
    return preprocessor, num_features + cat_features

# Model Training and Evaluation
def train_model(X_train, y_train, X_test, y_test, preprocessor, feature_names):
    # Use Random Forest as the best model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the preprocessor on the training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Train the model
    model.fit(X_train_transformed, y_train)
    
    # Evaluate the model
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRandom Forest:")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Feature Importance for Random Forest
    model_feature_importance = model.feature_importances_
    feature_importance = sorted(zip(feature_names, model_feature_importance), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")
    
    return preprocessor, model

# Save Preprocessor and Model Separately
def save_components(preprocessor, model):
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(model, 'random_forest_model.pkl')
    print("\nPreprocessor saved as 'preprocessor.pkl'")
    print("Random Forest model saved as 'random_forest_model.pkl'")

# Main Function
def main():
    # Load and preprocess data
    df = load_data()
    df = feature_engineering(df)
    
    # Prepare target and features
    y = df['Calories_sqrt']
    X = df.drop(columns=['User_ID', 'Calories', 'log_calories', 'Calories_sqrt'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get preprocessor and feature names
    preprocessor, feature_names = get_preprocessor()
    
    # Train the best model
    preprocessor, model = train_model(X_train, y_train, X_test, y_test, preprocessor, feature_names)
    
    # Save the preprocessor and model separately
    save_components(preprocessor, model)

# Entry point
if __name__ == "__main__":
    main()