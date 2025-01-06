import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st


def train_model(players_df):
    # Merge the datasets
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'])
    players_df['market_value_in_eur'] = pd.to_numeric(players_df['market_value_in_eur'], errors='coerce')
    players_df.dropna(subset=['market_value_in_eur', 'highest_market_value_in_eur'], inplace=True)

    # Feature engineering
    players_df['age'] = (pd.Timestamp.now() - players_df['date_of_birth']).dt.days // 365
    players_df['value_decline_ratio'] = players_df['market_value_in_eur'] / players_df['highest_market_value_in_eur']

    # Model features
    features = ['market_value_in_eur', 'highest_market_value_in_eur', 'position', 'age', 'height_in_cm']

    # Simulate a target column for demonstration
    players_df['undervalued'] = (
                players_df['market_value_in_eur'] < players_df['highest_market_value_in_eur'] * 0.7).astype(int)

    X = players_df[features]
    y = players_df['undervalued']

    # Column transformations
    categorical_features = ['position']
    numerical_features = ['market_value_in_eur', 'highest_market_value_in_eur', 'age', 'height_in_cm']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    pipeline.fit(X_train, y_train)

    return pipeline


# Function to load the trained model
def load_model():
    model_path = 'rf_model.pkl'
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None


# Function to make a prediction
def make_prediction(model, input_data):
    return model.predict(input_data)[0], model.predict_proba(input_data)[0]
