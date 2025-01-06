import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Train the model
def train_model(appearances_df, games_df):
    # Merge datasets
    data = pd.merge(appearances_df, games_df, on="game_id", suffixes=('_player', '_game'))

    # Feature Engineering
    data['goal_scored'] = (data['goals'] > 0).astype(int)  # Target variable
    features = ['player_club_id', 'assists', 'minutes_played', 'home_club_goals', 'away_club_goals']

    # Handling missing values
    data = data[features + ['goal_scored']].dropna()

    # Splitting dataset
    X = data[features]
    y = data['goal_scored']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balancing the dataset with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Balancing the dataset with undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    # Calculate class weights
    class_weights = {0: 1, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}

    # Train Gradient Boosting model
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_resampled, y_resampled)

    # Evaluate model
    y_pred = gbm.predict(X_test)
    y_pred_proba = gbm.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    # Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': gbm.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    return gbm, feature_importances

# Save the trained model
def save_model(model, model_path='gbm_model.pkl'):
    joblib.dump(model, model_path)
    print("Model saved successfully.")

# Load a trained model
def load_model(model_path='gbm_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None

# Predict using the model
def make_prediction(model, input_data):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    return prediction, prediction_proba

# Plot feature importance
def plot_feature_importance(feature_importances):
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values(by='Importance', ascending=True).plot.barh(x='Feature', y='Importance', legend=False, color='skyblue')
    plt.title('Feature Importance - Gradient Boosting Classifier', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()
