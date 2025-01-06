import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load and merge datasets
def load_and_merge_data(club_games_df, clubs_df, competitions_df):
    clubs = clubs_df
    merged_df = pd.merge(club_games_df, clubs, on='club_id', how='left')
    merged_df = pd.merge(merged_df, competitions_df, left_on='domestic_competition_id', right_on='competition_id',
                         how='left')
    return merged_df


# Feature engineering
def feature_engineering(merged_df):
    label_encoder = LabelEncoder()
    merged_df['hosting'] = label_encoder.fit_transform(merged_df['hosting'].map({'Home': 1, 'Away': 0}))
    merged_df['goal_difference'] = merged_df['own_goals'] - merged_df['opponent_goals']
    merged_df['revenue'] = merged_df['stadium_seats'] * merged_df['hosting']
    merged_df['profitability'] = merged_df['revenue'] + merged_df['is_win'] * 10000
    return merged_df


# Prepare features and target
def prepare_data(merged_df, features, target):
    merged_df.dropna(subset=[target], inplace=True)
    X = merged_df[features]
    y = merged_df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train and evaluate model
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='poly')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared Score (R2): {r2}")

    return model, scaler


# Save the trained model
def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


# Load a trained model
def load_model(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found.")
        return None, None