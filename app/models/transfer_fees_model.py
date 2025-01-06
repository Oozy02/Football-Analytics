import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st


def train_model(transfers_df, players_df, appearances_df):
    # Data preprocessing
    transfers_df['transfer_date'] = pd.to_datetime(transfers_df['transfer_date'])
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'])
    players_df['age'] = (pd.Timestamp.now() - players_df['date_of_birth']).dt.days // 365

    players_selected = players_df[['player_id', 'age', 'market_value_in_eur']]
    appearances_selected = appearances_df[['player_id', 'goals', 'assists', 'minutes_played']]

    # Calculate performance metric
    appearances_selected['performance'] = (appearances_selected['goals'] + 0.5 * appearances_selected['assists']) / appearances_selected['minutes_played'] * 100

    # Aggregate performance metrics
    performance_metrics = appearances_selected.groupby('player_id').agg({
        'performance': 'mean',
        'goals': 'sum',
        'assists': 'sum',
        'minutes_played': 'sum'
    }).reset_index()

    # Merge datasets
    merged_data = transfers_df.merge(players_selected, on='player_id', how='left').merge(performance_metrics, on='player_id', how='left')
    merged_data.dropna(inplace=True)

    # Map average transfer fee per club as a feature
    club_influence_mapping = transfers_df.groupby('from_club_id')['transfer_fee'].mean()
    merged_data['club_influence'] = merged_data['from_club_id'].map(club_influence_mapping)
    overall_avg_transfer_fee = transfers_df['transfer_fee'].mean()
    merged_data['club_influence'].fillna(overall_avg_transfer_fee, inplace=True)

    # Filter rows with zero transfer fee
    merged_data = merged_data[merged_data['transfer_fee'] > 0].reset_index(drop=True)

    # Log transformation for skewed features
    merged_data['market_value_in_eur_y'] = np.log1p(merged_data['market_value_in_eur_y'])
    merged_data['transfer_fee'] = np.log1p(merged_data['transfer_fee'])

    # Scaling numerical features
    numerical_features = ['age', 'market_value_in_eur_y', 'performance', 'goals', 'assists', 'minutes_played', 'club_influence']
    scaler = StandardScaler()
    merged_data[numerical_features] = scaler.fit_transform(merged_data[numerical_features])

    # Define features (X) and target (y)
    X = merged_data[numerical_features]
    y = merged_data['transfer_fee']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_regressor.fit(X_train, y_train)

    # Save model
    joblib.dump(xgb_regressor, 'xgb_player_model.pkl')

    return xgb_regressor


def load_model():
    model_path = 'xgb_player_model.pkl'
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None


def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]
