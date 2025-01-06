import numpy as np
import streamlit as st
import pandas as pd
import joblib
import models.undervalued_model as rf
import models.transfer_fees_model as xgb
import models.profitability_model as svr
import models.player_goal_model as gbm
from database.database import create_db, insert_rf_prediction, insert_xgb_prediction, insert_svr_prediction, \
    get_past_predictions, insert_gbm_prediction, get_data_from_db, check_data_exists, load_csv_into_db

# Ensure the necessary files and database exist
create_db()

# Sidebar for page navigation
page = st.sidebar.selectbox("Select Page", ["Player Undervaluation Predictor", "Transfer Fee Predictor", "Profitability Predictor", "Goal Scoring Predictor", "Train Model"])

if page == "Player Undervaluation Predictor":
    st.title("Player Undervaluation Predictor")

    model = rf.load_model()
    if model:
        st.sidebar.header("Input Player Details")
        market_value = st.sidebar.number_input("Market Value in EUR", min_value=0.0, value=1000000.0)
        highest_market_value = st.sidebar.number_input("Highest Market Value in EUR", min_value=0.0, value=2000000.0)
        position = st.sidebar.selectbox("Position", ["Goalkeeper", "Defender", "Midfield", "Attack"])
        age = st.sidebar.slider("Age", 15, 45, 25)
        height = st.sidebar.number_input("Height in cm", 150, 220, 180)

        if st.sidebar.button("Predict"):
            with st.spinner("Predicting..."):
                input_data = pd.DataFrame({
                    "market_value_in_eur": [market_value],
                    "highest_market_value_in_eur": [highest_market_value],
                    "position": [position],
                    "age": [age],
                    "height_in_cm": [height]
                })
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                insert_rf_prediction(market_value, highest_market_value, position, age, height, prediction,
                                     max(prediction_proba))

                st.write("### Prediction Result")
                if prediction == 1:
                    st.success("The player is undervalued.")
                else:
                    st.info("The player is not undervalued.")
                st.write(f"### Prediction Confidence: {max(prediction_proba) * 100:.2f}%")

        if st.sidebar.checkbox("Show Past Predictions"):
            with st.spinner("Loading past predictions..."):
                st.write("### Past Predictions")
                st.write(get_past_predictions(model_type="rf"))

elif page == "Transfer Fee Predictor":
    st.title("Transfer Fee Predictor")

    model = xgb.load_model()
    if model:
        st.sidebar.header("Input Player Details")
        age = st.sidebar.slider("Age", 15, 45, 25)
        market_value = st.sidebar.number_input("Market Value in EUR", min_value=0.0, value=1000000.0)
        performance = st.sidebar.slider("Performance Rating (0.0 - 1.0)", 0.0, 1.0, 0.75)
        goals = st.sidebar.number_input("Goals Scored", min_value=0, value=10)
        assists = st.sidebar.number_input("Assists", min_value=0, value=5)
        minutes_played = st.sidebar.number_input("Minutes Played", min_value=0, value=1800)
        club_influence = st.sidebar.number_input("Club Influence (1-5)", min_value=1.0, max_value=5.0, value=3.0)

        if st.sidebar.button("Predict"):
            with st.spinner("Predicting..."):
                input_data = pd.DataFrame({
                    "age": [age],
                    "market_value_in_eur_y": [market_value],
                    "performance": [performance],
                    "goals": [goals],
                    "assists": [assists],
                    "minutes_played": [minutes_played],
                    "club_influence": [club_influence]
                })
                prediction = model.predict(input_data)[0]
                original_prediction = np.expm1(prediction)
                insert_xgb_prediction(age, market_value, performance, goals, assists, minutes_played, club_influence,
                                      original_prediction)

                st.write("### Predicted Transfer Fee")
                st.success(f"Predicted Fee: €{original_prediction:,.2f}")

        if st.sidebar.checkbox("Show Past Predictions"):
            with st.spinner("Loading past predictions..."):
                st.write("### Past Predictions")
                st.write(get_past_predictions(model_type="xgb"))

elif page == "Profitability Predictor":
    st.title("Profitability Predictor")

    model, scaler = svr.load_model("svr_model.pkl", "scaler.pkl")
    if model and scaler:
        st.sidebar.header("Input Match Details")
        stadium_seats = st.sidebar.number_input("Stadium Seats", min_value=0, value=50000)
        hosting = st.sidebar.selectbox("Hosting", ["Home", "Away"])
        goal_difference = st.sidebar.number_input("Goal Difference", min_value=-10, max_value=10, value=0)
        is_win = st.sidebar.checkbox("Win", value=False)

        if st.sidebar.button("Predict"):
            with st.spinner("Predicting..."):
                hosting_encoded = 1 if hosting == "Home" else 0
                input_data = pd.DataFrame({
                    "stadium_seats": [stadium_seats],
                    "hosting": [hosting_encoded],
                    "goal_difference": [goal_difference],
                    "revenue": [stadium_seats * hosting_encoded],
                    "profitability": [stadium_seats * hosting_encoded + (is_win * 10000)]
                })
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)[0]

                insert_svr_prediction(stadium_seats, hosting, goal_difference, is_win, prediction)

                st.write("### Predicted Profitability")
                st.success(f"Predicted Profitability: €{prediction:,.2f}")

        if st.sidebar.checkbox("Show Past Predictions"):
            with st.spinner("Loading past predictions..."):
                st.write("### Past Predictions")
                st.write(get_past_predictions(model_type="svr"))

elif page == "Goal Scoring Predictor":
    st.title("Goal Scoring Predictor")

    model = gbm.load_model()
    if model:
        st.sidebar.header("Input Match Details")
        player_club_id = st.sidebar.number_input("Player Club ID", min_value=1, value=100)
        assists = st.sidebar.number_input("Assists", min_value=0, value=1)
        minutes_played = st.sidebar.number_input("Minutes Played", min_value=0, value=90)
        home_club_goals = st.sidebar.number_input("Home Club Goals", min_value=0, value=2)
        away_club_goals = st.sidebar.number_input("Away Club Goals", min_value=0, value=1)

        if st.sidebar.button("Predict"):
            with st.spinner("Predicting..."):
                input_data = pd.DataFrame({
                    "player_club_id": [player_club_id],
                    "assists": [assists],
                    "minutes_played": [minutes_played],
                    "home_club_goals": [home_club_goals],
                    "away_club_goals": [away_club_goals]
                })
                prediction, prediction_proba = gbm.make_prediction(model, input_data)

                insert_gbm_prediction(player_club_id, assists, minutes_played, home_club_goals, away_club_goals,
                                      prediction, max(prediction_proba))

                st.write("### Prediction Result")
                if prediction == 1:
                    st.success("The player is likely to score a goal.")
                else:
                    st.info("The player is unlikely to score a goal.")
                st.write(f"### Prediction Confidence: {max(prediction_proba) * 100:.2f}%")

        if st.sidebar.checkbox("Show Past Predictions"):
            with st.spinner("Loading past predictions..."):
                st.write("### Past Predictions")
                st.write(get_past_predictions(model_type="gbm"))

elif page == "Train Model":
    st.title("Train the Model")
    password = st.text_input("Enter Admin Password", type="password")

    if password == "admin_password":
        st.success("Admin Verified")

        model_to_train = st.selectbox("Select Model to Train", [
            "Player Undervaluation (Random Forest)",
            "Transfer Fee (XGBoost)",
            "Profitability (SVR)",
            "Goal Scoring (Gradient Boosting)"
        ])

        if st.button("Train Selected Model"):
            with st.spinner("Training the model..."):
                try:
                    csv_files = {
                        "players": "archive/players.csv",
                        "player_valuations": "archive/player_valuations.csv",
                        "transfers": "archive/transfers.csv",
                        "appearances": "archive/appearances.csv",
                        "club_games": "archive/club_games.csv",
                        "competitions": "archive/competitions.csv",
                        "games": "archive/games.csv",
                        "clubs": "archive/clubs.csv"
                    }

                    for table_name, csv_file in csv_files.items():
                        if not check_data_exists(table_name):
                            load_csv_into_db(csv_file, table_name)
                            st.write(f"Loaded data into `{table_name}` table.")

                    if model_to_train == "Player Undervaluation (Random Forest)":
                        data = get_data_from_db("players")
                        rf_model = rf.train_model(data)
                        joblib.dump(rf_model, "rf_model.pkl")
                        st.success("Random Forest model trained and saved.")

                    elif model_to_train == "Transfer Fee (XGBoost)":
                        transfers = get_data_from_db("transfers")
                        players = get_data_from_db("players")
                        appearances = get_data_from_db("appearances")
                        xgb_model = xgb.train_model(transfers, players, appearances)
                        joblib.dump(xgb_model, "xgb_model.pkl")
                        st.success("XGBoost model trained and saved.")

                    elif model_to_train == "Profitability (SVR)":
                        club_games = get_data_from_db("club_games")
                        competitions = get_data_from_db("competitions")
                        clubs = get_data_from_db("clubs")
                        merged_df = svr.load_and_merge_data(club_games, clubs, competitions)
                        merged_df = svr.feature_engineering(merged_df)
                        X_train, X_test, y_train, y_test = svr.prepare_data(
                            merged_df, ["stadium_seats", "hosting", "goal_difference", "revenue"], "profitability"
                        )
                        svr_model, scaler = svr.train_and_evaluate_model(X_train, y_train, X_test, y_test)
                        svr.save_model(svr_model, scaler, "svr_model.pkl", "scaler.pkl")
                        st.success("SVR model trained and saved.")

                    elif model_to_train == "Goal Scoring (Gradient Boosting)":
                        appearances = get_data_from_db("appearances")
                        games = get_data_from_db("games")
                        gbm_model, _ = gbm.train_model(appearances, games)
                        joblib.dump(gbm_model, "gbm_model.pkl")
                        st.success("Gradient Boosting model trained and saved.")

                except FileNotFoundError as e:
                    st.error(f"File not found: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Incorrect admin password. Access denied.")
