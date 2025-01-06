import sqlite3
import pandas as pd

# Function to create the database and tables
def create_db():
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()

    # Table for Random Forest predictions
    c.execute('''CREATE TABLE IF NOT EXISTS rf_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_value REAL,
                    highest_market_value REAL,
                    position TEXT,
                    age INTEGER,
                    height REAL,
                    prediction INTEGER,
                    prediction_confidence REAL)''')

    # Table for XGBoost predictions
    c.execute('''CREATE TABLE IF NOT EXISTS xgb_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    age REAL,
                    market_value REAL,
                    performance REAL,
                    goals INTEGER,
                    assists INTEGER,
                    minutes_played INTEGER,
                    club_influence REAL,
                    transfer_fee_prediction REAL)''')

    # Table for SVR predictions
    c.execute('''CREATE TABLE IF NOT EXISTS svr_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stadium_seats REAL,
                    hosting TEXT,
                    goal_difference REAL,
                    is_win INTEGER,
                    predicted_profitability REAL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS gbm_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_club_id INTEGER,
                    assists INTEGER,
                    minutes_played INTEGER,
                    home_club_goals INTEGER,
                    away_club_goals INTEGER,
                    prediction INTEGER,
                    prediction_confidence REAL)''')

    conn.commit()
    conn.close()

# Function to insert Random Forest prediction
def insert_rf_prediction(market_value, highest_market_value, position, age, height, prediction, prediction_confidence):
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()
    c.execute('''INSERT INTO rf_predictions (market_value, highest_market_value, position, age, height, prediction, prediction_confidence)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (market_value, highest_market_value, position, age, height, prediction, prediction_confidence))
    conn.commit()
    conn.close()

# Function to insert XGBoost prediction
def insert_xgb_prediction(age, market_value, performance, goals, assists, minutes_played, club_influence, transfer_fee_prediction):
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()
    c.execute('''INSERT INTO xgb_predictions (age, market_value, performance, goals, assists, minutes_played, club_influence, transfer_fee_prediction)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (age, market_value, performance, goals, assists, minutes_played, club_influence, transfer_fee_prediction))
    conn.commit()
    conn.close()

# Function to retrieve past predictions
def get_past_predictions(model_type="rf"):
    conn = sqlite3.connect("player_predictions.db")

    # Determine the table name based on the model type
    if model_type == "rf":
        table_name = "rf_predictions"
    elif model_type == "xgb":
        table_name = "xgb_predictions"
    elif model_type == "svr":
        table_name = "svr_predictions"
    elif model_type == "gbm":
        table_name = "gbm_predictions"
    else:
        raise ValueError("Invalid model type. Choose 'rf', 'xgb','svr' or 'gbm'.")

    # Retrieve the last 10 predictions from the corresponding table
    df_predictions = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    return df_predictions



# Function to insert SVR prediction
def insert_svr_prediction(stadium_seats, hosting, goal_difference, is_win, predicted_profitability):
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()
    c.execute('''INSERT INTO svr_predictions (stadium_seats, hosting, goal_difference, is_win, predicted_profitability)
                 VALUES (?, ?, ?, ?, ?)''',
              (stadium_seats, hosting, goal_difference, is_win, predicted_profitability))
    conn.commit()
    conn.close()

# Function to insert GBM prediction
def insert_gbm_prediction(player_club_id, assists, minutes_played, home_club_goals, away_club_goals, prediction, prediction_confidence):
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()
    c.execute('''INSERT INTO gbm_predictions (player_club_id, assists, minutes_played, home_club_goals, away_club_goals, prediction, prediction_confidence)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (player_club_id, assists, minutes_played, home_club_goals, away_club_goals, prediction, prediction_confidence))
    conn.commit()
    conn.close()

# Check if data exists in a table
def check_data_exists(table_name):
    conn = sqlite3.connect("player_predictions.db")
    c = conn.cursor()
    c.execute(f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    table_exists = c.fetchone()[0]
    conn.close()
    return table_exists > 0

# Load CSV into database
def load_csv_into_db(csv_file, table_name):
    conn = sqlite3.connect("player_predictions.db")
    data = pd.read_csv(csv_file)
    data.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

# Fetch data from the database
def get_data_from_db(table_name):
    conn = sqlite3.connect("player_predictions.db")
    data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return data
