import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import plotly.express as px

# Step 1: Download the SQLite database from GitHub
db_url = "https://raw.githubusercontent.com/chiragpalan/time_series_prediction_v1/main/joined_data.db"
db_file = "joined_data.db"
os.makedirs("models", exist_ok=True)

# Download the database
import requests
response = requests.get(db_url)
with open(db_file, 'wb') as f:
    f.write(response.content)

# Step 2: Connect to the database and get all table names
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = [row[0] for row in cursor.fetchall()]

def preprocess_data(df):
    """Cleans the data and prepares it for training."""
    df = df.dropna()  # Drop rows with NaN values
    if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df.set_index(df.columns[0], inplace=True)
    df.sort_index(inplace=True)
    return df

def train_random_forest(X, y):
    """Trains a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Step 3: Iterate over each table to train the Random Forest and predict
for table in table_names:
    print(f"Processing table: {table}")
    try:
        # Load and preprocess the data
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        df = preprocess_data(df)

        # Prepare features and target
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # Last column as target

        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = train_random_forest(X_train, y_train)

        # Save the model to disk
        model_path = f"models/{table}_rf_model.pkl"
        joblib.dump(model, model_path)

        # Make predictions on the entire dataset
        y_pred = model.predict(X)

        # Save predictions to CSV
        results = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        }, index=df.index)
        results.to_csv(f"{table}_predictions.csv")

        # Create and save interactive chart as HTML
        fig = px.line(results, title=f"Actual vs Predicted for {table}",
                      labels={'index': 'Date'}, template='plotly_white')
        fig.update_traces(mode='lines+markers')
        fig.write_html(f"{table}_predictions.html")  # Save chart as HTML

        print(f"Completed processing for table: {table}")

    except Exception as e:
        print(f"Error processing table '{table}': {str(e)}")

# Close the database connection
conn.close()
print("Training and prediction process completed.")
