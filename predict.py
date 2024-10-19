import os
import pandas as pd
import sqlite3
import requests
import pickle
import plotly.express as px

# Step 1: Download the database file from GitHub
db_url = "https://raw.githubusercontent.com/chiragpalan/time_series_prediction_v1/main/joined_data.db"
db_file = "joined_data.db"
response = requests.get(db_url)
with open(db_file, 'wb') as f:
    f.write(response.content)

# Step 2: Download models from the GitHub repo
models_url = "https://github.com/chiragpalan/model_pipeline_v1/raw/main/"
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

# Step 3: Connect to the database and get all table names
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = [row[0] for row in cursor.fetchall()]

# Function to download a model by name
def download_model(model_name):
    model_url = f"{models_url}{model_name}.pkl"
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    return model_path

# Preprocessing function to clean data
def preprocess_data(df):
    df = df.dropna()  # Drop rows with null values
    if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df.set_index(df.columns[0], inplace=True)
    df.sort_index(inplace=True)
    return df

# Function to create and save interactive chart as HTML
def plot_predictions(actual, predicted, index, title, output_file):
    results = pd.DataFrame({'Actual': actual, 'Predicted': predicted}, index=index)
    fig = px.line(results, title=title, labels={'index': 'Date'}, template='plotly_white')
    fig.update_traces(mode='lines+markers')
    fig.write_html(output_file)  # Save chart as HTML

# Iterate through tables, predict, and generate charts
for table in table_names:
    print(f"Processing table: {table}")
    try:
        # Download the corresponding model
        model_path = download_model(table)

        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the data from the table
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        df = preprocess_data(df)

        # Prepare features and target
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # Last column as target

        # Make predictions
        y_pred = model.predict(X)

        # Save the chart as HTML
        html_file = f"{table}_predictions.html"
        plot_predictions(y, y_pred, df.index, f"Actual vs Predicted for {table}", html_file)

        # Save predictions to CSV (optional)
        prediction_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=df.index)
        prediction_df.to_csv(f"{table}_predictions.csv")

    except Exception as e:
        print(f"Error processing table '{table}': {str(e)}")

# Close the database connection
conn.close()
print("Prediction process completed.")
