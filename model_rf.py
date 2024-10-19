import pandas as pd
import plotly.express as px
import sqlite3
import requests

# Step 1: Download the database file
url = 'https://raw.githubusercontent.com/chiragpalan/time_series_prediction_v1/main/joined_data.db'
response = requests.get(url)

# Step 2: Save the database to a local file
with open('joined_data.db', 'wb') as f:
    f.write(response.content)

# Step 1: Connect to the database and fetch all table names
# db_path = './abc/database.db'  # Adjust to your database path
conn = sqlite3.connect("joined_data.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = [row[0] for row in cursor.fetchall()]

# Function to preprocess data
def preprocess_data(df):
    # Convert first column to datetime
    if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    
    # Set the first column as the index
    df.set_index(df.columns[0], inplace=True)

    # Convert non-target columns to numeric or factorize them
    for col in df.columns[:-1]:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in non-target columns
    df = df.dropna(subset=df.columns[:-1])

    # Ensure the target column has no NaNs
    if df[df.columns[-1]].isna().any():
        print(f"Warning: Dropping rows with NaN in the target column for {df.columns[-1]}")
        df = df.dropna(subset=[df.columns[-1]])

    return df

# Function to plot actual vs predicted values with actual dates on x-axis
def plot_predictions(actual, predicted, index, title):
    # Create a DataFrame for plotting
    results = pd.DataFrame({'Actual': actual, 'Predicted': predicted}, index=index)
    
    # Create an interactive plot with actual dates on x-axis
    fig = px.line(
        results,
        title=title,
        labels={'index': 'Date'},
        template='plotly_white'
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Date', yaxis_title='Values')
    fig.show()

# Process each table and make predictions for the entire dataset
for table in table_names:
    print(f"Predicting entire dataset for table: {table}")

    # Load data from the table
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)

    # Preprocess data to handle non-numeric columns and set index
    df_clean = preprocess_data(df)

    # Sort by index (assuming the index is datetime)
    df_clean = df_clean.sort_index()

    # Prepare features and target for the entire dataset
    X_full = df_clean.iloc[:, :-1].values  # All columns except the last one
    y_full = df_clean.iloc[:, -1].values   # Target column (last column)

    # Predict using the trained model
    model = models[table]
    y_pred_full = model.predict(X_full)

    # Plot actual vs predicted values with actual dates on the x-axis
    plot_predictions(y_full, y_pred_full, df_clean.index, f'Actual vs Predicted for {table}')

    # Save predictions to a CSV (optional)
    prediction_full_df = pd.DataFrame({'Actual': y_full, 'Predicted': y_pred_full}, index=df_clean.index)
    prediction_full_df.to_csv(f'{table}_full_data_predictions.csv')

print("Predictions for entire datasets completed and plots displayed.")
