import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define file paths
DATA_FILE = 'data.csv'
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'rf_model')

def train_model():
    """
    Loads data, trains a RandomForestClassifier model, and saves it.
    """
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file {DATA_FILE} not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Preprocessing data and engineering features...")
    # Convert timestamp to datetime objects
    # Specify dayfirst=True for formats like '28-02-2018 21:00' and '1/3/2018 12:00'
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

    # Drop rows where timestamp conversion failed
    df.dropna(subset=['timestamp'], inplace=True)

    # Feature Engineering: Extract datetime components
    # These features are based on the ones used in app.py's prediction logic
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['weekofyear'] = df['timestamp'].dt.isocalendar().week.astype(int) # Matches .dt.weekofyear behavior in newer pandas

    # Define features (X) and target (y)
    # Based on final.iloc[:,[1,2,3,4,6,10,11]] in app.py which corresponds to:
    # month, day, hour, dayofyear, weekofyear, latitude, longitude
    feature_cols = ['month', 'day', 'hour', 'dayofyear', 'weekofyear', 'latitude', 'longitude']
    
    # Check if all feature columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"Error: Missing feature columns in the data: {missing_features}")
        return

    X = df[feature_cols]

    target_cols = ['act379', 'act13', 'act279', 'act323', 'act363', 'act302']
    # Check if all target columns exist
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        print(f"Error: Missing target columns in the data: {missing_targets}")
        return
        
    y = df[target_cols]

    print(f"Training RandomForestClassifier model...")
    # Initialize and train the model
    # Using default parameters for RandomForestClassifier as in the original app (not specified)
    # n_estimators=10 was the default in older scikit-learn versions like 0.19.1.
    # Newer versions default to 100. Let's use 100 for better performance,
    # or you can pin it to 10 if you want to be closer to a potential old default.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    print("Model training complete.")

    # Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    print(f"Saving model to {MODEL_FILE}...")
    try:
        joblib.dump(model, MODEL_FILE)
        print(f"Model successfully saved to {MODEL_FILE}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    train_model()
