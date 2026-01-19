import pandas as pd
import numpy as np
import os

def load_data(filepath='data/AKH_WQI.csv'):
    """
    Loads the AKH_WQI dataset, separates features from targets,
    and converts WQI values into classification labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find the file: {filepath}. Please ensure it is in the data/ folder.")

    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Define your specific input features (X)
    # Note: 'WQI' is excluded here because it is the target
    feature_cols = [
        'PH', 'Temp', 'Turbidity', 'TSS', 'BOD5', 
        'COD', 'DO', 'Amoni', 'Phosphat', 'Coliforms'
    ]
    
    # Check if all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from the CSV: {missing_cols}")

    # Extract Features (X)
    X = df[feature_cols].values
    
    # Generate Target Labels (y) from WQI column
    # Logic: 0: Excellent (<=25), 1: Good (25-50), 2: Bad (>50)
    def classify_wqi(wqi_value):
        if wqi_value <= 25:
            return 0  # Excellent
        elif wqi_value <= 50:
            return 1  # Good
        else:
            return 2  # Bad

    y = df['WQI'].apply(classify_wqi).values

    print(f"Successfully loaded {len(df)} samples.")
    return X, y, feature_cols