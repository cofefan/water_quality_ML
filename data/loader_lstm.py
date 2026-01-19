import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def create_sequences(data, target, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
        labels.append(target[i + window_size])
    return torch.tensor(sequences).float(), torch.tensor(labels).long()

def load_lstm_data(filepath, window_size=5):
    df = pd.read_csv(filepath)
    feature_cols = ['PH', 'Temp', 'Turbidity', 'TSS', 'BOD5', 'COD', 'DO', 'Amoni', 'Phosphat', 'Coliforms']
    
    # Scale data (Mandatory for LSTMs)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Convert WQI to labels
    def classify(wqi):
        if wqi <= 25: return 0
        elif wqi <= 50: return 1
        else: return 2
    
    targets = df['WQI'].apply(classify).values
    
    X, y = create_sequences(scaled_data, targets, window_size)
    return X, y, feature_cols