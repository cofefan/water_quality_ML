import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Import local modules
from data.loader import load_data
from data.loader_lstm import load_lstm_data

from models.random_forest import WaterQualityRF
from models.xgboost_model import WaterQualityXGB
from models.lstm_model import WaterQualityLSTM

def main():
    print("--- Starting High-Iteration Model Comparison ---")
    results = []

    # 1. Prepare Data
    X, y, _ = load_data('data/AKH_WQI.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- MODEL 1: Random Forest (Increased Complexity) ---
    # We increase n_estimators so the "forest" has more trees to vote.
    print("Training Random Forest (500 Trees)...")
    rf = WaterQualityRF(n_estimators=500) 
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, rf_pred),
        'F1-Score': f1_score(y_test, rf_pred, average='macro')
    })

    # --- MODEL 2: XGBoost (Increased Iterations & Lower Learning Rate) ---
    # By lowering learning_rate and increasing n_estimators, the model 
    # takes smaller, more precise steps toward the minimum error.
    print("Training XGBoost (1000 Iterations)...")
    xgb = WaterQualityXGB(n_estimators=1000, learning_rate=0.01) 
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    results.append({
        'Model': 'XGBoost',
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred, average='macro')
    })

    # --- MODEL 3: LSTM (Increased Epochs) ---
    # We increase EPOCHS from 50 to 200 to give the Neural Network more time to 
    # adjust its internal weights via backpropagation.
    print("Training LSTM (200 Epochs)...")
    WINDOW_SIZE = 5
    X_l, y_l, _ = load_lstm_data('data/AKH_WQI.csv', window_size=WINDOW_SIZE)
    split = int(0.8 * len(X_l))
    X_train_l, X_test_l = X_l[:split], X_l[split:]
    y_train_l, y_test_l = y_l[:split], y_l[split:]

    model_l = WaterQualityLSTM(input_size=10, hidden_size=64, num_layers=2, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_l.parameters(), lr=0.001)

    # The training loop: Each epoch is one full pass through the dataset
    for epoch in range(200): 
        model_l.train()
        optimizer.zero_grad()
        out = model_l(X_train_l)
        loss = criterion(out, y_train_l)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"   LSTM Epoch [{epoch+1}/200] complete...")

    model_l.eval()
    with torch.no_grad():
        test_out = model_l(X_test_l)
        _, predicted_l = torch.max(test_out.data, 1)
        results.append({
            'Model': 'LSTM',
            'Accuracy': accuracy_score(y_test_l.numpy(), predicted_l.numpy()),
            'F1-Score': f1_score(y_test_l.numpy(), predicted_l.numpy(), average='macro')
        })

    # 2. Final Plotting
    res_df = pd.DataFrame(results)
    print("\n--- FINAL COMPARISON TABLE ---")
    print(res_df)

    res_df.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar', figsize=(10, 6), color=['#3498db', '#e74c3c'])
    plt.title("Performance Comparison (High-Iteration Training)")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()