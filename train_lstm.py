import torch
import torch.nn as nn
import torch.optim as optim
from data.loader_lstm import load_lstm_data
from models.lstm_model import WaterQualityLSTM
from evaluate import print_evaluation

# Hyperparameters
WINDOW_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 50

def train():
    # 1. Load data
    X, y, _ = load_lstm_data('data/AKH_WQI.csv', window_size=WINDOW_SIZE)
    
    # Split into train/test (Simple split for time-series)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Initialize Model
    model = WaterQualityLSTM(input_size=10, hidden_size=HIDDEN_SIZE, 
                             num_layers=NUM_LAYERS, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("--- Training LSTM ---")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        print_evaluation(y_test.numpy(), predicted.numpy())

if __name__ == "__main__":
    train()