import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import our new XGBoost module
from data.loader import load_data
from models.xgboost_model import WaterQualityXGB
from evaluate import print_evaluation, plot_confusion_matrix, plot_feature_importance

def main():
    # 1. Load Data
    print("--- 1. Loading AKH_WQI Data (XGBoost Pipeline) ---")
    try:
        X, y, feature_names = load_data('data/AKH_WQI.csv')
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    accuracies = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []
    feature_importance_sum = np.zeros(len(feature_names))

    print("\n--- 2. Starting Stratified K-Fold Training ---")
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and Train XGBoost
        xgb_model = WaterQualityXGB(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=5
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = xgb_model.predict(X_test_scaled)
        
        # Evaluate Fold
        acc, f1 = print_evaluation(y_test, y_pred, fold_index=i+1)
        accuracies.append(acc)
        f1_scores.append(f1)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        feature_importance_sum += xgb_model.get_feature_importance()

    # 3. Final Results
    print("\n--- 3. Final XGBoost Evaluation ---")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Macro F1: {np.mean(f1_scores):.4f}")
    
    # 4. Visualizations
    avg_feature_importance = feature_importance_sum / 5
    plot_feature_importance(avg_feature_importance, feature_names)
    plot_confusion_matrix(all_y_true, all_y_pred)

if __name__ == "__main__":
    main()