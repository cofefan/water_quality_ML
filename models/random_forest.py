from sklearn.ensemble import RandomForestClassifier

class WaterQualityRF:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced', # Handles potential imbalance in your real data
            n_jobs=-1                # Use all CPU cores
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        return self.model.feature_importances_