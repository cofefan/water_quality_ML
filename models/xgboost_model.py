from xgboost import XGBClassifier
import numpy as np

class WaterQualityXGB:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        """
        XGBoost Wrapper for Water Quality Classification.
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective='multi:softprob', # Multi-class classification
            num_class=3,                # 0, 1, 2
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        return self.model.feature_importances_