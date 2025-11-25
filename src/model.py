import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class HullBiHybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, weights, lgbm_params, ridge_alpha):
        self.weights = weights
        self.lgbm_params = lgbm_params
        self.ridge_alpha = ridge_alpha
        self.tree = None
        self.linear = None
        self.scaler_linear = StandardScaler()

    def fit(self, X, y):
        # Tree
        self.tree = lgb.LGBMRegressor(**self.lgbm_params)
        self.tree.fit(X, y)
        
        # Linear
        self.linear = Ridge(alpha=self.ridge_alpha)
        self.linear.fit(self.scaler_linear.fit_transform(X.fillna(0)), y)
        
        return self

    def predict(self, X):
        p_tree = self.tree.predict(X)
        p_linear = self.linear.predict(self.scaler_linear.transform(X.fillna(0)))
        
        w_tree = self.weights.get('tree', 0.9)
        w_lin = self.weights.get('linear', 0.1)

        return (w_tree * p_tree + w_lin * p_linear)