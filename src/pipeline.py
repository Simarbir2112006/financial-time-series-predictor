import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
import lightgbm as lgb
from src.config import SEED


class BaseFE(ABC, BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    @abstractmethod
    def transform(self, X): pass

class MemoryReducer(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if df[c].dtype == 'float64': df[c] = df[c].astype('float32')
            if df[c].dtype == 'int64': df[c] = df[c].astype('int32')

        return df

class Imputer(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.ffill().bfill()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())

        return df

class Winsorizer(BaseFE):
    def __init__(self, percentile=0.01) -> None: 
        self.p = percentile

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 120: return df 
        numeric = df.select_dtypes(include=[np.number]).columns

        for c in numeric:
            if c in ['date_id','forward_returns','is_scored','prediction']:
                continue

            series = df[c]
            lower, upper = series.quantile(self.p), series.quantile(1-self.p)
            df[c] = series.clip(lower, upper)

        return df

class LagFeat(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'forward_returns' not in df.columns:
            return df
        
        df = df.copy()
        df['lagged_forward_returns'] = df['forward_returns'].shift(1)
        df['target_mom_3d'] = df['forward_returns'].rolling(3).mean()
        df['target_mom_7d'] = df['forward_returns'].rolling(7).mean()
        df['target_vol_10d'] = df['forward_returns'].rolling(10).std()
        df['target_vol_20d'] = df['forward_returns'].rolling(20).std()

        return df

class GroupStatsGenerator(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        families = {'M': [], 'E': [], 'P': [], 'V': [], 'S': []}

        for c in df.columns:
            if len(c)>=2 and c[0] in families and c[1:].isdigit():
                families[c[0]].append(c)

        for fam, cols in families.items():
            if len(cols)==0:
                continue

            block = df[cols]
            df[f'{fam}_mean_index'] = block.mean(axis=1)
            df[f'{fam}_std_index']  = block.std(axis=1)
            df[f'{fam}_z_strength'] = (block.sub(block.mean(axis=1), axis=0)
                                            .div(block.std(axis=1)+1e-6, axis=0)
                                            .mean(axis=1))
            
        return df

class InteractionEngine(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = [('P_mean_index', 'M_mean_index'), ('V_mean_index', 'S_mean_index')]

        for a,b in pairs:
            if a in df.columns and b in df.columns:
                df[f'inter_{a}_x_{b}'] = df[a] * df[b]

        return df

class RollingStatsEngineer(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        M_cols = [c for c in df.columns if c.startswith('M')][:10]

        for c in M_cols:
            rm20 = df[c].rolling(20).mean()
            rs20 = df[c].rolling(20).std()
            df[f'{c}_rm5'] = df[c].rolling(5).mean()
            df[f'{c}_rm20'] = rm20
            df[f'{c}_z20'] = (df[c]-rm20)/(rs20+1e-6)

        return df

class Encoder(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date_id' not in df.columns: 
            return df
        
        df = df.copy()

        for period in [7, 21, 63]:
            df[f'sin_{period}'] = np.sin(2*np.pi*df['date_id']/period)
            df[f'cos_{period}'] = np.cos(2*np.pi*df['date_id']/period)

        return df

class VolatilityBinner(BaseFE):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'V_mean_index' not in df.columns:
            df['vol_regime']=1

            return df
        
        df = df.copy()

        if len(df)>50:
            try:
                df['vol_regime'] = pd.qcut(df['V_mean_index'], q=3, labels=False, duplicates='drop')
            except:
                df['vol_regime']=1

        else: 
            df['vol_regime']=1

        return df

class HullMasterPipeline:
    def __init__(self):
        self.steps = [
            Imputer(), 
            Winsorizer(), 
            LagFeat(),
            GroupStatsGenerator(), 
            InteractionEngine(), 
            RollingStatsEngineer(),
            Encoder(), 
            VolatilityBinner(), 
            MemoryReducer()
        ]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.transform(df)

        return df

def select_features(train_eng, top_k=60):
    drop_cols = ['date_id','forward_returns','risk_free_rate',
                 'market_forward_excess_returns','is_scored','prediction']
    
    feature_cols = [c for c in train_eng.columns if c not in drop_cols]
    X = train_eng[feature_cols].fillna(0)
    y = train_eng['forward_returns']

    # LGBM Importance
    temp_lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.01, max_depth=-1, random_state=SEED, n_jobs=-1)
    temp_lgb.fit(X, y)
    lgb_imp = temp_lgb.feature_importances_

    # Ridge Importance
    temp_ridge = Ridge(alpha=10)
    temp_ridge.fit(X, y)
    ridge_imp = np.abs(temp_ridge.coef_)

    # Correlation
    corr_list = []
    for c in feature_cols:
        try: corr_list.append(abs(np.corrcoef(train_eng[c].fillna(0), y)[0,1]))
        except: corr_list.append(0.0)
    corr_imp = np.array(corr_list)

    # Normalize & Blend
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    final_imp = 0.55 * norm(lgb_imp) + 0.25 * norm(ridge_imp) + 0.20 * norm(corr_imp)
    
    sorted_idx = np.argsort(final_imp)[::-1]
    selected = [feature_cols[i] for i in sorted_idx[:top_k]]
    return selected