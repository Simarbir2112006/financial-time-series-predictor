import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

# Ensure artifacts dir exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Model & Training Config
SEED = 42
TARGET_VOL_ANNUAL = 0.15 
TARGET_VOL_DAILY = TARGET_VOL_ANNUAL / (252 ** 0.5)

WINNING_WEIGHTS = {'tree': 0.9, 'linear': 0.1}

WINNING_LGBM_PARAMS = {
    'n_estimators': 1000, 'learning_rate': 0.005, 'max_depth': 6,
    'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'verbosity': -1, 'random_state': SEED,
    'n_jobs': -1
}
WINNING_RIDGE_ALPHA = 10.0