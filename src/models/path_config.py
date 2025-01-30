# config.py
import os

# Define paths for data
DATA_DIR = 'data/output'
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.parquet')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.parquet')

# Define paths for models
OUTPUT_DIR = 'output/model'
DEFAULT_MODEL_DIR = os.path.join(OUTPUT_DIR, 'default') # this is for production
RETRAINED_MODEL_DIR = os.path.join(OUTPUT_DIR, 'retrained') # this is for testing and retraining

# Ensure directories exist
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
os.makedirs(RETRAINED_MODEL_DIR, exist_ok=True)
