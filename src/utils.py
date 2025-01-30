# utils.py
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from src.models.model import TicketSalesTransformer

# Configuration for paths and other parameters
DATA_PATH = 'data/output/X_train.parquet'
TRANSFORMER_MODEL_BOWL_0_PATH = 'output/model/default/model_bowl_0.pth'
TRANSFORMER_MODEL_BOWL_1_PATH = 'output/model/default/model_bowl_1.pth'
POISSON_MODEL_PATH = 'output/model/default/poisson_model.joblib'
OUTPUT_FILENAME_TEMPLATE = 'output/prediction/predicted_tickets_sales_{date}.csv'
PREDICTION_INPUT_PATH = 'data/prediction_input/*.csv'
PREPROCESSOR_PATH = 'output/preprocessor.pkl'

# Define feature lists
numeric_feats = [
    'revenue_to_date', 's/t-rate', 'opens', 'holds', 'pminventory',
    'ticket_sold-total', 'ticket_sold-last_7days', 'ticket_sold-yesterday', 'host_sold-total', 'host_sold-last_7days',
    'host_sold-yesterday', 'resale_sold-2_days_ago', 'resale_sold-last_7days', 'resale_sold-total', 'resale_asp-2_days_ago',
    'resale_asp-last_7days', 'resale_asp-total', 'number_of_postings', 'median_posting_price', 'posting_below_cp',
    'lowest_post_price', 'highest_post_price', 'host_sold_at_current_price',
    'unique_views', 'tkt_qty'
]

categorical_feats = ['month', 'opponent']

passthrough_feats = [
    'weekend_game', 'days_until_game', 'opponent_rank', 'van_rank', 'bowl_location', 
    'price_code_ordinal', 'host_sold_agg_last_day', 'times_above_floor', 'inventory_normalized'
]

# Transformer model parameters
input_dim = 72  # Set according to your data
embed_dim = 64
num_heads = 8
num_layers = 3
output_dim = 1
max_length = 30
sequence_length = 3

# Transformer Training parameters
epochs = 20
lr = 0.00001

def load_data(file_pattern=PREDICTION_INPUT_PATH):
    input_files = glob.glob(file_pattern)
    all_data = pd.concat([pd.read_csv(file) for file in input_files])
    return all_data

def preprocess_train_data(X_train, preprocessor_path='output/preprocessor.pkl'):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats),
            ('pass', 'passthrough', passthrough_feats)
        ],
        remainder='drop'
    )
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, preprocessor_path)
    return preprocessor

def load_transformer_model(model_path, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length):
    model = TicketSalesTransformer(input_dim, embed_dim, num_heads, num_layers, output_dim, max_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_poisson_model(model_path):
    return joblib.load(model_path)

def pad_or_truncate_sequence(seq, max_length, padding_value=0):
    if seq.shape[0] > max_length:
        return seq[-max_length:]
    else:
        return np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), mode='constant', constant_values=padding_value)

def create_sequences(data, opponent_col, date_col, price_code_ordinal_col, sequence_length, target_column, max_length):
    sequences = []
    targets = []
    grouped = data.groupby(opponent_col)
    for opponent, group in grouped:
        group = group.sort_values(by=[date_col, price_code_ordinal_col])
        for idx in range(len(group)):
            end_idx = idx + 1
            start_idx = max(0, end_idx - sequence_length)
            seq_data = group.iloc[start_idx:end_idx].drop(columns=[date_col, target_column, opponent_col, price_code_ordinal_col])
            target_data = group.iloc[idx][target_column]
            padded_seq = pad_or_truncate_sequence(seq_data.values, max_length)
            sequences.append(padded_seq)
            targets.append(target_data)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def get_earliest_date(file_path):
    data = pd.read_parquet(file_path)
    earliest_date = data['calculate_date'].min()
    return earliest_date

# Update START_DATE dynamically
START_DATE = get_earliest_date(DATA_PATH)