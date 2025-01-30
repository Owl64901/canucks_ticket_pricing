from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
import src.models.path_config as path_config
import os

def main(output_dir): 
    # Load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH).values.ravel()

    y_train = y_train.reshape(-1, 1)

    # Define feature preprocessing
    numeric_feats = ['revenue_to_date', 's/t-rate', 'opens', 'holds', 'pminventory', 'prp_currentprice', 'prp_forwardtix', 'prp_forwardrev', 
                     'ticket_sold-total', 'ticket_sold-last_7days', 'ticket_sold-yesterday', 'host_sold-total', 'host_sold-last_7days', 
                     'host_sold-yesterday', 'resale_sold-2_days_ago', 'resale_sold-last_7days', 'resale_sold-total', 'resale_asp-2_days_ago', 
                     'resale_asp-last_7days', 'resale_asp-total', 'last_price', 'number_of_postings', 'median_posting_price', 'posting_below_cp', 
                     'lowest_post_price', 'highest_post_price', 'host_sold_at_current_price', 'days_until_game', 'opponent_rank', 'van_rank', 
                     'tickets_sold_2_days_before_today', 'tickets_sold_3_days_before_today', 'tickets_sold_4_days_before_today', 'tickets_sold_5_days_before_today', 
                     'tickets_sold_6_days_before_today', 'tickets_sold_7_days_before_today', 'weekend_game', 'days_until_game', 
                     'host_sold_agg_last_day', 'unique_views', 'tkt_qty']
    categorical_feats = ['month', 'opponent']
    passthrough_feats = ['weekend_game', 'bowl_location', 'price_code_ordinal', 'times_above_floor', 'inventory_normalized']

    # Preprocessing for numeric data: imputation + scaling
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: one-hot encoding
    categorical_transformer = OneHotEncoder()

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, categorical_feats),
            ('pass', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('identity', 'passthrough')
            ]), passthrough_feats)
    ])

    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # Define the TabNet model (assuming hyperparameters are tuned)
    tabnet_params = {
        'n_d': 16,
        'n_a': 16,
        'n_steps': 5,
        'gamma': 1.5,
        'n_independent': 2,
        'n_shared': 2,
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2),
        'scheduler_params': {"step_size":50, "gamma":0.9},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'mask_type': 'entmax',  # This can be "sparsemax", "entmax" or other
        'input_dim': X_train_preprocessed.shape[1],
        'output_dim': 1,
        'device_name': 'auto'
    }
    model = TabNetRegressor(**tabnet_params)

    # Fit the model
    model.fit(
        X_train=X_train_preprocessed,
        y_train=y_train,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # Predict and evaluate
    y_train_pred = model.predict(X_train_preprocessed)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {rmse_train}")

    # Save the model and preprocessor
    model.save_model(os.path.join(output_dir, 'tabnet_model.zip'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)
