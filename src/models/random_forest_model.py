from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import joblib
import pandas as pd
import numpy as np
from ..cross_validation import eval_model
import src.models.path_config as path_config
import os
import warnings
warnings.filterwarnings('ignore')

def main(output_dir): 
    # load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH)

    # Define the features, may consider moving to these lists to another script
    numeric_feats = ['revenue_to_date', 's/t-rate', 'opens', 'holds', 'pminventory', 'prp_currentprice', 'prp_forwardtix', 'prp_forwardrev', 
                     'ticket_sold-total', 'ticket_sold-last_7days', 'ticket_sold-yesterday', 'host_sold-total', 'host_sold-last_7days', 
                     'host_sold-yesterday', 'resale_sold-2_days_ago', 'resale_sold-last_7days', 'resale_sold-total', 'resale_asp-2_days_ago', 
                     'resale_asp-last_7days', 'resale_asp-total', 'last_price', 'number_of_postings', 'median_posting_price', 'posting_below_cp', 
                     'lowest_post_price', 'highest_post_price', 'host_sold_at_current_price', 'days_until_game', 'opponent_rank', 'van_rank', 
                     'tickets_sold_2_days_before_today', 'tickets_sold_3_days_before_today', 'tickets_sold_4_days_before_today', 'tickets_sold_5_days_before_today', 
                     'tickets_sold_6_days_before_today', 'tickets_sold_7_days_before_today', 'weekend_game', 'days_until_game', 'opponent_rank', 'van_rank', 
                     'host_sold_agg_last_day', 'unique_views', 'tkt_qty']
    categorical_feats = ['month', 'opponent']
    passthrough_feats = ['weekend_game', 'bowl_location', 'price_code_ordinal', 'times_above_floor', 'inventory_normalized']
    all_columns = set(X_train.columns)
    rest_columns = set(numeric_feats + categorical_feats + passthrough_feats)
    drop_feats = list(all_columns - rest_columns)

    # Define the numeric transformer with imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # imputation on numeric features
        ('scaler', StandardScaler()),  # scaling on numeric features
    ])

    # Define the column transformer
    ct = make_column_transformer(
        (numeric_transformer, numeric_feats),
        (OneHotEncoder(), categorical_feats),  # OHE on categorical features
        ("passthrough", passthrough_feats),
        ("drop", drop_feats),  # drop the drop features
    )

    # Use the best known hyperparameters
    best_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 4
    }

    # Create and fit the model
    model = Pipeline([
        ('preprocessor', ct),
        ('model', RandomForestRegressor(**best_params, random_state=42))
    ])

    model.fit(X_train, y_train.values.ravel())

    # Predict and evaluate
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {rmse_train}")

    eval_model(model, X_train, y_train)

    # Save the model
    joblib.dump(model, os.path.join(output_dir, 'random_forest_model.joblib'))

if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)
