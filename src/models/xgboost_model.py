from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import pandas as pd
import numpy as np
from ..cross_validation import eval_model
from xgboost import XGBRegressor
import src.models.path_config as path_config
import os
import warnings
warnings.filterwarnings('ignore')

def main(output_dir): 
    # load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH)

    # Define the features, may consider moving to these lists to another script
    numeric_feats = ['revenue_to_date', 's/t-rate', 'opens', 'holds', 
                'prp_currentprice', 'prp_forwardtix', 'prp_forwardrev', 'ticket_sold-total', 
                'ticket_sold-last_7days', 'ticket_sold-yesterday', 'host_sold-total', 'host_sold-last_7days', 'host_sold-yesterday', 
                'resale_sold-2_days_ago', 'resale_sold-last_7days', 'resale_sold-total', 'resale_asp-2_days_ago', 'resale_asp-last_7days',
                'resale_asp-total', 'last_price', 'number_of_postings', 'median_posting_price', 'posting_below_cp', 'lowest_post_price',
                'highest_post_price', 'host_sold_at_current_price', 'days_until_game', 'opponent_rank', 'van_rank',
                'tickets_sold_2_days_before_today', 'tickets_sold_3_days_before_today', 'tickets_sold_4_days_before_today', 'tickets_sold_5_days_before_today', 
                'tickets_sold_6_days_before_today', 'tickets_sold_7_days_before_today', 'weekend_game', 'days_until_game',
                'host_sold_agg_last_day', 'unique_views', 'tkt_qty']
    categorical_feats = ['price_code', 'month']
    passthrough_feats = ['weekend_game', 'bowl_location']
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
    
    # Create a pipeline with the column transformer and Random Forest regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', ct),
        ('model', XGBRegressor(n_estimators=1500, max_depth=3, eta=0.025,
                               colsample_bytree=0.15, random_state=123))
    ])

    # Evaluate the model using custom cross-validation
    eval_model(pipeline, X_train, y_train)

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train.values.ravel())

    # Predict on the training data
    y_train_pred = pipeline.predict(X_train)

    # Calculate the RMSE on training data
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {rmse_train}")

    # Export the trained model
    joblib.dump(pipeline, os.path.join(output_dir, 'xgboost_model.joblib'))


if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)