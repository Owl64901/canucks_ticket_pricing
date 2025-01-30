# poisson_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from ..cross_validation import eval_model
from sklearn.metrics import mean_squared_error
import joblib
import src.models.path_config as path_config
import os

def main(output_dir):
    # Load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH)

    # Features used
    numeric_feats = ['s/t-rate', 'opens', 'host_sold-total', 'host_sold-last_7days', 
        'host_sold-yesterday', 'resale_sold-last_7days', 'resale_sold-total', 
        'resale_asp-last_7days', 'resale_asp-total', 'number_of_postings', 'median_posting_price',
        'posting_below_cp', 'lowest_post_price', 'highest_post_price', 'host_sold_at_current_price', 
        'days_until_game', 'opponent_rank', 'van_rank','host_sold_agg_last_day','unique_views', 
        'price_code_ordinal', 'tickets_sold_2_days_before_today', 'tickets_sold_3_days_before_today', 
        'tickets_sold_4_days_before_today', 'tickets_sold_5_days_before_today', 
        'tickets_sold_6_days_before_today', 'tickets_sold_7_days_before_today'
    ]

    categorical_feats = ['month', 'opponent']

    passthrough_feats = ['bowl_location', 'weekend_game', 'inventory_normalized', 'times_above_floor']
    
    all_columns = set(X_train.columns)
    rest_columns = set(numeric_feats + categorical_feats + passthrough_feats)
    drop_feats = list(all_columns - rest_columns)   

    # Define the model
    alpha = 1

    preprocessor = ColumnTransformer(
        transformers=[
            ('drop', 'drop', drop_feats), 
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), numeric_feats),
            ('boolean','passthrough', passthrough_feats),
            ('categorical', Pipeline([
                ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore'))
            ]), categorical_feats)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poisson_regressor', PoissonRegressor(alpha=alpha))
    ])

    # Fit the model
    model.fit(X_train, y_train.to_numpy().ravel())

    # Evaluate: Training Score
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {rmse_train}")

    # Evaluate: Cross-validation
    eval_model(model, X_train, y_train)

    # Feature coefficients
    cat_feature_names = model.named_steps['preprocessor'].transformers_[3][1].named_steps['onehot'].get_feature_names_out(categorical_feats)
    feature_names = np.concatenate((numeric_feats, passthrough_feats, cat_feature_names))
    coefficients = model.named_steps['poisson_regressor'].coef_
    df_coef = pd.DataFrame({'feature': feature_names, 'coefficients': coefficients})
    df_coef['magnitude'] = df_coef['coefficients'].abs()
    df_coef = df_coef.sort_values(by='magnitude', ascending=False)

    # Export the model & feature coefficients dataframe 
    joblib.dump(model, os.path.join(output_dir, 'poisson_model.joblib'))
    df_coef.to_csv(os.path.join(path_config.DATA_DIR, 'poisson_feature_importances.csv'), index=False)

if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)
