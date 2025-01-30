import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ..cross_validation import eval_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import joblib
import src.models.path_config as path_config
import os

def main(output_dir): 
    
    # Model: Degree 2 polynomial transformation with Ridge and alpha = 1559.945

    # load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH)

    # Features used 
    numeric_feats = ['cap', 's/t-rate', 'opens', 'prp_forwardtix', 'ticket_sold-total', 'ticket_sold-last_7days', 
    'ticket_sold-yesterday', 'host_sold-total', 'host_sold-last_7days', 'host_sold-yesterday', 'resale_sold-last_7days', 
    'resale_sold-total', 'resale_asp-last_7days', 'resale_asp-total', 
    'initial_price', 'last_price', 'number_of_postings', 'median_posting_price', 
    'posting_below_cp', 'lowest_post_price', 'highest_post_price', 'host_sold_at_current_price', 
    'days_until_game', 'opponent_rank', 'van_rank', 
    'host_sold_agg_last_day', 'unique_views']

    categorical_feats = ['month', 'opponent', 'price_code']
    binary_feats = ['weekend_game']

    all_columns = set(X_train.columns)
    rest_columns = set(numeric_feats + categorical_feats + binary_feats)
    drop_feats = list(all_columns - rest_columns)

    # DEFINE THE MODEL 
    # hyperparameters
    alpha_val = 1559.945
    deg_val = 2

    # model structure
    preprocessor = ColumnTransformer(
        transformers=[
            ('drop', 'drop', drop_feats), 
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('poly', PolynomialFeatures(degree=deg_val)),
                ('scaler', StandardScaler())
            ]), numeric_feats),
            ('boolean','passthrough', binary_feats),
            ('categorical', Pipeline([
                ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore'))
            ]), categorical_feats)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha_val)) 
    ])

    # Fit the model on the training data 
    pipeline.fit(X_train, y_train)

    # Evaluate: Training Score 
    y_train_pred = pipeline.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)

    print(f"Training RMSE: {rmse_train}")
    print(f"Training R^2: {r2_train}")

    # Evalulate: Cross-validation 
    eval_model(pipeline, X_train, y_train)

    # Export the model
    joblib.dump(pipeline, os.path.join(output_dir, 'polynomial_model.joblib'))

if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)

