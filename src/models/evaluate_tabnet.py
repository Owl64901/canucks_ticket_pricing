import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor

def sliding_window_cv(X, y, train_days, validation_days):
    """
    Ensure y is properly accessed as a Series.
    """
    X['calculate_date'] = pd.to_datetime(X['calculate_date'])
    X = X.sort_values('calculate_date')
    y = y.reindex(X.index)

    dates = X['calculate_date'].dt.date.unique()

    for i in range(train_days, len(dates), validation_days):
        train_dates = pd.to_datetime(dates[:i])
        validation_dates = pd.to_datetime(dates[i:i+validation_days])
        X_train = X[X['calculate_date'].isin(train_dates)]
        y_train = y.loc[X_train.index]
        X_validation = X[X['calculate_date'].isin(validation_dates)]
        y_validation = y.loc[X_validation.index]

        yield X_train, y_train, X_validation, y_validation

def eval_model(model, preprocessor, X_train, y_train):
    """
    Adjusted evaluation to handle preprocessing and TabNet model.
    """
    validation_scores = []
    cv_generator = sliding_window_cv(X_train, y_train, 270, 60)

    for X_train_cv, y_train_cv, X_validation_cv, y_validation_cv in cv_generator:
        # Preprocess the data
        X_train_preprocessed = preprocessor.transform(X_train_cv)
        X_validation_preprocessed = preprocessor.transform(X_validation_cv)

        # Fit and predict
        model.fit(X_train_preprocessed, y_train_cv.values)
        predictions = model.predict(X_validation_preprocessed)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_validation_cv.values, predictions))
        validation_scores.append(rmse)

        print(f'Training date range: {X_train_cv.calculate_date.min()} - {X_train_cv.calculate_date.max()}')
        print(f'Validation date range: {X_validation_cv.calculate_date.min()} - {X_validation_cv.calculate_date.max()}')
        print(f'Validation Root Mean Squared Error: {rmse}')

    mean_score = np.mean(validation_scores)
    std_dev_score = np.std(validation_scores)

    print(f'All Validation Scores: {validation_scores}')
    print(f'Mean Validation Score: {mean_score}')
    print(f'Standard Deviation of Validation Scores: {std_dev_score}')

def main():
    # Load the data
    X_train = pd.read_parquet('data/output/X_train.parquet')
    y_train = pd.read_parquet('data/output/y_train.parquet')  # Ensure this is a Series

    # Load the pre-trained TabNet model
    model = TabNetRegressor()
    model.load_model('output/model/tabnet_model.zip.zip')

    # Load the preprocessor
    preprocessor = joblib.load('output/model/preprocessor.pkl')

    # Evaluate the model
    eval_model(model, preprocessor, X_train, y_train)

if __name__ == "__main__":
    main()




