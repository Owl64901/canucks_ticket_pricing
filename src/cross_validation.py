# cross_validation.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def sliding_window_cv(X, y, train_days, validation_days):
    """
    Perform sliding window cross-validation on time series data.

    Parameters:
    X (pd.DataFrame): The features in the training set.
    y (pd.Series): The target variable in the training set.
    train_days (int): The number of days to use for training in each iteration.
    validation_days (int): The number of days to use for validation in each iteration.

    Yields:
    tuple: A tuple containing the training and validation sets for X and y in each iteration.
    """
    X['calculate_date'] = pd.to_datetime(X['calculate_date'])
    X = X.sort_values('calculate_date')
    y = y.loc[X.index]

    dates = X['calculate_date'].dt.date.unique()

    for i in range(train_days, len(dates), validation_days):
        train_dates = pd.to_datetime(dates[:i])
        validation_dates = pd.to_datetime(dates[i:i+validation_days])
        X_train = X[X['calculate_date'].isin(train_dates)]
        y_train = y.loc[X_train.index]
        X_validation = X[X['calculate_date'].isin(validation_dates)]
        y_validation = y.loc[X_validation.index]

        yield X_train, y_train, X_validation, y_validation

def eval_model(pipeline, X_train, y_train):
    """
    Evaluate a model using sliding window cross-validation and print the validation scores.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): The pipeline to evaluate.
    X_train (pd.DataFrame): The features in the training set.
    y_train (pd.Series): The target variable in the training set.
    """
    validation_scores = []
    cv_generator = sliding_window_cv(X_train, y_train, 270, 60)

    for X_train, y_train, X_validation, y_validation in cv_generator:
        pipeline.fit(X_train, y_train.to_numpy().ravel())
        predictions = pipeline.predict(X_validation)
        rmse = np.sqrt(mean_squared_error(y_validation, predictions))
        validation_scores.append(rmse)

        print(f'Training date range: {X_train.calculate_date.min()} - {X_train.calculate_date.max()}')
        print(f'Validation date range: {X_validation.calculate_date.min()} - {X_validation.calculate_date.max()}')
        print(f'Validation Root Mean Squared Error: {rmse}')

    mean_score = np.mean(validation_scores)
    std_dev_score = np.std(validation_scores)

    print(f'All Validation Scores: {validation_scores}')
    print(f'Mean Validation Score: {mean_score}')
    print(f'Standard Deviation of Validation Scores: {std_dev_score}')
