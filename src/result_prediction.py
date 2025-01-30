import pandas as pd
import numpy as np
import torch
import joblib
from src.utils import (
    load_data,
    preprocess_train_data,
    create_sequences,
    load_transformer_model,
    DATA_PATH,
    TRANSFORMER_MODEL_BOWL_0_PATH,
    TRANSFORMER_MODEL_BOWL_1_PATH,
    POISSON_MODEL_PATH,
    OUTPUT_FILENAME_TEMPLATE,
    START_DATE,
    input_dim,
    embed_dim,
    num_heads,
    num_layers,
    output_dim,
    max_length,
    sequence_length
)
from src.preprocess_feature_engineering import preprocess_data, feature_engineer_data

device = torch.device('cpu')

def preprocess_data_for_model(preprocessor, filtered_X_test):
    """
    Preprocesses test data for input to the Transformer model.

    Args:
    - preprocessor (object): Preprocessor object fitted on training data.
    - filtered_X_test (DataFrame): Test data containing features for prediction.

    Returns:
    DataFrame: Preprocessed test data with transformed features and additional columns.
    """
    filtered_X_test['date'] = pd.to_datetime(filtered_X_test['calculate_date'])
    filtered_X_test = filtered_X_test.sort_values(by=['date', 'price_code'])
    filtered_X_test['original_opponent'] = filtered_X_test['opponent'].copy()
    filtered_X_test['price_code_ordinal'] = filtered_X_test['price_code_ordinal'].copy()

    transformed_filtered_X_test = preprocessor.transform(filtered_X_test)
    transformed_filtered_X_test_df = pd.DataFrame(transformed_filtered_X_test)
    transformed_filtered_X_test_df['date'] = filtered_X_test['date'].values
    transformed_filtered_X_test_df['target'] = np.nan
    transformed_filtered_X_test_df['original_opponent'] = filtered_X_test['original_opponent'].values
    transformed_filtered_X_test_df['price_code_ordinal'] = filtered_X_test['price_code_ordinal'].values

    return transformed_filtered_X_test_df

def predict_with_transformer_model(model, preprocessor, filtered_X_test, sequence_length):
    """
    Generates predictions using the Transformer model for each sequence in filtered test data.

    Args:
    - model (nn.Module): Transformer model instance.
    - preprocessor (object): Preprocessor object fitted on training data.
    - filtered_X_test (DataFrame): Test data containing features for prediction.
    - sequence_length (int): Length of sequences used for input to the Transformer.

    Returns:
    list: Predicted ticket sales for each sequence in filtered test data.
    """
    transformed_filtered_X_test_df = preprocess_data_for_model(preprocessor, filtered_X_test)
    sequences, _ = create_sequences(
        transformed_filtered_X_test_df,
        'original_opponent',
        'date',
        'price_code_ordinal',
        sequence_length,
        'target',
        max_length
    )

    model.eval()
    transformer_predictions = []
    for sequence in sequences:
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(sequence_tensor).cpu().numpy().mean()
            transformer_predictions.append(prediction)

    return transformer_predictions

def predict_with_poisson_model(filtered_X_test, poisson_model):
    """
    Generates predictions using the Poisson regression model.

    Args:
    - filtered_X_test (DataFrame): Test data containing features for prediction.
    - poisson_model (object): Trained Poisson regression model.

    Returns:
    ndarray: Predicted ticket sales for each entry in filtered test data.
    """
    processed_entries = poisson_model.named_steps['preprocessor'].transform(filtered_X_test)
    poisson_predictions = poisson_model.named_steps['poisson_regressor'].predict(processed_entries)
    return poisson_predictions

def make_predictions_and_export(filtered_X_test, preprocessor, transformer_model_bowl_0, transformer_model_bowl_1, poisson_model, output_filename):
    """
    Orchestrates the prediction process using Transformer and Poisson models, and exports predictions to a CSV file.

    Args:
    - filtered_X_test (DataFrame): Test data containing features for prediction.
    - preprocessor (object): Preprocessor object fitted on training data.
    - transformer_model_bowl_0 (nn.Module): Transformer model for bowl location 0.
    - transformer_model_bowl_1 (nn.Module): Transformer model for bowl location 1.
    - poisson_model (object): Trained Poisson regression model.
    - output_filename (str): File path to save the predictions CSV.

    Returns:
    None
    """
    transformer_predictions = []
    poisson_predictions = predict_with_poisson_model(filtered_X_test, poisson_model)
    
    for idx, row in filtered_X_test.iterrows():
        opponent = row['opponent']
        price_code = row['price_code']
        bowl_location = row['bowl_location']

        if bowl_location == 0:
            transformer_model = transformer_model_bowl_0
        else:
            transformer_model = transformer_model_bowl_1

        filtered_row = filtered_X_test[
            (filtered_X_test['opponent'] == opponent) &
            (filtered_X_test['price_code'] == price_code)
        ]
        transformer_price_predictions = predict_with_transformer_model(transformer_model, preprocessor, filtered_row, sequence_length)
        transformer_predictions.append(transformer_price_predictions[0])  # Assuming we take the first prediction

    filtered_X_test['predicted_tickets_sold_transformer'] = transformer_predictions
    filtered_X_test['predicted_tickets_sold_poisson'] = poisson_predictions

    filtered_X_test.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

if __name__ == "__main__":
    all_data = load_data()
    all_data = preprocess_data(all_data, START_DATE)
    all_data = feature_engineer_data(all_data)

    last_calculate_date = all_data['calculate_date'].max().strftime('%Y-%m-%d')
    filtered_X_test = all_data[all_data['calculate_date'] == last_calculate_date]

    X_train = pd.read_parquet(DATA_PATH)
    preprocessor = preprocess_train_data(X_train)

    transformer_model_bowl_0 = load_transformer_model(
        TRANSFORMER_MODEL_BOWL_0_PATH, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length
    )
    transformer_model_bowl_1 = load_transformer_model(
        TRANSFORMER_MODEL_BOWL_1_PATH, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length
    )
    poisson_model = joblib.load(POISSON_MODEL_PATH)

    output_filename = OUTPUT_FILENAME_TEMPLATE.format(date=last_calculate_date)

    make_predictions_and_export(filtered_X_test, preprocessor, transformer_model_bowl_0, transformer_model_bowl_1, poisson_model, output_filename)
