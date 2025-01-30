"""
Train and evaluate a Transformer model for predicting ticket sales based on historical data.

This script preprocesses training data, creates sequences for training and validation,
and trains a Transformer model separately for two bowl locations. It saves the best
trained models and evaluates their performance using RMSE on both training and validation sets.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from ..utils import preprocess_train_data, create_sequences, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length, sequence_length, epochs, lr
from src.models.model import TicketSalesTransformer
import src.models.path_config as path_config
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

# Negative Binomial Loss Function with Fixed Dispersion
def negative_binomial_loss_fixed_dispersion(predictions, targets, dispersion=1.0):
    """
    Computes the negative binomial loss function with a fixed dispersion parameter.

    Args:
    - predictions (torch.Tensor): Predicted values from the model.
    - targets (torch.Tensor): Actual target values.
    - dispersion (float, optional): Dispersion parameter for the negative binomial distribution.
                                    Default is 1.0.

    Returns:
    torch.Tensor: Negative of the mean negative binomial loss.
    """
    mu = predictions
    alpha = torch.tensor(dispersion, dtype=mu.dtype, device=mu.device)
    t1 = torch.lgamma(targets + alpha) - torch.lgamma(alpha) - torch.lgamma(targets + 1)
    t2 = alpha * torch.log(alpha / (alpha + mu))
    t3 = targets * torch.log(mu / (alpha + mu))
    loss = t1 + t2 + t3
    return -loss.mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, clip_value=1.0, patience=4):
    """
    Trains the Transformer model using the specified train and validation data loaders.

    Args:
    - model (nn.Module): Transformer model instance.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - criterion (function): Loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer for training the model.
    - epochs (int): Number of epochs for training.
    - clip_value (float, optional): Value for gradient clipping. Default is 1.0.
    - patience (int, optional): Number of epochs to wait before early stopping. Default is 4.

    Returns:
    None
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for sequences, targets, _ in train_loader:
            sequences = sequences.to(device)
            targets = targets.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            if torch.isnan(loss).any():
                print("Found NaN in loss, skipping update")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_sequences, val_targets, _ in val_loader:
                val_sequences = val_sequences.to(device)
                val_targets = val_targets.unsqueeze(1).to(device)
                val_outputs = model(val_sequences)
                val_loss = criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        print(f'Learning rate: {scheduler.optimizer.param_groups[0]["lr"]}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping!')
            break

def batchify(data, batch_size):
    # Divides the data into batches of specified size.
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def evaluate_model(model, X_batches, y_batches, device):
    """
    Evaluates the Transformer model using RMSE (Root Mean Squared Error) on given batches of data.

    Args:
    - model (nn.Module): Trained Transformer model instance.
    - X_batches (list): List of batches of input sequences.
    - y_batches (list): List of batches of target values.
    - device (torch.device): Device to perform computations on (cpu or cuda).

    Returns:
    float: Average RMSE across all batches.
    """
    model.eval()
    rmse = 0
    with torch.no_grad():
        for X_batch, y_batch in zip(X_batches, y_batches):
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            y_pred_batch = model(X_batch).cpu().numpy()
            rmse += np.sqrt(mean_squared_error(y_batch.cpu().numpy(), y_pred_batch))
    rmse /= len(X_batches)
    return rmse

def main(output_dir): 
    # load the data
    X_train = pd.read_parquet(path_config.X_TRAIN_PATH)
    y_train = pd.read_parquet(path_config.Y_TRAIN_PATH)

    preprocessor = preprocess_train_data(X_train)

    for bowl_location in [0, 1]:
        print(f"Processing bowl location: {bowl_location}")
        data = X_train[X_train['bowl_location'] == bowl_location].copy()
        y_train_bowl = y_train.loc[data.index]
        data['target'] = y_train_bowl.values
        data['date'] = pd.to_datetime(data['calculate_date'])
        data = data.sort_values(by=['date', 'price_code'])
        data['original_opponent'] = data['opponent'].copy()
        data['price_code_ordinal'] = data['price_code_ordinal'].copy()

        preprocessed_data = preprocessor.transform(data)
        preprocessed_data = pd.DataFrame(preprocessed_data)
        preprocessed_data['date'] = data['date'].values
        preprocessed_data['target'] = data['target'].values
        preprocessed_data['original_opponent'] = data['original_opponent'].values
        preprocessed_data['price_code_ordinal'] = data['price_code_ordinal'].values
        preprocessed_data = preprocessed_data.fillna(0)

        sequences, targets = create_sequences(preprocessed_data, 'original_opponent', 'date', 'price_code_ordinal', sequence_length, 'target', max_length)

        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = TicketSalesTransformer(input_dim, embed_dim, num_heads, num_layers, output_dim, max_length)
        model = model.to(device)

        criterion = negative_binomial_loss_fixed_dispersion
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, patience=3)

        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_bowl_{bowl_location}.pth'))

        train_rmse = evaluate_model(model, batchify(X_train_seq, 8), batchify(y_train_seq, 8), device)
        val_rmse = evaluate_model(model, batchify(X_val_seq, 8), batchify(y_val_seq, 8), device)
        print(f"Training RMSE for bowl {bowl_location}: {train_rmse}")
        print(f"Validation RMSE for bowl {bowl_location}: {val_rmse}")

if __name__ == "__main__":
    # select the output folder (default or retrained)
    output_dir = path_config.RETRAINED_MODEL_DIR
    main(output_dir)
