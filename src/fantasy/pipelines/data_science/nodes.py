import pandas as pd
import torch
import torch.nn as nn
import logging
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torchcp
import torch
import torch.nn as nn
from torchcp.regression.predictors import SplitPredictor
from torchcp.regression.loss import QuantileLoss
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(LSTMModel, self).__init__()
        
        # LSTM layer for sequential features
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully connected layers for processing LSTM output
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, seq_input):
        # LSTM layer
        lstm_out, _ = self.lstm(seq_input)
        lstm_out = lstm_out[:, -1, :]  # Use the last hidden state

        # Fully connected layers
        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        
        return x
    

import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(X_bat: pd.DataFrame, y_bat: pd.DataFrame, split_params: dict):
    """
    Splits the input data into training, calibration, and testing sets.

    Args:
        X_bat (pd.DataFrame): Input features DataFrame.
        y_bat (pd.DataFrame): Target DataFrame.
        split_params (dict): Dictionary containing 'test_size', 'cali_size', and 'random_state' for splitting.

    Returns:
        tuple: X_train, X_test, X_cali, y_train, y_test, y_cali DataFrames.
    """
    test_size = split_params['test_size']  # Fraction of data used for testing (including calibration)
    cali_size = split_params['cali_size']  # Fraction of test data used for calibration
    random_state = split_params['random_state']

    # First, split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_bat, y_bat, test_size=test_size, random_state=random_state
    )

    # Further split the test set into calibration and final test sets
    X_test, X_cali, y_test, y_cali = train_test_split(
        X_test, y_test, test_size=cali_size, random_state=random_state
    )

    return X_train, X_test, X_cali, y_train, y_test, y_cali



def _split_seq(X: pd.DataFrame, seq_col_start: str):
    """
    Splits the input DataFrame into sequential feature sets.

    Args:
        X (pd.DataFrame): Input features DataFrame.
        seq_col_start (str): Prefix to identify sequential feature columns.

    Returns:
        tuple: Sequential DataFrame.
    """
    # Identify sequential feature columns
    sequence_columns = [col for col in X.columns if col.startswith(seq_col_start)]

    # Split into sequential features
    X_seq = X[sequence_columns]

    return X_seq


def _convert_to_tensors(X_seq_df: pd.DataFrame, y_df: pd.DataFrame, model_params):
    """
    Converts DataFrame inputs to PyTorch tensors, reshaping the sequential features appropriately.

    Args:
        X_seq_df (pd.DataFrame): Sequential features DataFrame.
        y_df (pd.DataFrame): Target DataFrame containing batting points.
        sequence_length (int): Number of past matches in the sequence.
        num_features_per_match (int): Number of features per match.

    Returns:
        tuple: PyTorch tensors for sequential features and targets.
    """
    sequence_length = model_params['sequence_length']
    num_features_per_match = model_params['num_features_per_match']

    # Reshape sequential features to (samples, sequence_length, num_features_per_match)
    X_seq = X_seq_df.values.reshape(-1, sequence_length, num_features_per_match)
    y = y_df.values

    # Convert to PyTorch tensors
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    return X_seq_tensor, y_tensor


def train_lstm_model(X_train: pd.DataFrame, y_train: pd.DataFrame, split_params: dict, model_params: dict, train_params: dict):
    """
    Trains an LSTM model using sequential features from the input data.

    Args:
        X_train (pd.DataFrame): Training DataFrame containing combined sequential features.
        y_train (pd.DataFrame): Target DataFrame containing the target values.
        split_params (dict): Parameters for splitting features with keys:
                             - 'seq_col_start': Prefix for sequential feature columns.
        model_params (dict): Parameters for the LSTM model with keys:
                             - 'sequence_length': Number of past matches in the sequence.
                             - 'num_features_per_match': Number of features per match.
        train_params (dict): Training parameters with keys:
                             - 'epochs': Number of training epochs.
                             - 'batch_size': Batch size for training.
                             - 'learning_rate': Learning rate for the optimizer.

    Returns:
        nn.Module: Trained LSTM model.
    """
    logger = logging.getLogger(__name__)
    
    # Step 1: Split into sequential features
    seq_col_start = split_params['seq_col_start']

    X_seq_train = _split_seq(X_train, seq_col_start)

    # Step 2: Convert to tensors
    X_seq_train_tensor, y_train_tensor = _convert_to_tensors(X_seq_train, y_train, model_params)

    # Step 3: Create DataLoader for training
    train_dataset = TensorDataset(X_seq_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)

    # Step 4: Define the LSTM model
    model = LSTMModel(
        sequence_length=model_params['sequence_length'], 
        n_features=model_params['num_features_per_match']
    )

    # Step 5: Set the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    # Step 6: Training loop
    logging.info("Starting training process...")  # Log the start of training
    for epoch in range(train_params['epochs']):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        for seq_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(seq_batch)

            # Compute the loss
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f'Epoch [{epoch + 1}/{train_params["epochs"]}], Loss: {avg_loss:.4f}')

    logger.info("Training completed.")  # Log the completion of training

    return model



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from torchcp.regression.predictors import SplitPredictor

def calibrate_lstm_model(model, X_cali, y_cali, X_test, y_test, params):
    """
    Calibrates an LSTM model using Split Conformal Prediction and tests it on the provided test set.
    
    Args:
        model (nn.Module): The LSTM model to be calibrated.
        X_cali (pd.DataFrame or np.ndarray): Calibration data of shape (num_players, num_features * seq_length).
        y_cali (pd.DataFrame or np.ndarray): Calibration labels.
        X_test (pd.DataFrame or np.ndarray): Test features data.
        y_test (pd.DataFrame or np.ndarray): Test labels data.
        params (dict): Dictionary containing the following keys:
            - 'seq_length' (int): Length of the input sequence.
            - 'n_features' (int): Number of features per time step.
            - 'significance_level' (float): The desired significance level for conformal prediction (default: 0.1 for 90% confidence interval).
            - 'batch_size' (int): Batch size for DataLoader.

    Returns:
        SplitPredictor: The calibrated split conformal prediction model.
        List of tuples: Each tuple contains the prediction interval for a batch.
    """
    logger = logging.getLogger(__name__)
    
    # Extract parameters from the dictionary
    seq_length = params['seq_length']
    n_features = params['n_features']
    significance_level = params['significance_level']
    batch_size = params['batch_size']
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert X_cali and y_cali to PyTorch tensors and reshape X_cali
    X_cali = torch.tensor(X_cali.values, dtype=torch.float32).reshape(-1, seq_length, n_features)
    y_cali = torch.tensor(y_cali.values, dtype=torch.float32).reshape(-1, 1) 
    calib_dataset = TensorDataset(X_cali, y_cali)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)

    # Prepare the test data
    X_test = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, seq_length, n_features)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)  
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the SplitPredictor model with the given LSTM model
    split_predictor = SplitPredictor(model)

    # Calibrate the SplitPredictor model using the calibration data
    split_predictor.calibrate(calib_loader, significance_level)

    logger.info(f"Model calibrated with {len(X_cali)} calibration examples and {1 - significance_level:.0%} confidence level.")
    
    # Evaluate the model on the test set
    logger.info(f"Evaluation on test set: {split_predictor.evaluate(test_loader)}")

    return split_predictor


