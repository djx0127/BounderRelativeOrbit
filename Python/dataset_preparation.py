"""
dataset_preparation.py
----------------------
Prepare dataset for Conditional Variational Autoencoder (CVAE) training.
Loads MATLAB .mat data containing 4D state parameters and 2D feature parameters,
normalizes them, splits into training/validation/test sets, and saves as PyTorch tensors.

Author: Jixin Ding
Date: 25-10-16
"""

import torch
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

def load_and_prepare_dataset(mat_file, input_keys=('X',), target_keys=('Y',)):
    """
    Load data from a .mat file and normalize to [0, 1].
    
    Args:
        mat_file (str): Path to MATLAB file containing dataset.
        input_keys (tuple): Keys for input variables (e.g., 4D states).
        target_keys (tuple): Keys for output variables (e.g., 2D features).
    
    Returns:
        dict: Dictionary containing normalized tensors and normalization parameters.
    """
    data = loadmat(mat_file)
    
    X = np.concatenate([data[k] for k in input_keys], axis=1)
    Y = np.concatenate([data[k] for k in target_keys], axis=1)

    # Normalize both X and Y to [0, 1]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    Y_min, Y_max = Y.min(axis=0), Y.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-9)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-9)

    # Split dataset into train/val/test (80/10/10)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    dataset = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "Y_train": torch.tensor(Y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "Y_val": torch.tensor(Y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "Y_test": torch.tensor(Y_test, dtype=torch.float32),
        "X_min": X_min, "X_max": X_max,
        "Y_min": Y_min, "Y_max": Y_max
    }

    return dataset

if __name__ == "__main__":
    mat_file = "PCM_dataset.mat"  # modify to your dataset name
    dataset = load_and_prepare_dataset(mat_file, input_keys=('X',), target_keys=('Y',))
    torch.save(dataset, "prepared_dataset.pt")
    print("✅ Dataset prepared and saved as prepared_dataset.pt")