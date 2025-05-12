"""
Assignment 1 – Train/Test Split Function

Write a function to randomly shuffle and split (X, y) into train and test sets.

Parameters:
- X: np.ndarray, shape (n_samples, n_features)
- y: np.ndarray, shape (n_samples,)
- test_ratio: float (e.g. 0.2)

Returns:
- X_train, X_test, y_train, y_test: tuple of arrays

Function Signature:
def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""
import numpy as np

def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(X) == len(y), "X and y must have same number of samples"
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)  # karıştır
    test_size = int(n_samples * test_ratio)     # test boyutu hesapla
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
