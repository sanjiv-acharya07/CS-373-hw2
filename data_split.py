'''
-----------------------------------------------------------------------------------
This file will split the original dataset into a training set and a validation set.
PLEASE DO NOT CHANGE ANYTHING!
-----------------------------------------------------------------------------------
'''

import numpy as np

def shuffle_data(X, y, random_state=None):
    """
    Shuffle the data.

    Args:
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )
        seed: int or None

    Returns:
        X: shuffled data
        y: shuffled labels
    """
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def my_train_valid_split(X, y, val_size=0.2, shuffle=True, random_state=42):
    """
    Split the data into training and test sets.

    Args:
        X: numpy array of shape (N, D)
        y: numpy array of shape (N, )
        val_size: float, percentage of data to use as test set
        shuffle: bool, whether to shuffle the data or not
        seed: int or None

    Returns:
        X_train: numpy array of shape (N_train, D)
        X_val: numpy array of shape (N_val, D)
        y_train: numpy array of shape (N_train, )
        y_val: numpy array of shape (N_val, )
    """

    if shuffle:
        X, y = shuffle_data(X, y, random_state)

    n_train_samples = int(X.shape[0] * (1 - val_size))
    X_train, X_val = X[:n_train_samples], X[n_train_samples:]
    y_train, y_val = y[:n_train_samples], y[n_train_samples:]

    return X_train, X_val, y_train, y_val