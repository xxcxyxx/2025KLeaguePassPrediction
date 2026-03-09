import numpy as np


def euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))


def mean_euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(euclidean_distance(y_true, y_pred).mean())