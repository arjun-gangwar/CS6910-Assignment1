# helper functions
import numpy as np

def one_hot_encode(y: np.array, n_class: int):      # (B,1)
    b_size = y.shape[0]
    encoded = np.zeros((b_size, n_class))
    encoded[np.arange(b_size), y] = 1
    return encoded