import numpy as np


def mecdf(x_val: np.ndarray, t: np.ndarray) -> float:
    """Computes the multivariate empirical cdf of x_val at t.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        t: A numpy array of shape (num_samples_t, dim) representing the point at which to evaluate
            the cdf.

    Returns:
        The multivariate empirical cdf of x_val at t.
    """
    lower = (x_val <= t) * 1.0
    return np.mean(np.prod(lower, axis=1))
