from typing import Callable, Optional

import numpy as np

from .ks_1samp import ks_1samp
from .ks_2samp import ks_2samp


def mkstest(
    x_val: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    y_cdf: Optional[Callable] = None,
    alpha: float = 0.05,
    verbose: bool = False,
) -> bool:
    """Extended Kolmogorov-Smirnov test for the one and two-sample cases.

    Args:
        x_val (array_like): first data sample.
        y_val (array_like): second data sample.
        y_cdf (callable): mcdf of the distribution to test against in the one-sample case.
        alpha (float): significance level.
        verbose (bool): whether to print the test statistic and the critical value.

    Returns:
        bool: True if the null hypothesis is rejected, False otherwise.
    """
    if not isinstance(alpha, float):
        raise ValueError("alpha must be float")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")
    if y_val is None and y_cdf is None:
        raise ValueError("y_val and y_cdf cannot both be None")

    if y_cdf:
        return ks_1samp(x_val, y_cdf, alpha=alpha, verbose=verbose)
    return ks_2samp(x_val, y_val, alpha=alpha, verbose=verbose)
