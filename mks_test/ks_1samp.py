from typing import Callable

import numpy as np

from .utils import mecdf


def ks_1samp(
    x_val: np.ndarray,
    f_y: Callable,
    alpha: float,
    asymptotic: bool = False,
    verbose: bool = False,
) -> bool:
    """Performs a multivariate one-sample extension of the Kolmogorov-Smirnov test.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        f_y: The cdf of the distribution to test against.
        alpha: The significance level.
        asymptotic: Whether to use the asymptotic approximation or not.
        verbose: Whether to print the test statistic and the critical value.

    Returns:
        A boolean indicating whether the null hypothesis is rejected.
    """
    num_samples_x, dim = x_val.shape

    z = np.zeros((num_samples_x, dim, dim))
    for h in range(dim):
        ind = np.argsort(x_val[:, h])[::-1]
        temp = np.take(x_val, ind, axis=0)
        z[:, :, h] = temp
        for i in range(dim):
            for j in range(num_samples_x - 1, -1, -1):
                if j == num_samples_x - 1:
                    runmax = temp[num_samples_x - 1, i]
                else:
                    runmax = max(runmax, temp[j, i])
                z[j, i, h] = runmax

    diff = np.zeros((num_samples_x, dim))
    for h in range(dim):
        for i in range(num_samples_x):
            val = np.abs(mecdf(x_val, z[i, :, h]) - f_y(z[i, :, h])) * (
                round(num_samples_x * mecdf(x_val, z[i, :, h])) == num_samples_x - i
            )
            diff[i, h] = val
            if h == 0:
                diff[i, h] = max(
                    diff[i, h],
                    np.abs(mecdf(x_val, x_val[i, :]) - f_y(x_val[i, :])),
                )
    KS = np.max(diff)
    if asymptotic:
        KS_critical_val = np.sqrt(-np.log(alpha / (2 * dim)) * (0.5 / num_samples_x))
    else:
        KS_critical_val = np.sqrt(
            -np.log(alpha / (2 * (num_samples_x + 1) * dim)) * (0.5 / num_samples_x)
        )

    if verbose:
        print("test statistic: ", KS)
        print("test statistic critical value: ", KS_critical_val)
    return KS > KS_critical_val
