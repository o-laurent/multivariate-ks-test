# Multivariate extension of the Kolmogorov-Smirnov test

This repository implements the extension of the KS test proposed by Michael Naaman in 
[On the tight constant in the multivariate Dvoretzky–Kiefer–Wolfowitz inequality](https://www.sciencedirect.com/science/article/pii/S016771522100050X/pdf)
in the one and two-sample cases.

This package was translated from the original MATLAB code provided by the author.

Please feel free to open an issue if you have any problems or questions.

## Installation

Clone the repository and install it with pip:

```bash

cd multivariate-ks-test && pip install .
```

Please raise an issue if you want to install it from PyPI.

## Usage example

```python
import numpy as np

from mks_test import mkstest

# Generate two samples from a 5D Normal distribution
n = 100
d = 5
mu = np.zeros(d)
sigma = np.eye(d)
X = np.random.multivariate_normal(mu, sigma, n)
Y = np.random.multivariate_normal(mu, sigma, n)

# Compute the test statistic
mkstest(X, Y, alpha=0.05, verbose=True)
# returns False - you can't reject that the two samples are drawn from the same distribution
```

## Reference

If you find this code useful, you may cite the following paper:

```latex
@article{naaman2021tight,
  title={On the tight constant in the multivariate Dvoretzky--Kiefer--Wolfowitz inequality},
  author={Naaman, Michael},
  journal={Statistics \& Probability Letters},
  year={2021},
}
```
