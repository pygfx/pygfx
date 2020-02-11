"""Global configuration for pytest"""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def predictable_random_numbers():
    """
    Called at start of each test, guarantees that calls to random produce the same output over subsequent tests runs,
    see http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.seed.html
    """
    np.random.seed(0)
