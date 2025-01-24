"""Global configuration for pytest"""

import os
import numpy as np
import pytest


# System fonts can interfere with screenshots that expect a particular font to be used
os.environ["PYGFX_DISABLE_SYSTEM_FONTS"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-screenshots",
        action="store_true",
        dest="regenerate_screenshots",
        default=False,
    )


@pytest.fixture(autouse=True)
def predictable_random_numbers():
    """
    Called at start of each test, guarantees that calls to random produce the same output over subsequent tests runs,
    see http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.seed.html
    """
    np.random.seed(0)


@pytest.fixture(autouse=True, scope="session")
def numerical_exceptions():
    """
    Ensure any numerical errors raise a warning in our test suite
    The point is that we enforce such cases to be handled explicitly in our code
    Preferably using local `with np.errstate(...)` constructs
    """
    np.seterr(all="raise")
