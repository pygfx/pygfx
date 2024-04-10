"""Global configuration for pytest"""

import numpy as np
import pytest


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


@pytest.fixture(autouse=True, scope="session")
def force_llvm_adapter():
    import wgpu
    import pygfx as gfx
    adapters = wgpu.gpu.enumerate_adapters()
    adapters_llvm = [a for a in adapters if "llvmpipe" in a.summary.lower()]
    if adapters_llvm:
        gfx.renderers.wgpu.select_adapter(adapters_llvm[0])
    else:
        pytest.skip(reason="No LLVMpipe adapter found")
