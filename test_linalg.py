"""
Small script to test linalg.
"""
import linalg

import pytest


def test_clamp():
    """Test the clamp function."""
    # Value already within limits
    np.assert_equal(linalg.clamp(0.5, 0, 1), 0.5)
    # Value equal to one limit
    np.assert_equal(linalg.clamp(0, 0, 1), 0)
    # Value too low
    np.assert_equal(linalg.clamp(-0.1, 0, 1), 0)
    # Value too high
    np.assert_equal(linalg.clamp(1.1, 0, 1), 1)


def test_lerp():
    """Test the Vector3.lerp function."""
    # Value equal to lower boundary
    np.assert_equal(linalg.Vector3.lerp(1, 2, 0), 1)
    # Value equal to upper boundary
    np.assert_equal(linalg.Vector3.lerp(1, 2, 1), 2)
    # Value within range
    np.assert_equal(linalg.Vector3.lerp(1, 2, 0.4), 1.4)