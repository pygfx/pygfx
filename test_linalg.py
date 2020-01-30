"""
Small script to test linalg.
"""
import linalg

import numpy as np
import pytest


def test_clamp():
    """Test the clamp function."""
    assert linalg.clamp(0.5, 0, 1) == 0.5, "Value already within limits"
    assert linalg.clamp(0, 0, 1) == 0, "Value equal to one limit"    
    assert linalg.clamp(-0.1, 0, 1) == 0, "Value too low"
    assert linalg.clamp(1.1, 0, 1) == 1, "Value too high"


def test_vector3():
    """Test the Vector3.lerp function."""
    # Value equal to lower boundary
    v1 = linalg.Vector3()
    assert v1.x == 0
    assert v1.y == 0
    assert v1.z == 0
    
    x, y, z = 1, 2 ,3
    v2 = linalg.Vector3(x, y, z)
    assert v2.x == x
    assert v2.y == y
    assert v2.z == z
