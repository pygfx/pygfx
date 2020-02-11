from visvis2 import linalg


def test_clamp():
    """Test the clamp function."""
    assert linalg.clamp(0.5, 0, 1) == 0.5, "Value already within limits"
    assert linalg.clamp(0, 0, 1) == 0, "Value equal to one limit"    
    assert linalg.clamp(-0.1, 0, 1) == 0, "Value too low"
    assert linalg.clamp(1.1, 0, 1) == 1, "Value too high"
