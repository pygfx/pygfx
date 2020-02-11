from visvis2 import linalg


def test_vector3():
    """Test the Vector3.lerp function."""
    v1 = linalg.Vector3()
    assert v1.x == 0
    assert v1.y == 0
    assert v1.z == 0
    
    x, y, z = 1, 2 ,3
    v2 = linalg.Vector3(x, y, z)
    assert v2.x == x
    assert v2.y == y
    assert v2.z == z
