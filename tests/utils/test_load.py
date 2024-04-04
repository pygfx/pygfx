import pytest

import pygfx as gfx


BASE_URL = "https://github.com/mikedh/trimesh/raw/main/models"
FILES_TO_TEST = [
    "wallhole.obj",
    "CesiumMilkTruck.glb",
    "nested.glb",
    "cube_blender_uv.ply",
]
URLS = [f"{BASE_URL}/{file}" for file in FILES_TO_TEST]


@pytest.mark.parametrize("url", URLS)
def test_load_meshes(url):
    # Test loading meshes via trimesh
    mesh = gfx.load_mesh(url, remote_ok=True)

    assert isinstance(mesh, list)
    assert all([isinstance(m, gfx.Mesh) for m in mesh])


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("flatten", (True, False))
def test_load_scenes(url, flatten, ):
    # Test loading scenes via trimesh
    mesh = gfx.load_scene(url, flatten=flatten, remote_ok=True)

    assert isinstance(mesh, gfx.Scene)
