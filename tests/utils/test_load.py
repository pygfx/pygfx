import pytest

import pygfx as gfx


BASE_URL = "https://github.com/mikedh/trimesh/raw/main/models"
FILES_TO_TEST = [
    "wallhole.obj",
    "CesiumMilkTruck.glb",
    "nested.glb",
    "cube_blender_uv.ply",
    "chair_model.binvox",
]
URLS = [f"{BASE_URL}/{file}" for file in FILES_TO_TEST]


@pytest.mark.parametrize("url", URLS)
def test_load_meshes(url):
    # The .binvox format is expected to fail because trimesh
    # loads it as VoxelGrid, not as a mesh
    if url.endswith(".binvox"):
        with pytest.raises(ValueError):
            mesh = gfx.load_mesh(url, remote_ok=True)
        return

    # Test loading meshes via trimesh
    mesh = gfx.load_mesh(url, remote_ok=True)

    assert isinstance(mesh, list)
    assert all([isinstance(m, gfx.Mesh) for m in mesh])


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("flatten", (True, False))
def test_load_scenes(
    url,
    flatten,
):
    # Test loading scenes via trimesh
    mesh = gfx.load_scene(url, flatten=flatten, remote_ok=True)

    assert isinstance(mesh, gfx.Scene)
