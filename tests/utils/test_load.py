import pytest
import httpx
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


@pytest.mark.xfail(raises=httpx.ConnectError)
@pytest.mark.parametrize("url", URLS)
def test_load_meshes(url):
    # The .binvox format only has a volume
    if url.endswith(".binvox"):
        mesh = gfx.load_mesh(url, remote_ok=True)
        assert mesh == []
        return

    # Test loading meshes via trimesh
    mesh = gfx.load_mesh(url, remote_ok=True)

    assert isinstance(mesh, list)
    assert len(mesh) > 0
    assert all(isinstance(m, gfx.Mesh) for m in mesh)


@pytest.mark.xfail(raises=httpx.ConnectError)
@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("flatten", (True, False))
def test_load_scenes(
    url,
    flatten,
):
    # Test loading scenes via trimesh
    scene = gfx.load_scene(url, flatten=flatten, remote_ok=True)

    assert isinstance(scene, gfx.Scene)
    assert len(scene.children) > 0

    if flatten and url.endswith(".binvox"):
        assert any(isinstance(ob, gfx.Volume) for ob in scene.children)
