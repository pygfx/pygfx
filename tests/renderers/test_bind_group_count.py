"""Test that the shaders do not need more bind groups than WebGPU guarantees."""

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import get_shared
from pygfx.renderers.wgpu.shaders.meshshader import MeshPhysicalShader


def test_physical_material_bind_group_count():
    # WebGPU guarantees 4 bind groups (maxBindGroups). The last one is for
    # the render state (lights and shadows), which leaves 3 for the shader.
    tex = gfx.Texture(np.zeros((4, 4, 4), np.uint8), dim=2)
    material = gfx.MeshPhysicalMaterial(
        map=gfx.TextureMap(tex),
        roughness_map=gfx.TextureMap(tex),
        clearcoat=1.0,
        clearcoat_map=gfx.TextureMap(tex),
        iridescence=1.0,
        iridescence_map=gfx.TextureMap(tex),
    )
    mesh = gfx.Mesh(gfx.box_geometry(), material)
    scene = gfx.Scene()
    scene.add(mesh)

    shader = MeshPhysicalShader(mesh)
    bindings = shader.get_bindings_info(mesh, get_shared(), scene)

    assert len(bindings) <= 3
    assert set(bindings) <= {0, 1, 2}

    # The bindings of the physical material follow those of the standard one
    names = [b.name for b in bindings[2].values()]
    assert "t_roughness_map" in names
    assert "t_clearcoat_map" in names
    assert "t_iridescence_map" in names
    assert sorted(bindings[2]) == list(range(len(names)))
