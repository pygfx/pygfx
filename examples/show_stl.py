"""
Demonstrates show utility with an STL file
"""
from pathlib import Path

import numpy as np
import trimesh

import pygfx as gfx


TEAPOT = Path(__file__).parent / "models" / "teapot.stl"

teapot = trimesh.load(TEAPOT)

mesh = gfx.Mesh(
    gfx.Geometry.from_trimesh(teapot),
    gfx.MeshPhongMaterial(),
)

# TODO: I'd rather change the camera up vector to be (0, 0, 1) for this model
# but it's not clear to me how to do that
rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(np.pi * 1.5))
mesh.rotation.multiply(rot)

if __name__ == "__main__":
    gfx.show(mesh)
