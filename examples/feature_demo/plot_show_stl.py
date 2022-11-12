"""
Show STL File via gfx.show
==========================

Demonstrates show utility with an STL file
"""
from pathlib import Path

import trimesh

import pygfx as gfx


TEAPOT = Path(__file__).parents[1] / "models" / "teapot.stl"

teapot = trimesh.load(TEAPOT)

mesh = gfx.Mesh(
    gfx.trimesh_geometry(teapot),
    gfx.MeshPhongMaterial(),
)

if __name__ == "__main__":
    gfx.show(mesh, up=gfx.linalg.Vector3(0, 0, 1))
