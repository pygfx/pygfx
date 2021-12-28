"""
Demonstrates show utility with an STL file
"""
from pathlib import Path

import trimesh

import pygfx as gfx


TEAPOT = Path(__file__).parent / "models" / "teapot.stl"
teapot = trimesh.load(TEAPOT)

scene = gfx.Scene()

mesh = gfx.Mesh(
    gfx.trimesh_geometry(teapot),
    gfx.MeshPhongMaterial(),
)
scene.add(mesh)

box = gfx.BoxHelper()
box.set_object_world(mesh)
scene.add(box)


if __name__ == "__main__":
    gfx.show(scene, up=gfx.linalg.Vector3(0, 0, 1))
