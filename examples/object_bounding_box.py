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
mesh.rotation.set_from_euler(gfx.linalg.Euler(0.71, 0.91))
scene.add(mesh)

box_world = gfx.BoxHelper()
box_world.set_object_world(mesh)
scene.add(box_world)

box_local = gfx.BoxHelper()
box_local.set_object_local(mesh)
mesh.add(box_local)

box_local.material.color = (0, 1, 0, 1)


if __name__ == "__main__":
    gfx.show(scene, up=gfx.linalg.Vector3(0, 0, 1))
