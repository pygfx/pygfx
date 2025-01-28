"""
Skinned Mesh
============

This example shows the rendering of a skinned mesh with a skeleton and bones.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import math
import time
import numpy as np
import pygfx as gfx
import pylinalg as la
from wgpu.gui.auto import WgpuCanvas, run


def create_geometry(sizing):
    geometry = gfx.cylinder_geometry(
        5, 5, sizing["height"], 8, sizing["segment_height"], open_ended=True
    )
    positions = geometry.positions

    skin_indices = np.zeros((positions.nitems, 4), dtype=np.uint32)
    skin_weights = np.zeros((positions.nitems, 4), dtype=np.float32)

    for i in range(positions.nitems):
        vertex = positions.data[i]
        z = vertex[2] + sizing["half_height"]
        skin_index = z // sizing["segment_height"]
        skin_weight = (z % sizing["segment_height"]) / sizing["segment_height"]

        skin_indices[i] = [skin_index, skin_index + 1, 0, 0]
        skin_weights[i] = [1 - skin_weight, skin_weight, 0, 0]

    geometry.skin_indices = gfx.Buffer(skin_indices)
    geometry.skin_weights = gfx.Buffer(skin_weights)

    return geometry


def create_bones(sizing):
    bones = []
    prev_bone = gfx.Bone()
    bones.append(prev_bone)

    prev_bone.local.position = (0, 0, -sizing["half_height"])

    for _ in range(sizing["segment_count"]):
        bone = gfx.Bone()
        bone.local.position = (0, 0, sizing["segment_height"])
        bones.append(bone)

        prev_bone.add(bone)
        prev_bone = bone

    return bones


def create_mesh(geometry, bones):
    material = gfx.MeshNormalMaterial()

    material.flat_shading = True

    mesh = gfx.SkinnedMesh(geometry, material)
    skeleton = gfx.Skeleton(bones)

    mesh.add(bones[0])
    mesh.bind(skeleton)

    mesh.local.rotation = la.quat_from_euler((-math.pi / 2, 0, 0))
    return mesh


segment_height = 8
segment_count = 4
height = segment_height * segment_count
half_height = height * 0.5

sizing = {
    "segment_height": segment_height,
    "segment_count": segment_count,
    "height": height,
    "half_height": half_height,
}

canvas = WgpuCanvas(size=(640, 480), max_fps=60, title="Skinnedmesh")

renderer = gfx.WgpuRenderer(canvas)

camera = gfx.PerspectiveCamera(75, 640 / 480, depth_range=(0.1, 200))
camera.local.position = (0, 30, 30)
camera.look_at((0, 0, 0))

scene = gfx.Scene()

geometry = create_geometry(sizing)
bones = create_bones(sizing)
mesh = create_mesh(geometry, bones)

scene.add(mesh)

skeleton_helper = gfx.SkeletonHelper(mesh)
scene.add(skeleton_helper)

gfx.OrbitController(camera, register_events=renderer)


def animate():
    t = time.time()
    for bone in mesh.skeleton.bones:
        rotation_y = math.sin(t) * 2 / len(mesh.skeleton.bones)
        bone.local.rotation = la.quat_from_euler((0, rotation_y, 0))

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
