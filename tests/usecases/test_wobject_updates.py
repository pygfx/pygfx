"""
Tests that updates stuff on the world-object and then verify that pipeline
mechanics applies changes as expected. Prevent unexpected recompiles
(expensive), but also make sure changes actually have effect.

The tests in this module are not exhaustive. Let's add tests for cases of
interests, and regression tests when a bug is fixed.

These tests touch parts of the materials, geometry, buffers, the trackable
object, and the pipeline update mechanics.
"""

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
from pygfx.renderers.wgpu.engine.pipeline import get_pipeline_container_group
from pygfx.renderers.wgpu.engine.renderstate import get_renderstate

from ..testutils import can_use_wgpu_lib
import pytest


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


class PipelineSnapshotter:
    def __init__(self, renderer, scene, world_object):
        self.world_object = world_object
        self.renderer = renderer
        flat = renderer._get_flat_scene(scene, None)
        self.env = get_renderstate(flat.lights, renderer._blender)
        self._snapshot()

    def get_shaders_pipelines_bindings(self):
        # Prime the blender
        self.renderer._blender.ensure_target_size((100, 100))

        # Get the pipeline container. In the call below, the pipeline is updated,
        # the equivalent what happens during a call. This means we don't need to
        # perform an actual draw.
        pipeline_container_group = get_pipeline_container_group(
            self.world_object, self.env
        )
        pipeline_container = pipeline_container_group.render_containers[0]

        # Store shaders
        shaders = pipeline_container.wgpu_pipelines.copy()

        # Store pipelines
        pipelines = pipeline_container.wgpu_pipelines.copy()

        # Store bindings:  bind group -> binding id -> Binding
        bindings = {k: v.copy() for k, v in pipeline_container.bindings_dicts.items()}

        return shaders, pipelines, bindings

    def _snapshot(self):
        x = self.get_shaders_pipelines_bindings()
        self.prev_shaders, self.prev_pipelines, self.prev_bindings = x

    def check(self, *, shaders_same, pipelines_same, bindings_same):
        shaders, pipelines, bindings = self.get_shaders_pipelines_bindings()
        prev_shaders, prev_pipelines, prev_bindings = (
            self.prev_shaders,
            self.prev_pipelines,
            self.prev_bindings,
        )

        if shaders_same:
            assert shaders == prev_shaders
        else:
            assert shaders != prev_shaders

        if pipelines_same:
            assert pipelines == prev_pipelines
        else:
            assert pipelines != prev_pipelines

        if bindings_same:
            assert bindings == prev_bindings
        else:
            assert bindings != prev_bindings

        self._snapshot()


def get_pipelines(world_object):
    pipeline_container_group = world_object._wgpu_pipeline_container_group
    pipeline_container = pipeline_container_group.render_containers[0]
    env_hashes = list(pipeline_container.wgpu_shaders.keys())
    assert len(env_hashes) == 1
    env_hash = env_hashes[0]
    return pipeline_container.wgpu_pipelines[env_hash].copy()


def get_bindings(world_object):
    pipeline_container_group = world_object._wgpu_pipeline_container_group
    pipeline_container = pipeline_container_group.render_containers[0]
    # bind group -> binding id -> Binding
    return {k: v.copy() for k, v in pipeline_container.bindings_dicts.items()}


def test_updating_image_material_map():
    renderer = gfx.renderers.WgpuRenderer(RenderCanvas(), blend_mode="ordered2")
    scene = gfx.Scene()

    # Create an image
    im = gfx.Image(
        gfx.Geometry(grid=np.random.uniform(0, 1, size=(24, 24, 1)).astype(np.float32)),
        gfx.ImageBasicMaterial(map=gfx.cm.plasma),
    )
    scene.add(im)

    # Snapshot - change map - compare

    snapshotter = PipelineSnapshotter(renderer, scene, im)

    im.material.map = gfx.cm.viridis

    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=False)


def test_updating_mesh_material_color():
    renderer = gfx.renderers.WgpuRenderer(RenderCanvas(), blend_mode="ordered2")
    scene = gfx.Scene()

    # Create a mesh
    mesh = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color=(1, 0, 0), color_mode="uniform"),
    )
    scene.add(mesh)

    snapshotter = PipelineSnapshotter(renderer, scene, mesh)

    # Sanity check
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)

    # Changing to another opaque color does not invoke a change for the pipeline
    mesh.material.color = (0, 1, 0)
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)

    # Named colours should be opaque too
    mesh.material.color = "red"
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)

    # Using a transparent color does require a change, bc it needs a shader for the transparency pass
    mesh.material.color = (1, 1, 0, 0.5)
    snapshotter.check(shaders_same=False, pipelines_same=False, bindings_same=True)

    # Going back to a transparent color does NOT invoke a change, because the shader's still there
    mesh.material.color = "red"
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)


def test_updating_mesh_material_opacity():
    renderer = gfx.renderers.WgpuRenderer(RenderCanvas(), blend_mode="ordered2")
    scene = gfx.Scene()

    # Create a mesh
    mesh = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color=(1, 0, 0), color_mode="uniform"),
    )
    scene.add(mesh)

    snapshotter = PipelineSnapshotter(renderer, scene, mesh)

    # Sanity check
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)

    # Making the mesh transparent does require a change, bc it needs a shader for the transparency pass
    mesh.material.opacity = 0.5
    snapshotter.check(shaders_same=False, pipelines_same=False, bindings_same=True)

    # Going back to a transparent color does NOT invoke a change, because the shader's still there
    mesh.material.opacity = 1
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=True)


def test_updating_mesh_geometry_color():
    renderer = gfx.renderers.WgpuRenderer(RenderCanvas(), blend_mode="ordered2")
    scene = gfx.Scene()

    # Create a mesh
    mesh = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color_mode="vertex"),
    )
    mesh.geometry.colors = gfx.Buffer(np.random.uniform(size=(24, 3)).astype("f4"))
    scene.add(mesh)

    snapshotter = PipelineSnapshotter(renderer, scene, mesh)

    # Changing to a new set of colors only changes the bindings
    mesh.geometry.colors = gfx.Buffer(np.random.uniform(size=(24, 3)).astype("f4"))
    snapshotter.check(shaders_same=True, pipelines_same=True, bindings_same=False)

    # Changing to transparent colors invokes a shader recompile
    mesh.geometry.colors = gfx.Buffer(np.random.uniform(size=(24, 4)).astype("f4"))
    snapshotter.check(shaders_same=False, pipelines_same=False, bindings_same=False)

    # This is not because an alpha channel was added; also happens for grayscale
    mesh.geometry.colors = gfx.Buffer(np.random.uniform(size=(24, 1)).astype("f4"))
    snapshotter.check(shaders_same=False, pipelines_same=False, bindings_same=False)
