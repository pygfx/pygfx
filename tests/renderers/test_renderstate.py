import gc

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu.engine.renderstate import (
    get_renderstate,
    _renderstate_instance_cache as renderstate_cache,
)

from ..testutils import can_use_wgpu_lib
import pytest


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


render_tex = gfx.Texture(dim=2, size=(10, 10, 1), format=wgpu.TextureFormat.rgba8unorm)


def test_renderstate_reuse1():
    renderer1 = gfx.renderers.WgpuRenderer(render_tex)
    renderer2 = gfx.renderers.WgpuRenderer(render_tex)
    scene1 = gfx.Scene()
    scene2 = gfx.Scene()

    env1 = get_renderstate(scene1, renderer1._blender)
    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1 is env2


def test_renderstate_reuse2():
    renderer1 = gfx.renderers.WgpuRenderer(render_tex)

    scene1 = gfx.Scene()

    scene2 = gfx.Scene()
    scene2.add(gfx.AmbientLight())
    scene2.add(gfx.DirectionalLight())

    scene3 = gfx.Scene()
    scene3.add(gfx.AmbientLight())
    scene3.add(gfx.DirectionalLight())

    scene4 = gfx.Scene()
    scene4.add(gfx.AmbientLight())
    scene4.add(gfx.AmbientLight())
    scene4.add(gfx.DirectionalLight())

    scene5 = gfx.Scene()
    scene5.add(gfx.AmbientLight())
    scene5.add(gfx.AmbientLight())
    scene5.add(gfx.DirectionalLight())
    scene5.add(gfx.DirectionalLight())

    env1 = get_renderstate(scene1, renderer1._blender)
    env2 = get_renderstate(scene2, renderer1._blender)
    env3 = get_renderstate(scene3, renderer1._blender)
    env4 = get_renderstate(scene4, renderer1._blender)
    env5 = get_renderstate(scene5, renderer1._blender)

    assert env1 is not env2, "env1 and env2 have different number of lights"
    assert env2 is env3, "env2 and env3 have same number of lights"
    assert env2 is env4, (
        "env2 and env4 have same number of lights, ambient lights dont count"
    )
    assert env2 is not env5, "env2 and env5 have different number of lights"


def prepare_for_cleanup():
    renderer1 = gfx.renderers.WgpuRenderer(render_tex)
    renderer2 = gfx.renderers.WgpuRenderer(render_tex)
    renderer1.blend_mode = "ordered1"
    renderer2.blend_mode = "ordered2"
    scene1 = gfx.Scene()
    scene2 = gfx.Scene()

    env1 = get_renderstate(scene1, renderer1._blender)
    assert env1.hash in renderstate_cache

    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1 is not env2
    assert env1.hash in renderstate_cache
    assert env2.hash in renderstate_cache

    return renderer1, renderer2, scene1, scene2, env1, env2


def test_renderstate_cleanup_noop():
    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1.hash in renderstate_cache
    assert env2.hash in renderstate_cache


def test_renderstate_cleanup_by_scene_del():
    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    del scene1
    gc.collect()

    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1.hash not in renderstate_cache
    assert env2.hash in renderstate_cache


def test_renderstate_cleanup_by_renderer_del():
    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    del renderer1
    gc.collect()

    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1.hash not in renderstate_cache
    assert env2.hash in renderstate_cache


def test_renderstate_cleanup_by_scene_change():
    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    # cannot test this yet, as currently the scene does not result in state


def test_environment_cleanup_by_renderer_change():
    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    renderer1.blend_mode = "weighted"

    # Rendering render2 doesn't check renderer1
    env2 = get_renderstate(scene2, renderer2._blender)
    assert env1.hash in renderstate_cache
    assert env2.hash in renderstate_cache

    # But rendering with renderer1 does
    env3 = get_renderstate(scene1, renderer1._blender)
    assert env1.hash not in renderstate_cache
    assert env2.hash in renderstate_cache
    assert env3.hash in renderstate_cache
    assert env3 is not env1
    assert env3 is not env2


if __name__ == "__main__":
    pytest.main(["-x", __file__])
