import gc

import wgpu
import pygfx as gfx
from pygfx.renderers.wgpu._environment import Environment, environment_manager

import pytest


render_tex = gfx.Texture(dim=2, size=(10, 10, 1), format=wgpu.TextureFormat.rgba8unorm)


def test_environment_reuse():
    renderer1 = gfx.renderers.WgpuRenderer(render_tex)
    renderer2 = gfx.renderers.WgpuRenderer(render_tex)
    scene1 = gfx.Scene()
    scene2 = gfx.Scene()

    env1 = environment_manager.get_environment(renderer1, scene1)
    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1 is env2


def prepare_for_cleanup():
    renderer1 = gfx.renderers.WgpuRenderer(render_tex)
    renderer2 = gfx.renderers.WgpuRenderer(render_tex)
    renderer1.blend_mode = "ordered1"
    renderer2.blend_mode = "ordered2"
    scene1 = gfx.Scene()
    scene2 = gfx.Scene()

    env1 = environment_manager.get_environment(renderer1, scene1)
    assert env1.hash in environment_manager.environments

    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1 is not env2
    assert env1.hash in environment_manager.environments
    assert env2.hash in environment_manager.environments

    return renderer1, renderer2, scene1, scene2, env1, env2


def test_environment_cleanup_noop():

    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1.hash in environment_manager.environments
    assert env2.hash in environment_manager.environments


def test_environment_cleanup_by_scene_del():

    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    del scene1
    gc.collect()

    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1.hash not in environment_manager.environments
    assert env2.hash in environment_manager.environments


def test_environment_cleanup_by_renderer_del():

    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    del renderer1
    gc.collect()

    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1.hash not in environment_manager.environments
    assert env2.hash in environment_manager.environments


def test_environment_cleanup_by_scene_change():

    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    # cannot test this yet, as currently the scene does not result in state


def test_environment_cleanup_by_renderer_change():

    renderer1, renderer2, scene1, scene2, env1, env2 = prepare_for_cleanup()

    renderer1.blend_mode = "weighted"

    # Rendering render2 doesnt check renderer1
    env2 = environment_manager.get_environment(renderer2, scene2)
    assert env1.hash in environment_manager.environments
    assert env2.hash in environment_manager.environments

    # But rendering with renderer1 does
    env3 = environment_manager.get_environment(renderer1, scene1)
    assert env1.hash not in environment_manager.environments
    assert env2.hash in environment_manager.environments
    assert env3.hash in environment_manager.environments
    assert env3 is not env1
    assert env3 is not env2


if __name__ == "__main__":
    pytest.main(["-x", __file__])
