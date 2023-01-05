"""

Spotlights
==========


Spotlights example.
"""
# sphinx_gallery_pygfx_render = True

import time
import math
import random

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


def update_value(target, dist):
    for k, v in dist.items():
        if isinstance(v, dict):
            update_value(getattr(target, k), v)
        else:
            setattr(target, k, v)

    return dist


def get_value(target, template):
    vals = {}
    for k, v in template.items():
        if isinstance(v, dict):
            vals[k] = get_value(getattr(target, k), v)
        else:
            vals[k] = getattr(target, k)

    return vals


def lerp(start_vals, to_vals, t):
    current_vals = {}
    for key, start_value in start_vals.items():
        if isinstance(start_value, (int, float)):
            to_value = to_vals[key]
            value = start_value + (to_value - start_value) * t
            current_vals[key] = value

        elif isinstance(start_value, dict):
            current_vals[key] = lerp(start_value, to_vals[key], t)

    return current_vals


class Tween:
    def __init__(self, target):
        self.target = target
        self.starttime = 0
        self._lerp_func = None

    def to(self, to: dict, duration):
        self.to_val = to
        self.duration = duration
        self.starttime = time.time()
        self.start_val = get_value(self.target, to)
        return self

    def lerp_func(self, func):
        self._lerp_func = func
        return self

    def update(self):
        t = (time.time() - self.starttime) / self.duration
        if t < 1:
            if self._lerp_func:
                t = self._lerp_func(t)
            current_val = lerp(self.start_val, self.to_val, t)
            update_value(self.target, current_val)

    @property
    def since_last_start(self):
        return time.time() - self.starttime


def init_scene():
    renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())

    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(35, 16 / 9, 1, 2000)
    camera.position.set(46, 22, -21)

    controller = gfx.OrbitController(camera.position.clone())
    controller.add_default_event_handlers(renderer, camera)

    controller.target.set(0, 7, 0)

    floor = gfx.Mesh(
        gfx.plane_geometry(2000, 2000),
        gfx.MeshPhongMaterial(color="#808080", side="Front"),
    )

    floor.rotation.set_from_euler(gfx.linalg.Euler(-math.pi / 2))
    floor.position.set(0, -0.05, 0)

    ambient = gfx.AmbientLight("#111111")

    def create_spot_light(color) -> gfx.SpotLight:
        return gfx.SpotLight(color, 2000, angle=0.3, penumbra=0.2, decay=2)

    spot_light1 = create_spot_light("#ff7f00")
    spot_light2 = create_spot_light("#00ff7f")
    spot_light3 = create_spot_light("#7f00ff")

    spot_light1.position.set(15, 40, 45)
    spot_light2.position.set(0, 40, 35)
    spot_light3.position.set(-15, 40, 45)

    spot_light1.add(gfx.SpotLightHelper())
    spot_light2.add(gfx.SpotLightHelper())
    spot_light3.add(gfx.SpotLightHelper())

    scene.add(floor)
    scene.add(ambient)
    scene.add(spot_light1, spot_light2, spot_light3)

    tweens = [Tween(spot_light1), Tween(spot_light2), Tween(spot_light3)]

    def animate():
        controller.update_camera(camera)

        for tween in tweens:
            if tween.since_last_start > 5:
                tween.to(
                    {
                        "angle": random.random() * 0.7 + 0.1,
                        "penumbra": random.random() + 1,
                        "position": {
                            "x": random.random() * 30 - 15,
                            "y": random.random() * 10 + 15,
                            "z": random.random() * 30 - 15,
                        },
                    },
                    random.random() * 3 + 2,
                ).lerp_func(lambda t: t * (2 - t))

            tween.update()

        renderer.render(scene, camera)
        renderer.request_draw()

    renderer.request_draw(animate)
    return renderer


if __name__ == "__main__":
    renderer = init_scene()
    run()
