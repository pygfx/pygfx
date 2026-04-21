"""
World map projections
=====================

Example showing a world map with different projections.

The idea is that you can switch between different map projections like
Mercator, Hammer, etc, and also 3D spherical model.

The cool thing is that you can express all your data in lat/lon, and it will
be nicely projected together.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'


################################################################################
# .. note::
#
#   To run this example, you need a model from the source repo's example
#   folder. If you are running this example from a local copy of the code (dev
#   install) no further actions are needed. Otherwise, you may have to replace
#   the path below to point to the location of the model.

import os
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"


################################################################################

import time

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


# Load public domain coastline data from naturalearthdata.com.
# Original source: https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip
# Transformed to a numpy array using geopandas and then stored as npz
lonlat = np.load(model_dir / "coastlines.npz")["lonlat"]


# The names of the transforms in this example.
# For each transform, there is a corresponding shader in the WGSL code below.
transform_names = [
    "mercator",
    "hammer",
    "winkel_tripel",
    "azimuthal_equidistant",
    "transverse_mercator",
    "sphere",
]


WGSL = """

fn mercator_projection(pos: vec3f) -> vec3f {
    // Source: Wikipedia - this is web-mercator
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    let max_lat: f32 = 1.4844222;  // = 2*atan(e**pi)-pi/2 ≈ 85°
    let clamped_lat = clamp(lat, -max_lat, max_lat);
    let y = log(tan(0.25 * PI + 0.5 * clamped_lat));
    return vec3f(lon, y, pos.z);
}


fn hammer_projection(pos: vec3f) -> vec3f {
    // Source: Glumpy
    let B = 2.0;
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    let cos_lat = cos(lat);
    let sin_lat = sin(lat);
    let cos_lon = cos(lon/B);
    let sin_lon = sin(lon/B);
    let d = sqrt(1.0 + cos_lat * cos_lon);
    let x = (B * 1.4844222 * cos_lat * sin_lon) / d;
    let y =     (1.4844222 * sin_lat) / d;
    return vec3f(x, y, pos.z);
}


fn winkel_tripel_projection(pos: vec3f) -> vec3f {
    // Source: Wikipedia
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    let x1 = lon * cos(lat / 2.0);
    let y1 = lat;
    let alpha = acos(cos(lat) * cos(lon / 2.0));
    let sinc = select(1.0, sin(alpha) / alpha, alpha > 1e-6);
    let x2 = 2.0 * cos(lat) * sin(lon / 2.0) / sinc;
    let y2 = sin(lat) / sinc;
    let x = 0.5 * (x1 + x2);
    let y = 0.5 * (y1 + y2);
    return vec3f(x, y, pos.z);
}


fn azimuthal_equidistant_projection(pos: vec3f) -> vec3f {
    // Source: ChatGTP
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let c = (3.14159265359 * 0.5) - lat;
    let k = c / sin(c);
    let x = k * cos_lat * sin(lon);
    let y = -k * cos_lat * cos(lon);
    return vec3f(x, y, pos.z);
}


const k0 = 0.75;
const a  = 1.00;

fn cosh(x: f32) -> f32 { return 0.5 * (exp(x)+exp(-x)); }
fn sinh(x: f32) -> f32 { return 0.5 * (exp(x)-exp(-x)); }

fn nonlin_forward(lambda: f32, phi: f32) -> vec2f {
    let x = 0.5*k0*log((1.0+sin(lambda)*cos(phi)) / (1.0 - sin(lambda)*cos(phi)));
    let y = k0*a*atan2(tan(phi), cos(lambda));
    return vec2f(x, y);
}

fn transverse_mercator_projection(pos: vec3f) -> vec3f {
    // Source: Glumpy
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    return vec3f(nonlin_forward(lon, lat), pos.z);
}


fn sphere_projection(pos: vec3f) -> vec3f {
    let lon = pos.x * PI / 180;
    let lat = pos.y * PI / 180;
    let elevation = pos.z;
    let clat = cos(lat);
    //let radius = 6367444.0 + elevation; // meters
    let radius = 2.0 + elevation;  // relative units
    return vec3f(
        radius * clat * cos(lon),
        radius * clat * sin(lon),
        radius * sin(lat),
    );
}


// Combining the above
fn nonlinear_transform(pos: vec3f) -> vec3f {
    return (
        COMBINE
    );
}

""".replace(
    "COMBINE",
    " + ".join(
        f"{name}_projection(pos) * u_wobject.{name}_factor" for name in transform_names
    ),
)


class MapProjectedPoints(gfx.Points):
    """Points subclass with support for multiple map transforms, and
    the ability to smoothly transition between them, using a uniform
    buffer with factors for each transform.
    """

    uniform_type = dict(
        gfx.Mesh.uniform_type,
        **{n + "_factor": "f4" for n in transform_names},
    )

    transition_time = 2.0  # seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nonlinear_transform = WGSL

        self._old_factors = {n: 0.0 for n in transform_names}
        self._new_factors = {n: 0.0 for n in transform_names}
        self.uniform_buffer.data["mercator_factor"] = 1.0
        self._set_time = 0

    def _update_object(self):
        super()._update_object()

        time_delta = time.perf_counter() - self._set_time
        f = min(1.0, time_delta / self.transition_time)

        for name in transform_names:
            v1 = self._old_factors[name]
            v2 = self._new_factors[name]
            self.uniform_buffer.data[name + "_factor"] = (1 - f) * v1 + f * v2

        self.uniform_buffer.update_full()

    def select_projection(self, name):
        assert name in transform_names
        self._old_factors = {
            n: self.uniform_buffer.data[n + "_factor"] for n in transform_names
        }
        self._new_factors = {n: 0.0 for n in transform_names}
        self._new_factors[name] = 1.0
        self._set_time = time.perf_counter()


# Setup visuzaliation

canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

camera = gfx.OrthographicCamera(maintain_aspect=True)
camera.show_object((0, 0, 0, 3), up=(0, 0, 1))
controller = gfx.OrbitController(camera, register_events=renderer)


# Add points for the lon/lat lines

density = 10

point_positions1 = []
for lat in range(-90, 90, 10):
    x = np.linspace(-180, 180, 360 * density, dtype=np.float32)
    y = np.full_like(x, lat)
    point_positions1.append(np.column_stack([x, y, np.zeros_like(x)]))
for lon in range(-180, 180, 10):
    y = np.linspace(-90, 90, 180 * density, dtype=np.float32)
    x = np.full_like(y, lon)
    point_positions1.append(np.column_stack([x, y, np.zeros_like(x)]))
point_positions1 = np.vstack(point_positions1)

points1 = MapProjectedPoints(
    gfx.Geometry(positions=point_positions1),
    gfx.PointsMaterial(size=1.0, color="#777"),
)
scene.add(points1)


# Add points for the coastlines. Turn line-segments into pointset.

point_positions2 = []
for i in range(lonlat.shape[0] - 1):
    p1, p2 = lonlat[i], lonlat[i + 1]
    if np.isnan(p1[0]) or np.isnan(p2[0]):
        continue
    dist = np.linalg.norm(p2 - p1)
    n = int(np.ceil(dist * density))
    x = np.linspace(p1[0], p2[0], n, np.float32)
    y = np.linspace(p1[1], p2[1], n, np.float32)
    point_positions2.append(np.column_stack([x, y, np.zeros_like(x)]))
point_positions2 = np.vstack(point_positions2)

points2 = MapProjectedPoints(
    gfx.Geometry(positions=point_positions2),
    gfx.PointsMaterial(size=3.0, color="#aaf"),
)
scene.add(points2)


# A scene for the text overlay

screen_scene = gfx.Scene()
screen_camera = gfx.ScreenCoordsCamera(invert_y=True)
for i, name in enumerate(transform_names):
    t = gfx.Text(
        text=f"{i + 1}: {name}", screen_space=True, font_size=16, anchor="top-left"
    )
    t.local.position = 10, i * 20, 0
    screen_scene.add(t)


# Event handling


@canvas.add_event_handler("pointer_move")
def on_move(event):
    if event["x"] < 200 and event["y"] < len(transform_names) * 20:
        canvas.set_cursor("pointer")
    else:
        canvas.set_cursor("default")


@canvas.add_event_handler("pointer_down")
def on_click(event):
    if event["x"] < 200:
        index = int(event["y"] / 20)
        set_projection(index)


@canvas.add_event_handler("key_down")
def on_key(event):
    if event["key"] in "123456789":
        index = int(event["key"]) - 1
        set_projection(index)


def set_projection(index):
    if index < len(transform_names):
        transform = transform_names[index]
        for i, name in enumerate(transform_names):
            screen_scene.children[i].set_text(f"{i + 1}: {name}")
        screen_scene.children[index].set_markdown(f"**{index + 1}: {transform}**")
        for ob in [points1, points2]:
            ob.select_projection(transform)
        canvas.request_draw()


set_projection(0)


def animate():
    renderer.render(scene, camera, flush=False)
    renderer.render(screen_scene, screen_camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
