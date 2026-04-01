"""Example showing a world map.

The idea is that you can switch between different map projections like
Mercator, Hammer, etc, and also 3D spherical and ellipsoid.

The cool think is that you can express all your data in lat/lon, and it will
be nicely projected along.
"""


# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


# Read data from CSV
# Original source: https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip

# import geopandas as gpd
# import pandas as pd
#
# # Download Natural Earth coastline (low-res ~110m)
# url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip"
#
# gdf = gpd.read_file(url)
#
# # Break multi-lines into simple segments
# gdf = gdf.explode(index_parts=False)
#
# rows = []
# for geom in gdf.geometry:
#     coords = list(geom.coords)
#     for lon, lat in coords:
#         rows.append((lon, lat))
#     rows.append((None, None))  # separator between segments
#
# df = pd.DataFrame(rows, columns=["lon", "lat"])
#
# # Optional: drop separators if you don't need segmentation
# # df = df.dropna()
#
# df.to_csv("real_coastline.csv", index=False)

longlat = []
with open("real_coastline.csv", "rt") as f:
    for line in f.readlines():
        if line.startswith("lon") or "," not in line:
            continue
        parts = line.strip().split(",")
        assert len(parts) == 2
        try:
            val = float(parts[0]), float(parts[1])
        except ValueError:
            val = np.nan, np.nan
        longlat.append(val)

longlat = np.array(longlat, np.float32)


# Setup visuzaliation

canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background.from_color("#000")


long = longlat[:, 0]
lat = longlat[:, 1]
positions = np.column_stack([long, lat, np.zeros_like(long)])

line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(thickness=4.0, color="#aaf"),
)
scene.add(background, line)

camera = gfx.OrthographicCamera(maintain_aspect=True, nonlinear="ll2sphere")

camera.show_object((0, 0, 0, 60_00_000))

controller = gfx.OrbitController(camera, register_events=renderer)



def animate():
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
