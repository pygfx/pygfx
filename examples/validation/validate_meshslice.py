"""
Mesh slice
==========

Example to validate the mesh slice rendering.

This particular configuration is a sphere with a dent in it, which
displayed multiple different artifacts (missing line pieces and spurious
points) with the initial mesh-slice implementation. By making this a
validation example we avoid regressions w.r.t. these artifacts.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import base64

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)


# Data - base64 is a little more compact than array literals.

positions_encoded = """
AAAAAGKxtj7rJW8/AAAAAGKxtj7rJW+/AAAAAGKxtr7rJW8/AAAAAGKxtr7rJW+/6yVvPwAAAABi
sbY+6yVvPwAAAABisba+6yVvvwAAAABisbY+6yVvvwAAAABisba+YrG2Puslbz8AAAAAAXu2Pvgq
WL+fIuO4YrG2vuslbz8AAAAAAXu2vvgqWL+fIuO4UJYGPwAAAABAxFk/UJYGPwAAAABAxFm/UJYG
vwAAAABAxFk/UJYGvwAAAABAxFm/QMRZP1CWBj8AAAAAQMRZP1CWBr8AAAAAQMRZv1CWBj8AAAAA
QMRZv1CWBr8AAAAAAAAAAEDEWT9QlgY/AAAAAEDEWT9Qlga/AAAAAEDEWb9QlgY/DKc0r4twGb9u
6Ay/O80TPzvNEz87zRM/O80TPzvNEz87zRO/O80TPzvNE787zRM/O80TPzvNE787zRO/O80TvzvN
Ez87zRM/O80TvzvNEz87zRO/O80TvzvNE787zRM/O80TvzvNE787zRO/AAAAAAAAAAAAAIA/FRSO
Pp/cQD5wLHE/FRSOvp/cQD5wLHE/AAAAAMooIz/ZREU/ejeePgAAAD+9G08/ejeevgAAAD+9G08/
AAAAAAAAAAAAAIC/FRSOPp/cQD5wLHG/FRSOvp/cQD5wLHG/AAAAAMooIz/ZREW/ejeePgAAAD+9
G0+/ejeevgAAAD+9G0+/FRSOPp/cQL5wLHE/FRSOvp/cQL5wLHE/AAAAAMooI7/ZREU/ejeePgAA
AL+9G08/ejeevgAAAL+9G08/FRSOPp/cQL5wLHG/FRSOvp/cQL5wLHG/AAAAAJU7Hr+lykW/ejee
PgAAAL+9G0+/ejeevgAAAL+9G0+/AACAPwAAAAAAAAAA2URFPwAAAADKKCM/cCxxPxUUjj6f3EA+
cCxxPxUUjr6f3EA+vRtPP3o3nj4AAAA/vRtPP3o3nr4AAAA/2URFPwAAAADKKCO/cCxxPxUUjj6f
3EC+cCxxPxUUjr6f3EC+vRtPP3o3nj4AAAC/vRtPP3o3nr4AAAC/AACAvwAAAAAAAAAA2URFvwAA
AADKKCM/cCxxvxUUjj6f3EA+cCxxvxUUjr6f3EA+vRtPv3o3nj4AAAA/vRtPv3o3nr4AAAA/2URF
vwAAAADKKCO/cCxxvxUUjj6f3EC+cCxxvxUUjr6f3EC+vRtPv3o3nj4AAAC/vRtPv3o3nr4AAAC/
AAAAAAAAgD8AAAAAyigjP9lERT8AAAAAn9xAPnAscT8VFI4+n9xAPnAscT8VFI6+AAAAP70bTz96
N54+AAAAP70bTz96N56+8EOKLfYSF7/SF408yigjP9lERb8AAAAA5m1APraQXr+3JpA+/6U9PoFd
0L6mVJu+AAAAP70bT796N54+TNH/PntTQb8SCqC+yigjv9lERT8AAAAAn9xAvnAscT8VFI4+n9xA
vnAscT8VFI6+AAAAv70bTz96N54+AAAAv70bTz96N56+yigjv9lERb8AAAAA5m1AvraQXr+3JpA+
/6U9voFd0L6mVJu+AAAAv70bT796N54+TNH/vntTQb8SCqC+sA0VP0sHnD5I9UA/sA0VP0sHnL5I
9UA/sA0VP0sHnD5I9UC/sA0VP0sHnL5I9UC/sA0Vv0sHnD5I9UA/sA0Vv0sHnL5I9UA/sA0Vv0sH
nD5I9UC/sA0Vv0sHnL5I9UC/SPVAP7ANFT9LB5w+SPVAP7ANFT9LB5y+SPVAP7ANFb9LB5w+SPVA
P7ANFb9LB5y+SPVAv7ANFT9LB5w+SPVAv7ANFT9LB5y+SPVAv7ANFb9LB5w+SPVAv7ANFb9LB5y+
SwecPkj1QD+wDRU/Swecvkj1QD+wDRU/SwecPkj1QD+wDRW/Swecvkj1QD+wDRW/SwecPkj1QL+w
DRU/Swecvkj1QL+wDRU/CtKbPtugNb9T3xW/CtKbvtugNb9T3xW/
"""

faces_encoded = """
DAAAACEAAAAsAAAAAAAAACAAAAAhAAAAAgAAACwAAAAgAAAAIQAAACAAAAAsAAAADAAAACwAAABj
AAAAAgAAAC8AAAAsAAAAGgAAAGMAAAAvAAAALAAAAC8AAABjAAAADAAAAGMAAAA3AAAAGgAAADsA
AABjAAAABAAAADcAAAA7AAAAYwAAADsAAAA3AAAADAAAADcAAABiAAAABAAAADoAAAA3AAAAGAAA
AGIAAAA6AAAANwAAADoAAABiAAAADAAAAGIAAAAhAAAAGAAAACQAAABiAAAAAAAAACEAAAAkAAAA
YgAAACQAAAAhAAAADQAAADEAAAAnAAAAAwAAACYAAAAxAAAAAQAAACcAAAAmAAAAMQAAACYAAAAn
AAAADQAAACcAAABkAAAAAQAAACoAAAAnAAAAGQAAAGQAAAAqAAAAJwAAACoAAABkAAAADQAAAGQA
AAA8AAAAGQAAAD8AAABkAAAABQAAADwAAAA/AAAAZAAAAD8AAAA8AAAADQAAADwAAABlAAAABQAA
AEAAAAA8AAAAGwAAAGUAAABAAAAAPAAAAEAAAABlAAAADQAAAGUAAAAxAAAAGwAAADQAAABlAAAA
AwAAADEAAAA0AAAAZQAAADQAAAAxAAAADgAAAC0AAAAiAAAAAgAAACAAAAAtAAAAAAAAACIAAAAg
AAAALQAAACAAAAAiAAAADgAAACIAAABmAAAAAAAAACUAAAAiAAAAHAAAAGYAAAAlAAAAIgAAACUA
AABmAAAADgAAAGYAAABCAAAAHAAAAEUAAABmAAAABgAAAEIAAABFAAAAZgAAAEUAAABCAAAADgAA
AEIAAABnAAAABgAAAEYAAABCAAAAHgAAAGcAAABGAAAAQgAAAEYAAABnAAAADgAAAGcAAAAtAAAA
HgAAADAAAABnAAAAAgAAAC0AAAAwAAAAZwAAADAAAAAtAAAADwAAACgAAAAyAAAAAQAAACYAAAAo
AAAAAwAAADIAAAAmAAAAKAAAACYAAAAyAAAADwAAADIAAABpAAAAAwAAADUAAAAyAAAAHwAAAGkA
AAA1AAAAMgAAADUAAABpAAAADwAAAGkAAABHAAAAHwAAAEsAAABpAAAABwAAAEcAAABLAAAAaQAA
AEsAAABHAAAADwAAAEcAAABoAAAABwAAAEoAAABHAAAAHQAAAGgAAABKAAAARwAAAEoAAABoAAAA
DwAAAGgAAAAoAAAAHQAAACsAAABoAAAAAQAAACgAAAArAAAAaAAAACsAAAAoAAAAEAAAADgAAAA9
AAAABAAAADYAAAA4AAAABQAAAD0AAAA2AAAAOAAAADYAAAA9AAAAEAAAAD0AAABrAAAABQAAAD8A
AAA9AAAAGQAAAGsAAAA/AAAAPQAAAD8AAABrAAAAEAAAAGsAAABNAAAAGQAAAFEAAABrAAAACAAA
AE0AAABRAAAAawAAAFEAAABNAAAAEAAAAE0AAABqAAAACAAAAFAAAABNAAAAGAAAAGoAAABQAAAA
TQAAAFAAAABqAAAAEAAAAGoAAAA4AAAAGAAAADoAAABqAAAABAAAADgAAAA6AAAAagAAADoAAAA4
AAAAEQAAAD4AAAA5AAAABQAAADYAAAA+AAAABAAAADkAAAA2AAAAPgAAADYAAAA5AAAAEQAAADkA
AABsAAAABAAAADsAAAA5AAAAGgAAAGwAAAA7AAAAOQAAADsAAABsAAAAEQAAAGwAAABTAAAAGgAA
AFYAAABsAAAACQAAAFMAAABWAAAAbAAAAFYAAABTAAAAEQAAAFMAAABtAAAACQAAAFcAAABTAAAA
GwAAAG0AAABXAAAAUwAAAFcAAABtAAAAEQAAAG0AAAA+AAAAGwAAAEAAAABtAAAABQAAAD4AAABA
AAAAbQAAAEAAAAA+AAAAEgAAAEgAAABDAAAABwAAAEEAAABIAAAABgAAAEMAAABBAAAASAAAAEEA
AABDAAAAEgAAAEMAAABuAAAABgAAAEUAAABDAAAAHAAAAG4AAABFAAAAQwAAAEUAAABuAAAAEgAA
AG4AAABYAAAAHAAAAFsAAABuAAAACgAAAFgAAABbAAAAbgAAAFsAAABYAAAAEgAAAFgAAABvAAAA
CgAAAFwAAABYAAAAHQAAAG8AAABcAAAAWAAAAFwAAABvAAAAEgAAAG8AAABIAAAAHQAAAEoAAABv
AAAABwAAAEgAAABKAAAAbwAAAEoAAABIAAAAEwAAAEQAAABJAAAABgAAAEEAAABEAAAABwAAAEkA
AABBAAAARAAAAEEAAABJAAAAEwAAAEkAAABxAAAABwAAAEsAAABJAAAAHwAAAHEAAABLAAAASQAA
AEsAAABxAAAAEwAAAHEAAABdAAAAHwAAAGEAAABxAAAACwAAAF0AAABhAAAAcQAAAGEAAABdAAAA
EwAAAF0AAABwAAAACwAAAGAAAABdAAAAHgAAAHAAAABgAAAAXQAAAGAAAABwAAAAEwAAAHAAAABE
AAAAHgAAAEYAAABwAAAABgAAAEQAAABGAAAAcAAAAEYAAABEAAAAFAAAAE4AAABZAAAACAAAAEwA
AABOAAAACgAAAFkAAABMAAAATgAAAEwAAABZAAAAFAAAAFkAAABzAAAACgAAAFsAAABZAAAAHAAA
AHMAAABbAAAAWQAAAFsAAABzAAAAFAAAAHMAAAAjAAAAHAAAACUAAABzAAAAAAAAACMAAAAlAAAA
cwAAACUAAAAjAAAAFAAAACMAAAByAAAAAAAAACQAAAAjAAAAGAAAAHIAAAAkAAAAIwAAACQAAABy
AAAAFAAAAHIAAABOAAAAGAAAAFAAAAByAAAACAAAAE4AAABQAAAAcgAAAFAAAABOAAAAFQAAAFoA
AABPAAAACgAAAEwAAABaAAAACAAAAE8AAABMAAAAWgAAAEwAAABPAAAAFQAAAE8AAAB0AAAACAAA
AFEAAABPAAAAGQAAAHQAAABRAAAATwAAAFEAAAB0AAAAFQAAAHQAAAApAAAAGQAAACoAAAB0AAAA
AQAAACkAAAAqAAAAdAAAACoAAAApAAAAFQAAACkAAAB1AAAAAQAAACsAAAApAAAAHQAAAHUAAAAr
AAAAKQAAACsAAAB1AAAAFQAAAHUAAABaAAAAHQAAAFwAAAB1AAAACgAAAFoAAABcAAAAdQAAAFwA
AABaAAAAFgAAAF4AAABUAAAACwAAAFIAAABeAAAACQAAAFQAAABSAAAAXgAAAFIAAABUAAAAFgAA
AFQAAAB2AAAACQAAAFYAAABUAAAAGgAAAHYAAABWAAAAVAAAAFYAAAB2AAAAFgAAAHYAAAAuAAAA
GgAAAC8AAAB2AAAAAgAAAC4AAAAvAAAAdgAAAC8AAAAuAAAAFgAAAC4AAAB3AAAAAgAAADAAAAAu
AAAAHgAAAHcAAAAwAAAALgAAADAAAAB3AAAAFgAAAHcAAABeAAAAHgAAAGAAAAB3AAAACwAAAF4A
AABgAAAAdwAAAGAAAABeAAAAFwAAAFUAAABfAAAACQAAAFIAAABVAAAACwAAAF8AAABSAAAAVQAA
AFIAAABfAAAAFwAAAF8AAAB5AAAACwAAAGEAAABfAAAAHwAAAHkAAABhAAAAXwAAAGEAAAB5AAAA
FwAAAHkAAAAzAAAAHwAAADUAAAB5AAAAAwAAADMAAAA1AAAAeQAAADUAAAAzAAAAFwAAADMAAAB4
AAAAAwAAADQAAAAzAAAAGwAAAHgAAAA0AAAAMwAAADQAAAB4AAAAFwAAAHgAAABVAAAAGwAAAFcA
AAB4AAAACQAAAFUAAABXAAAAeAAAAFcAAABVAAAA
"""

positions_bytes = base64.decodebytes(positions_encoded.strip().encode())
faces_bytes = base64.decodebytes(faces_encoded.strip().encode())

positions = np.frombuffer(positions_bytes, np.float32).reshape(-1, 3)
faces = np.frombuffer(faces_bytes, np.int32).reshape(-1, 3)


# Vis

scene = gfx.Scene()

# Add color and texcoords. These are not used, but it means that all
# color_mode's are supported and can quickly be tested.
# NOTE: when showing a color per face, some line pieces are split into two colors.
# This is because the plane runs exactly over an edge in the mesh, so the observed
# line piece has contributions from two faces.
coords = np.linspace(0, 1, len(faces), dtype=np.float32)
colors = np.random.uniform(0, 1, (len(faces), 4)).astype(np.float32)
colors[:, 3] = 1
map = gfx.cm.viridis

mesh = gfx.Mesh(
    gfx.Geometry(positions=positions, indices=faces, colors=colors, texcoords=coords),
    gfx.MeshSliceMaterial(thickness=10, color="cyan", map=map, color_mode="uniform"),
)
mesh.material.plane = -1, 0, 0, 0  # yz
scene.add(mesh)

camera = gfx.OrthographicCamera(5, 5)
camera.show_object(scene, view_dir=(-1, 0, 0), up=(0, 0, 1))

canvas.request_draw(lambda: renderer.render(scene, camera))


@mesh.add_event_handler("pointer_move")
def show_pick_coords(e):
    print("face:", e.pick_info["face_index"], ":", e.pick_info["face_coord"])


if __name__ == "__main__":
    print(__doc__)
    loop.run()
