"""
GPU selection
=============

Demonstrate how a specific GPU can be selected, or how a GPU can be
selected using a power preference.

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import wgpu
import pygfx as gfx


# You can express whether you prefer a "high-performance" (default) or "low-power" device.
# Note that the power preference is ignored if select_adapter() is used.
gfx.renderers.wgpu.select_power_preference("high-performance")

# Get a list of all adapters from wgpu
adapters = wgpu.gpu.enumerate_adapters_sync()

# Show the options
print("Available adapters:")
for a in adapters:
    print(a.summary)

# The best way to select an adapter is highly dependent on the use-case.
# Here we prefer a Geforce GPU, but fallback to the default if its not available.
adapters_geforce = [a for a in adapters if "geforce" in a.summary.lower()]
if adapters_geforce:
    gfx.renderers.wgpu.select_adapter(adapters_geforce[0])


# Draw a cube
cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(cube)

    # Show what adapter was selected
    wgpu.diagnostics.pygfx_adapter_info.print_report()
