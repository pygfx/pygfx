"""
A Method to Subtract Background from an Image
================================================

This example shows how to subtract a background from an image.

The background is created via the mipmap levels of the image so that the background does not need to be
computed per pixel during each render cycle. The mipmap level is sampled from to create the background.

One can adjust the background level via the dropdown menu. Each level subtracts a subsampled background
from the image with the subsampling increasing by a power of 2 with each level. Level 0 is the original
image with no background subtraction.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


# Load image
im = iio.imread("imageio:hubble_deep_field.png").astype(np.float32)
X, Y = np.meshgrid(
    np.arange(im.shape[1]) - im.shape[1] / 2, np.arange(im.shape[0]) - im.shape[0] / 2
)
radius = np.sqrt(X**2 + Y**2)
im *= 1 - radius[..., np.newaxis] / radius.max()
im = im.astype(np.uint8)

canvas_size = im.shape[0], im.shape[1]
canvas = WgpuCanvas(size=canvas_size, max_fps=999)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
camera = gfx.OrthographicCamera(canvas_size[0], canvas_size[1])
camera.local.y = canvas_size[1] / 2
camera.local.scale_y = -1
camera.local.x = canvas_size[0] / 2
controller = gfx.PanZoomController(camera, register_events=renderer)


image_texture = gfx.Texture(im, dim=2, generate_mipmaps=True)


class BackGroundRemovedImageMaterial(gfx.ImageBasicMaterial):
    """
    An image that has the background removed.
    """

    uniform_type = dict(
        gfx.ImageBasicMaterial.uniform_type,
        background_level="u4",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_level = 0

    @property
    def background_level(self):
        return self.uniform_buffer.data["background_level"]

    @background_level.setter
    def background_level(self, value):
        self.uniform_buffer.data["background_level"] = int(value)
        self.uniform_buffer.update_range()


@register_wgpu_render_function(gfx.Image, BackGroundRemovedImageMaterial)
class BackGroundRemovedImageShader(ImageShader):
    def __init__(self, wobject):
        super().__init__(wobject)

    def get_code(self):
        code = super().get_code()
        code = code.replace(
            """
    let value = sample_im(varyings.texcoord.xy, sizef);
""",
            """
    // Get the image value via bilinear interpolation
    var value: vec4<f32> = textureSampleLevel(t_img, s_img, varyings.texcoord.xy, 0.);

    let background_level = f32(u_material.background_level);
    if background_level != 0.0 {
        // Get the background value via bilinear interpolation
        let background = textureSampleLevel(t_img, s_img, varyings.texcoord.xy, background_level);

        value = vec4<f32>(
            (value.rgb - background.rgb),
            value.a
        );
    }
""",
        )
        return code


image = gfx.Image(
    gfx.Geometry(grid=image_texture),
    BackGroundRemovedImageMaterial(clim=(0, 255), interpolation="linear"),
)
scene.add(image)

current_background_index = 0
current_image_index = 0


def draw_imgui():
    global current_background_index
    global current_image_index
    global im, image_texture

    imgui.new_frame()

    imgui.set_next_window_size((400, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)

    if is_expand:
        background_levels = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        # Background level selection dropdown
        changed, current_background_index = imgui.combo(
            "Background Level",
            current_background_index,
            background_levels,
            len(background_levels),
        )

        if changed:
            image.material.background_level = np.int32(
                background_levels[current_background_index]
            )

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


# Create GUI renderer
gui_renderer = ImguiRenderer(renderer.device, canvas)


def animate():
    renderer.render(scene, camera)
    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


gui_renderer.set_gui(draw_imgui)

if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
