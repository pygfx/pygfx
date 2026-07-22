"""
PMREM Generator Test
====================

Test the PMREM generator by showing the cube texture of each mip level.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import imageio as iio
import cv2
import numpy as np
import pygfx as gfx
import wgpu
from pygfx.utils.pmrem_generator import generate_pmrem
from pygfx.renderers.wgpu.engine.shared import get_shared


def show_cube_texture(
    texture: gfx.Texture, mip_level, title: str = "Image", size=(1024, 768)
):
    """
    The layout of the cubemap looks like this:
    ┌────┬────┬────┬────┐
    │    │ +Y │    │    │
    ├────┼────┼────┼────┤
    │ -X │ +Z │ +X │ -Z │
    ├────┼────┼────┼────┤
    │    │ -Y │    │    │
    └────┴────┴────┴────┘
    """

    device = get_shared().device
    bytes_per_pixel = 8  # rgba16float
    mip_size = texture.size[0] // (2**mip_level)

    data = device.queue.read_texture(
        {
            "texture": texture._wgpu_object,
            "mip_level": mip_level,
            "origin": (0, 0, 0),
        },
        {
            "offset": 0,
            "bytes_per_row": bytes_per_pixel * mip_size,
            "rows_per_image": mip_size,
        },
        (mip_size, mip_size, 6),
    )
    data = np.frombuffer(data, np.float16).reshape(6, mip_size, mip_size, 4)

    # linear -> sRGB
    data = np.where(
        data <= 0.0031308, data * 12.92, 1.055 * np.power(data, 1 / 2.4) - 0.055
    )

    # to unit8
    data = (data * 255).astype(np.uint8)

    posx, negx, posy, negy, posz, negz = data

    h, w, c = posx.shape[:3]

    big_image = np.zeros((3 * h, 4 * w, c), dtype=data.dtype)

    big_image[0:h, w : 2 * w] = posy
    big_image[h : 2 * h, 0:w] = negx
    big_image[h : 2 * h, w : 2 * w] = posz
    big_image[h : 2 * h, 2 * w : 3 * w] = posx
    big_image[h : 2 * h, 3 * w : 4 * w] = negz
    big_image[2 * h : 3 * h, w : 2 * w] = negy

    big_image = cv2.resize(big_image, size)

    # rgba -> bgra
    big_image = big_image[..., [2, 1, 0, 3]]
    cv2.imshow(title, big_image)


if __name__ == "__main__":
    # Read cube image and turn it into a 3D image (a 4d array)
    env_img = iio.imread("imageio:meadow_cube.jpg")
    cube_size = env_img.shape[1]
    env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

    # Create environment map
    env_tex = gfx.Texture(
        env_img,
        dim=2,
        size=(cube_size, cube_size, 6),
        usage=wgpu.TextureUsage.COPY_SRC,
        colorspace="srgb",
        generate_mipmaps=False,
    )

    pmrem_texture, cube_texture = generate_pmrem(env_tex)

    for i in range(6):
        show_cube_texture(pmrem_texture, i, f"mip_{i}", size=(512, 384))

    cv2.waitKey(0)
