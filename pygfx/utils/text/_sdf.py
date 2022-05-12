import freetype
import numpy as np
import threading

# todo: this is just a test script and should be removed


class Font:
    def __init__(self, filename):

        self.filename = filename
        self.face = freetype.Face(filename)
        self.face.set_pixel_sizes(100, 100)
        # self.face.set_char_size(20 * 64)
        self._lock = threading.RLock()

    def shape(self, c):
        self.face.get_char_index(c)

    def render_bitmap(self, glyph_index, mode="normal"):
        render_mode = getattr(freetype, f"FT_RENDER_MODE_{mode.upper()}")
        with self._lock:
            self.face.load_char(glyph_index, freetype.FT_LOAD_DEFAULT)
            self.face.glyph.render(render_mode)
            bitmap = self.face.glyph.bitmap
            return np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)

    def render_sdf(self, glyph_index):
        return self.render_bitmap(glyph_index, "sdf")


f = Font("/Users/almar/dev/py/pygfx/pygfx/pkg_resources/NotoSans-Regular.ttf")
# f = Font("/Users/almar/dev/py/pygfx/pygfx/pkg_resources/DejaVuSans.ttf")
face = f.face
print(f.face.height)

a = f.render_sdf("W")
print(a.shape, f.face.get_advance(f.face.get_char_index("M"), 0))
# a = f.render_sdf("i")
print(a.shape, f.face.get_advance(f.face.get_char_index("i"), 0))

import imageio

imageio.imwrite("/Users/Almar/Desktop/tmp.png", a)

##
import matplotlib.pyplot as plt

plt.imshow(gfx.utils.text.atlas._array)
plt.show()
