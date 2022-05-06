"""
Module implementing the listen and detection of fonts.

Mostly a stub for now.

* Visvis uses a combo of fc-match and manual matching
  https://github.com/almarklein/visvis/blob/master/text/text_freetype.py#L326
* Vipsy uses a mix of gdi32 on Windows, quartz on MacOS, fontconfig on Linux.
    https://github.com/vispy/vispy/blob/main/vispy/util/fonts/_win32.py
    https://github.com/vispy/vispy/blob/main/vispy/util/fonts/_freetype.py
* Matplotlib uses a platform agnostic approach that searches for fonts in known
  locations and scores each font based on its name. More manual work, but feels
  the most stable.
  https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/font_manager.py

So far this is a minimal version of the MPL approach.
"""

from .. import get_resource_filename


# todo: math fonts? (MPL makes that part of the FontProperties)
# todo: caching


class FontProps:
    """
    A class for storing font properties.
    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification.

    Parameters:
        family (str, tuple): The name of a font, or a list of font names
            in decreasing order of priority. The items may include a
            generic font family name, either 'sans-serif', 'serif',
            'cursive', 'fantasy', or 'monospace'.
        style (str): Either 'normal', 'italic' or 'oblique'.
        variant (str): Either 'normal' or 'small-caps'.
        stretch (float, str): A numeric value in the range 0-1000 or one of
            'ultra-condensed', 'extra-condensed', 'condensed',
            'semi-condensed', 'normal', 'semi-expanded', 'expanded',
            'extra-expanded' or 'ultra-expanded'.
        weight (float, str): A numeric value in the range 0-1000 or one of
            'ultralight', 'light', 'normal', 'regular', 'book',
            'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
            'heavy', 'extra bold', 'black'.
        size (float, str): Either a relative value of 'xx-small', 'x-small',
            'small', 'medium', 'large', 'x-large', 'xx-large' or an
            absolute font size, e.g., 10.
    """

    def __init__(
        self,
        family=None,
        *,
        style=None,
        variant=None,
        stretch=None,
        weight=None,
        size=None,
    ):
        if family is None:
            family = font_manager.get_default_font_props().family
        if style is None:
            style = font_manager.get_default_font_props().style
        if variant is None:
            variant = font_manager.get_default_font_props().variant
        if stretch is None:
            stretch = font_manager.get_default_font_props().stretch
        if weight is None:
            weight = font_manager.get_default_font_props().weight
        if size is None:
            size = font_manager.get_default_font_props().size

        # todo: checks
        # todo: use @properties
        self.family = family
        self.style = style
        self.variant = variant
        self.stretch = stretch
        self.weight = weight
        self.size = size


class FontManager:
    def __init__(self):
        self._default_font_props = FontProps(
            "noto sans",
            style="normal",
            variant="normal",
            stretch="normal",
            weight="normal",
            size=12,
        )

    def get_default_font_props(self):
        return self._default_font_props  # todo: are font props immutable?

    def set_default_font(self, font_props):
        global default_font
        if not isinstance(font_props, FontProps):
            x = font_properties.__class__.__name__
            raise TypeError(f"set_default_font() requires a FontProps object, not {x}")
        self._default_font_props = font_props

    def find_font(self, font_props):
        """Find the font that best matches the given properties. Returns the filename
        for the corresponding font file.
        """
        # todo: not implemented :)
        return get_resource_filename("NotoSans-Regular.ttf")


# Instantiate the global/default font manager
font_manager = FontManager()
find_font = font_manager.find_font
