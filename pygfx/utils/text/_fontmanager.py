"""
Module implementing the listing and detection of fonts.

Mostly a stub for now.

* Visvis uses a combo of fc-match and manual matching
  https://github.com/almarklein/visvis/blob/master/text/text_freetype.py#L326
* Vispy uses a mix of gdi32 on Windows, quartz on MacOS, fontconfig on Linux.
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
import os

import freetype


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
        self._index_available_fonts()

    def _index_available_fonts(self):

        # Collect font files
        # todo: does the order matter, or only for the default font?
        font_list = []

        fonts_dir = get_resource_filename("")
        font_files_in_dir = [
            os.path.join(fonts_dir, fname)
            for fname in sorted(os.listdir(fonts_dir))
            if fname.endswith((".ttf", ".otf"))
        ]

        # The default default font goes first
        filename = os.path.join(fonts_dir, "NotoSans-Regular.ttf")
        self._default_default_font = FontInfo(filename)
        font_list.append(self._default_default_font)

        # Then add all the other default font files
        for filename in font_files_in_dir:
            fi = FontInfo(filename)
            if (
                fi.name.startswith("Noto")
                and fi.name != self._default_default_font.name
            ):
                font_list.append(fi)

        # Then add the other font files
        for filename in font_files_in_dir:
            if not fi.name.startswith("Noto"):
                font_list.append(fi)

        name_to_font_info = {}
        default_fonts = []
        for fi in font_list:
            if fi.name not in name_to_font_info:
                name_to_font_info[fi.name] = fi
            if fi.name.startswith("Noto"):
                default_fonts.append(fi)

        # For our default font, create a dict mapping codepoints to font filenames
        # Could use a smart/sparse data structure here, but it's only for our default
        # font, so not that important (full Noto incl CJK would be about 2 MB). More
        # important is that it's fast to read from.
        default_font_map = {}

        for codepoint in self._default_default_font.codepoints:
            x = default_font_map.setdefault(codepoint, [])
            x.append(self._default_default_font)

        for fi in default_fonts:
            for codepoint in fi.codepoints:
                if codepoint not in self._default_default_font.codepoints:
                    x = default_font_map.setdefault(codepoint, [])
                    x.append(fi)

        self._font_list = font_list
        self._name_to_font_info = name_to_font_info
        self._default_font_map = default_font_map

    def get_default_font_props(self):
        return self._default_font_props  # todo: are font props immutable?

    def set_default_font(self, font_props):
        global default_font
        if not isinstance(font_props, FontProps):
            x = font_props.__class__.__name__
            raise TypeError(f"set_default_font() requires a FontProps object, not {x}")
        self._default_font_props = font_props

    def select_fonts_for_codepoint(self, codepoint, family):
        """Select the fonts that support the given codepoint."""
        familie_names = (family,) if isinstance(family, str) else tuple(family)
        fonts = []
        # Add fonts from given families that support the code point
        for family in familie_names:
            name = normalize_font_name(family)
            try:
                font_info = self._name_to_font_info[name]
            except KeyError:
                continue
            if font_info.has_codepoint(codepoint):
                fonts.append(font_info)
        # Add default font
        for font_info in self._default_font_map.get(codepoint, ()):
            fonts.append(font_info)
        if not fonts:
            fonts.append(self._default_default_font)
        return fonts

    def select_font_for_text(self, text, family):
        """Select the (best) font for the given text. Returns a list of (text, fontname)
        tuples, because different characters in the text may require different fonts.
        """
        # TODO: how to deal with codepoints that are not even in the default fonts
        text_pieces = []
        last_i, i = 0, 1
        fonts = self.select_fonts_for_codepoint(ord(text[0]), family)
        for i in range(1, len(text)):
            codepoint = ord(text[i])
            new_fonts = [font for font in fonts if font.has_codepoint(codepoint)]
            if new_fonts:
                fonts = new_fonts
            else:
                text_pieces.append((text[last_i:i], fonts[0]))
                last_i = i
                fonts = self.select_fonts_for_codepoint(codepoint, family)
        text_pieces.append((text[last_i : i + 1], fonts[0]))
        return text_pieces


def normalize_font_name(name):
    # Normalize the name - this could use some more sophistication
    name2 = name.split(".")[0]
    name3 = name2.replace("-", " ").replace("Regular", "")
    return "".join(part[0].upper() + part[1:] for part in name3.split())


class FontInfo:
    """Object to store information on a font file."""

    def __init__(self, filename):
        # Get name
        self._filename = filename
        self._name = normalize_font_name(os.path.basename(filename))

        # Collect codepoints
        face = freetype.Face(filename)
        # todo: use a data structure that stores codepoints more efficiently
        self._codepoints = set(i for i, _ in face.get_chars())

    def __repr__(self):
        return f"<FontInfo {self.name} at 0x{hex(id(self))}>"

    def __hash__(self):
        return id(self)

    @property
    def filename(self):
        return self._filename

    @property
    def name(self):
        return self._name

    @property
    def codepoints(self):
        return self._codepoints

    def has_codepoint(self, codepoint):
        return codepoint in self._codepoints


# Instantiate the global/default font manager
font_manager = FontManager()
