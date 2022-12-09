"""
Module implementing the listing and detection of fonts.

Mostly a stub for now. Needs cleanup once we know how to proceed ...

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
import os
import json

from .. import logger, get_resources_dir
from ._fontfinder import FontFile, get_all_fonts


# Weight names according to CSS and the OpenType spec.
weight_dict = {
    "thin": 100,
    "hairline": 100,
    "ultralight": 200,
    "extralight": 200,
    "light": 300,
    "normal": 400,
    "regular": 400,
    "medium": 500,
    "semibold": 600,
    "demibold": 600,
    "bold": 700,
    "extrabold": 800,
    "ultrabold": 800,
    "black": 900,
    "heavy": 900,
}


style_dict = {
    "normal": "normal",
    "regular": "normal",
    "italic": "italic",
    "oblique": "oblique",
    "slanted": "oblique",
}


class FontProps:
    """
    A class for storing font properties.
    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification.

    Parameters:
        family (str, tuple): The name of a font, or a list of font names
            in decreasing order of priority.
        style (str): Either 'normal', 'italic' or 'oblique'.
        weight (float, str): A numeric value in the range 100-900 or one of
            'ultralight', 'light', 'normal', 'regular', 'medium',
            'semibold', 'demibold', 'bold', 'extra bold', 'black'.
    """

    def __init__(
        self,
        family=None,
        *,
        style=None,
        weight=None,
    ):

        # Check family
        if family is None:
            family = font_manager.get_default_font_props().family
        else:
            if not isinstance(family, str):
                cls = type(family).__name__
                raise TypeError(f"Font family must be str, not '{cls}'")

        # Check style
        if style is None:
            style = font_manager.get_default_font_props().style
        else:
            if not isinstance(style, str):
                cls = type(style).__name__
                raise TypeError("Font style must be str, not '{cls}'")
            try:
                style = style_dict[style.lower()]
            except KeyError:
                raise TypeError(f"Style string not known: '{style}'")

        # Check weight
        if weight is None:
            weight = font_manager.get_default_font_props().weight
        else:
            if isinstance(weight, str):
                try:
                    weight = weight_dict[weight.lower()]
                except KeyError:
                    raise TypeError(f"Weight string not known: '{weight}'")
            elif isinstance(weight, int):
                weight = min(900, max(100, weight))
            else:
                raise TypeError("Weight must be an int (100-900) or a string.")

        self._kwargs = {
            "family": family,
            "style": style,
            "weight": weight,
        }

    def copy(self, **kwargs):
        """Make a copy of the font prop, with given kwargs replaced."""
        d = self._kwargs.copy()
        for k, v in kwargs.items():
            if v is not None:
                d[k] = v
        return self.__class__(**d)

    @property
    def family(self):
        """The font family, e.g. "NotoSans" or "Arial". Can also be a tuple
        to indicate fallback fonts.
        """
        return self._kwargs["family"]

    @property
    def style(self):
        """The style, one of "normal", "italic", or "oblique"."""
        return self._kwargs["style"]

    @property
    def weight(self):
        """The weight, as a number between 100-900."""
        return self._kwargs["weight"]


class FontManager:
    def __init__(self):
        self._default_font_props = FontProps(
            "noto sans",
            style="normal",
            weight="regular",
        )
        self._index_available_fonts()
        self._warned_for_codepoints = set()
        self._warned_for_font_names = set()

    def add_font_file(self, filename):
        ff = FontFile(filename)
        self._name_to_font[ff.name] = ff

    def _index_available_fonts(self):

        # Get a dict of FontFile objects
        self._name_to_font = {ff.name: ff for ff in get_all_fonts()}

        # Load default font index
        index_filename = os.path.join(get_resources_dir(), "noto_default_index.json")
        with open(index_filename, "rt", encoding="utf-8") as f:
            index = json.load(f)

        # Get default names (family_name-style_name), and create a map to lookup the fname
        default_names = [x.split(".")[0] for x in index["fonts"]]
        self._default_name_to_fname = {
            name: fname for name, fname in zip(default_names, index["fonts"])
        }

        # Create a dict that maps codepoint to tuple of names.
        self._default_font_map = {}
        for k, v in index["index"].items():
            codepoint = int(k)
            self._default_font_map[codepoint] = tuple(default_names[i] for i in v)

        # The main font of the default font is the fallback of fallbacks :)
        self._default_main_font = self._name_to_font["NotoSans-Regular"]

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
            name = "".join(x[0].upper() + x[1:] for x in family.split())
            if "-" not in name:
                name = name + "-Regular"
            try:
                ff = self._name_to_font[name]
            except KeyError:
                continue
            if ff.has_codepoint(codepoint):
                fonts.append(ff)
        # Add default font
        default_names = self._default_font_map.get(codepoint, ())
        for name in default_names:
            try:
                fonts.append(self._name_to_font[name])
            except KeyError:
                continue
        if not fonts:
            fonts.append(self._default_main_font)
            self._produce_font_warning(codepoint, default_names)
        return fonts

    def _produce_font_warning(self, codepoint, default_names):
        if codepoint not in self._warned_for_codepoints:
            self._warned_for_codepoints.add(codepoint)
            codepoint_repr = "U+" + hex(codepoint)[2:]
            if not default_names:
                msg = f"No font available for {chr(codepoint)} ({codepoint_repr})."
                logger.warning(msg)
            elif not any(name in self._warned_for_font_names for name in default_names):
                self._warned_for_font_names.update(default_names)
                msg = f"Fonts to support {chr(codepoint)} ({codepoint_repr}) can be installed via:\n"
                for name in default_names:
                    fname = self._default_name_to_fname[name]
                    msg += f"    https://pygfx.github.io/noto-mirror/#{fname}\n"
                logger.warning(msg)

    def select_font(self, text, family):
        """Select the (best) font for the given text. Returns a list of (text, fontname)
        tuples, because different characters in the text may require different fonts.
        """
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


# Instantiate the global/default font manager
font_manager = FontManager()
