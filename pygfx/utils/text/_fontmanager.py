import os
import json

from .. import logger, get_resources_dir
from ._fontfinder import FontFile, get_all_fonts, weight_dict, style_dict


# Allow "slanted" to be requested. It will request a regular font,
# and the glyphs will be slanted in the shader.
style_dict = style_dict.copy()
style_dict["slanted"] = "slanted"


class FontProps:
    """
    An object for storing font properties. Typically used as a request for an actual font.

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
            family = font_manager.default_font_props.family
        elif isinstance(family, str):
            family = (family,)
        elif isinstance(family, (tuple, list)):
            for x in family:
                if not isinstance(x, str):
                    cls = type(x).__name__
                    raise TypeError(f"Font family must be str, not '{cls}'")
            family = tuple(family)
        else:
            cls = type(family).__name__
            raise TypeError(f"Font family must be str or tuple-of-str, not '{cls}'")

        # Check style
        if style is None:
            style = font_manager.default_font_props.style
        elif isinstance(style, str):
            try:
                style = style_dict[style.lower()]
            except KeyError:
                raise TypeError(f"Style string not known: '{style}'")
        else:
            cls = type(style).__name__
            raise TypeError("Font style must be str, not '{cls}'")

        # Check weight
        if weight is None:
            weight = font_manager.default_font_props.weight
        elif isinstance(weight, str):
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

    def __repr__(self):
        fam = ", ".join(repr(x) for x in self.family)
        return f"<FontProps {fam} at 0x{hex(id(self))}>"

    def copy(self, **kwargs):
        """Make a copy of the font prop, with given kwargs replaced."""
        d = self._kwargs.copy()
        for k, v in kwargs.items():
            if v is not None:
                d[k] = v
        return self.__class__(**d)

    @property
    def family(self):
        """A tuple of font family strings, e.g. "Noto Sans" or "Arial"."""
        return self._kwargs["family"]

    @property
    def style(self):
        """The style, one of "normal", "italic", or "oblique"."""
        return self._kwargs["style"]

    @property
    def weight(self):
        """The font weight, as a number between 100-900."""
        return self._kwargs["weight"]


class FontManager:
    """Storage and discovery of text glyphs.

    The main purpose of the font manager is font selection, i.e. selecting a
    font based on the text to be rendered and a font_props object.

    This manager uses a default font set based on the Noto fonts, allowing a
    very complete Unicode coverage, including rare and ancient scripts. Users
    probably don't have the full set of Noto fonts installed. When a font is
    missing that the manager knows would support the text, a useful error
    message is produced, that includes a link to where the font can be installed
    from.

    There is a singleton instance of this class at ``pygfx.utils.text.font_manager``.
    """

    def __init__(self):
        self._default_font_props = FontProps((), style="normal", weight="regular")
        self._warned_for_codepoints = set()
        self._family_to_font = {}  # name -> style -> FontFile
        self._load_default_font_index()
        self._load_fonts()

    @property
    def default_font_props(self):
        """The default font properties."""
        # Note: could implement set_default_font(font_props) at some point ...
        return self._default_font_props

    def add_font_file(self, font_file):
        """Add the given font_file to the collection of fonts. The
        font_file can be a filename or a FontFile object. Returns the
        FontFile object for the font.
        """
        # Obtain FontFile object
        if isinstance(font_file, FontFile):
            ff = font_file
        elif isinstance(font_file, str):
            ff = FontFile(font_file)
        else:
            raise TypeError("add_font_file() expects FontFile or str filename.")

        # Select on family name
        variants = self._family_to_font.setdefault(ff.family, {})

        # Warn for duplicates
        if ff.variant in variants:
            old = variants[ff.variant]
            logger.debug(f"Duplicate font {ff.name} ({old.filename} -> {ff.filename})")

        # Store
        variants[ff.variant] = ff
        return ff

    def _load_fonts(self):
        # Populate the dict of font objects
        for ff in get_all_fonts():
            self.add_font_file(ff)

        # The main font of the default font is the fallback of fallbacks.
        # We copy the fontfile so we can detect when it's used to show tofu's.
        ff = self._family_to_font["Noto Sans"]["Regular"]
        self._fallback_font = FontFile(ff.filename, ff.family, ff.variant)

    def _load_default_font_index(self):
        # Load the json
        index_filename = os.path.join(get_resources_dir(), "noto_default_index.json")
        with open(index_filename, "rt", encoding="utf-8") as f:
            ob = json.load(f)
        families, fnames, index = ob["families"], ob["filenames"], ob["index"]

        # Create a map to lookup the fname from the family
        self._default_family_to_fname = {
            family: fname for family, fname in zip(families, fnames)
        }

        # Create a dict that maps codepoint to tuple of names
        self._default_font_map = {}
        for k, v in index.items():
            codepoint = int(k)
            self._default_font_map[codepoint] = tuple(families[i] for i in v)

    def get_fonts(self):
        """Get a list of all registered FontFile objects. E.g. to show a list
        of all available fonts:
        ``for ff in font_manager.get_fonts(): print(ff.family, "-", ff.variant)``
        """
        fonts = []
        for family in sorted(self._family_to_font.keys()):
            for ff in self._family_to_font[family].values():
                fonts.append(ff)
        return fonts

    def select_font(self, text, font_props):
        """Select the (best) fonts for the given text. Returns a list of (text, font_file)
        tuples, because different characters in the text may require different fonts.
        """

        # The selection strategy depends on whether preferred fonts are
        # given. If this is the case, the preferred font should be used
        # where possible. This applies per character. For the remaining
        # text, the default font is used. In this case, any font from
        # the default set is fine. We just try to make the pieces as
        # long as possible. For characters that we cannot render, the
        # fallback font is used - we need *a* font to render the tofu's ...

        fallback_font = self._fallback_font

        # Select fonts that match the preferred font_props
        preferred_fonts = []
        for family in font_props.family:
            # Get variants for the given family
            try:
                variants = self._family_to_font[family]
            except KeyError:
                continue
            # Select best variant
            best_score, best_ff = 0, None
            for ff in variants.values():
                score = (
                    int(font_props.style == ff.style)
                    + 600
                    - abs(ff.weight - font_props.weight)
                )
                if score > best_score:
                    best_score, best_ff = score, ff
            # Select
            if best_ff:
                preferred_fonts.append(best_ff)

        # First apply the most-preferred font to each character.
        # If not preferred_fonts are supplied, we can take a shortcut.
        text_pieces1 = []
        if preferred_fonts:
            codepoint = ord(text[0])
            last_font = self._select_preferred_font_for_codepoint(
                codepoint, preferred_fonts
            )
            last_i = i = 0
            for i in range(1, len(text)):
                codepoint = ord(text[i])
                font = self._select_preferred_font_for_codepoint(
                    codepoint, preferred_fonts
                )
                if font is not last_font:
                    text_pieces1.append((text[last_i:i], last_font))
                    last_i = i
                    last_font = font
            text_pieces1.append((text[last_i : i + 1], last_font))
        else:
            text_pieces1.append((text, None))

        failed_codepoints = []

        # Now process the pieces that don't have a font yet, using the default fonts.
        text_pieces2 = []
        for text, font in text_pieces1:
            if font is not None:
                text_pieces2.append((text, font))
            else:
                codepoint = ord(text[0])
                fonts = self._select_default_fonts_for_codepoint(codepoint)
                last_i = i = 0
                for i in range(1, len(text)):
                    codepoint = ord(text[i])
                    new_fonts = [ff for ff in fonts if ff.has_codepoint(codepoint)]
                    if new_fonts:
                        # Our selection of fonts is still nonzero
                        fonts = new_fonts
                    else:
                        if not fonts:
                            failed_codepoints.append(codepoint)
                            fonts = self._select_default_fonts_for_codepoint(codepoint)
                            if not fonts:
                                continue
                            last_font = fallback_font
                        else:
                            last_font = fonts[0]
                            fonts = self._select_default_fonts_for_codepoint(codepoint)
                        text_pieces2.append((text[last_i:i], last_font))
                        last_i = i
                last_font = fonts[0] if fonts else fallback_font
                text_pieces2.append((text[last_i : i + 1], last_font))

        # Did we encounter characters that we cannot render?
        failed_texts = [text for text, font in text_pieces2 if font is fallback_font]
        if failed_texts:
            codepoints = {ord(c) for c in "".join(failed_texts)}
            if any(cp not in self._warned_for_codepoints for cp in codepoints):
                self._warned_for_codepoints.update(codepoints)
                logger.warning(self._produce_font_warning(*failed_texts))

        return text_pieces2

    def _select_preferred_font_for_codepoint(self, codepoint, preferred_fonts):
        # Select one font, in order of preference
        for ff in preferred_fonts:
            if ff.has_codepoint(codepoint):
                return ff
        return None

    def _select_default_fonts_for_codepoint(self, codepoint):
        # Select all fonts capable of rendering this character
        fonts = []
        default_families = self._default_font_map.get(codepoint, ())
        for family in default_families:
            try:
                ff = self._family_to_font[family]["Regular"]
            except KeyError:
                continue
            if ff.has_codepoint(codepoint):
                fonts.append(ff)
        return fonts

    def _produce_font_warning(self, *failed_texts):
        # Get the codepoints that failed
        codepoints = list({ord(c) for c in "".join(failed_texts)})

        # Collect families. We try to find a set of families that supports all
        # the failed characters. In theory we could analyse things more and
        # show the minimal set of fonts that supports all chars.
        all_families = set()
        min_families = set(self._default_font_map.get(codepoints[0], ()))
        for cp in codepoints:
            families = self._default_font_map.get(cp, ())
            all_families.update(families)
            min_families.intersection_update(families)

        # Define what to report
        msg = f"Cannot render chars '{' '.join(failed_texts)}'. "
        fonts_to_link = set()
        if min_families:
            fonts_to_link = min_families
            if len(min_families) == 1:
                msg += "To fix this, install the following font:"
            else:
                msg += "To fix this, install any of the following fonts:"
        elif all_families:
            fonts_to_link = all_families
            msg += "To fix this, install (some) of the following fonts:"
        else:
            msg += "Even the Noto font set does not support these characters."

        # Show links
        msg += "\n"
        for family in sorted(fonts_to_link):
            fname = self._default_family_to_fname[family]
            msg += f"    https://pygfx.github.io/noto-mirror/#{fname}\n"

        # We return a string, making this method easy to test
        return msg


# Instantiate the global/default font manager
font_manager = FontManager()
