import time

from pytest import raises

from pygfx.utils.text import font_manager, FontManager, FontProps, FontFile


def test_font_props():
    # Defaults
    fp = FontProps()
    assert fp.family == ()
    assert fp.weight == 400
    assert fp.style == "normal"

    fp = FontProps("Arial", weight="bold", style="oblique")
    assert fp.family == ("Arial",)
    assert fp.weight == 700
    assert fp.style == "oblique"
    assert "Arial" in repr(fp)

    fp = FontProps(("Arial", "Tahoma"), weight=250, style="regular")
    assert fp.family == ("Arial", "Tahoma")
    assert fp.weight == 250
    assert fp.style == "normal"
    assert "Arial" in repr(fp) and "Tahoma" in repr(fp)

    # Fails
    with raises(TypeError):
        FontProps(3)
    with raises(TypeError):
        FontProps(("Arial", 3))
    with raises(TypeError):
        FontProps("Arial", weight=())
    with raises(TypeError):
        FontProps("Arial", weight="notaweight")
    with raises(TypeError):
        FontProps("Arial", style=())
    with raises(TypeError):
        FontProps("Arial", style="notastyle")


def test_select_font():
    # A simple text, that can be rendered with the main font
    text = "HelloWorld"

    pieces = font_manager.select_font(text, FontProps())
    assert len(pieces) == 1
    assert isinstance(pieces[0], tuple)
    assert pieces[0][0] == text
    assert isinstance(pieces[0][1], FontFile)

    # A text with both Latin and Arabic, needs two fonts
    text = "Hello World مرحبا بالعالم"

    pieces = font_manager.select_font(text, FontProps())
    assert len(pieces) == 2
    assert isinstance(pieces[0], tuple)
    assert isinstance(pieces[1], tuple)
    assert isinstance(pieces[0][1], FontFile)
    assert isinstance(pieces[1][1], FontFile)
    assert pieces[0][0] == "Hello World "
    assert pieces[1][0] == "مرحبا بالعالم"

    # this test ensures there isn't any of by one error in the font selection
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(1, 26):
        this_text = text[:i]
        pieces = font_manager.select_font(this_text, FontProps("Arial"))
        assert pieces[0][0] == this_text


def test_font_fallback1():
    # The preferred order of fonts. The default font is implicitly appended.
    # The nonexistint font will always be skipped.
    families = "DoesNotExist Sans", "Humor Sans"

    # Simple, select Humor Sans
    text = "Hello"
    pieces = font_manager.select_font(text, FontProps(families))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Humor Sans"

    # Humor Sans only supports Latin, so fallback here.
    text = "Привет"
    pieces = font_manager.select_font(text, FontProps(families))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans"

    #  Humor Sans will be used for the latic chars, and Noto Sans
    # for the Cyrillic chars.
    text = "Hello Привет"
    pieces = font_manager.select_font(text, FontProps(families))
    assert len(pieces) == 2
    assert pieces[0][1].family == "Humor Sans"
    assert pieces[1][1].family == "Noto Sans"

    # Make it a bit harder by including Arabic
    text = "Hello Привет مرحبا"
    pieces = font_manager.select_font(text, FontProps(families))
    assert len(pieces) == 4
    assert pieces[0][1].family == "Humor Sans"
    assert pieces[1][1].family == "Noto Sans"
    assert pieces[3][1].family == "Noto Sans Arabic"
    # Yes, four pieces, because the space is in Humor Sans
    assert pieces[2][1].family == "Humor Sans"
    assert pieces[2][0] == " "


def test_font_fallback2():
    # A text that requires the Arabic font, with a dash in between.
    # That dash is supported by the Arabic font, but also by Noto Sans.
    text = "مرحبا-مرحبا"

    # If we use the default fonts, we get a single piece
    pieces = font_manager.select_font(text, FontProps())
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans Arabic"

    # Default font rules apply when preferred font does not "kick in"
    pieces = font_manager.select_font(text, FontProps("NotAnActualFont"))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans Arabic"

    # However, if a preferred font is given, the select_font() tries
    # to use that wherever it can, so also for the dash.
    pieces = font_manager.select_font(text, FontProps("Humor Sans"))
    assert len(pieces) == 3
    assert pieces[0][1].family == "Noto Sans Arabic"
    assert pieces[1][1].family == "Humor Sans"
    assert pieces[1][0] == "-"
    assert pieces[2][1].family == "Noto Sans Arabic"

    # Also if the preferred font is the main default
    pieces = font_manager.select_font(text, FontProps("Noto Sans"))
    assert len(pieces) == 3
    assert pieces[0][1].family == "Noto Sans Arabic"
    assert pieces[1][1].family == "Noto Sans"
    assert pieces[1][0] == "-"
    assert pieces[2][1].family == "Noto Sans Arabic"

    # But not when we prefer Arabic
    pieces = font_manager.select_font(text, FontProps("Noto Sans Arabic"))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans Arabic"

    # Also not when we use the default name
    # Ok, this works simply because "sans" in not a valid family name
    pieces = font_manager.select_font(text, FontProps("sans"))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans Arabic"


def test_tofu():
    fallback_font = font_manager._fallback_font

    # Multiple unsupported chars should be kept together
    text = "\uf000\uf001\uf002"
    pieces = font_manager.select_font(text, FontProps())
    assert len(pieces) == 1
    assert pieces[0][1] is fallback_font

    # Also when surrounded
    text = "abc\uf000\uf001\uf002abc\uf000\uf001\uf002abc"
    pieces = font_manager.select_font(text, FontProps())
    assert len(pieces) == 5
    assert pieces[0][1] is not fallback_font
    assert pieces[1][1] is fallback_font
    assert pieces[2][1] is not fallback_font
    assert pieces[3][1] is fallback_font
    assert pieces[4][1] is not fallback_font

    # Also when preferred font is used
    text = "abc\uf000\uf001\uf002abc\uf000\uf001\uf002abc"
    pieces = font_manager.select_font(text, FontProps("Humor Sans"))
    assert len(pieces) == 5
    assert pieces[0][1] is not fallback_font
    assert pieces[1][1] is fallback_font
    assert pieces[2][1] is not fallback_font
    assert pieces[3][1] is fallback_font
    assert pieces[4][1] is not fallback_font


def test_missing_font_hints():
    def dedent(s):
        lines = [line for line in s.splitlines() if line.strip()]
        indent = min([len(line) - len(line.lstrip()) for line in lines])
        return "\n".join(line[indent:] for line in lines) + "\n"

    assert font_manager._produce_font_warning("abc") == dedent(
        """
    Cannot render chars 'abc'. To fix this, install the following font:
        https://pygfx.github.io/noto-mirror/#NotoSans-Regular.ttf
    """
    )

    assert font_manager._produce_font_warning("foo", "bar") == dedent(
        """
    Cannot render chars 'foo bar'. To fix this, install the following font:
        https://pygfx.github.io/noto-mirror/#NotoSans-Regular.ttf
    """
    )

    assert font_manager._produce_font_warning("こんにちは世界") == dedent(
        """
    Cannot render chars 'こんにちは世界'. To fix this, install any of the following fonts:
        https://pygfx.github.io/noto-mirror/#NotoSansHK-Regular.otf
        https://pygfx.github.io/noto-mirror/#NotoSansJP-Regular.otf
        https://pygfx.github.io/noto-mirror/#NotoSansKR-Regular.otf
        https://pygfx.github.io/noto-mirror/#NotoSansSC-Regular.otf
        https://pygfx.github.io/noto-mirror/#NotoSansTC-Regular.otf
    """
    )

    assert font_manager._produce_font_warning("\uf000a") == dedent(
        """
    Cannot render chars 'a'. To fix this, install (some) of the following fonts:
        https://pygfx.github.io/noto-mirror/#NotoSans-Regular.ttf
    """
    )

    assert font_manager._produce_font_warning("\uf000\uf001") == dedent(
        """
    Cannot render chars ''. Even the Noto font set does not support these characters.
    """
    )


class FakeFontFile(FontFile):
    def __init__(self, family, variant, codepoints):
        super("", family, variant)

    @property
    def family(self):
        return self._family

    @property
    def variant(self):
        return self._variant


def test_add_font_file():
    with raises(TypeError):
        font_manager.add_font_file(42)
    with raises(Exception):  # noqa: B017 - FreeType error
        font_manager.add_font_file("not a filename")

    # Add font by filename
    ff = font_manager.add_font_file(font_manager._fallback_font.filename)
    assert ff.family == "Noto Sans"


def test_selecting_font_props():
    font_manager = FontManager()

    codepoints = {ord(c) for c in "abcdefghijklmnopqrtsuvwxyz"}
    foo_fonts = [
        FontFile("", "Foo Sans", "Regular", codepoints),
        FontFile("", "Foo Sans", "Bold", codepoints),
        FontFile("", "Foo Sans", "Italic", codepoints),
        FontFile("", "Foo Sans", "Bold Italic", codepoints),
    ]

    for ff in foo_fonts:
        assert ff not in font_manager.get_fonts()

    for ff in foo_fonts:
        font_manager.add_font_file(ff)

    for ff in foo_fonts:
        assert ff in font_manager.get_fonts()

    # Won't select on codepoints that our fonts do not support
    pieces = font_manager.select_font("ABC", FontProps("Foo Sans"))
    assert len(pieces) == 1
    assert pieces[0][1].family == "Noto Sans"

    # But this will select the regular
    pieces = font_manager.select_font("abc", FontProps("Foo Sans"))
    assert len(pieces) == 1
    assert pieces[0][1] is foo_fonts[0]

    # So do thin requests
    for w in (100, 200, 300, 400, 500):
        fp = FontProps("Foo Sans", weight=w)
        pieces = font_manager.select_font("abc", fp)
        assert len(pieces) == 1
        assert pieces[0][1] is foo_fonts[0]

    # And these will select the bold
    for w in (600, 700, 800, 900):
        fp = FontProps("Foo Sans", weight=w)
        pieces = font_manager.select_font("abc", fp)
        assert len(pieces) == 1
        assert pieces[0][1] is foo_fonts[1]

    # These select regular italic
    for w in (100, 200, 300, 400, 500):
        fp = FontProps("Foo Sans", weight=w, style="italic")
        pieces = font_manager.select_font("abc", fp)
        assert len(pieces) == 1
        assert pieces[0][1] is foo_fonts[2]

    # These select bold italic
    for w in (600, 700, 800, 900):
        fp = FontProps("Foo Sans", weight=w, style="italic")
        pieces = font_manager.select_font("abc", fp)
        assert len(pieces) == 1
        assert pieces[0][1] is foo_fonts[3]


def check_speed():
    text = "HelloWorld"

    t0 = time.perf_counter()
    for _i in range(1000):
        font_manager.select_font(text, FontProps())
    dt = time.perf_counter() - t0
    print(
        f"select_font: {1000 * dt:0.1f} ms total",
        f"{1000 * dt / (10000):0.3f} ms per char",
    )

    # About 0.00 ms (0.3 us), so this  negligible.


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

    check_speed()
