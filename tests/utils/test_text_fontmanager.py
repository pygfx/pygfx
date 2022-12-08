import time

from pygfx.utils.text import font_manager, FontFile


def test_select_font():

    # A simple text, that can be rendered with the main font
    text = "HelloWorld"

    pieces = font_manager.select_font(text, ())
    assert len(pieces) == 1
    assert isinstance(pieces[0], tuple)
    assert pieces[0][0] == text
    assert isinstance(pieces[0][1], FontFile)

    # A text with both Latin and Arabic, needs two fonts
    text = "Hello World مرحبا بالعالم"

    pieces = font_manager.select_font(text, ())
    assert len(pieces) == 2
    assert isinstance(pieces[0], tuple)
    assert isinstance(pieces[1], tuple)
    assert isinstance(pieces[0][1], FontFile)
    assert isinstance(pieces[1][1], FontFile)
    assert pieces[0][0] == "Hello World "
    assert pieces[1][0] == "مرحبا بالعالم"


def check_speed():
    text = "HelloWorld"

    t0 = time.perf_counter()
    for i in range(1000):
        font_manager.select_font(text, ())
    dt = time.perf_counter() - t0
    print(
        f"select_font: {1000*dt:0.1f} ms total", f"{1000*dt/(10000):0.3f} ms per char"
    )

    # About 0.00 ms (0.3 us), so this  negligible.


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")

    check_speed()
