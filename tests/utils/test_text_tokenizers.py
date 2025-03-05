from pygfx.utils.text import tokenize_markdown, tokenize_text


def test_tokenize_text():
    text = "foo bar  spam\teggs"
    parts = list(tokenize_text(text))
    assert text == "".join(p[1] for p in parts), "total is not original"
    assert [p[0] for p in parts] == [
        "word",
        "ws",
        "word",
        "ws",
        "word",
        "ws",
        "word",
    ]


def test_tokenize_text_with_newline():
    text = (
        " \n  \n\n"  # Some newlines and spaces before the text starts.
        "  Lorem ipsum\n"  # Some space at the very beginning of the line
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "py gfx\n"  # a line with exactly 1 word (with a non breaking space inside)
        "last line  \n"  # a line with some space at the end
        "\n  \n\n"  # Some newlines and space at the end
    )
    parts = list(tokenize_text(text))
    assert text == "".join(p[1] for p in parts), "total is not original"
    assert [p[0] for p in parts] == [
        "ws",
        "nl",
        "ws",
        "nl",
        "ws",
        "word",
        "ws",
        "word",
        "nl",
        "word",
        "ws",
        "word",
        "ws",
        "word",
        "nl",
        "word",
        "ws",
        "word",
        "nl",
        "word",
        "ws",
        "word",
        "ws",
        "nl",
        "ws",
        "nl",
    ]


def test_tokenize_markdown():
    # Test regular text
    text = "foo bar  spam\teggs"
    parts = list(tokenize_markdown(text))
    assert text == "".join(p[1] for p in parts), "total is not original"
    assert [p[0] for p in parts] == ["word", "ws", "word", "ws", "word", "ws", "word"]

    # Test stars to make bold text. Note the comma being marked as punctuation.
    # We need that to check when a stars token can be used to start/end a bold/italic section.
    text = "**hello world**, there"
    parts = list(tokenize_markdown(text))
    assert text == "".join(p[1] for p in parts), "total is not original"
    assert [p[0] for p in parts] == [
        "stars",
        "word",
        "ws",
        "word",
        "stars",
        "other",
        "ws",
        "word",
    ]


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
