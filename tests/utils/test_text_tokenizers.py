from pygfx.utils.text import tokenize_text, tokenize_markdown


def test_tokenize_text():
    text = "foo bar  spam\teggs"
    parts = list(tokenize_text(text))
    assert text == "".join(p[1] for p in parts), "total is not original"
    assert [p[0] for p in parts] == [
        "other",
        "ws",
        "other",
        "ws",
        "other",
        "ws",
        "other",
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
        "punctuation",
        "ws",
        "word",
    ]


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(f"{ob.__name__} ...")
            ob()
    print("done")
