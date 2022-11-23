import re

text_prog = re.compile(r"\s+")


def tokenize_text(text):
    """Splits the text in pieces of whitespace and non-whitespace."""
    pos = 0
    while True:
        match = text_prog.search(text, pos)

        if not match:
            other = text[pos:]
            if other:
                yield "other", other
            break

        other = text[pos : match.start()]
        if other:
            yield "other", other

        s = match.group()
        yield "space", s
        pos = match.end()


punctuation = r"[\,\.\;\:\?\!\…]"
markdown_prog = re.compile(r"(\s)+|(\d|\w|-|_)+|(\*)+|(" + punctuation + ")")


def tokenize_markdown(text):
    """Splits the text in pieces of: whitespace, words, stars, punctuation, other."""
    pos = 0
    while True:
        match = markdown_prog.search(text, pos)

        if not match:
            other = text[pos:]
            if other:
                yield "other", other
            break

        other = text[pos : match.start()]
        if other:
            yield "", other

        s = match.group()

        if match.group(1):
            yield "space", s
        elif match.group(2):
            yield "word", s
        elif match.group(3):
            yield "stars", s
        elif match.group(4):
            yield "punctuation", s
        else:
            yield "other", s  # should not happen, but just in case

        pos = match.end()
