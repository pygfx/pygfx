import re

text_prog = re.compile(
    r"(\n)+"  # newline (nl) -- match first so that it doesn't get included in \s
    r"|([ \r\f\t])+"  # whitespace (ws) (except for newline as it is matched above)
    r"|(\S)+"  # not whitespace
)


def tokenize_text(text):
    """Splits the text in pieces of "ws" and "other" (whitespace and non-whitespace)."""
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
        if match.group(1):
            yield "nl", s
        elif match.group(2):
            yield "ws", s
        else:
            yield "other", s

        pos = match.end()


punctuation = r"[\,\.\;\:\?\!\…]"
markdown_prog = re.compile(r"(\s)+|(\d|\w|-|_)+|(\*)+|(" + punctuation + ")")


def tokenize_markdown(text):
    """Splits the text in pieces of: ws, words, stars, punctuation, other."""
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
            yield "ws", s
        elif match.group(2):
            yield "word", s
        elif match.group(3):
            yield "stars", s
        elif match.group(4):
            yield "punctuation", s
        else:
            yield "other", s  # should not happen, but just in case

        pos = match.end()
