import re


matchers = {
    "nl": r"(\n)+",  # newline (nl) -- match first so that it doesn't get included in \s
    "ws": r"([ \r\f\t])+",  # whitespace (ws) (except for newline as it is matched above)
    "word": r"(\w)+",
    "number": r"(\d)+",
    "punctuation": r"([\,\.\;\:\?\!\…])+",
    "stars": r"(\*)+",
}

text_groups = ["nl", "ws", "word"]
text_prog = re.compile("|".join(matchers[group] for group in text_groups))

markdown_groups = ["nl", "ws", "word", "stars"]
markdown_prog = re.compile("|".join(matchers[group] for group in markdown_groups))


def tokenize_text(text):
    """Splits the text in pieces of "nl", "ws" "word", and "other"."""
    return _tokenze(text, text_groups, text_prog)


def tokenize_markdown(text):
    """Splits the text in pieces of "nl", "ws" "word", "stars", and "other"."""
    return _tokenze(text, markdown_groups, markdown_prog)


def _tokenze(text, groups, prog):
    pos = 0
    while True:
        match = prog.search(text, pos)

        if not match:
            other = text[pos:]
            if other:
                yield "other", other
            break

        other = text[pos : match.start()]
        if other:
            yield "other", other

        s = match.group()
        for group_index, group_name in enumerate(groups, 1):
            if match.group(group_index):
                yield group_name, s
                break
        else:
            yield "other", s

        pos = match.end()
