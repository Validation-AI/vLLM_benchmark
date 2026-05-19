import shlex
import re

target_delimiter = ','

def split_ignoring_quotes(s, delimiter=target_delimiter):
    if not s:
        return []

    placeholder = "__EMPTY__PLACEHOLDER__"

    escaped = re.escape(delimiter)
    pattern = f'({escaped}){escaped}+'
    s_with_placeholders = re.sub(pattern, r'\1' + placeholder + delimiter, s)

    lexer = shlex.shlex(s_with_placeholders.strip(), posix=True)
    lexer.whitespace = delimiter
    lexer.whitespace_split = True
    lexer.quotes = '"'
    parts = list(lexer)

    return ["" if p == placeholder else p for p in parts]
