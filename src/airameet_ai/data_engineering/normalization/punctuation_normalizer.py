"""Punctuation and annotation handling from AI-DS-02."""

REMOVABLE_PUNCTUATION = str.maketrans(
    dict.fromkeys(('"', "“", "”", "«", "»", "(", ")", "{", "}"))
)


def normalize_punctuation(text: str) -> str:
    """Remove selected quote/bracket characters and paired outer quotes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.translate(REMOVABLE_PUNCTUATION)
    if len(text) >= 2 and (text[0], text[-1]) in [("'", "'"), ("‘", "’")]:
        text = text[1:-1]
    return " ".join(text.split())
