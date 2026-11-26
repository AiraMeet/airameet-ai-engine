"""Arabic transformations preserved from the AI-DS-02 notebook."""

import re

ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652]")
ARABIC_CHAR_TRANSLATION = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}
)


def remove_arabic_diacritics(text: str) -> str:
    """Remove the Arabic diacritics matched by the notebook's regex."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return ARABIC_DIACRITICS_PATTERN.sub("", text)


def normalize_arabic_orthography(text: str) -> str:
    """Apply the notebook's character map and remove tatweel."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.translate(ARABIC_CHAR_TRANSLATION).replace("ـ", "")
