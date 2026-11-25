"""AI-DS-02 normalization regression fixtures copied from its notebook."""

import pytest

from airameet_ai.data_engineering.conversion import (
    normalize_arabic_orthography,
    normalize_english,
    normalize_punctuation,
    normalize_text,
    normalize_unicode,
    remove_arabic_diacritics,
)

NOTEBOOK_FIXTURES = [
    ("أحمد", "احمد"),
    ("إسلام", "اسلام"),
    ("آدم", "ادم"),
    ("ٱلْحَمْدُ", "الحمد"),
    ("على", "علي"),
    ("مدرسة", "مدرسه"),
    ("جميلة", "جميله"),
    ("هُدى", "هدي"),
    ("شيء", "شيء"),
    ("مسؤول", "مسؤول"),
    ("مُحَمَّد", "محمد"),
    ("كِتَاب", "كتاب"),
    ("مَدْرَسَة", "مدرسه"),
    ("العَرَبِيَّة", "العربيه"),
    ("شُكْرًا", "شكرا"),
    ("PyTorch", "PyTorch"),
    ("TensorFlow", "TensorFlow"),
    ("Machine Learning", "Machine Learning"),
    ("Deep Learning", "Deep Learning"),
    ("SQL Server", "SQL Server"),
    ("GPT-6", "GPT-6"),
    ("HTTP/2", "HTTP/2"),
    ("C++", "C++"),
    ("Node.js", "Node.js"),
    ("Python 3.12", "Python 3.12"),
    ("CUDA 12.4", "CUDA 12.4"),
    ("Whisper Large-v3", "Whisper Large-v3"),
    ("API", "API"),
    ("Layer Normalization", "Layer Normalization"),
    ("don't remove user's data", "don't remove user's data"),
    ("أهلاً بـ GPT-6", "اهلا ب GPT-6"),
    ("استخدم PyTorch مع CUDA", "استخدم PyTorch مع CUDA"),
    ("العَرَبِيَّة و English", "العربيه و English"),
    ("إزاي تستخدم Python؟", "ازاي تستخدم Python؟"),
    ("Layer Normalization في PyTorch", "Layer Normalization في PyTorch"),
    ("Hello يا أحمد", "Hello يا احمد"),
    ("ال+weekends", "ال+weekends"),
    ("ال+hobbies", "ال+hobbies"),
    ("Okay ماشي", "Okay ماشي"),
    ("Python 3.12 مع GPT-6", "Python 3.12 مع GPT-6"),
    ("  Hello   World  ", "Hello World"),
    ("طيب\nبتعمل إيه؟", "طيب بتعمل ايه؟"),
    ("Layer\tNormalization", "Layer Normalization"),
    ("\ufeffHello World", "Hello World"),
    ("", ""),
    ("HTTP/2 - GPT-6", "HTTP/2 - GPT-6"),
    ('"Hello World"', "Hello World"),
    ("(PyTorch 2.5)", "PyTorch 2.5"),
    ("[LAUGHTER] okay شكرا", "[LAUGHTER] okay شكرا"),
    ("Hello, world!", "Hello, world!"),
]

TECHNICAL_FIXTURES = [
    "PyTorch 2.5",
    "TensorFlow",
    "Python 3.12",
    "GPT-6",
    "HTTP/2",
    "C++",
    "Node.js",
    "CUDA 12.4",
    "Whisper Large-v3",
    "don't",
    "user's account",
    "[LAUGHTER]",
    "ال+weekends",
]

IDEMPOTENCY_FIXTURES = [
    "أَحْمَد بيستخدم Python 3.12",
    '"Hello" مع PyTorch 2.5',
    "[LAUGHTER] و [HESITATION]",
    "ال+weekends و HTTP/2",
    "don't remove user's data",
]


@pytest.mark.parametrize(("original", "expected"), NOTEBOOK_FIXTURES)
def test_notebook_regression_fixtures(original: str, expected: str) -> None:
    assert normalize_text(original) == expected


@pytest.mark.parametrize("original", TECHNICAL_FIXTURES)
def test_notebook_technical_preservation(original: str) -> None:
    assert normalize_text(original) == original


@pytest.mark.parametrize("original", IDEMPOTENCY_FIXTURES)
def test_notebook_idempotency(original: str) -> None:
    assert normalize_text(normalize_text(original)) == normalize_text(original)


@pytest.mark.parametrize(
    "function",
    [
        normalize_text,
        normalize_unicode,
        remove_arabic_diacritics,
        normalize_arabic_orthography,
        normalize_punctuation,
        normalize_english,
    ],
)
def test_normalizers_reject_non_string_input(function) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        function(None)


def test_individual_normalization_stages() -> None:
    assert normalize_unicode("\ufeffA\n  B") == "A B"
    assert remove_arabic_diacritics("مُحَمَّد") == "محمد"
    assert normalize_arabic_orthography("أحمدـى") == "احمدي"
    assert normalize_punctuation('"(hello)"') == "hello"
    assert normalize_english("GPT-6 / Python 3.12") == "GPT-6 / Python 3.12"


def test_known_prompt_regressions() -> None:
    assert normalize_text("أَحْمَد") == "احمد"
    assert normalize_text("إِسْلَام") == "اسلام"
    assert normalize_text("آدم") == "ادم"
    assert normalize_text("ٱلْحَمْدُ") == "الحمد"
    assert normalize_text('"أهلاً بـ GPT-6"') == "اهلا ب GPT-6"


def test_large_text_field_is_processed_without_truncation() -> None:
    source = "GPT-6 " * 20_000 + "أَحْمَد"
    result = normalize_text(source)
    assert result == "GPT-6 " * 20_000 + "احمد"
