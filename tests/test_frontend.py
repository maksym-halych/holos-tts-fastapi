from unicodedata import normalize

from holos_tts_fastapi.frontend import (
    normalize_chunk,
    split_to_parts,
    text_to_ipa_chunks,
)


def test_long_sentences_split():
    sentence = "А" * 60 + ". "
    text = sentence + sentence
    parts = split_to_parts(text)
    assert len(parts) == 2


def test_short_sentences_are_grouped():
    text = "Привіт. Як справи? Добре. "
    parts = split_to_parts(text)
    assert len(parts) == 1


def test_newline_becomes_sentence_break():
    text = "Перше речення\nДруге речення."
    parts = split_to_parts(text, group=False)
    assert parts == ["Перше речення.", " Друге речення."]


def test_punctuation_without_trailing_space_does_not_split():
    text = "Це приклад.Без пробілу після крапки."
    parts = split_to_parts(text)
    assert len(parts) == 1


def test_manual_stress_marker_becomes_combining_acute():
    result = normalize_chunk("Ярослава Му+дрого")
    assert "́" in result
    assert "+" not in result


def test_em_dash_becomes_colon_pause():
    result = normalize_chunk("слово — слово")
    assert ": " in result
    assert "—" not in result


def test_terminal_punctuation_is_appended():
    result = normalize_chunk("без крапки в кінці")
    assert result.endswith(".")


def test_quotes_are_stripped():
    result = normalize_chunk('текст "у лапках"')
    assert '"' not in result


def test_stress_before_nfkc_matches_manual_composition():
    result = normalize_chunk("Му+дрого")
    assert result == normalize("NFKC", "Му" + "́" + "дрого.")


def test_digits_do_not_raise():
    chunks = text_to_ipa_chunks("У 2026 році все буде добре.")
    assert isinstance(chunks, list)


def test_latin_text_does_not_raise():
    chunks = text_to_ipa_chunks("Це Ukrainian text змішаний з latin.")
    assert isinstance(chunks, list)


def test_out_of_vocabulary_word_does_not_raise():
    chunks = text_to_ipa_chunks("Абвгдеєжзблаблабла невідоме слово.")
    assert isinstance(chunks, list)


def test_whitespace_only_chunk_is_skipped():
    chunks = text_to_ipa_chunks("   ")
    assert chunks == []
