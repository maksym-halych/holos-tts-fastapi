import re
from unicodedata import normalize

from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol

_DASH_VARIANTS_RE = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_SPLIT_SYMBOLS = ".?!:"

_stressify = Stressifier()


def split_to_parts(text: str, group: bool = True) -> list[str]:
    text = re.sub(r"(\w+[^.,!:?\-])\n", r"\1. ", text)
    text = text.replace("\n", " ")
    parts = [""]
    index = 0
    last = len(text) - 1
    for i, s in enumerate(text):
        parts[index] += s
        if s in _SPLIT_SYMBOLS and i < last and text[i + 1] == " ":
            if group and len(parts[index]) <= 100:
                continue
            index += 1
            parts.append("")
    return parts


def normalize_chunk(chunk: str) -> str:
    chunk = chunk.strip()
    chunk = chunk.replace('"', "")
    if not chunk:
        return chunk
    chunk = chunk.replace("+", StressSymbol.CombiningAcuteAccent)
    chunk = normalize("NFKC", chunk)
    chunk = _DASH_VARIANTS_RE.sub("-", chunk)
    if chunk[-1] not in ".?!:-":
        chunk += "."
    chunk = re.sub(r" - ", ": ", chunk)
    return chunk


def text_to_ipa_chunks(text: str) -> list[str]:
    ipa_chunks = []
    for raw_chunk in split_to_parts(text):
        chunk = normalize_chunk(raw_chunk)
        if not chunk:
            continue
        stressed = _stressify(chunk)
        phonemes = ipa(stressed)
        if phonemes:
            ipa_chunks.append(phonemes)
    return ipa_chunks
