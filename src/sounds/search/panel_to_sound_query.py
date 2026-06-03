"""Convert a comic panel into a CLAP sound query — OCR SFX first, CLIP tags as fallback.

Two LLM-free signals feed one CLAP text query:
  1. OCR onomatopoeia ("BANG", "SMASH") -> curated audio descriptors. This is the
     strongest, most explicit cue (the artist drew the sound), so it wins.
  2. CLIP image tagging (depicted content: rocket, dog, water) -> used only when
     the panel has no drawn SFX text.

Returns None when neither fires, so the UI can stay honestly silent instead of
querying with raw dialogue (which returns confident-looking noise — verified).
"""
from __future__ import annotations

import re

from src.sounds.search.clip_tagger import tag_image_vector

# Curated comic onomatopoeia -> audio descriptor words. Extend freely.
ONOMATOPOEIA_HINTS = {
    # impacts / combat
    "BANG": "gunshot", "BLAM": "gunshot", "POW": "punch impact", "BAM": "impact hit",
    "WHAM": "heavy impact hit", "BOOM": "explosion boom", "KABOOM": "explosion blast",
    "KAPOW": "explosion impact", "CRASH": "crash breaking debris", "SMASH": "smashing impact glass",
    "CRACK": "crack snap", "THUD": "thud impact", "THUNK": "thud impact", "CLANG": "metal clang",
    "CLANK": "metal clank", "RATTAT": "rapid gunfire", "KRAK": "crack impact",
    # motion / air
    "WHOOSH": "whoosh swoosh", "SWOOSH": "whoosh swoosh", "ZIP": "fast whoosh", "VROOM": "engine revving",
    # electric / sci-fi
    "ZAP": "electric zap spark", "ZZZT": "electric buzz spark", "BZZT": "electric buzz",
    # vocal / animal
    "GRR": "animal growl", "GRRR": "animal growl", "ROAR": "loud roar", "WUF": "dog bark",
    "ARF": "dog bark", "WOOF": "dog bark", "MEOW": "cat meow", "HISS": "hiss",
    "AIE": "scream", "AIEE": "scream", "EEK": "scream", "OHH": "groan", "UGH": "grunt groan",
    "UNH": "grunt", "OOF": "grunt", "ARGH": "groan grunt",
    # environment
    "SPLASH": "water splash", "DRIP": "water drip", "RUMBLE": "low rumble thunder",
    "SCREECH": "screech brakes metal", "RING": "bell ring", "KNOCK": "knock on door",
    "TICK": "clock ticking", "CREAK": "creak door",
}

_DEFAULT_FILTERS = {"duration_sec": {"$lte": 10}}


def _collapse(token: str) -> str:
    """GRRRR -> GRR, OHHHH -> OHH, AIEEE -> AIE (cap runs at 2)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", token)


def ocr_terms(text: str) -> list[str]:
    """Matched, de-duplicated onomatopoeia descriptors found in OCR text."""
    tokens = re.findall(r"[A-Za-z]+", (text or "").upper())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        for cand in (tok, _collapse(tok)):
            term = ONOMATOPOEIA_HINTS.get(cand)
            if term and term not in seen:
                seen.add(term)
                out.append(term)
            if term:
                break
    return out


def panel_to_sound_query(panel: dict, image_vec=None, top_k_tags: int = 3) -> dict | None:
    """Build a sound query packet from a panel, or None if there's no cue.

    panel: {"ocr_text": ..., "search_text": ...}
    image_vec: the panel's stored OpenCLIP `image_dense` vector (optional).
    """
    # 1. OCR onomatopoeia wins when present.
    text = f"{panel.get('ocr_text') or ''} {panel.get('search_text') or ''}"
    terms = ocr_terms(text)
    if terms:
        return {
            "sound_query": "sound effect " + " ".join(terms),
            "source": "ocr",
            "matched": terms,
            "filters": dict(_DEFAULT_FILTERS),
        }

    # 2. Fall back to what the panel depicts (CLIP image tags).
    if image_vec is not None:
        tags = tag_image_vector(image_vec, top_k=top_k_tags)
        if tags:
            return {
                "sound_query": "sound of " + " ".join(tags),
                "source": "image",
                "matched": tags,
                "filters": dict(_DEFAULT_FILTERS),
            }

    # 3. No drawn SFX, no confident visual content -> stay silent.
    return None
