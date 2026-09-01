"""
IntentParser — turns a free-form text description into a ParsedIntent.

Prototype implementation: deterministic keyword/regex rules with a simple
confidence score per field. This is intentionally NOT an LLM call yet, but
the public contract — `parse(text, has_media) -> ParsedIntent` — is exactly
what an LLM-backed implementation would also expose, so swapping the body
of `parse()` for a structured-output LLM call later requires no changes
anywhere else in the pipeline (see ARCHITECTURE.md section 7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Audience, Occasion, ParsedIntent, Platform, Tone

# --------------------------------------------------------------------------
# Keyword tables. Order matters within a category: first match wins, so put
# more specific phrases before generic ones.
# --------------------------------------------------------------------------

OCCASION_KEYWORDS: list[tuple[Occasion, list[str]]] = [
    (Occasion.BIRTHDAY, ["birthday", "bday", "turning ", "b-day"]),
    (Occasion.WEDDING, ["wedding", "getting married", "our marriage", "bride", "groom"]),
    (Occasion.ANNIVERSARY, ["anniversary"]),
    (Occasion.MEMORIAL, ["memorial", "in loving memory", "celebration of life", "funeral", "passed away", "in memory of"]),
    (Occasion.GRADUATION, ["graduation", "graduating", "grad video", "diploma", "class of "]),
    (Occasion.RETIREMENT, ["retirement", "retiring"]),
    (Occasion.BABY, ["baby shower", "gender reveal", "new baby", "expecting", "pregnancy announcement"]),
    (Occasion.HOLIDAY, ["christmas", "hanukkah", "thanksgiving", "new year", "halloween", "holiday card", "easter"]),
    (Occasion.PRODUCT_PROMO, ["product launch", "our new product", "product promo", "promo for", "promo video", "sale", "discount", "shop now", "our store", "business promo"]),
    (Occasion.REAL_ESTATE, ["real estate", "open house", "for sale", "listing", "sq ft", "square feet", "new home tour", "property tour"]),
    (Occasion.TRAVEL, ["vacation", "trip to", "travel recap", "our trip", "road trip"]),
]

TONE_KEYWORDS: list[tuple[Tone, list[str]]] = [
    (Tone.FUN, ["fun", "funny", "silly", "playful", "hype", "energetic", "party"]),
    (Tone.EMOTIONAL, ["emotional", "heartfelt", "touching", "tearjerker", "sentimental", "in loving memory", "miss you"]),
    (Tone.ELEGANT, ["elegant", "classy", "sophisticated", "chic", "luxury", "upscale"]),
    (Tone.CALM, ["calm", "peaceful", "relaxing", "soft", "gentle", "serene"]),
    (Tone.BOLD, ["bold", "epic", "powerful", "dramatic", "intense"]),
    (Tone.PROFESSIONAL, ["professional", "corporate", "business", "polished and professional"]),
    (Tone.WARM, ["warm", "loving", "cozy", "sweet", "wholesome"]),
]

AUDIENCE_KEYWORDS: list[tuple[Audience, list[str]]] = [
    (Audience.FAMILY, ["my mom", "my dad", "my mother", "my father", "grandma", "grandpa", "family", "my son", "my daughter", "my wife", "my husband", "my parents", "my sister", "my brother"]),
    (Audience.KIDS, ["kids", "children", "toddler", "my kid", "for the kids"]),
    (Audience.CUSTOMERS, ["customers", "clients", "shoppers", "our audience", "subscribers"]),
    (Audience.COLLEAGUES, ["colleagues", "coworkers", "my team", "the office", "employees"]),
    (Audience.FRIENDS, ["friends", "my friend", "buddies", "squad"]),
]

# Two tiers: "strong" signals are unambiguous platform names and are checked
# first, across all platforms, before any "weak" generic term (like "reel",
# which appears inside the common phrase "highlight reel" and would
# otherwise misfire as a vertical-video signal for a YouTube request).
PLATFORM_STRONG_KEYWORDS: list[tuple[Platform, list[str]]] = [
    (Platform.VERTICAL, ["tiktok", "instagram"]),
    (Platform.HORIZONTAL, ["youtube"]),
]

PLATFORM_WEAK_KEYWORDS: list[tuple[Platform, list[str]]] = [
    (Platform.VERTICAL, ["reel", "reels", "ig story", "stories", "shorts", "vertical"]),
    (Platform.HORIZONTAL, ["landscape", "horizontal", "widescreen", "tv"]),
    (Platform.SQUARE, ["square", "facebook feed", "instagram feed", "insta post"]),
]

_LENGTH_WORD_MAP = {
    "short": 15,
    "quick": 15,
    "medium": 30,
    "long": 60,
}

_NUMBER_SECONDS_RE = re.compile(r"(\d{1,3})\s*(?:-|to)?\s*(?:sec|secs|second|seconds|s)\b", re.IGNORECASE)
_NUMBER_MINUTES_RE = re.compile(r"(\d{1,2})\s*(?:min|mins|minute|minutes)\b", re.IGNORECASE)

# Recognizes "for my mom", "for my son Alex", "for grandma" -> subject phrase
_SUBJECT_RE = re.compile(
    r"for\s+(my\s+\w+|our\s+\w+|grandma|grandpa|the\s+team|the\s+family)\b",
    re.IGNORECASE,
)

_SUBJECT_DISPLAY_MAP = {
    "my mom": "Mom", "my mother": "Mom", "my dad": "Dad", "my father": "Dad",
    "my wife": "your wife", "my husband": "your husband",
    "my son": "your son", "my daughter": "your daughter",
    "my sister": "your sister", "my brother": "your brother",
    "my friend": "your friend", "my grandma": "Grandma", "my grandpa": "Grandpa",
    "grandma": "Grandma", "grandpa": "Grandpa",
    "our team": "the team", "the team": "the team", "the family": "the family",
}


@dataclass
class _Match:
    value: str
    confidence: float


def _keyword_match(text_lower: str, table) -> _Match | None:
    for value, phrases in table:
        for phrase in phrases:
            if phrase.lower() in text_lower:
                return _Match(value=value, confidence=0.9)
    return None


class IntentParser:
    """Rule-based intent parser. See module docstring for the swap plan."""

    def parse(self, text: str, has_media: bool = False) -> ParsedIntent:
        text_lower = text.lower()

        occasion_match = _keyword_match(text_lower, OCCASION_KEYWORDS)
        tone_match = _keyword_match(text_lower, TONE_KEYWORDS)
        audience_match = _keyword_match(text_lower, AUDIENCE_KEYWORDS)
        platform_match = _keyword_match(text_lower, PLATFORM_STRONG_KEYWORDS) or _keyword_match(
            text_lower, PLATFORM_WEAK_KEYWORDS
        )

        occasion = occasion_match.value if occasion_match else Occasion.GENERAL_CELEBRATION
        audience = audience_match.value if audience_match else Audience.GENERAL
        platform = platform_match.value if platform_match else Platform.VERTICAL

        tone, tone_confidence = self._resolve_tone(tone_match, occasion)

        length_seconds, length_confidence = self._resolve_length(text_lower)

        subject_name = self._extract_subject(text)

        keywords = self._extract_keywords(text_lower)

        confidence = {
            "occasion": occasion_match.confidence if occasion_match else 0.3,
            "tone": tone_confidence,
            "audience": audience_match.confidence if audience_match else 0.3,
            "platform": platform_match.confidence if platform_match else 0.4,
            "length_seconds": length_confidence,
        }

        return ParsedIntent(
            raw_text=text,
            occasion=occasion,
            tone=tone,
            audience=audience,
            platform=platform,
            length_seconds=length_seconds,
            subject_name=subject_name,
            keywords=keywords,
            has_user_media=has_media,
            confidence=confidence,
        )

    # -- helpers -----------------------------------------------------------

    def _resolve_tone(self, tone_match: _Match | None, occasion: Occasion) -> tuple[Tone, float]:
        if tone_match:
            return tone_match.value, tone_match.confidence
        # No explicit tone stated -> infer a sensible default from occasion.
        default_by_occasion = {
            Occasion.BIRTHDAY: Tone.FUN,
            Occasion.WEDDING: Tone.ELEGANT,
            Occasion.ANNIVERSARY: Tone.WARM,
            Occasion.MEMORIAL: Tone.EMOTIONAL,
            Occasion.GRADUATION: Tone.BOLD,
            Occasion.RETIREMENT: Tone.WARM,
            Occasion.BABY: Tone.WARM,
            Occasion.HOLIDAY: Tone.WARM,
            Occasion.PRODUCT_PROMO: Tone.BOLD,
            Occasion.REAL_ESTATE: Tone.ELEGANT,
            Occasion.TRAVEL: Tone.FUN,
            Occasion.GENERAL_CELEBRATION: Tone.WARM,
            Occasion.OTHER: Tone.WARM,
        }
        return default_by_occasion.get(occasion, Tone.WARM), 0.5

    def _resolve_length(self, text_lower: str) -> tuple[int, float]:
        m = _NUMBER_SECONDS_RE.search(text_lower)
        if m:
            return int(m.group(1)), 0.95
        m = _NUMBER_MINUTES_RE.search(text_lower)
        if m:
            return int(m.group(1)) * 60, 0.95
        for word, seconds in _LENGTH_WORD_MAP.items():
            if word in text_lower:
                return seconds, 0.7
        return 30, 0.4  # default: medium length

    def _extract_subject(self, text: str) -> str | None:
        m = _SUBJECT_RE.search(text)
        if not m:
            return None
        phrase = m.group(1).lower()
        return _SUBJECT_DISPLAY_MAP.get(phrase, phrase.replace("my ", "your ").title())

    def _extract_keywords(self, text_lower: str) -> list[str]:
        # Very small heuristic: pull out capitalized-looking proper nouns are
        # lost once lowercased, so instead grab a few salient short tokens
        # (nouns/adjectives) that aren't stopwords/already-classified terms.
        stopwords = {
            "a", "an", "the", "for", "my", "our", "of", "to", "and", "with",
            "on", "in", "is", "video", "make", "create", "want", "please",
            "seconds", "second", "sec", "secs", "minute", "minutes", "min",
        }
        tokens = re.findall(r"[a-zA-Z']+", text_lower)
        seen = []
        for tok in tokens:
            if tok in stopwords or len(tok) < 3:
                continue
            if tok not in seen:
                seen.append(tok)
            if len(seen) >= 8:
                break
        return seen
