from app.intent_parser import IntentParser
from app.models import Audience, Occasion, Platform, Tone


def _parse(text, has_media=False):
    return IntentParser().parse(text, has_media=has_media)


def test_fun_birthday_for_mom_instagram_30s():
    intent = _parse("Fun birthday video for my mom, 30 seconds, for Instagram")
    assert intent.occasion == Occasion.BIRTHDAY
    assert intent.tone == Tone.FUN
    assert intent.audience == Audience.FAMILY
    assert intent.platform == Platform.VERTICAL
    assert intent.length_seconds == 30
    assert intent.subject_name == "Mom"


def test_elegant_wedding_youtube_60s():
    intent = _parse("Elegant wedding highlight reel for YouTube, about 60 seconds")
    assert intent.occasion == Occasion.WEDDING
    assert intent.tone == Tone.ELEGANT
    assert intent.platform == Platform.HORIZONTAL
    assert intent.length_seconds == 60


def test_memorial_defaults_to_emotional_tone_when_unstated():
    intent = _parse("A celebration of life video for my grandpa")
    assert intent.occasion == Occasion.MEMORIAL
    # Tone not explicitly stated -> inferred default for memorial.
    assert intent.tone == Tone.EMOTIONAL
    assert intent.subject_name == "Grandpa"


def test_product_promo_short_tiktok():
    intent = _parse("Quick TikTok promo for our new coffee product, 15s")
    assert intent.occasion == Occasion.PRODUCT_PROMO
    assert intent.platform == Platform.VERTICAL
    assert intent.length_seconds == 15


def test_real_estate_horizontal_default_length():
    intent = _parse("Real estate listing tour video, professional, for YouTube")
    assert intent.occasion == Occasion.REAL_ESTATE
    assert intent.tone == Tone.PROFESSIONAL
    assert intent.platform == Platform.HORIZONTAL
    # No explicit length stated -> falls back to default medium length.
    assert intent.length_seconds == 30
    assert intent.confidence["length_seconds"] < 0.5


def test_minutes_are_converted_to_seconds():
    intent = _parse("Graduation video, 1 minute long, for my daughter")
    assert intent.length_seconds == 60
    assert intent.occasion == Occasion.GRADUATION
    assert intent.subject_name == "your daughter"


def test_retirement_office_colleagues():
    intent = _parse("Retirement video for my colleagues in the office, warm and heartfelt")
    assert intent.occasion == Occasion.RETIREMENT
    assert intent.audience == Audience.COLLEAGUES
    assert intent.tone in (Tone.WARM, Tone.EMOTIONAL)


def test_unrecognized_text_gets_safe_defaults():
    intent = _parse("asdkjhaskjdh random gibberish text")
    assert intent.occasion == Occasion.GENERAL_CELEBRATION
    assert intent.platform == Platform.VERTICAL
    assert intent.length_seconds == 30
    assert intent.confidence["occasion"] < 0.5


def test_has_media_flag_is_passed_through():
    intent = _parse("Birthday video for my mom", has_media=True)
    assert intent.has_user_media is True
