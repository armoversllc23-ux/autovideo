"""
Phase 5 verification: a broad scenario matrix across occasions, tones,
platforms, lengths, and with/without user photos, run through the full
pipeline end-to-end. This is the acceptance sweep referenced in
ARCHITECTURE.md's Phase 5 criteria: every realistic combination the brief
calls out must produce a valid, correctly-shaped render without raising.
"""
from pathlib import Path

import pytest
from PIL import Image

from app.intent_parser import IntentParser
from app.media_selector import MediaSelector
from app.models import Platform
from app.renderer import Renderer
from app.storyboard_generator import StoryboardGenerator

SCENARIOS = [
    ("Fun birthday video for my mom, 30 seconds, for Instagram", False),
    ("Elegant wedding highlight reel for YouTube, about 60 seconds", False),
    ("A celebration of life video for my grandpa", False),
    ("Quick TikTok promo for our new coffee shop, 15s", False),
    ("Real estate listing tour video, professional, for YouTube", False),
    ("Graduation video for my daughter, 1 minute long", True),
    ("Retirement video for my colleagues in the office, warm and heartfelt", True),
    ("Baby shower announcement video, gentle and calm, 20 seconds", False),
    ("Happy Holidays video for the family, square for Facebook", False),
    ("Our trip to Japan, fun travel recap, 45 seconds", True),
]


@pytest.mark.parametrize("text,with_photo", SCENARIOS)
def test_scenario_end_to_end(tmp_path, text, with_photo):
    uploaded = []
    if with_photo:
        photo = tmp_path / "photo.jpg"
        Image.new("RGB", (500, 400), (80, 140, 200)).save(photo)
        uploaded = [photo]

    intent = IntentParser().parse(text, has_media=bool(uploaded))
    storyboard = StoryboardGenerator().generate(intent)
    storyboard = MediaSelector().plan_media(storyboard, uploaded_media=uploaded)

    # -- structural acceptance criteria --------------------------------
    assert 3 <= len(storyboard.scenes) <= 6
    assert storyboard.scenes[0].role.value == "intro"
    assert storyboard.scenes[-1].role.value == "closing"
    for scene in storyboard.scenes:
        assert scene.caption
        assert scene.narration
        assert scene.duration_seconds > 0
        assert scene.visual.source_type is not None

    # -- rendered-output acceptance criteria ---------------------------
    job_dir = tmp_path / "render"
    meta = Renderer().render(storyboard, job_dir, resolution_scale=0.2)

    base_w, base_h = {
        Platform.VERTICAL: (1080, 1920),
        Platform.HORIZONTAL: (1920, 1080),
        Platform.SQUARE: (1080, 1080),
    }[intent.platform]
    assert meta.width == pytest.approx(base_w * 0.2, abs=2)
    assert meta.height == pytest.approx(base_h * 0.2, abs=2)
    assert meta.file_size_bytes > 500
    assert abs(meta.duration_seconds - storyboard.total_duration_seconds) < 2.0
