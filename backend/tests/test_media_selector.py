from pathlib import Path

from app.intent_parser import IntentParser
from app.media_selector import MediaSelector
from app.models import CropStrategy, SceneRole, VisualSourceType
from app.storyboard_generator import StoryboardGenerator


def _storyboard(text):
    intent = IntentParser().parse(text)
    return StoryboardGenerator().generate(intent)


def test_with_user_photos_every_scene_uses_user_media_round_robin(tmp_path):
    sb = _storyboard("Birthday video for my mom, 60 seconds")
    photos = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for p in photos:
        p.write_bytes(b"fake-jpeg-bytes")

    result = MediaSelector().plan_media(sb, uploaded_media=photos)

    assert len(result.scenes) >= 3
    for i, scene in enumerate(result.scenes):
        assert scene.visual.source_type == VisualSourceType.USER_MEDIA
        assert scene.visual.crop_strategy == CropStrategy.FACE_AWARE
        # Round-robins through the 2 provided photos.
        assert Path(scene.visual.user_media_ref) == photos[i % len(photos)]


def test_without_photos_product_promo_prefers_stock():
    sb = _storyboard("Product promo video for our new coffee, TikTok, 15 seconds")
    result = MediaSelector().plan_media(sb, uploaded_media=[])
    assert all(s.visual.source_type == VisualSourceType.STOCK for s in result.scenes)
    assert all(s.visual.stock_query for s in result.scenes)


def test_without_photos_real_estate_prefers_stock():
    sb = _storyboard("Real estate listing tour for YouTube, 30 seconds")
    result = MediaSelector().plan_media(sb, uploaded_media=[])
    assert all(s.visual.source_type == VisualSourceType.STOCK for s in result.scenes)


def test_without_photos_birthday_uses_template_and_ai_for_highlight():
    sb = _storyboard("Fun birthday video for my mom, 30 seconds, for Instagram")
    result = MediaSelector().plan_media(sb, uploaded_media=[])

    highlight_scenes = [s for s in result.scenes if s.role == SceneRole.HIGHLIGHT]
    non_highlight = [s for s in result.scenes if s.role != SceneRole.HIGHLIGHT]

    assert all(s.visual.source_type == VisualSourceType.AI_GENERATED for s in highlight_scenes)
    assert all(s.visual.ai_prompt for s in highlight_scenes)
    assert all(s.visual.source_type == VisualSourceType.TEMPLATE for s in non_highlight)
    assert all(s.visual.template_id for s in non_highlight)


def test_plan_media_does_not_mutate_input_storyboard():
    sb = _storyboard("Birthday video for my mom, 30 seconds")
    original_visuals = [s.visual.source_type for s in sb.scenes]
    MediaSelector().plan_media(sb, uploaded_media=[])
    # sb itself must be unchanged (StoryboardGenerator's output is treated
    # as immutable input by every later stage).
    assert [s.visual.source_type for s in sb.scenes] == original_visuals
