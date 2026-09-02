from pathlib import Path

from app.intent_parser import IntentParser
from app.media_selector import MediaSelector
from app.models import CropStrategy, VisualSourceType
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


def test_without_photos_every_scene_gets_a_real_stock_query():
    sb = _storyboard("Product promo video for our new coffee, TikTok, 15 seconds")
    result = MediaSelector().plan_media(sb, uploaded_media=[])
    assert all(s.visual.source_type == VisualSourceType.STOCK for s in result.scenes)
    assert all(s.visual.stock_query for s in result.scenes)


def test_without_photos_real_estate_gets_relevant_query():
    sb = _storyboard("Real estate listing tour for YouTube, 30 seconds")
    result = MediaSelector().plan_media(sb, uploaded_media=[])
    assert all(s.visual.source_type == VisualSourceType.STOCK for s in result.scenes)
    assert any(
        "home" in s.visual.stock_query or "house" in s.visual.stock_query or "living room" in s.visual.stock_query
        for s in result.scenes
    )


def test_without_photos_birthday_scenes_vary_across_roles():
    sb = _storyboard("Fun birthday video for my mom, 30 seconds, for Instagram")
    result = MediaSelector().plan_media(sb, uploaded_media=[])

    assert all(s.visual.source_type == VisualSourceType.STOCK for s in result.scenes)
    assert all(s.visual.stock_query for s in result.scenes)
    # Not every scene should get the exact same query — some visual variety
    # across a multi-scene video.
    queries = [s.visual.stock_query for s in result.scenes]
    assert len(set(queries)) > 1


def test_variant_seed_changes_stock_queries():
    sb = _storyboard("Fun birthday video for my mom, 30 seconds, for Instagram")
    result_a = MediaSelector().plan_media(sb, uploaded_media=[], variant_seed=0)
    result_b = MediaSelector().plan_media(sb, uploaded_media=[], variant_seed=1)
    queries_a = [s.visual.stock_query for s in result_a.scenes]
    queries_b = [s.visual.stock_query for s in result_b.scenes]
    assert queries_a != queries_b


def test_plan_media_does_not_mutate_input_storyboard():
    sb = _storyboard("Birthday video for my mom, 30 seconds")
    original_visuals = [s.visual.source_type for s in sb.scenes]
    MediaSelector().plan_media(sb, uploaded_media=[])
    # sb itself must be unchanged (StoryboardGenerator's output is treated
    # as immutable input by every later stage).
    assert [s.visual.source_type for s in sb.scenes] == original_visuals
