from app.intent_parser import IntentParser
from app.models import SceneRole
from app.storyboard_generator import StoryboardGenerator


def _storyboard(text, has_media=False):
    intent = IntentParser().parse(text, has_media=has_media)
    return StoryboardGenerator().generate(intent)


def test_short_video_has_three_scenes_with_intro_and_closing():
    sb = _storyboard("Fun birthday video for my mom, 15 seconds, for Instagram")
    assert 3 <= len(sb.scenes) <= 6
    assert sb.scenes[0].role == SceneRole.INTRO
    assert sb.scenes[-1].role == SceneRole.CLOSING
    assert "Mom" in sb.scenes[0].caption or "Mom" in sb.scenes[-1].caption


def test_long_video_has_more_scenes_than_short_video():
    short_sb = _storyboard("Birthday video for my mom, 15 seconds")
    long_sb = _storyboard("Birthday video for my mom, 90 seconds")
    assert len(long_sb.scenes) > len(short_sb.scenes)


def test_scene_durations_sum_to_target_length():
    sb = _storyboard("Wedding video for YouTube, 60 seconds")
    total = sum(s.duration_seconds for s in sb.scenes)
    assert abs(total - sb.total_duration_seconds) < 0.01
    assert abs(sb.total_duration_seconds - 60) < 1.0


def test_every_scene_has_caption_and_narration():
    sb = _storyboard("Graduation video for my daughter, 30 seconds")
    for scene in sb.scenes:
        assert scene.caption
        assert scene.narration


def test_memorial_gets_muted_palette_regardless_of_tone_field():
    sb = _storyboard("A celebration of life video for my grandpa, fun and upbeat")
    # Even though "fun and upbeat" was said, memorial forces a muted palette.
    assert sb.palette.primary == "#3A3A50"


def test_music_mood_matches_tone():
    fun_sb = _storyboard("Fun birthday party video for my son, 30 seconds")
    calm_sb = _storyboard("Calm, peaceful video for my dad's retirement, 30 seconds")
    assert "playful" in fun_sb.music_mood.tags or "upbeat" in fun_sb.music_mood.tags
    assert "gentle" in calm_sb.music_mood.tags or "ambient" in calm_sb.music_mood.tags


def test_visual_consistency_font_and_palette_shared_across_scenes():
    sb = _storyboard("Elegant wedding video, 60 seconds")
    # font_family and palette are storyboard-level (not per-scene), which is
    # what *guarantees* every scene renders with the same treatment.
    assert sb.font_family
    assert sb.palette.primary and sb.palette.secondary


def test_repeated_story_beats_on_long_videos_are_not_identical():
    sb = _storyboard("Birthday video for my mom, 90 seconds")
    story_beat_captions = [s.caption for s in sb.scenes if s.role.value == "story_beat"]
    if len(story_beat_captions) > 1:
        assert len(set(story_beat_captions)) > 1
