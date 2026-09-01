"""
Renderer tests actually invoke FFmpeg and Pillow and assert on the real
output file (resolution, playability via ffprobe, rough duration) rather
than mocking them out — this is the "placeholder renderer that produces a
simple but structured, and in this prototype fully real, video" from the
brief. Kept fast by rendering at a small `resolution_scale` and with short
target lengths.
"""
import json
import subprocess

import pytest
from PIL import Image

from app.intent_parser import IntentParser
from app.media_selector import MediaSelector
from app.models import Platform
from app.renderer import Renderer
from app.storyboard_generator import StoryboardGenerator


def _pipeline(text, uploaded_media=None):
    intent = IntentParser().parse(text, has_media=bool(uploaded_media))
    storyboard = StoryboardGenerator().generate(intent)
    storyboard = MediaSelector().plan_media(storyboard, uploaded_media=uploaded_media or [])
    return storyboard


def _ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        stdout=subprocess.PIPE, text=True,
    )
    return json.loads(out.stdout)


def test_render_without_user_photos_vertical_short(tmp_path):
    sb = _pipeline("Fun birthday video for my mom, 12 seconds, for Instagram")
    assert sb.intent.platform == Platform.VERTICAL

    meta = Renderer().render(sb, tmp_path, resolution_scale=0.25)  # 270x480, fast

    assert meta.width == 270
    assert meta.height == 480
    probe = _ffprobe(meta.path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(video_stream["width"]) == 270
    assert int(video_stream["height"]) == 480
    assert audio_stream is not None
    # Rendered duration should be within ~1s of the requested storyboard length.
    assert abs(float(probe["format"]["duration"]) - sb.total_duration_seconds) < 1.5


def test_render_horizontal_youtube():
    sb = _pipeline("Wedding highlight video for YouTube, 12 seconds")
    assert sb.intent.platform == Platform.HORIZONTAL

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        meta = Renderer().render(sb, Path(d), resolution_scale=0.25)
        assert meta.width / meta.height == pytest.approx(1920 / 1080, rel=0.02)
        probe = _ffprobe(meta.path)
        assert probe["streams"]


def test_render_with_user_photos(tmp_path):
    photo_path = tmp_path / "mom.jpg"
    Image.new("RGB", (800, 600), (200, 100, 50)).save(photo_path)

    sb = _pipeline("Birthday video for my mom, 12 seconds", uploaded_media=[photo_path])
    assert sb.scenes[0].visual.user_media_ref == str(photo_path)

    out_dir = tmp_path / "job"
    meta = Renderer().render(sb, out_dir, resolution_scale=0.25)

    assert meta.file_size_bytes > 0
    probe = _ffprobe(meta.path)
    assert probe["format"]["format_name"]


def test_output_file_size_is_reasonable_not_empty_or_huge(tmp_path):
    sb = _pipeline("Retirement video for my colleagues, 12 seconds")
    meta = Renderer().render(sb, tmp_path, resolution_scale=0.25)
    # A few-second, low-res clip should be well under 5MB and clearly not 0.
    assert 1000 < meta.file_size_bytes < 5_000_000


def test_different_lengths_produce_proportionally_longer_output(tmp_path):
    short_sb = _pipeline("Birthday video for my mom, 12 seconds")
    long_sb = _pipeline("Birthday video for my mom, 45 seconds")

    short_meta = Renderer().render(short_sb, tmp_path / "short", resolution_scale=0.2)
    long_meta = Renderer().render(long_sb, tmp_path / "long", resolution_scale=0.2)

    assert long_meta.duration_seconds > short_meta.duration_seconds
