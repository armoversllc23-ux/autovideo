"""
Core data structures shared across the whole pipeline.

These Pydantic models are deliberately the *single* source of truth for
shape: the same classes are used as API request/response schemas (FastAPI
serializes them straight to/from JSON) and as the plain-data contracts
passed between IntentParser -> StoryboardGenerator -> MediaSelector ->
Renderer. See ARCHITECTURE.md section 4 for the rationale.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Enums — the fixed vocabulary the whole system reasons in.
# --------------------------------------------------------------------------

class Occasion(str, Enum):
    BIRTHDAY = "birthday"
    WEDDING = "wedding"
    ANNIVERSARY = "anniversary"
    MEMORIAL = "memorial"
    GRADUATION = "graduation"
    RETIREMENT = "retirement"
    HOLIDAY = "holiday"
    BABY = "baby_announcement"
    PRODUCT_PROMO = "product_promo"
    REAL_ESTATE = "real_estate"
    TRAVEL = "travel_recap"
    GENERAL_CELEBRATION = "general_celebration"
    OTHER = "other"


class Tone(str, Enum):
    FUN = "fun"
    EMOTIONAL = "emotional"
    ELEGANT = "elegant"
    CALM = "calm"
    BOLD = "bold"
    WARM = "warm"
    PROFESSIONAL = "professional"


class Audience(str, Enum):
    FAMILY = "family"
    FRIENDS = "friends"
    CUSTOMERS = "customers"
    COLLEAGUES = "colleagues"
    KIDS = "kids"
    GENERAL = "general"


class Platform(str, Enum):
    VERTICAL = "vertical"      # TikTok / Instagram Reels & Stories
    HORIZONTAL = "horizontal"  # YouTube
    SQUARE = "square"          # Instagram feed / Facebook


class SceneRole(str, Enum):
    INTRO = "intro"
    HIGHLIGHT = "highlight"
    STORY_BEAT = "story_beat"
    CLOSING = "closing"


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    SLIDE = "slide"
    ZOOM = "zoom"


class VisualSourceType(str, Enum):
    USER_MEDIA = "user_media"
    TEMPLATE = "template"
    STOCK = "stock"
    AI_GENERATED = "ai_generated"


class CropStrategy(str, Enum):
    CENTER = "center"
    FACE_AWARE = "face_aware"
    TOP = "top"
    CUSTOM = "custom"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    STORYBOARDING = "storyboarding"
    SELECTING_MEDIA = "selecting_media"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


# --------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------

class ParsedIntent(BaseModel):
    raw_text: str
    occasion: Occasion = Occasion.GENERAL_CELEBRATION
    tone: Tone = Tone.WARM
    audience: Audience = Audience.GENERAL
    platform: Platform = Platform.VERTICAL
    length_seconds: int = 30
    subject_name: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    has_user_media: bool = False
    # Per-field confidence in [0, 1]. Lets a future LLM-backed parser (or a
    # human reviewing logs) see which fields were guessed vs. clearly stated.
    confidence: dict[str, float] = Field(default_factory=dict)
    # Explicit user overrides from the (hidden-by-default) "Advanced" panel.
    # Later pipeline stages honor these if present; the prototype frontend
    # never sets them, but the plumbing exists so adding that panel later
    # requires zero pipeline changes.
    overrides: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Storyboard
# --------------------------------------------------------------------------

class ColorPalette(BaseModel):
    primary: str
    secondary: str
    accent: str
    text_on_dark: str = "#FFFFFF"
    text_on_light: str = "#111111"


class MusicMood(BaseModel):
    tags: list[str]           # e.g. ["uplifting", "medium-tempo"]
    tempo_bpm_range: tuple[int, int] = (90, 120)


class VisualPlan(BaseModel):
    source_type: VisualSourceType = VisualSourceType.TEMPLATE
    user_media_ref: Optional[str] = None   # filename/path of an uploaded asset
    template_id: Optional[str] = None
    stock_query: Optional[str] = None      # abstraction only, stubbed in prototype
    ai_prompt: Optional[str] = None        # abstraction only, stubbed in prototype
    crop_strategy: CropStrategy = CropStrategy.CENTER


class Scene(BaseModel):
    index: int
    role: SceneRole
    caption: str
    narration: Optional[str] = None
    duration_seconds: float
    visual: VisualPlan = Field(default_factory=VisualPlan)
    transition_in: TransitionType = TransitionType.CROSSFADE


class StoryboardPlan(BaseModel):
    intent: ParsedIntent
    scenes: list[Scene]
    palette: ColorPalette
    font_family: str
    music_mood: MusicMood
    total_duration_seconds: float


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------

class MusicTrack(BaseModel):
    track_id: str
    tags: list[str]
    tempo_bpm: int
    # In production this would be a URL/S3 key to a licensed audio file.
    # In the prototype it's a recipe for a synthesized placeholder tone bed.
    synth_recipe: dict


# --------------------------------------------------------------------------
# Render job
# --------------------------------------------------------------------------

class RenderOutputMeta(BaseModel):
    width: int
    height: int
    duration_seconds: float
    file_size_bytes: int
    codec: str = "h264"
    path: str


class RenderJob(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress_message: str = "Queued..."
    storyboard: Optional[StoryboardPlan] = None
    output_meta: Optional[RenderOutputMeta] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=False)
