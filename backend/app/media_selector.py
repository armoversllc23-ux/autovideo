"""
MediaSelector — decides, per scene, where the *visual* comes from.

Rule (per the brief):
  - If the user uploaded photos/clips, use them (cycled across scenes),
    with face-aware crop framing.
  - Otherwise, fetch a real, relevant stock photo (see stock_media.py) keyed
    to the occasion/tone/keywords — a genuinely photographic backdrop, not a
    placeholder card. The Renderer falls back to a gradient template only if
    the actual photo fetch fails (offline, no results, etc.), so this path
    always degrades gracefully rather than ever failing a render.

Notes:
  - Real face detection is still stubbed: `crop_strategy` is set to
    FACE_AWARE whenever user media is used, but the Renderer currently
    performs a plain center-crop with a marked TODO for where a
    face-detection model (e.g. mediapipe) would plug in.
  - `stock_query` is now a real search string sent to stock_media.py, built
    per occasion with per-scene-role variety so a multi-scene video doesn't
    show the same photo four times in a row.

Public contract: `plan_media(storyboard, uploaded_media) -> StoryboardPlan`
returns a NEW StoryboardPlan with every scene's `visual` field filled in.
"""
from __future__ import annotations

from pathlib import Path

from .models import CropStrategy, Occasion, SceneRole, StoryboardPlan, VisualPlan, VisualSourceType

# Base search terms per occasion — chosen for what real, freely-licensed
# stock libraries actually have good coverage of (generic, widely-
# photographed scenes/subjects), not overly specific phrasing that would
# return zero results.
_OCCASION_QUERY_TERMS: dict[Occasion, list[str]] = {
    Occasion.BIRTHDAY: ["birthday party celebration", "birthday cake candles", "confetti party balloons"],
    Occasion.WEDDING: ["wedding couple", "wedding rings flowers", "wedding celebration reception"],
    Occasion.ANNIVERSARY: ["couple celebrating together", "romantic dinner candles", "holding hands sunset"],
    Occasion.MEMORIAL: ["candle light peaceful", "sunset horizon calm", "white flowers remembrance"],
    Occasion.GRADUATION: ["graduation cap ceremony", "students celebrating graduation", "diploma achievement"],
    Occasion.RETIREMENT: ["office celebration party", "handshake congratulations", "sunset new beginning"],
    Occasion.BABY: ["newborn baby nursery", "baby shower decorations", "tiny baby feet"],
    Occasion.HOLIDAY: ["holiday lights festive", "christmas decorations cozy", "family gathering winter"],
    Occasion.PRODUCT_PROMO: ["product photography studio", "modern retail shop", "flat lay product"],
    Occasion.REAL_ESTATE: ["modern home interior", "house exterior sunny", "living room bright"],
    Occasion.TRAVEL: ["travel landscape adventure", "road trip scenic view", "airplane window travel"],
    Occasion.GENERAL_CELEBRATION: ["celebration friends toast", "confetti party lights", "people laughing together"],
    Occasion.OTHER: ["lifestyle everyday moment", "warm sunlight window", "hands together community"],
}

# Which query variant (index into the occasion's list above) each scene
# role prefers, for a bit of intentional visual rhythm: INTRO opens on the
# "headline" image, HIGHLIGHT gets the most celebratory/dramatic one, etc.
_ROLE_QUERY_INDEX = {
    SceneRole.INTRO: 0,
    SceneRole.HIGHLIGHT: 1,
    SceneRole.STORY_BEAT: 2,
    SceneRole.CLOSING: 0,
}


class MediaSelector:
    """See module docstring for the selection rules."""

    def plan_media(
        self,
        storyboard: StoryboardPlan,
        uploaded_media: list[Path] | None = None,
        variant_seed: int = 0,
    ) -> StoryboardPlan:
        uploaded_media = uploaded_media or []
        new_scenes = []

        for i, scene in enumerate(storyboard.scenes):
            if uploaded_media:
                visual = self._plan_from_user_media(uploaded_media, i, variant_seed)
            else:
                visual = self._plan_without_user_media(storyboard, scene, i, variant_seed)
            new_scenes.append(scene.model_copy(update={"visual": visual}))

        return storyboard.model_copy(update={"scenes": new_scenes})

    # -- strategies ----------------------------------------------------

    def _plan_from_user_media(self, uploaded_media: list[Path], scene_index: int, variant_seed: int = 0) -> VisualPlan:
        # Round-robin: cycles back through the user's photos if there are
        # fewer photos than scenes, so every scene still gets a visual.
        # variant_seed offsets the starting point, so "Try a different
        # version" pairs different photos with different scenes without
        # touching the script.
        media_path = uploaded_media[(scene_index + variant_seed) % len(uploaded_media)]
        return VisualPlan(
            source_type=VisualSourceType.USER_MEDIA,
            user_media_ref=str(media_path),
            crop_strategy=CropStrategy.FACE_AWARE,
        )

    def _plan_without_user_media(self, storyboard: StoryboardPlan, scene, scene_index: int, variant_seed: int = 0) -> VisualPlan:
        intent = storyboard.intent
        role = scene.role
        query = self._build_stock_query(intent, role, scene_index, variant_seed)
        return VisualPlan(
            source_type=VisualSourceType.STOCK,
            stock_query=query,
            crop_strategy=CropStrategy.CENTER,
        )

    def _build_stock_query(self, intent, role: SceneRole, scene_index: int, variant_seed: int) -> str:
        variants = _OCCASION_QUERY_TERMS.get(intent.occasion, _OCCASION_QUERY_TERMS[Occasion.OTHER])
        # Repeated STORY_BEAT scenes in a longer video should cycle through
        # the occasion's variants rather than repeat the same photo.
        base_idx = _ROLE_QUERY_INDEX.get(role, 2)
        idx = (base_idx + scene_index + variant_seed) % len(variants)
        query = variants[idx]

        # A user-supplied keyword (e.g. "coffee shop", a name, a place) adds
        # real specificity when present — most useful for promo/other scenes
        # where the occasion alone is generic.
        extra_keyword = next(
            (k for k in intent.keywords if k not in query.split()), None
        )
        if extra_keyword and intent.occasion in {Occasion.PRODUCT_PROMO, Occasion.OTHER, Occasion.GENERAL_CELEBRATION}:
            query = f"{query} {extra_keyword}"
        return query
