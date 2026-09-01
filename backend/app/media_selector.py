"""
MediaSelector — decides, per scene, where the *visual* comes from.

Rule (per the brief):
  - If the user uploaded photos/clips, use them (cycled across scenes),
    with face-aware crop framing.
  - Otherwise, choose between a template background, a stock asset, or a
    (future) AI-generated image, depending on occasion.

Prototype implementation notes:
  - Real face detection is stubbed: `crop_strategy` is set to FACE_AWARE
    whenever user media is used, but the Renderer currently performs a
    plain center-crop with a clearly marked TODO for where a face-detection
    model (e.g. mediapipe) would plug in to shift the crop box.
  - Stock/AI retrieval is stubbed: `stock_query`/`ai_prompt` are populated
    with what a real API call would be given, so the abstraction boundary
    is real, but the Renderer paints a clearly-labeled placeholder card
    instead of fetching/generating an actual image (no network dependency,
    no licensing questions, for this prototype).

Public contract: `plan_media(storyboard, uploaded_media) -> StoryboardPlan`
returns a NEW StoryboardPlan with every scene's `visual` field filled in.
"""
from __future__ import annotations

from pathlib import Path

from .models import CropStrategy, Occasion, SceneRole, StoryboardPlan, VisualPlan, VisualSourceType

# Occasions where, absent user photos, a relevant stock photo/clip is a much
# stronger default than an abstract template (a listing needs to look like
# a home; a product promo needs to look like a product).
_STOCK_PREFERRED_OCCASIONS = {Occasion.PRODUCT_PROMO, Occasion.REAL_ESTATE}


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
                visual = self._plan_without_user_media(storyboard, scene.role, i, variant_seed)
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

    def _plan_without_user_media(
        self, storyboard: StoryboardPlan, role: SceneRole, scene_index: int, variant_seed: int = 0
    ) -> VisualPlan:
        intent = storyboard.intent
        if intent.occasion in _STOCK_PREFERRED_OCCASIONS:
            query_subject = intent.keywords[0] if intent.keywords else intent.occasion.value
            return VisualPlan(
                source_type=VisualSourceType.STOCK,
                stock_query=f"{intent.occasion.value.replace('_', ' ')} {query_subject}".strip(),
                crop_strategy=CropStrategy.CENTER,
            )

        if role == SceneRole.HIGHLIGHT:
            # Demonstrates the AI-generated-imagery interface: the single
            # most important scene gets a bespoke prompt built from the
            # storyboard's own caption, ready for an image-gen API later.
            highlight_caption = next(
                (s.caption for s in storyboard.scenes if s.role == SceneRole.HIGHLIGHT), ""
            )
            prompt = (
                f"{intent.tone.value} {intent.occasion.value.replace('_', ' ')} scene, "
                f"'{highlight_caption}', cinematic lighting, no text"
            )
            return VisualPlan(source_type=VisualSourceType.AI_GENERATED, ai_prompt=prompt)

        return VisualPlan(
            source_type=VisualSourceType.TEMPLATE,
            template_id=f"gradient_{intent.tone.value}_{(scene_index + variant_seed) % 3}",
            crop_strategy=CropStrategy.CENTER,
        )
