"""
StoryboardGenerator — turns a ParsedIntent into a full StoryboardPlan:
3-6 scenes with roles/captions/narration, an emotional arc, a color palette,
typography, per-scene pacing, and a music mood.

Prototype implementation: deterministic templates keyed by (occasion, tone),
with slot-filling for the subject name and salient keywords. The public
contract — `generate(intent) -> StoryboardPlan` — is what an LLM-backed
scriptwriter would also expose (see ARCHITECTURE.md section 7): a future
version can call an LLM to fill in `caption`/`narration` text while keeping
scene count, roles, palette, pacing, and music-mood logic here (or also
delegated), because every stage returns the same `StoryboardPlan` shape.
"""
from __future__ import annotations

from .models import (
    ColorPalette,
    MusicMood,
    Occasion,
    ParsedIntent,
    Scene,
    SceneRole,
    StoryboardPlan,
    Tone,
    TransitionType,
    VisualPlan,
)

# --------------------------------------------------------------------------
# Palette + typography per tone. Occasion can nudge palette (see _palette_for).
# --------------------------------------------------------------------------

_PALETTE_BY_TONE: dict[Tone, ColorPalette] = {
    Tone.FUN: ColorPalette(primary="#FF5D8F", secondary="#FFC93C", accent="#4CC9F0"),
    Tone.EMOTIONAL: ColorPalette(primary="#5B4B8A", secondary="#F2C6C2", accent="#FFFFFF"),
    Tone.ELEGANT: ColorPalette(primary="#1B1B1B", secondary="#C9A959", accent="#F5F1E8"),
    Tone.CALM: ColorPalette(primary="#7FA9A0", secondary="#E8E3D3", accent="#FFFFFF"),
    Tone.BOLD: ColorPalette(primary="#E63946", secondary="#1D3557", accent="#F1FAEE"),
    Tone.WARM: ColorPalette(primary="#E08E45", secondary="#F4D35E", accent="#FFF8EE"),
    Tone.PROFESSIONAL: ColorPalette(primary="#1F3A5F", secondary="#4A90D9", accent="#F4F6F8"),
}

_FONT_BY_TONE: dict[Tone, str] = {
    Tone.FUN: "Poppins-Bold",
    Tone.EMOTIONAL: "PlayfairDisplay-Regular",
    Tone.ELEGANT: "PlayfairDisplay-Regular",
    Tone.CALM: "Lato-Regular",
    Tone.BOLD: "Montserrat-Black",
    Tone.WARM: "Quicksand-Medium",
    Tone.PROFESSIONAL: "Inter-Medium",
}

_MUSIC_BY_TONE: dict[Tone, MusicMood] = {
    Tone.FUN: MusicMood(tags=["upbeat", "playful"], tempo_bpm_range=(115, 130)),
    Tone.EMOTIONAL: MusicMood(tags=["gentle", "reflective", "slow"], tempo_bpm_range=(60, 80)),
    Tone.ELEGANT: MusicMood(tags=["elegant", "sparse", "piano"], tempo_bpm_range=(80, 100)),
    Tone.CALM: MusicMood(tags=["gentle", "ambient", "slow"], tempo_bpm_range=(65, 85)),
    Tone.BOLD: MusicMood(tags=["epic", "driving", "cinematic"], tempo_bpm_range=(120, 140)),
    Tone.WARM: MusicMood(tags=["uplifting", "warm", "medium-tempo"], tempo_bpm_range=(95, 115)),
    Tone.PROFESSIONAL: MusicMood(tags=["clean", "corporate", "medium-tempo"], tempo_bpm_range=(100, 118)),
}


def _palette_for(intent: ParsedIntent) -> ColorPalette:
    base = _PALETTE_BY_TONE[intent.tone]
    if intent.occasion == Occasion.MEMORIAL:
        # Force a gentle, muted palette regardless of detected tone, since
        # this occasion has a near-universal visual convention.
        return ColorPalette(primary="#3A3A50", secondary="#D9CBB8", accent="#FFFFFF")
    return base


# --------------------------------------------------------------------------
# Scene count + role sequence by target length.
# --------------------------------------------------------------------------

def _scene_roles_for_length(length_seconds: int) -> list[SceneRole]:
    if length_seconds <= 18:
        return [SceneRole.INTRO, SceneRole.HIGHLIGHT, SceneRole.CLOSING]
    if length_seconds <= 40:
        return [SceneRole.INTRO, SceneRole.HIGHLIGHT, SceneRole.STORY_BEAT, SceneRole.CLOSING]
    if length_seconds <= 75:
        return [
            SceneRole.INTRO, SceneRole.HIGHLIGHT, SceneRole.STORY_BEAT,
            SceneRole.STORY_BEAT, SceneRole.CLOSING,
        ]
    return [
        SceneRole.INTRO, SceneRole.HIGHLIGHT, SceneRole.STORY_BEAT,
        SceneRole.STORY_BEAT, SceneRole.STORY_BEAT, SceneRole.CLOSING,
    ]


# Relative weight of each role's duration (intro/closing shorter, story
# beats and highlight get more room to breathe). Normalized to total length.
_ROLE_WEIGHT = {
    SceneRole.INTRO: 0.8,
    SceneRole.HIGHLIGHT: 1.3,
    SceneRole.STORY_BEAT: 1.1,
    SceneRole.CLOSING: 0.9,
}

_TRANSITION_BY_ROLE = {
    SceneRole.INTRO: TransitionType.CUT,
    SceneRole.HIGHLIGHT: TransitionType.ZOOM,
    SceneRole.STORY_BEAT: TransitionType.CROSSFADE,
    SceneRole.CLOSING: TransitionType.CROSSFADE,
}


# --------------------------------------------------------------------------
# Caption / narration templates, keyed by occasion. `{subject}` and
# `{keyword}` are slot-filled. Each template list must have an entry usable
# for every SceneRole that can appear (see _scene_roles_for_length).
# --------------------------------------------------------------------------

def _default_subject(intent: ParsedIntent) -> str:
    if intent.subject_name:
        return intent.subject_name
    fallback = {
        Occasion.MEMORIAL: "a life well lived",
        Occasion.PRODUCT_PROMO: "something new",
        Occasion.REAL_ESTATE: "your next home",
    }
    return fallback.get(intent.occasion, "you")


_TEMPLATES: dict[Occasion, dict[SceneRole, list[tuple[str, str]]]] = {
    Occasion.BIRTHDAY: {
        SceneRole.INTRO: [("Happy Birthday, {subject}! \U0001F389", "Today we're celebrating {subject}.")],
        SceneRole.HIGHLIGHT: [("So many reasons to celebrate", "Every year with {subject} is a gift.")],
        SceneRole.STORY_BEAT: [
            ("Here's to more memories together", "We can't wait to make more memories with {subject}."),
            ("So loved, today and always", "{subject}, you are so loved."),
            ("Another year of laughter ahead", "Here's to another year of laughter with {subject}."),
        ],
        SceneRole.CLOSING: [("Happy Birthday! \U0001F382", "Happy Birthday, {subject} — we love you.")],
    },
    Occasion.WEDDING: {
        SceneRole.INTRO: [("Two hearts, one story", "This is where their story began.")],
        SceneRole.HIGHLIGHT: [("A love worth celebrating", "A love worth celebrating, today and always.")],
        SceneRole.STORY_BEAT: [
            ("Forever starts today", "From this day forward."),
            ("Two families, one story", "Two families becoming one."),
            ("Every love story is beautiful", "But this one is our favorite."),
        ],
        SceneRole.CLOSING: [("Congratulations!", "Congratulations to the happy couple.")],
    },
    Occasion.ANNIVERSARY: {
        SceneRole.INTRO: [("Happy Anniversary!", "Celebrating another year of love.")],
        SceneRole.HIGHLIGHT: [("Still going strong", "Still going strong, year after year.")],
        SceneRole.STORY_BEAT: [
            ("Here's to many more", "Here's to many more years together."),
            ("Still writing the story", "Still writing the best story together."),
            ("Love that only grows", "A love that only grows with time."),
        ],
        SceneRole.CLOSING: [("Happy Anniversary!", "Happy Anniversary — we love you both.")],
    },
    Occasion.MEMORIAL: {
        SceneRole.INTRO: [("In loving memory of {subject}", "In loving memory of {subject}.")],
        SceneRole.HIGHLIGHT: [("A life full of love", "{subject} touched so many lives.")],
        SceneRole.STORY_BEAT: [
            ("Forever in our hearts", "We carry {subject} with us, always."),
            ("The memories remain", "The memories of {subject} will never fade."),
            ("Gone but never forgotten", "{subject} is gone but never forgotten."),
        ],
        SceneRole.CLOSING: [("Forever remembered", "Forever loved, forever remembered.")],
    },
    Occasion.GRADUATION: {
        SceneRole.INTRO: [("The journey continues", "{subject} did it!")],
        SceneRole.HIGHLIGHT: [("So proud of this moment", "So proud of everything {subject} accomplished.")],
        SceneRole.STORY_BEAT: [
            ("The best is yet to come", "This is just the beginning for {subject}."),
            ("Years of hard work paid off", "All those late nights led to this."),
            ("Onward to what's next", "The world is waiting, {subject}."),
        ],
        SceneRole.CLOSING: [("Congrats, Grad!", "Congratulations, {subject} — go make us proud.")],
    },
    Occasion.RETIREMENT: {
        SceneRole.INTRO: [("Cheers to the next chapter", "It's time to celebrate {subject}.")],
        SceneRole.HIGHLIGHT: [("Decades of hard work", "Thank you for everything, {subject}.")],
        SceneRole.STORY_BEAT: [
            ("Now for some well-earned rest", "Enjoy every moment of what's next."),
            ("A career to be proud of", "What a career to look back on."),
            ("New adventures await", "Here's to whatever comes next, {subject}."),
        ],
        SceneRole.CLOSING: [("Happy Retirement!", "Happy Retirement, {subject}!")],
    },
    Occasion.BABY: {
        SceneRole.INTRO: [("A new little one is here", "Introducing the newest member of the family.")],
        SceneRole.HIGHLIGHT: [("So much love already", "So much love waiting for this little one.")],
        SceneRole.STORY_BEAT: [
            ("Tiny fingers, big dreams", "A whole new adventure begins."),
            ("Our family is growing", "Our hearts just got a little bigger."),
            ("So much love to give", "There's already so much love for this little one."),
        ],
        SceneRole.CLOSING: [("Welcome to the world!", "Welcome to the world, little one.")],
    },
    Occasion.HOLIDAY: {
        SceneRole.INTRO: [("Season's Greetings!", "Wishing you joy this season.")],
        SceneRole.HIGHLIGHT: [("Making memories together", "The best part of the season: being together.")],
        SceneRole.STORY_BEAT: [
            ("Warm wishes to you", "Sending warm wishes your way."),
            ("Gathered with the ones we love", "Nothing beats being together this time of year."),
            ("Cheers to the season", "Cheers to good times and good company."),
        ],
        SceneRole.CLOSING: [("Happy Holidays!", "Happy Holidays, from our family to yours.")],
    },
    Occasion.PRODUCT_PROMO: {
        SceneRole.INTRO: [("Something new has arrived", "Meet {keyword}.")],
        SceneRole.HIGHLIGHT: [("Made for you", "Designed with you in mind.")],
        SceneRole.STORY_BEAT: [
            ("Don't miss out", "Available now, for a limited time."),
            ("Loved by people like you", "See why people can't stop talking about it."),
            ("Quality you can feel", "Made with quality you can feel."),
        ],
        SceneRole.CLOSING: [("Get yours today", "Shop now and see for yourself.")],
    },
    Occasion.REAL_ESTATE: {
        SceneRole.INTRO: [("Welcome home", "Welcome to {subject}.")],
        SceneRole.HIGHLIGHT: [("Every detail, thoughtfully designed", "Every room tells a story.")],
        SceneRole.STORY_BEAT: [
            ("Room for every moment", "Space to live, love, and grow."),
            ("A neighborhood you'll love", "In a neighborhood you'll love coming home to."),
            ("Picture yourself here", "Can you picture yourself here?"),
        ],
        SceneRole.CLOSING: [("Schedule your tour today", "Reach out today to schedule a tour.")],
    },
    Occasion.TRAVEL: {
        SceneRole.INTRO: [("Our adventure begins", "Come along on our trip.")],
        SceneRole.HIGHLIGHT: [("Unforgettable moments", "So many unforgettable moments.")],
        SceneRole.STORY_BEAT: [
            ("Every mile worth it", "Every mile was worth it."),
            ("Getting a little lost, on purpose", "Some of the best moments were the unplanned ones."),
            ("New places, new stories", "Every stop had its own story."),
        ],
        SceneRole.CLOSING: [("Until next time", "Until the next adventure.")],
    },
    Occasion.GENERAL_CELEBRATION: {
        SceneRole.INTRO: [("A moment worth celebrating", "This is a moment worth celebrating.")],
        SceneRole.HIGHLIGHT: [("So much to be grateful for", "So much to be grateful for today.")],
        SceneRole.STORY_BEAT: [
            ("Here's to this moment", "Here's to this moment, and many more."),
            ("Worth celebrating together", "Some things are worth celebrating together."),
            ("A moment to remember", "This is one to remember."),
        ],
        SceneRole.CLOSING: [("Cheers!", "Cheers to {subject}.")],
    },
    Occasion.OTHER: {
        SceneRole.INTRO: [("A story worth telling", "Here's a story worth telling.")],
        SceneRole.HIGHLIGHT: [("The moments that matter", "The moments that matter most.")],
        SceneRole.STORY_BEAT: [
            ("Made with care", "Made with care, just for you."),
            ("A story still unfolding", "This story is still being written."),
            ("Every detail matters", "Every detail here matters."),
        ],
        SceneRole.CLOSING: [("Thanks for watching", "Thanks for being part of this.")],
    },
}


class StoryboardGenerator:
    """Deterministic, template-driven storyboard builder. See module docstring."""

    def generate(self, intent: ParsedIntent) -> StoryboardPlan:
        roles = _scene_roles_for_length(intent.length_seconds)
        durations = self._durations_for(roles, intent.length_seconds)
        subject = _default_subject(intent)
        keyword = intent.keywords[0] if intent.keywords else "our story"

        scenes: list[Scene] = []
        template_for_occasion = _TEMPLATES.get(intent.occasion, _TEMPLATES[Occasion.OTHER])

        # Track which template variant index to use per role, so repeated
        # STORY_BEAT roles (long videos) don't repeat the exact same line.
        role_use_count: dict[SceneRole, int] = {}

        for i, (role, duration) in enumerate(zip(roles, durations)):
            variants = template_for_occasion.get(role) or _TEMPLATES[Occasion.OTHER][role]
            use_idx = role_use_count.get(role, 0) % len(variants)
            role_use_count[role] = role_use_count.get(role, 0) + 1
            caption_tpl, narration_tpl = variants[use_idx]

            caption = caption_tpl.format(subject=subject, keyword=keyword)
            narration = narration_tpl.format(subject=subject, keyword=keyword)

            scenes.append(
                Scene(
                    index=i,
                    role=role,
                    caption=caption,
                    narration=narration,
                    duration_seconds=round(duration, 2),
                    visual=VisualPlan(),  # filled in later by MediaSelector
                    transition_in=_TRANSITION_BY_ROLE[role],
                )
            )

        return StoryboardPlan(
            intent=intent,
            scenes=scenes,
            palette=_palette_for(intent),
            font_family=_FONT_BY_TONE[intent.tone],
            music_mood=_MUSIC_BY_TONE[intent.tone],
            total_duration_seconds=round(sum(durations), 2),
        )

    def _durations_for(self, roles: list[SceneRole], target_total: float) -> list[float]:
        weights = [_ROLE_WEIGHT[r] for r in roles]
        weight_sum = sum(weights)
        raw = [target_total * (w / weight_sum) for w in weights]
        # Enforce a sane minimum so no scene is a flash-cut.
        min_duration = 2.0
        raw = [max(min_duration, d) for d in raw]
        # Rescale to hit the target total exactly (minimum-floor can drift it).
        scale = target_total / sum(raw) if sum(raw) else 1.0
        return [d * scale for d in raw]
