"""Pure classification of Home Assistant state snapshots.

The adapter boundary is intentionally small: callers provide a mapping with a
``state`` value, optional ``attributes``, and optional entity metadata.  No
Home Assistant objects or imports are required here.  The results describe
observable context; they do not assert why a person chose that context.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar


class SecurityState(str, Enum):
    """Security state categories, ordered by the safety meaning of the state."""

    EMERGENCY = "emergency"
    ARMED_HOME = "armed_home"
    ARMED_NIGHT = "armed_night"
    ARMED_AWAY = "armed_away"
    DISARMED = "disarmed"
    PROBLEM = "problem"
    UNKNOWN = "unknown"


class MediaState(str, Enum):
    """Media categories inferred from a media-player snapshot."""

    VIDEO = "video"
    MOVIE = "movie"
    TV = "tv"
    MUSIC = "music"
    AUDIO = "audio"
    PODCAST = "podcast"
    GAME = "game"
    IDLE = "idle"
    UNKNOWN = "unknown"


class OpeningState(str, Enum):
    """Observable states for doors, windows, covers, and garage openings."""

    OPEN = "open"
    CLOSED = "closed"
    OPENING = "opening"
    CLOSING = "closing"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    JAMMED = "jammed"
    UNKNOWN = "unknown"


class SemanticIntent(str, Enum):
    """Conservative semantic labels for observable context.

    These values are context labels, not claims about a user's motivation.
    App-name and title fallbacks therefore retain lower confidence and a
    heuristic reason in their classification result.
    """

    EMERGENCY = "emergency"
    SECURITY_ARMED = "security_armed"
    SECURITY_DISARMED = "security_disarmed"
    SECURITY_PROBLEM = "security_problem"
    MEDIA_VIDEO = "video"
    MEDIA_AUDIO = "audio"
    MEDIA_GAME = "game"
    MEDIA_IDLE = "media_idle"
    OPENING_OPEN = "opening_open"
    OPENING_CLOSED = "opening_closed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Small HA-shaped value object accepted by the pure classifiers."""

    state: object | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    entity_id: str | None = None
    domain: str | None = None


CategoryT = TypeVar("CategoryT")


@dataclass(frozen=True, slots=True)
class Classification(Generic[CategoryT]):
    """A deterministic category plus auditable evidence."""

    category: CategoryT
    confidence: float
    reasons: tuple[str, ...]
    provenance: tuple[str, ...]
    semantic_intents: tuple[SemanticIntent, ...]
    context_only: bool = False

    @property
    def state(self) -> CategoryT:
        """Alias for callers that naturally refer to a classified state."""
        return self.category

    @property
    def semantic_intent(self) -> SemanticIntent | None:
        """Return the primary descriptive semantic label, if one exists."""
        return self.semantic_intents[0] if self.semantic_intents else None

    @property
    def is_known(self) -> bool:
        """Whether this result has enough evidence to be used as context."""
        return (
            self.confidence > 0 and getattr(self.category, "value", None) != "unknown"
        )

    @property
    def fails_closed(self) -> bool:
        """Whether the result intentionally authorizes no context action."""
        return not self.is_known


SecurityClassification = Classification[SecurityState]
MediaClassification = Classification[MediaState]
OpeningClassification = Classification[OpeningState]


@dataclass(frozen=True, slots=True)
class ContextClassification:
    """Combined context with safety-first semantic ordering."""

    security: SecurityClassification
    media: MediaClassification
    opening: OpeningClassification
    semantic_intents: tuple[SemanticIntent, ...]
    primary_semantic_intent: SemanticIntent
    confidence: float
    reasons: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def semantic_intent(self) -> SemanticIntent:
        """Return the highest-priority descriptive context label."""
        return self.primary_semantic_intent

    @property
    def emergency(self) -> bool:
        """Whether the combined snapshot contains a confirmed emergency."""
        return self.security.category is SecurityState.EMERGENCY


_UNKNOWN_TEXT = frozenset({"", "none", "null", "unknown", "unavailable"})
_IDLE_MEDIA_STATES = frozenset({"idle", "off", "standby"})
_ACTIVE_MEDIA_STATES = frozenset({"playing", "paused", "buffering", "on"})
_OPENING_DEVICE_CLASSES = frozenset(
    {"door", "window", "opening", "garage", "cover", "gate"},
)
_SAFETY_DEVICE_CLASSES = frozenset(
    {"safety", "smoke", "gas", "carbon_monoxide", "heat", "alarm"},
)
_SECURITY_STATUS_KEYS = (
    "alarm_state",
    "security_state",
    "alarm_status",
    "security_status",
)
_MEDIA_CONTENT_KEYS = ("media_content_type", "media_type", "content_type")
_MEDIA_APP_KEYS = ("app_id", "app_name")
_MEDIA_TITLE_KEYS = ("media_title", "title")


def _normalise(value: object) -> str:
    """Return a stable comparison form without coercing arbitrary objects."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _text(value: object) -> str:
    """Return trimmed text, preserving non-string values as unavailable."""
    return value.strip() if isinstance(value, str) else ""


def _snapshot_parts(snapshot: StateSnapshot | Mapping[str, object]) -> StateSnapshot:
    """Normalize a mapping or value object without importing HA types."""
    if isinstance(snapshot, StateSnapshot):
        return snapshot
    if not isinstance(snapshot, Mapping):
        return StateSnapshot()

    raw_attributes = snapshot.get("attributes")
    attributes: dict[str, object] = (
        dict(raw_attributes) if isinstance(raw_attributes, Mapping) else {}
    )
    for key in (
        "alarm_state",
        "security_state",
        "alarm_status",
        "security_status",
        "media_content_type",
        "media_type",
        "content_type",
        "app_id",
        "app_name",
        "media_title",
        "title",
        "device_class",
    ):
        if key in snapshot and key not in attributes:
            attributes[key] = snapshot[key]

    entity_id = snapshot.get("entity_id")
    domain = snapshot.get("domain")
    return StateSnapshot(
        state=snapshot.get("state"),
        attributes=attributes,
        entity_id=entity_id if isinstance(entity_id, str) else None,
        domain=domain if isinstance(domain, str) else None,
    )


def _attribute(snapshot: StateSnapshot, key: str) -> object:
    """Read one attribute without treating a missing value as meaningful."""
    return snapshot.attributes.get(key)


def _domain(snapshot: StateSnapshot) -> str:
    """Use explicit domain metadata, falling back to the entity-id prefix."""
    if snapshot.domain:
        return _normalise(snapshot.domain)
    if isinstance(snapshot.entity_id, str) and "." in snapshot.entity_id:
        return _normalise(snapshot.entity_id.split(".", 1)[0])
    return ""


def _provenance(snapshot: StateSnapshot, keys: tuple[str, ...]) -> tuple[str, ...]:
    """List supplied fields in a stable order for audit output."""
    fields: list[str] = []
    if snapshot.state is not None:
        fields.append("state")
    fields.extend(
        f"attributes.{key}" for key in keys if _text(_attribute(snapshot, key))
    )
    if snapshot.entity_id:
        fields.append("entity_id")
    if snapshot.domain:
        fields.append("domain")
    if _text(_attribute(snapshot, "device_class")):
        fields.append("attributes.device_class")
    return tuple(dict.fromkeys(fields))


def _has_token(value: str, tokens: frozenset[str]) -> bool:
    """Match a normalized state token, including compound alarm states."""
    normalized = _normalise(value)
    return normalized in tokens or bool(set(normalized.split("_")) & tokens)


def _security_candidate(
    value: object,
    *,
    domain: str,
    device_class: str,
) -> SecurityState | None:
    """Map an explicit security value, or return ``None`` when ambiguous."""
    normalized = _normalise(value)
    if normalized in _UNKNOWN_TEXT:
        return SecurityState.UNKNOWN

    if _has_token(
        normalized,
        frozenset({"triggered", "alarm", "fire", "panic", "emergency"}),
    ):
        return SecurityState.EMERGENCY

    armed = {
        "armed_home": SecurityState.ARMED_HOME,
        "armed_night": SecurityState.ARMED_NIGHT,
        "armed_away": SecurityState.ARMED_AWAY,
    }
    category = armed.get(normalized)
    if category is None and _has_token(
        normalized,
        frozenset({"problem", "fault", "error"}),
    ):
        category = SecurityState.PROBLEM
    if category is None and normalized == "disarmed":
        category = SecurityState.DISARMED
    if (
        category is None
        and normalized == "off"
        and (domain == "alarm_control_panel" or device_class in {"alarm", "security"})
    ):
        category = SecurityState.DISARMED
    if (
        category is None
        and normalized == "on"
        and (
            device_class in _SAFETY_DEVICE_CLASSES or domain == "alarm_control_panel"
        )
    ):
        category = SecurityState.EMERGENCY
    return category


def _security_result(
    category: SecurityState,
    *,
    reason: str,
    provenance: tuple[str, ...],
    confidence: float,
) -> SecurityClassification:
    semantic = {
        SecurityState.EMERGENCY: SemanticIntent.EMERGENCY,
        SecurityState.ARMED_HOME: SemanticIntent.SECURITY_ARMED,
        SecurityState.ARMED_NIGHT: SemanticIntent.SECURITY_ARMED,
        SecurityState.ARMED_AWAY: SemanticIntent.SECURITY_ARMED,
        SecurityState.DISARMED: SemanticIntent.SECURITY_DISARMED,
        SecurityState.PROBLEM: SemanticIntent.SECURITY_PROBLEM,
        SecurityState.UNKNOWN: SemanticIntent.UNKNOWN,
    }[category]
    return Classification(
        category=category,
        confidence=confidence,
        reasons=(reason,),
        provenance=provenance,
        semantic_intents=(semantic,),
    )


def classify_security_state(
    snapshot: StateSnapshot | Mapping[str, object],
) -> SecurityClassification:
    """Classify alarm/security state with emergency-first precedence."""
    parts = _snapshot_parts(snapshot)
    device_class = _normalise(_attribute(parts, "device_class"))
    domain = _domain(parts)
    values: list[tuple[str, object, float]] = [("state", parts.state, 1.0)]
    values.extend(
        (
            f"attributes.{key}",
            _attribute(parts, key),
            0.95,
        )
        for key in _SECURITY_STATUS_KEYS
        if _attribute(parts, key) is not None
    )

    candidates = [
        (
            field,
            value,
            confidence,
            _security_candidate(value, domain=domain, device_class=device_class),
        )
        for field, value, confidence in values
        if value is not None
    ]
    emergency = next(
        (
            candidate
            for candidate in candidates
            if candidate[3] is SecurityState.EMERGENCY
        ),
        None,
    )
    if emergency is not None:
        field, value, confidence, _ = emergency
        return _security_result(
            SecurityState.EMERGENCY,
            reason=f"{field} reported emergency condition '{_text(value)}'; emergency outranks all other context",
            provenance=_provenance(parts, _SECURITY_STATUS_KEYS),
            confidence=confidence,
        )

    for category in (
        SecurityState.ARMED_HOME,
        SecurityState.ARMED_NIGHT,
        SecurityState.ARMED_AWAY,
        SecurityState.PROBLEM,
        SecurityState.DISARMED,
    ):
        match = next(
            (candidate for candidate in candidates if candidate[3] is category), None,
        )
        if match is not None:
            field, value, confidence, _ = match
            return _security_result(
                category,
                reason=f"{field} reported '{_text(value)}'",
                provenance=_provenance(parts, _SECURITY_STATUS_KEYS),
                confidence=confidence,
            )

    reason = "security state is missing, unknown, unavailable, or not an explicit security value"
    if parts.state is not None:
        reason = f"security state '{_text(parts.state) or '<non-text>'}' is not safely classifiable"
    return _security_result(
        SecurityState.UNKNOWN,
        reason=reason,
        provenance=_provenance(parts, _SECURITY_STATUS_KEYS),
        confidence=0.0,
    )


def _content_category(value: object) -> MediaState | None:
    """Map an explicit media content type before considering app heuristics."""
    normalized = _normalise(value)
    categories = {
        **dict.fromkeys(("video", "clip", "visual"), MediaState.VIDEO),
        **dict.fromkeys(("movie", "film"), MediaState.MOVIE),
        **dict.fromkeys(
            ("tv", "tv_show", "tvshow", "show", "series", "episode"),
            MediaState.TV,
        ),
        **dict.fromkeys(
            ("music", "song", "track", "album", "playlist"),
            MediaState.MUSIC,
        ),
        **dict.fromkeys(("audio", "audiobook", "sound"), MediaState.AUDIO),
        **dict.fromkeys(("podcast", "podcasts"), MediaState.PODCAST),
        **dict.fromkeys(
            ("game", "gaming", "videogame", "video_game"),
            MediaState.GAME,
        ),
    }
    return categories.get(normalized)


def _app_category(value: object) -> MediaState | None:
    """Map common app IDs/names as a lower-confidence fallback."""
    normalized = _normalise(value)
    compact = normalized.replace("_", "")
    if any(
        marker in compact
        for marker in (
            "netflix",
            "plex",
            "youtube",
            "disney",
            "disneyplus",
            "primevideo",
            "amazonvideo",
            "appletv",
            "tvapp",
            "samsungtv",
        )
    ):
        return MediaState.VIDEO
    if any(
        marker in compact
        for marker in (
            "spotify",
            "applemusic",
            "applepodcasts",
            "podcast",
            "pocketcasts",
        )
    ):
        return (
            MediaState.PODCAST
            if "podcast" in compact or "casts" in compact
            else MediaState.MUSIC
        )
    if any(
        marker in compact
        for marker in (
            "playstation",
            "ps4",
            "ps5",
            "xbox",
            "nintendo",
            "steam",
            "game",
        )
    ):
        return MediaState.GAME
    return None


def _title_category(value: object) -> MediaState | None:
    """Use only unusually explicit title words as the weakest fallback."""
    normalized = _normalise(value)
    tokens = set(normalized.split("_"))
    category = None
    if "podcast" in tokens or "podcasts" in tokens:
        category = MediaState.PODCAST
    elif "movie" in tokens or "film" in tokens:
        category = MediaState.MOVIE
    elif "episode" in tokens or "season" in tokens or "tv" in tokens:
        category = MediaState.TV
    elif "game" in tokens or "gaming" in tokens:
        category = MediaState.GAME
    elif {"playlist", "album", "song", "track"} & tokens:
        category = MediaState.MUSIC
    elif "audio" in tokens:
        category = MediaState.AUDIO
    elif "video" in tokens:
        category = MediaState.VIDEO
    return category


def _media_semantic(category: MediaState) -> SemanticIntent:
    if category in {MediaState.VIDEO, MediaState.MOVIE, MediaState.TV}:
        return SemanticIntent.MEDIA_VIDEO
    if category in {MediaState.MUSIC, MediaState.AUDIO, MediaState.PODCAST}:
        return SemanticIntent.MEDIA_AUDIO
    if category is MediaState.GAME:
        return SemanticIntent.MEDIA_GAME
    if category is MediaState.IDLE:
        return SemanticIntent.MEDIA_IDLE
    return SemanticIntent.UNKNOWN


def _media_result(
    category: MediaState,
    *,
    reason: str,
    provenance: tuple[str, ...],
    confidence: float,
) -> MediaClassification:
    return Classification(
        category=category,
        confidence=confidence,
        reasons=(reason,),
        provenance=provenance,
        semantic_intents=(_media_semantic(category),),
    )


def classify_media_state(
    snapshot: StateSnapshot | Mapping[str, object],
) -> MediaClassification:
    """Classify media context using state, content type, app metadata, then title."""
    parts = _snapshot_parts(snapshot)
    state = _normalise(parts.state)
    provenance = _provenance(
        parts, _MEDIA_CONTENT_KEYS + _MEDIA_APP_KEYS + _MEDIA_TITLE_KEYS,
    )
    if state in _IDLE_MEDIA_STATES:
        return _media_result(
            MediaState.IDLE,
            reason=f"media player state '{_text(parts.state)}' is idle",
            provenance=provenance,
            confidence=1.0,
        )
    if state not in _ACTIVE_MEDIA_STATES:
        return _media_result(
            MediaState.UNKNOWN,
            reason="media player state is missing, unknown, unavailable, or not active",
            provenance=provenance,
            confidence=0.0,
        )

    for key in _MEDIA_CONTENT_KEYS:
        value = _attribute(parts, key)
        category = _content_category(value)
        if category is not None:
            return _media_result(
                category,
                reason=f"attributes.{key} explicitly reported '{_text(value)}'; explicit content type outranks app and title hints",
                provenance=provenance,
                confidence=1.0 if key == "media_content_type" else 0.95,
            )

    for key, confidence in (("app_id", 0.85), ("app_name", 0.75)):
        value = _attribute(parts, key)
        category = _app_category(value)
        if category is not None:
            return _media_result(
                category,
                reason=f"attributes.{key} '{_text(value)}' supplied a common-app heuristic; content type was not classifiable",
                provenance=provenance,
                confidence=confidence,
            )

    for key in _MEDIA_TITLE_KEYS:
        value = _attribute(parts, key)
        category = _title_category(value)
        if category is not None:
            return _media_result(
                category,
                reason=f"attributes.{key} '{_text(value)}' supplied a weak title heuristic; no explicit type or app match was available",
                provenance=provenance,
                confidence=0.55,
            )

    paused_note = "; paused state remains media context" if state == "paused" else ""
    return _media_result(
        MediaState.UNKNOWN,
        reason=f"media player is active but has no reliable content classification{paused_note}",
        provenance=provenance,
        confidence=0.0,
    )


def _opening_candidate(
    value: object,
    *,
    domain: str,
    device_class: str,
) -> OpeningState | None:
    """Map explicit opening/lock states while rejecting generic on/off states."""
    normalized = _normalise(value)
    if normalized in _UNKNOWN_TEXT:
        return OpeningState.UNKNOWN
    direct_states = {
        **dict.fromkeys(("open", "opened"), OpeningState.OPEN),
        **dict.fromkeys(("closed", "shut"), OpeningState.CLOSED),
        **dict.fromkeys(("opening", "opening_up"), OpeningState.OPENING),
        **dict.fromkeys(("closing", "closing_down"), OpeningState.CLOSING),
        **dict.fromkeys(("jammed", "stalled", "blocked"), OpeningState.JAMMED),
        "locked": OpeningState.LOCKED,
        "unlocked": OpeningState.UNLOCKED,
    }
    category = direct_states.get(normalized)
    if category is None and (
        device_class in _OPENING_DEVICE_CLASSES or domain in {"cover", "lock"}
    ):
        if normalized == "on":
            category = OpeningState.OPEN
        elif normalized == "off":
            category = OpeningState.CLOSED
    return category


def classify_opening_state(
    snapshot: StateSnapshot | Mapping[str, object],
) -> OpeningClassification:
    """Classify an opening for context only; never infer a user action."""
    parts = _snapshot_parts(snapshot)
    state = _text(parts.state)
    normalized_state = _normalise(parts.state)
    device_class = _normalise(_attribute(parts, "device_class"))
    domain = _domain(parts)
    candidate = _opening_candidate(
        parts.state,
        domain=domain,
        device_class=device_class,
    )
    provenance = _provenance(parts, ("device_class",))
    if candidate is None or candidate is OpeningState.UNKNOWN:
        reason = (
            "opening state is missing, unknown, unavailable, or not an opening entity"
        )
        if normalized_state:
            reason = f"opening state '{state}' is not safely classifiable"
        return Classification(
            category=OpeningState.UNKNOWN,
            confidence=0.0,
            reasons=(reason,),
            provenance=provenance,
            semantic_intents=(SemanticIntent.UNKNOWN,),
            context_only=True,
        )

    semantic = (
        SemanticIntent.OPENING_OPEN
        if candidate in {OpeningState.OPEN, OpeningState.OPENING}
        else SemanticIntent.OPENING_CLOSED
        if candidate in {OpeningState.CLOSED, OpeningState.CLOSING, OpeningState.LOCKED}
        else SemanticIntent.UNKNOWN
    )
    return Classification(
        category=candidate,
        confidence=1.0,
        reasons=(
            f"opening state '{state}' classified as {candidate.value}; context only",
        ),
        provenance=provenance,
        semantic_intents=(semantic,),
        context_only=True,
    )


def _nested_snapshot(
    snapshot: StateSnapshot | Mapping[str, object],
    names: tuple[str, ...],
) -> StateSnapshot | Mapping[str, object]:
    """Select an optional named section from a combined snapshot mapping."""
    if isinstance(snapshot, StateSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        for name in names:
            nested = snapshot.get(name)
            if isinstance(nested, (StateSnapshot, Mapping)):
                return nested
    return snapshot


def _unique(values: tuple[SemanticIntent, ...]) -> tuple[SemanticIntent, ...]:
    """Preserve deterministic order while removing duplicate labels."""
    return tuple(dict.fromkeys(values))


def classify_context(
    snapshot: StateSnapshot | Mapping[str, object],
) -> ContextClassification:
    """Classify security, media, and opening context with emergency precedence."""
    security = classify_security_state(
        _nested_snapshot(snapshot, ("security", "alarm", "security_state")),
    )
    media = classify_media_state(
        _nested_snapshot(snapshot, ("media", "media_player", "media_state")),
    )
    opening = classify_opening_state(
        _nested_snapshot(snapshot, ("opening", "door", "window", "cover", "garage")),
    )

    if security.category is SecurityState.EMERGENCY:
        semantic_intents = (SemanticIntent.EMERGENCY,)
        primary = SemanticIntent.EMERGENCY
        reasons = (
            *security.reasons,
            "security emergency outranks media and opening context",
        )
        confidence = security.confidence
    else:
        available_intents = tuple(
            intent
            for intent in (
                *security.semantic_intents,
                *media.semantic_intents,
                *opening.semantic_intents,
            )
            if intent is not SemanticIntent.UNKNOWN
        )
        semantic_intents = _unique(available_intents or (SemanticIntent.UNKNOWN,))
        primary = semantic_intents[0] if semantic_intents else SemanticIntent.UNKNOWN
        reasons = (*security.reasons, *media.reasons, *opening.reasons)
        known_confidences = [
            result.confidence
            for result in (security, media, opening)
            if result.is_known
        ]
        confidence = min(known_confidences) if known_confidences else 0.0

    provenance = tuple(
        dict.fromkeys((*security.provenance, *media.provenance, *opening.provenance)),
    )
    return ContextClassification(
        security=security,
        media=media,
        opening=opening,
        semantic_intents=semantic_intents,
        primary_semantic_intent=primary,
        confidence=confidence,
        reasons=reasons,
        provenance=provenance,
    )


def classify_security(
    snapshot: StateSnapshot | Mapping[str, object],
) -> SecurityClassification:
    """Short alias for :func:`classify_security_state`."""
    return classify_security_state(snapshot)


def classify_media(
    snapshot: StateSnapshot | Mapping[str, object],
) -> MediaClassification:
    """Short alias for :func:`classify_media_state`."""
    return classify_media_state(snapshot)


def classify_opening(
    snapshot: StateSnapshot | Mapping[str, object],
) -> OpeningClassification:
    """Short alias for :func:`classify_opening_state`."""
    return classify_opening_state(snapshot)


def classify_snapshot(
    snapshot: StateSnapshot | Mapping[str, object],
) -> ContextClassification:
    """Short alias for :func:`classify_context`."""
    return classify_context(snapshot)


__all__ = [
    "Classification",
    "ContextClassification",
    "MediaClassification",
    "MediaState",
    "OpeningClassification",
    "OpeningState",
    "SecurityClassification",
    "SecurityState",
    "SemanticIntent",
    "StateSnapshot",
    "classify_context",
    "classify_media",
    "classify_media_state",
    "classify_opening",
    "classify_opening_state",
    "classify_security",
    "classify_security_state",
    "classify_snapshot",
]
