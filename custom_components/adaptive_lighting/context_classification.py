"""Pure classification of Home Assistant state snapshots.

The adapter boundary is intentionally small: callers provide a mapping with a
``state`` value, optional ``attributes``, and optional entity metadata.  No
Home Assistant objects or imports are required here.  The results describe
observable context; they do not assert why a person chose that context.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
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


class WeatherDaylightState(str, Enum):
    """Coarse weather or daylight categories for advisory context."""

    STORM = "storm"
    RAIN = "rain"
    CLOUDY = "cloudy"
    BRIGHT = "bright"
    DIM = "dim"
    DARK = "dark"
    UNKNOWN = "unknown"


class HouseholdState(str, Enum):
    """Explicit aggregate household presence states."""

    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class ArrivalState(str, Enum):
    """Bounded arrival states; stale positive signals are not actionable."""

    RECENT = "recent"
    INACTIVE = "inactive"
    STALE = "stale"
    UNKNOWN = "unknown"


class SleepState(str, Enum):
    """Explicit sleep context used only for combined priority ordering."""

    SLEEPING = "sleeping"
    AWAKE = "awake"
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
    SLEEP = "sleep"
    RECENT_ARRIVAL = "recent_arrival"
    HOUSEHOLD_HOME = "household_home"
    HOUSEHOLD_AWAY = "household_away"
    WEATHER_STORM = "weather_storm"
    WEATHER_RAIN = "weather_rain"
    WEATHER_CLOUDY = "weather_cloudy"
    DAYLIGHT_BRIGHT = "daylight_bright"
    DAYLIGHT_DIM = "daylight_dim"
    DAYLIGHT_DARK = "daylight_dark"
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
WeatherDaylightClassification = Classification[WeatherDaylightState]
HouseholdClassification = Classification[HouseholdState]
ArrivalClassification = Classification[ArrivalState]
SleepClassification = Classification[SleepState]


@dataclass(frozen=True, slots=True)
class ContextClassification:
    """Combined context with safety-first semantic ordering."""

    security: SecurityClassification
    sleep: SleepClassification
    arrival: ArrivalClassification
    media: MediaClassification
    weather_daylight: WeatherDaylightClassification
    household: HouseholdClassification
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
_WEATHER_KEYS = (
    "weather_state",
    "condition",
    "cloud_coverage",
    "cloudiness",
    "precipitation",
    "precipitation_rate",
    "sun_elevation",
    "elevation",
    "solar_irradiance",
    "irradiance",
    "pv_power",
    "solar_power",
)
_HOUSEHOLD_KEYS = ("home", "occupied", "person_states", "persons", "people")
_ARRIVAL_KEYS = (
    "arrival",
    "arrived",
    "arrival_signal",
    "observed_at",
    "age",
    "age_seconds",
    "max_age",
    "max_age_seconds",
    "now",
    "evaluated_at",
)
_SLEEP_KEYS = ("sleep", "sleeping", "sleep_mode")


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
        *_SECURITY_STATUS_KEYS,
        *_MEDIA_CONTENT_KEYS,
        *_MEDIA_APP_KEYS,
        *_MEDIA_TITLE_KEYS,
        *_WEATHER_KEYS,
        *_HOUSEHOLD_KEYS,
        *_ARRIVAL_KEYS,
        *_SLEEP_KEYS,
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
        f"attributes.{key}"
        for key in keys
        if _attribute(snapshot, key) is not None
    )
    if snapshot.entity_id:
        fields.append("entity_id")
    if snapshot.domain:
        fields.append("domain")
    if _attribute(snapshot, "device_class") is not None:
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


def _number(value: object) -> float | None:
    """Parse a finite numeric sensor value without accepting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str) and _normalise(value) not in _UNKNOWN_TEXT:
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if isfinite(parsed) else None


def _first_numeric(
    snapshot: StateSnapshot,
    keys: tuple[str, ...],
) -> tuple[str, float] | None:
    """Return the first finite numeric attribute in deterministic key order."""
    for key in keys:
        value = _number(_attribute(snapshot, key))
        if value is not None:
            return f"attributes.{key}", value
    entity_id = _normalise(snapshot.entity_id)
    state_value = _number(snapshot.state)
    if state_value is not None and any(
        _normalise(key) in entity_id for key in keys
    ):
        return "state", state_value
    return None


def _weather_semantic(category: WeatherDaylightState) -> SemanticIntent:
    return {
        WeatherDaylightState.STORM: SemanticIntent.WEATHER_STORM,
        WeatherDaylightState.RAIN: SemanticIntent.WEATHER_RAIN,
        WeatherDaylightState.CLOUDY: SemanticIntent.WEATHER_CLOUDY,
        WeatherDaylightState.BRIGHT: SemanticIntent.DAYLIGHT_BRIGHT,
        WeatherDaylightState.DIM: SemanticIntent.DAYLIGHT_DIM,
        WeatherDaylightState.DARK: SemanticIntent.DAYLIGHT_DARK,
        WeatherDaylightState.UNKNOWN: SemanticIntent.UNKNOWN,
    }[category]


def _weather_result(
    category: WeatherDaylightState,
    *,
    confidence: float,
    reason: str,
    provenance: tuple[str, ...],
) -> WeatherDaylightClassification:
    return Classification(
        category=category,
        confidence=confidence,
        reasons=(reason,),
        provenance=provenance,
        semantic_intents=(_weather_semantic(category),),
        context_only=True,
    )


def classify_weather_daylight(  # noqa: PLR0911, PLR0912
    snapshot: StateSnapshot | Mapping[str, object],
) -> WeatherDaylightClassification:
    """Classify coarse weather/daylight context from explicit sensor evidence.

    Solar irradiance is direct outdoor light evidence.  Positive PV production
    is only a lower-confidence daylight proxy: it is never described as lux,
    illuminance, circadian, or melanopic evidence, and zero PV does not prove
    darkness.
    """
    parts = _snapshot_parts(snapshot)
    provenance = _provenance(parts, _WEATHER_KEYS)
    condition_values = (
        ("state", parts.state),
        ("attributes.weather_state", _attribute(parts, "weather_state")),
        ("attributes.condition", _attribute(parts, "condition")),
    )
    conditions = tuple(
        (field, _normalise(value), _text(value))
        for field, value in condition_values
        if _normalise(value) not in _UNKNOWN_TEXT
    )

    storm = next(
        (
            item
            for item in conditions
            if _has_token(
                item[1],
                frozenset({"storm", "stormy", "thunderstorm", "lightning"}),
            )
        ),
        None,
    )
    if storm is not None:
        return _weather_result(
            WeatherDaylightState.STORM,
            confidence=1.0,
            reason=f"{storm[0]} reported storm condition '{storm[2]}'",
            provenance=provenance,
        )

    rain = next(
        (
            item
            for item in conditions
            if _has_token(
                item[1],
                frozenset({"rain", "rainy", "pouring", "hail", "snowy"}),
            )
        ),
        None,
    )
    precipitation = _first_numeric(
        parts,
        ("precipitation", "precipitation_rate"),
    )
    if rain is not None or (precipitation is not None and precipitation[1] > 0):
        source, detail = (
            (rain[0], f"condition '{rain[2]}'")
            if rain is not None
            else (precipitation[0], f"positive precipitation {precipitation[1]:g}")
        )
        return _weather_result(
            WeatherDaylightState.RAIN,
            confidence=0.95,
            reason=f"{source} reported {detail}",
            provenance=provenance,
        )

    cloudy = next(
        (
            item
            for item in conditions
            if item[1] in {"cloudy", "partlycloudy", "partly_cloudy", "fog"}
        ),
        None,
    )
    cloud_coverage = _first_numeric(parts, ("cloud_coverage", "cloudiness"))
    if cloudy is not None or (
        cloud_coverage is not None and cloud_coverage[1] >= 70
    ):
        source, detail = (
            (cloudy[0], f"condition '{cloudy[2]}'")
            if cloudy is not None
            else (
                cloud_coverage[0],
                f"cloud coverage {cloud_coverage[1]:g}%",
            )
        )
        return _weather_result(
            WeatherDaylightState.CLOUDY,
            confidence=0.9,
            reason=f"{source} reported {detail}",
            provenance=provenance,
        )

    sun = _first_numeric(parts, ("sun_elevation", "elevation"))
    if sun is not None:
        if sun[1] <= -6:
            category = WeatherDaylightState.DARK
        elif sun[1] < 6:
            category = WeatherDaylightState.DIM
        else:
            category = WeatherDaylightState.BRIGHT
        return _weather_result(
            category,
            confidence=0.95,
            reason=f"{sun[0]} reported sun elevation {sun[1]:g} degrees",
            provenance=provenance,
        )

    irradiance = _first_numeric(parts, ("solar_irradiance", "irradiance"))
    if irradiance is not None and irradiance[1] > 0:
        category = (
            WeatherDaylightState.BRIGHT
            if irradiance[1] >= 120
            else WeatherDaylightState.DIM
        )
        return _weather_result(
            category,
            confidence=0.85,
            reason=f"{irradiance[0]} reported outdoor solar irradiance {irradiance[1]:g}",
            provenance=provenance,
        )

    pv = _first_numeric(parts, ("pv_power", "solar_power"))
    if pv is not None and pv[1] > 50:
        return _weather_result(
            WeatherDaylightState.BRIGHT,
            confidence=0.65,
            reason=f"{pv[0]} reported positive PV production {pv[1]:g}; PV is a daylight proxy, not lux or melanopic light",
            provenance=provenance,
        )

    clear_night = next((item for item in conditions if item[1] == "clear_night"), None)
    if clear_night is not None:
        return _weather_result(
            WeatherDaylightState.DARK,
            confidence=0.9,
            reason=f"{clear_night[0]} reported explicit clear-night condition",
            provenance=provenance,
        )
    sunny = next((item for item in conditions if item[1] in {"sunny", "clear"}), None)
    if sunny is not None:
        return _weather_result(
            WeatherDaylightState.BRIGHT,
            confidence=0.75,
            reason=f"{sunny[0]} reported '{sunny[2]}' without direct light measurement",
            provenance=provenance,
        )

    reason = "weather and daylight values are missing, unavailable, or not safely classifiable"
    if pv is not None and pv[1] == 0:
        reason = f"{pv[0]} reported zero PV; zero production does not prove darkness and is not lux or melanopic evidence"
    elif irradiance is not None and irradiance[1] == 0:
        reason = f"{irradiance[0]} reported zero irradiance without corroborating sun evidence"
    return _weather_result(
        WeatherDaylightState.UNKNOWN,
        confidence=0.0,
        reason=reason,
        provenance=provenance,
    )


def _explicit_boolean(value: object) -> bool | None:
    """Normalize only explicit boolean-like states."""
    if isinstance(value, bool):
        return value
    normalized = _normalise(value)
    if normalized in {"on", "true", "yes", "active"}:
        return True
    if normalized in {"off", "false", "no", "inactive"}:
        return False
    return None


def _household_boolean(value: object) -> bool | None:
    """Normalize explicit home/away vocabulary in household scope only."""
    explicit = _explicit_boolean(value)
    if explicit is not None:
        return explicit
    normalized = _normalise(value)
    if normalized in {"home", "present", "occupied"}:
        return True
    if normalized in {"away", "not_home", "absent", "vacant"}:
        return False
    return None


def _person_aggregate(value: object) -> HouseholdState | None:
    """Classify a complete sequence/mapping of explicit person states."""
    if isinstance(value, Mapping):
        states = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        states = tuple(value)
    else:
        return None
    if not states:
        return None
    normalized = tuple(
        _normalise(
            item.state
            if isinstance(item, StateSnapshot)
            else item.get("state")
            if isinstance(item, Mapping)
            else item,
        )
        for item in states
    )
    if any(item == "home" for item in normalized):
        return HouseholdState.HOME
    if all(item in {"away", "not_home"} for item in normalized):
        return HouseholdState.AWAY
    return HouseholdState.UNKNOWN


def classify_household_state(
    snapshot: StateSnapshot | Mapping[str, object],
) -> HouseholdClassification:
    """Classify home/away only from explicit boolean or person aggregates."""
    parts = _snapshot_parts(snapshot)
    provenance = _provenance(parts, _HOUSEHOLD_KEYS)
    direct_values = (
        ("state", parts.state),
        ("attributes.home", _attribute(parts, "home")),
        ("attributes.occupied", _attribute(parts, "occupied")),
    )
    for source, value in direct_values:
        explicit = _household_boolean(value)
        if explicit is not None:
            category = HouseholdState.HOME if explicit else HouseholdState.AWAY
            semantic = (
                SemanticIntent.HOUSEHOLD_HOME
                if explicit
                else SemanticIntent.HOUSEHOLD_AWAY
            )
            return Classification(
                category=category,
                confidence=1.0,
                reasons=(f"{source} explicitly reported {category.value}",),
                provenance=provenance,
                semantic_intents=(semantic,),
                context_only=True,
            )

    aggregate_values = (
        ("state", parts.state),
        ("person_states", _attribute(parts, "person_states")),
        ("persons", _attribute(parts, "persons")),
        ("people", _attribute(parts, "people")),
    )
    for key, value in aggregate_values:
        aggregate = _person_aggregate(value)
        if aggregate in {HouseholdState.HOME, HouseholdState.AWAY}:
            semantic = (
                SemanticIntent.HOUSEHOLD_HOME
                if aggregate is HouseholdState.HOME
                else SemanticIntent.HOUSEHOLD_AWAY
            )
            return Classification(
                category=aggregate,
                confidence=0.95,
                reasons=(f"{key} supplied an explicit person aggregate",),
                provenance=provenance,
                semantic_intents=(semantic,),
                context_only=True,
            )

    return Classification(
        category=HouseholdState.UNKNOWN,
        confidence=0.0,
        reasons=("household state is missing, unavailable, or not an explicit aggregate",),
        provenance=provenance,
        semantic_intents=(SemanticIntent.UNKNOWN,),
        context_only=True,
    )


def _timestamp(value: object) -> float | None:
    """Normalize numeric or ISO timestamps deterministically."""
    numeric = _number(value)
    if numeric is not None:
        return numeric
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and _normalise(value) not in _UNKNOWN_TEXT:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _arrival_signal(parts: StateSnapshot) -> tuple[str, bool] | None:
    """Find the first explicit arrival signal in stable priority order."""
    values = (
        ("state", parts.state),
        ("attributes.arrival", _attribute(parts, "arrival")),
        ("attributes.arrived", _attribute(parts, "arrived")),
        ("attributes.arrival_signal", _attribute(parts, "arrival_signal")),
    )
    for source, value in values:
        normalized = _normalise(value)
        if normalized in {"arrived", "arrival", "recent_arrival"}:
            return source, True
        explicit = _explicit_boolean(value)
        if explicit is not None:
            return source, explicit
    return None


def classify_arrival_state(  # noqa: PLR0912
    snapshot: StateSnapshot | Mapping[str, object],
    *,
    observed_at: object | None = None,
    age_seconds: object | None = None,
    max_age_seconds: object | None = None,
    now: object | None = None,
) -> ArrivalClassification:
    """Classify an explicit arrival only while its bounded lifetime is fresh."""
    parts = _snapshot_parts(snapshot)
    provenance = list(_provenance(parts, _ARRIVAL_KEYS))
    signal = _arrival_signal(parts)
    if signal is None:
        return Classification(
            category=ArrivalState.UNKNOWN,
            confidence=0.0,
            reasons=("arrival signal is missing, unavailable, or ambiguous",),
            provenance=tuple(provenance),
            semantic_intents=(SemanticIntent.UNKNOWN,),
        )
    signal_field, active = signal
    if not active:
        return Classification(
            category=ArrivalState.INACTIVE,
            confidence=1.0,
            reasons=(f"{signal_field} explicitly reported no active arrival",),
            provenance=tuple(provenance),
            semantic_intents=(SemanticIntent.UNKNOWN,),
        )

    effective_observed = (
        observed_at if observed_at is not None else _attribute(parts, "observed_at")
    )
    effective_age = age_seconds
    if effective_age is None:
        effective_age = _attribute(parts, "age_seconds")
    if effective_age is None:
        effective_age = _attribute(parts, "age")
    effective_max_age = max_age_seconds
    if effective_max_age is None:
        effective_max_age = _attribute(parts, "max_age_seconds")
    if effective_max_age is None:
        effective_max_age = _attribute(parts, "max_age")
    effective_now = now
    if effective_now is None:
        effective_now = _attribute(parts, "evaluated_at")
    if effective_now is None:
        effective_now = _attribute(parts, "now")

    if observed_at is not None:
        provenance.append("argument.observed_at")
    if age_seconds is not None:
        provenance.append("argument.age_seconds")
    if max_age_seconds is not None:
        provenance.append("argument.max_age_seconds")
    if now is not None:
        provenance.append("argument.now")

    age = _number(effective_age)
    if age is None and effective_observed is not None and effective_now is not None:
        observed_timestamp = _timestamp(effective_observed)
        now_timestamp = _timestamp(effective_now)
        if observed_timestamp is not None and now_timestamp is not None:
            age = now_timestamp - observed_timestamp
    max_age = _number(effective_max_age)
    provenance_tuple = tuple(dict.fromkeys(provenance))

    if max_age is None or max_age <= 0:
        return Classification(
            category=ArrivalState.UNKNOWN,
            confidence=0.0,
            reasons=("positive arrival has no valid bounded max_age",),
            provenance=provenance_tuple,
            semantic_intents=(SemanticIntent.UNKNOWN,),
        )
    if age is None or age < 0:
        return Classification(
            category=ArrivalState.UNKNOWN,
            confidence=0.0,
            reasons=("positive arrival has no valid deterministic age",),
            provenance=provenance_tuple,
            semantic_intents=(SemanticIntent.UNKNOWN,),
        )
    if age > max_age:
        return Classification(
            category=ArrivalState.STALE,
            confidence=0.0,
            reasons=(f"arrival age {age:g}s exceeds max_age {max_age:g}s; stale arrival fails closed",),
            provenance=provenance_tuple,
            semantic_intents=(SemanticIntent.UNKNOWN,),
        )

    confidence = max(0.5, 1.0 - (0.5 * age / max_age))
    return Classification(
        category=ArrivalState.RECENT,
        confidence=confidence,
        reasons=(f"explicit arrival is fresh at age {age:g}s within max_age {max_age:g}s",),
        provenance=provenance_tuple,
        semantic_intents=(SemanticIntent.RECENT_ARRIVAL,),
    )


def classify_sleep_state(
    snapshot: StateSnapshot | Mapping[str, object],
) -> SleepClassification:
    """Classify only an explicit sleep/wake signal for priority ordering."""
    parts = _snapshot_parts(snapshot)
    provenance = _provenance(parts, _SLEEP_KEYS)
    values = (
        ("state", parts.state),
        ("attributes.sleep", _attribute(parts, "sleep")),
        ("attributes.sleeping", _attribute(parts, "sleeping")),
        ("attributes.sleep_mode", _attribute(parts, "sleep_mode")),
    )
    for source, value in values:
        normalized = _normalise(value)
        if normalized in {"sleep", "sleeping", "asleep"}:
            active = True
        elif normalized in {"awake", "waking"}:
            active = False
        else:
            explicit = _explicit_boolean(value)
            if explicit is None:
                continue
            active = explicit
        category = SleepState.SLEEPING if active else SleepState.AWAKE
        semantics = (SemanticIntent.SLEEP,) if active else (SemanticIntent.UNKNOWN,)
        return Classification(
            category=category,
            confidence=1.0,
            reasons=(f"{source} explicitly reported {category.value}",),
            provenance=provenance,
            semantic_intents=semantics,
        )
    return Classification(
        category=SleepState.UNKNOWN,
        confidence=0.0,
        reasons=("sleep state is missing, unavailable, or ambiguous",),
        provenance=provenance,
        semantic_intents=(SemanticIntent.UNKNOWN,),
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
    """Classify context in a deterministic safety-first priority order."""
    security = classify_security_state(
        _nested_snapshot(snapshot, ("security", "alarm", "security_state")),
    )
    sleep = classify_sleep_state(
        _nested_snapshot(snapshot, ("sleep", "sleep_state", "sleep_mode")),
    )
    arrival = classify_arrival_state(
        _nested_snapshot(snapshot, ("arrival", "arrival_state")),
    )
    media = classify_media_state(
        _nested_snapshot(snapshot, ("media", "media_player", "media_state")),
    )
    weather_daylight = classify_weather_daylight(
        _nested_snapshot(
            snapshot,
            ("weather_daylight", "weather", "daylight"),
        ),
    )
    household = classify_household_state(
        _nested_snapshot(snapshot, ("household", "household_state", "home_state")),
    )
    opening = classify_opening_state(
        _nested_snapshot(snapshot, ("opening", "door", "window", "cover", "garage")),
    )

    if security.category is SecurityState.EMERGENCY:
        semantic_intents = (SemanticIntent.EMERGENCY,)
        primary = SemanticIntent.EMERGENCY
        reasons = (
            *security.reasons,
            "security emergency outranks media, sleep, arrival, weather, and other context",
        )
        confidence = security.confidence
    else:
        priority_results: list[Classification[object]] = []
        if sleep.category is SleepState.SLEEPING:
            priority_results.append(sleep)
        if security.is_known and security.category is not SecurityState.DISARMED:
            priority_results.append(security)
        if arrival.category is ArrivalState.RECENT:
            priority_results.append(arrival)
        if media.is_known and media.category is not MediaState.IDLE:
            priority_results.append(media)
        if weather_daylight.is_known:
            priority_results.append(weather_daylight)
        if security.category is SecurityState.DISARMED:
            priority_results.append(security)
        if media.category is MediaState.IDLE:
            priority_results.append(media)
        if household.is_known:
            priority_results.append(household)
        if opening.is_known:
            priority_results.append(opening)

        available_intents = tuple(
            intent
            for result in priority_results
            for intent in result.semantic_intents
            if intent is not SemanticIntent.UNKNOWN
        )
        semantic_intents = _unique(available_intents or (SemanticIntent.UNKNOWN,))
        primary = semantic_intents[0]
        confidence = priority_results[0].confidence if priority_results else 0.0
        reasons = (
            *security.reasons,
            *sleep.reasons,
            *arrival.reasons,
            *media.reasons,
            *weather_daylight.reasons,
            *household.reasons,
            *opening.reasons,
        )

    provenance = tuple(
        dict.fromkeys(
            (
                *security.provenance,
                *sleep.provenance,
                *arrival.provenance,
                *media.provenance,
                *weather_daylight.provenance,
                *household.provenance,
                *opening.provenance,
            ),
        ),
    )
    return ContextClassification(
        security=security,
        sleep=sleep,
        arrival=arrival,
        media=media,
        weather_daylight=weather_daylight,
        household=household,
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


def classify_weather(
    snapshot: StateSnapshot | Mapping[str, object],
) -> WeatherDaylightClassification:
    """Short alias for :func:`classify_weather_daylight`."""
    return classify_weather_daylight(snapshot)


def classify_household(
    snapshot: StateSnapshot | Mapping[str, object],
) -> HouseholdClassification:
    """Short alias for :func:`classify_household_state`."""
    return classify_household_state(snapshot)


def classify_arrival(
    snapshot: StateSnapshot | Mapping[str, object],
    *,
    observed_at: object | None = None,
    age_seconds: object | None = None,
    max_age_seconds: object | None = None,
    now: object | None = None,
) -> ArrivalClassification:
    """Short alias for :func:`classify_arrival_state`."""
    return classify_arrival_state(
        snapshot,
        observed_at=observed_at,
        age_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        now=now,
    )


def classify_snapshot(
    snapshot: StateSnapshot | Mapping[str, object],
) -> ContextClassification:
    """Short alias for :func:`classify_context`."""
    return classify_context(snapshot)


__all__ = [
    "ArrivalClassification",
    "ArrivalState",
    "Classification",
    "ContextClassification",
    "HouseholdClassification",
    "HouseholdState",
    "MediaClassification",
    "MediaState",
    "OpeningClassification",
    "OpeningState",
    "SecurityClassification",
    "SecurityState",
    "SemanticIntent",
    "SleepClassification",
    "SleepState",
    "StateSnapshot",
    "WeatherDaylightClassification",
    "WeatherDaylightState",
    "classify_arrival",
    "classify_arrival_state",
    "classify_context",
    "classify_household",
    "classify_household_state",
    "classify_media",
    "classify_media_state",
    "classify_opening",
    "classify_opening_state",
    "classify_security",
    "classify_security_state",
    "classify_sleep_state",
    "classify_snapshot",
    "classify_weather",
    "classify_weather_daylight",
]
