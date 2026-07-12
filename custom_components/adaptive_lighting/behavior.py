"""Pure semantic normalization for discrete light behavior events.

The temporal model owns learning, prediction, persistence, and feedback.  This
module only turns an event-shaped mapping into a bounded immutable observation
or an explicit rejection.  It has no Home Assistant imports and no side
effects, which keeps the boundary safe to use from replay tests and adapters.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MAX_CATEGORY_LENGTH = 64
MAX_MEDIA_APP_LENGTH = 48
MAX_ENTITY_ID_LENGTH = 128
MAX_CAPABILITIES = 8

_ACTIONS = {"on", "off"}
_HUMAN_MARKERS = {
    "human",
    "manual",
    "physical",
    "person",
    "remote_control",
    "user",
    "wall_switch",
}
_AUTOMATION_MARKERS = {
    "adaptive_lighting",
    "automation",
    "home_assistant",
    "integration",
    "scene",
    "scheduled",
    "script",
    "service",
    "system",
}
_SAFETY_MARKERS = {
    "alarm",
    "emergency",
    "evacuation",
    "fire",
    "grid_emergency",
    "panic",
    "safety",
    "security",
}
_BLOCKED_DOMAINS = {
    "cover",
    "door",
    "garage",
    "lock",
    "valve",
    "window",
}
_GOOD_NIGHT_MARKERS = {"goodnight", "good_night", "good-night"}
_MISSING = object()


def _text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _slug(
    value: Any,
    *,
    default: str = "unknown",
    maximum: int = MAX_CATEGORY_LENGTH,
) -> str:
    """Bound free-form categories so model keys cannot grow without limit."""
    if not isinstance(value, str):
        return default
    result = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return (result or default)[:maximum]


def _marker(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value))


def _first(mapping: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value) in {"1", "true", "yes", "on", "active"}
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0
    )


def _timestamp(value: Any) -> datetime | None:  # noqa: PLR0911
    """Parse only finite, timezone-aware timestamps and normalize them to UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(UTC)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _timestamp(parsed)
    return None


def _action(value: Any) -> str | None:
    if isinstance(value, bool):
        return "on" if value else "off"
    normalized = _text(value)
    return {"turn_on": "on", "turn_off": "off"}.get(normalized, normalized) or None


def _good_night(value: Any, mapping: Mapping[str, Any]) -> bool:
    candidates = (
        value,
        mapping.get("semantic_routine"),
        mapping.get("routine"),
        mapping.get("routine_name"),
        mapping.get("semantic_context"),
        mapping.get("context"),
    )
    markers = {_marker(item) for item in _GOOD_NIGHT_MARKERS}
    return any(_marker(item) in markers for item in candidates) or _truthy(
        _first(mapping, "good_night", "goodnight", default=False),
    )


def _safety_label(value: Any) -> bool:
    normalized = _marker(value)
    return bool(normalized) and any(marker in normalized for marker in _SAFETY_MARKERS)


def _automation_label(value: Any) -> bool:
    normalized = _marker(value)
    return any(_marker(marker) in normalized for marker in _AUTOMATION_MARKERS)


def _human_label(value: Any) -> bool:
    normalized = _marker(value)
    return normalized in {_marker(item) for item in _HUMAN_MARKERS}


def _day_type(
    mapping: Mapping[str, Any],
    timestamp: datetime,
) -> tuple[str, str, str | None]:
    holiday_value = _first(
        mapping,
        "public_holiday",
        "holiday",
        "is_holiday",
        default=False,
    )
    raw = _marker(_first(mapping, "day_type", "calendar_day_type", default=""))
    is_holiday = _truthy(holiday_value) or raw in {"holiday", "publicholiday"}
    if is_holiday:
        holiday = _slug(
            _first(
                mapping,
                "holiday_name",
                "holiday_provenance",
                default="public_holiday",
            ),
            default="public_holiday",
        )
        return "public_holiday", "weekend", holiday
    if raw in {"weekday", "weekend"}:
        return raw, raw, None
    inferred = "weekend" if timestamp.weekday() >= 5 else "weekday"
    return inferred, inferred, None


def _home_state(value: Any) -> str:
    if isinstance(value, bool):
        return "home" if value else "away"
    normalized = _marker(value)
    if normalized in {"home", "present", "occupied", "in"}:
        return "home"
    if normalized in {"away", "absent", "vacant", "out"}:
        return "away"
    return _slug(value)


def _arrival(value: Any) -> str:
    if isinstance(value, bool):
        return "recent_arrival" if value else "none"
    normalized = _marker(value)
    if normalized in {"arrival", "arrived", "justarrived", "recentarrival"}:
        return "recent_arrival"
    return _slug(value, default="none")


def _capabilities(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _first(mapping, "capabilities", "capability_flags", default=())
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, Sequence):
        return ()
    return tuple(_slug(item) for item in raw[:MAX_CAPABILITIES] if _slug(item))


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    """Normalized immutable event consumed by the temporal model."""

    entity_id: str
    action: str
    timestamp: datetime
    area: str
    zone: str
    semantic_routine: str
    day_type: str
    behavior_day_type: str
    holiday_provenance: str | None
    home_away: str
    arrival: str
    media_type: str
    media_app: str
    weather_band: str
    daylight_band: str
    preceding_event: str
    source: str
    provenance: str
    domain: str
    explicit_light_switch: bool
    supports_brightness: bool
    is_dimmable: bool
    capabilities: tuple[str, ...]

    @property
    def capability_flags(self) -> tuple[str, ...]:
        """Expose capability labels without returning mutable model state."""
        return self.capabilities


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Explicit acceptance result returned by the single-event normalizer."""

    accepted: bool
    reason: str
    observation: BehaviorObservation | None = None

    @property
    def ok(self) -> bool:
        """Return the acceptance flag under a short call-site-friendly name."""
        return self.accepted


def _reject(reason: str) -> NormalizationResult:
    return NormalizationResult(accepted=False, reason=reason)


def normalize_behavior_event(  # noqa: PLR0911
    mapping: Mapping[str, Any],
) -> NormalizationResult:
    """Normalize one attributable light on/off event without learning it."""
    if not isinstance(mapping, Mapping):
        return _reject("invalid_event")

    entity_id = _text(_first(mapping, "entity_id", "entity", "target", default=""))
    if not entity_id or len(entity_id) > MAX_ENTITY_ID_LENGTH or "." not in entity_id:
        return _reject("invalid_entity_id")
    domain = _text(_first(mapping, "domain", "entity_domain", default=""))
    inferred_domain = entity_id.split(".", 1)[0]
    if domain and domain != inferred_domain:
        return _reject("entity_domain_mismatch")
    domain = domain or inferred_domain
    device_type = _text(
        _first(mapping, "device_class", "actuator_type", "entity_type", default=""),
    )
    if domain in _BLOCKED_DOMAINS or device_type in _BLOCKED_DOMAINS:
        return _reject("blocked_actuator_domain")
    explicit_switch = _truthy(
        _first(
            mapping,
            "light_switch",
            "explicit_light_switch",
            "is_light_switch",
            default=False,
        ),
    )
    if domain == "switch" and not explicit_switch:
        return _reject("switch_requires_explicit_light_flag")
    if domain not in {"light", "switch"}:
        return _reject("not_a_light_like_entity")

    state = _text(_first(mapping, "entity_state", "availability", default=""))
    available = _first(mapping, "available", "is_available", default=True)
    if not _truthy(available) or state in {"unavailable", "unknown"}:
        return _reject("unavailable_or_unknown_entity")
    action = _action(_first(mapping, "action", "state", "desired_state", default=""))
    if action not in _ACTIONS:
        return _reject("invalid_action")

    timestamp = _timestamp(
        _first(mapping, "timestamp", "observed_at", "at", default=None),
    )
    if timestamp is None:
        return _reject("invalid_timestamp")
    routine_value = _first(
        mapping,
        "semantic_routine",
        "routine",
        "routine_name",
        default="ambient",
    )
    good_night = _good_night(routine_value, mapping)
    semantic_routine = (
        "good_night" if good_night else _slug(routine_value, default="ambient")
    )
    source = _slug(
        _first(mapping, "source", "origin", "actor", default=""),
        default="unknown",
    )
    provenance = _slug(
        _first(mapping, "provenance", "source_provenance", default=source),
        default=source,
    )
    labels = (
        source,
        provenance,
        _first(mapping, "source", "origin", "actor", default=""),
        _first(mapping, "provenance", "source_provenance", default=""),
        routine_value,
        _first(mapping, "semantic_context", "context", default=""),
    )
    if any(_safety_label(label) for label in labels):
        return _reject("safety_override")
    has_human = any(_human_label(label) for label in labels)
    has_automation = any(_automation_label(label) for label in labels)
    if not good_night and (has_automation or not has_human):
        return _reject("non_human_or_automation_source")
    if good_night and not (has_human or has_automation):
        return _reject("good_night_source_unattributed")

    calendar_day, behavior_day, holiday = _day_type(mapping, timestamp)
    area = _slug(_first(mapping, "area", "zone", "room", default="unknown"))
    supports_brightness = _truthy(
        _first(
            mapping,
            "supports_brightness",
            "brightness_supported",
            "dimmable",
            "is_dimmable",
            default=False,
        ),
    )
    observation = BehaviorObservation(
        entity_id=entity_id,
        action=action,
        timestamp=timestamp,
        area=area,
        zone=area,
        semantic_routine=semantic_routine,
        day_type=calendar_day,
        behavior_day_type=behavior_day,
        holiday_provenance=holiday,
        home_away=_home_state(
            _first(mapping, "home_away", "home_state", "presence", default="unknown"),
        ),
        arrival=_arrival(_first(mapping, "arrival", "arrival_state", default="none")),
        media_type=_slug(
            _first(mapping, "media_type", "media_category", "media", default="none"),
        ),
        media_app=_slug(
            _first(mapping, "media_app", "media_application", "app", default="none"),
            default="none",
            maximum=MAX_MEDIA_APP_LENGTH,
        ),
        weather_band=_slug(
            _first(mapping, "weather_band", "weather", default="unknown"),
        ),
        daylight_band=_slug(
            _first(
                mapping,
                "daylight_band",
                "weather_daylight_band",
                default="unknown",
            ),
        ),
        preceding_event=_slug(
            _first(mapping, "preceding_event", "previous_event", default="none"),
            default="none",
        ),
        source=source,
        provenance=provenance,
        domain=domain,
        explicit_light_switch=explicit_switch,
        supports_brightness=supports_brightness,
        is_dimmable=supports_brightness,
        capabilities=_capabilities(mapping),
    )
    reason = "accepted_good_night_routine" if good_night else "accepted_human_action"
    return NormalizationResult(accepted=True, reason=reason, observation=observation)


def normalize_good_night_routine(
    actions: Sequence[Any] | Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[BehaviorObservation, ...]:
    """Normalize the accepted targets of an explicit Good Night routine.

    Invalid targets are omitted from the returned tuple.  A scalar entity id
    means an off action, which mirrors the common multi-entity scene shape.
    """
    if isinstance(actions, Mapping):
        items: Sequence[Any] = (actions,)
    elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
        items = actions
    else:
        return ()
    if not isinstance(context, Mapping):
        return ()
    normalized: list[BehaviorObservation] = []
    for item in items:
        event = dict(context)
        if isinstance(item, Mapping):
            event.update(item)
        elif isinstance(item, str):
            event["entity_id"] = item
        else:
            continue
        event.setdefault("action", "off")
        event["semantic_routine"] = "good_night"
        event["routine"] = "good_night"
        event.setdefault("source", "scene.good_night")
        result = normalize_behavior_event(event)
        if result.accepted and result.observation is not None:
            normalized.append(result.observation)
    return tuple(normalized)


__all__ = [
    "MAX_CATEGORY_LENGTH",
    "MAX_ENTITY_ID_LENGTH",
    "MAX_MEDIA_APP_LENGTH",
    "BehaviorObservation",
    "NormalizationResult",
    "normalize_behavior_event",
    "normalize_good_night_routine",
]
