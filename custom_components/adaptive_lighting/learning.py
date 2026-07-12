"""Small, local preference-learning primitives for shadow-mode experiments.

The learner deliberately uses a bounded exponentially weighted average rather
than opaque reinforcement learning.  A shadow-mode lighting experiment needs
to be inspectable, deterministic, resettable, and safe when its data is sparse;
an offset table gives us those properties without a policy that can discover
surprising actions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

_SCHEMA_VERSION = 1
_DEFAULT_INTENT = "ambient"
_DEFAULT_TIME_BUCKET = "all"
_DEFAULT_DAYLIGHT_BAND = "unknown"
_HUMAN_SOURCES = frozenset(
    {
        "human",
        "manual",
        "physical",
        "person",
        "user",
        "wall_switch",
        "remote_control",
    },
)
_SAFETY_INTENTS = frozenset(
    {
        "alarm",
        "emergency",
        "good_night",
        "grid_emergency",
        "night_safety",
        "safety",
    },
)
def _text(value: Any, *, default: str = "") -> str:
    """Return a normalized, non-empty string or a default."""
    if not isinstance(value, str):
        return default
    value = value.strip().lower()
    return value or default


def _finite_number(value: Any) -> float | None:
    """Convert a JSON-like number while rejecting booleans and non-finite data."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


class OverrideSample:
    """A user change that may become preference-learning evidence.

    Brightness values are percentages in the inclusive range 0--100.  The
    class intentionally contains only JSON-friendly scalar fields so callers
    can persist samples without importing Home Assistant state objects.
    """

    __slots__ = (
        "baseline",
        "daylight_band",
        "duration_seconds",
        "intent",
        "safety_context",
        "selected",
        "source",
        "time_bucket",
        "zone",
    )

    def __init__(
        self,
        zone: str,
        baseline: float | None = None,
        selected: float | None = None,
        duration_seconds: float | None = None,
        source: str = "human",
        intent: str = _DEFAULT_INTENT,
        time_bucket: str = _DEFAULT_TIME_BUCKET,
        daylight_band: str = _DEFAULT_DAYLIGHT_BAND,
        safety_context: bool = False,
        *,
        previous_target: float | None = None,
        previous_brightness: float | None = None,
        user_selected_target: float | None = None,
        selected_brightness: float | None = None,
        chosen_brightness: float | None = None,
        duration_s: float | None = None,
        duration: float | None = None,
        context: str | None = None,
        actor: str | None = None,
    ) -> None:
        """Build a sample, accepting the plan's previous-target terminology."""
        if baseline is None:
            baseline = previous_target
            if baseline is None:
                baseline = previous_brightness
        if selected is None:
            selected = user_selected_target
            if selected is None:
                selected = selected_brightness
            if selected is None:
                selected = chosen_brightness
        if duration_seconds is None:
            duration_seconds = duration_s
            if duration_seconds is None:
                duration_seconds = duration
        if context is not None:
            intent = context
        if actor is not None:
            source = actor
        self.zone = zone
        self.baseline = baseline
        self.selected = selected
        self.duration_seconds = duration_seconds
        self.source = source
        self.intent = intent
        self.time_bucket = time_bucket
        self.daylight_band = daylight_band
        self.safety_context = safety_context

    @property
    def previous_target(self) -> float | None:
        """Alias matching the event terminology used by the design plan."""
        return self.baseline

    @property
    def user_selected_target(self) -> float | None:
        """Alias matching the event terminology used by the design plan."""
        return self.selected

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the sample."""
        return {
            "zone": self.zone,
            "baseline": self.baseline,
            "selected": self.selected,
            "duration_seconds": self.duration_seconds,
            "source": self.source,
            "intent": self.intent,
            "time_bucket": self.time_bucket,
            "daylight_band": self.daylight_band,
            "safety_context": self.safety_context,
        }


# PreferenceSample is a descriptive alias for callers that do not use the
# integration's event vocabulary.
PreferenceSample = OverrideSample


class PreferenceLearner:
    """Learn bounded brightness offsets from durable human overrides."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        max_offset: float = 20.0,
        min_duration_seconds: float = 30.0,
        *,
        alpha: float | None = None,
        minimum_duration_seconds: float | None = None,
    ) -> None:
        """Configure the learning rate, bounds, and durability threshold."""
        if alpha is not None:
            learning_rate = alpha
        if minimum_duration_seconds is not None:
            min_duration_seconds = minimum_duration_seconds
        if not 0 < learning_rate <= 1:
            raise ValueError
        if max_offset < 0 or not math.isfinite(max_offset):
            raise ValueError
        if min_duration_seconds < 0 or not math.isfinite(min_duration_seconds):
            raise ValueError

        self.learning_rate = float(learning_rate)
        self.max_offset = float(max_offset)
        self.min_duration_seconds = float(min_duration_seconds)
        self._offsets: dict[tuple[str, str, str, str], dict[str, float]] = {}

    @staticmethod
    def _key(
        zone: Any,
        intent: Any,
        time_bucket: Any,
        daylight_band: Any,
    ) -> tuple[str, str, str, str] | None:
        zone_name = _text(zone)
        if not zone_name:
            return None
        return (
            zone_name,
            _text(intent, default=_DEFAULT_INTENT),
            _text(time_bucket, default=_DEFAULT_TIME_BUCKET),
            _text(daylight_band, default=_DEFAULT_DAYLIGHT_BAND),
        )

    @staticmethod
    def _is_human_source(source: Any) -> bool:
        normalized = _text(source)
        if normalized in _HUMAN_SOURCES:
            return True
        return normalized.startswith(
            (
                "human:",
                "human_",
                "manual:",
                "manual_",
                "physical:",
                "physical_",
            ),
        )

    @staticmethod
    def _is_safety_context(sample: OverrideSample) -> bool:
        intent = _text(sample.intent)
        return bool(sample.safety_context) or intent in _SAFETY_INTENTS

    def _coerce_sample(self, sample: OverrideSample | Mapping[str, Any]) -> OverrideSample | None:
        if isinstance(sample, OverrideSample):
            return sample
        if not isinstance(sample, Mapping):
            return None

        return OverrideSample(
            zone=_mapping_value(sample, "zone", "room"),
            baseline=_mapping_value(
                sample,
                "baseline",
                "previous_target",
                "previous_brightness",
            ),
            selected=_mapping_value(
                sample,
                "selected",
                "user_selected_target",
                "selected_target",
                "user_brightness",
                "selected_brightness",
                "chosen_brightness",
            ),
            duration_seconds=_mapping_value(
                sample,
                "duration_seconds",
                "duration_s",
                "persisted_seconds",
                "hold_seconds",
                "duration",
            ),
            source=sample.get("source", sample.get("actor", "human")),
            intent=sample.get("intent", sample.get("context", _DEFAULT_INTENT)),
            time_bucket=sample.get("time_bucket", _DEFAULT_TIME_BUCKET),
            daylight_band=sample.get("daylight_band", _DEFAULT_DAYLIGHT_BAND),
            safety_context=sample.get(
                "safety_context",
                sample.get("is_safety_context", sample.get("safety", False)),
            ),
        )

    def record(self, sample: OverrideSample | Mapping[str, Any]) -> bool:
        """Record a sample if it is durable, human-attributed, and safe.

        Invalid or rejected samples are deliberately ignored and return
        ``False``.  This makes it safe for an event listener to pass through
        incomplete recorder rows without turning them into model state.
        """
        sample = self._coerce_sample(sample)
        if sample is None or not self._is_human_source(sample.source):
            return False
        if self._is_safety_context(sample):
            return False

        baseline = _finite_number(sample.baseline)
        selected = _finite_number(sample.selected)
        duration = _finite_number(sample.duration_seconds)
        if (
            baseline is None
            or selected is None
            or duration is None
            or not 0 <= baseline <= 100
            or not 0 <= selected <= 100
            or duration < self.min_duration_seconds
        ):
            return False

        key = self._key(
            sample.zone,
            sample.intent,
            sample.time_bucket,
            sample.daylight_band,
        )
        if key is None:
            return False

        observed_offset = max(-self.max_offset, min(self.max_offset, selected - baseline))
        state = self._offsets.setdefault(key, {"offset": 0.0, "count": 0.0})
        state["offset"] = max(
            -self.max_offset,
            min(
                self.max_offset,
                (1 - self.learning_rate) * state["offset"]
                + self.learning_rate * observed_offset,
            ),
        )
        state["count"] += 1
        return True

    # These aliases keep the primitive pleasant to use from event adapters.
    add_sample = record
    learn = record

    def get_offset(
        self,
        zone: str,
        intent: str = _DEFAULT_INTENT,
        time_bucket: str = _DEFAULT_TIME_BUCKET,
        daylight_band: str = _DEFAULT_DAYLIGHT_BAND,
    ) -> float:
        """Return the learned offset for an exact context, or zero if absent."""
        key = self._key(zone, intent, time_bucket, daylight_band)
        if key is None:
            return 0.0
        return self._offsets.get(key, {}).get("offset", 0.0)

    offset = get_offset

    def adjusted_target(
        self,
        baseline: float,
        zone: str,
        intent: str = _DEFAULT_INTENT,
        time_bucket: str = _DEFAULT_TIME_BUCKET,
        daylight_band: str = _DEFAULT_DAYLIGHT_BAND,
    ) -> float:
        """Apply a learned offset while keeping the target in 0--100."""
        value = _finite_number(baseline)
        if value is None:
            raise ValueError
        value = max(0.0, min(100.0, value))
        return max(
            0.0,
            min(
                100.0,
                value
                + self.get_offset(zone, intent, time_bucket, daylight_band),
            ),
        )

    apply = adjusted_target

    @property
    def sample_count(self) -> int:
        """Return the number of accepted samples."""
        return int(sum(state["count"] for state in self._offsets.values()))

    def export_state(self) -> dict[str, object]:
        """Export deterministic, JSON-safe learner state."""
        entries = [
            {
                "zone": key[0],
                "intent": key[1],
                "time_bucket": key[2],
                "daylight_band": key[3],
                "offset": state["offset"],
                "count": int(state["count"]),
            }
            for key, state in sorted(self._offsets.items())
        ]
        return {
            "version": _SCHEMA_VERSION,
            "config": {
                "learning_rate": self.learning_rate,
                "max_offset": self.max_offset,
                "min_duration_seconds": self.min_duration_seconds,
            },
            "entries": entries,
        }

    to_dict = export_state
    export = export_state

    def to_json(self) -> str:
        """Export state as stable JSON for a local file or HA storage adapter."""
        return json.dumps(self.export_state(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> PreferenceLearner:
        """Create a learner from a JSON string."""
        return cls.from_state(json.loads(payload))

    def import_json(self, payload: str, *, replace: bool = True) -> None:
        """Import a learner from a JSON string."""
        self.import_state(json.loads(payload), replace=replace)

    def import_state(self, payload: Mapping[str, Any], *, replace: bool = True) -> None:
        """Import state previously returned by :meth:`export_state`."""
        if not isinstance(payload, Mapping):
            raise TypeError
        version = payload.get("version", _SCHEMA_VERSION)
        if version != _SCHEMA_VERSION:
            raise ValueError
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise TypeError
        config = payload.get("config", {})
        if isinstance(config, Mapping):
            configured = type(self)(
                learning_rate=config.get("learning_rate", self.learning_rate),
                max_offset=config.get("max_offset", self.max_offset),
                min_duration_seconds=config.get(
                    "min_duration_seconds",
                    self.min_duration_seconds,
                ),
            )
            self.learning_rate = configured.learning_rate
            self.max_offset = configured.max_offset
            self.min_duration_seconds = configured.min_duration_seconds
        if replace:
            self._offsets.clear()

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            key = self._key(
                entry.get("zone"),
                entry.get("intent"),
                entry.get("time_bucket"),
                entry.get("daylight_band"),
            )
            offset = _finite_number(entry.get("offset"))
            count = _finite_number(entry.get("count", 0))
            if key is None or offset is None or count is None or count < 0:
                continue
            self._offsets[key] = {
                "offset": max(-self.max_offset, min(self.max_offset, offset)),
                "count": float(int(count)),
            }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> PreferenceLearner:
        """Create a learner from JSON-safe exported state."""
        config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(config, Mapping):
            config = {}
        learner = cls(
            learning_rate=config.get("learning_rate", 0.25),
            max_offset=config.get("max_offset", 20.0),
            min_duration_seconds=config.get("min_duration_seconds", 30.0),
        )
        learner.import_state(payload)
        return learner

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreferenceLearner:
        """Create a learner from a JSON-compatible mapping."""
        return cls.from_state(payload)

    def reset(self, zone: str | None = None, intent: str | None = None) -> int:
        """Reset all state or the selected zone/intent, returning entries removed."""
        if zone is None:
            removed = len(self._offsets)
            self._offsets.clear()
            return removed

        zone_name = _text(zone)
        intent_name = _text(intent) if intent is not None else None
        keys = [
            key
            for key in self._offsets
            if key[0] == zone_name and (intent_name is None or key[1] == intent_name)
        ]
        for key in keys:
            del self._offsets[key]
        return len(keys)

    clear = reset
