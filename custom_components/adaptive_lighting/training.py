"""Local, no-actuation training and commissioning for Adaptive Lighting.

This module is deliberately a small Home Assistant-facing adapter around the
pure :class:`PreferenceLearner`.  It owns the lifecycle of a local shadow
training session, but it does not know how to apply a learned value to a
light.  That separation is important: commissioning can be enabled and
inspected without granting the training code an actuation path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .learning import OverrideSample, PreferenceLearner

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = "adaptive_lighting_training"
DATA_VERSION = 1

PHASE_SHADOW_LEARNING = "shadow_learning"
PHASE_ACTIVE = "active"
VALID_PHASES = frozenset({PHASE_SHADOW_LEARNING, PHASE_ACTIVE})

DEFAULT_TRAINING_DURATION_DAYS = 7
DEFAULT_DURABILITY_SECONDS = 30.0
DEFAULT_MINIMUM_SAMPLES = 5
DEFAULT_MINIMUM_CONFIDENCE = 0.75
DAY_TYPE_WEEKDAY = "weekday"
DAY_TYPE_WEEKEND = "weekend"
DAY_TYPE_PUBLIC_HOLIDAY = "public_holiday"
VALID_DAY_TYPES = frozenset(
    {DAY_TYPE_WEEKDAY, DAY_TYPE_WEEKEND, DAY_TYPE_PUBLIC_HOLIDAY},
)

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
_BLOCKED_SOURCE_MARKERS = (
    "automation",
    "integration",
    "adaptive_lighting",
    "home_assistant",
    "script",
    "scene",
    "service",
    "scheduled",
    "system",
)
_SAFETY_MARKERS = frozenset(
    {
        "alarm",
        "emergency",
        "evacuation",
        "fire",
        "goodnight",
        "nightpath",
        "nightsafety",
        "safety",
        "security",
        "sleep",
    },
)
_MISSING = object()


def _finite_number(value: Any) -> float | None:
    """Return a finite float while treating booleans as invalid numbers."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any, default: str = "") -> str:
    """Normalize a scalar text value for matching and compact persistence."""
    if not isinstance(value, str):
        return default
    return value.strip().lower() or default


def _marker_text(value: Any) -> str:
    """Normalize separators so ``good-night`` and ``good_night`` match."""
    return re.sub(r"[^a-z0-9]+", "", _text(value))


def _truthy(value: Any) -> bool:
    """Parse the common JSON-ish truth values used by event metadata."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value) in {"1", "true", "yes", "on", "active"}
    return bool(value) if isinstance(value, (int, float)) else False


def _json_safe(value: Any, *, depth: int = 0) -> Any:  # noqa: PLR0911
    """Return a compact JSON-safe copy of event metadata.

    Arbitrary Home Assistant event objects are intentionally not retained.
    A shallow, bounded conversion prevents a diagnostic/export call from
    accidentally serializing a state object or an unbounded nested structure.
    """
    if depth > 3:
        return "…"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:24]]
    return str(value)


def _as_utc(value: datetime) -> datetime:
    """Normalize a datetime without relying on a local timezone assumption."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse the ISO strings used by the local store."""
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str):
        return None
    parsed = dt_util.parse_datetime(value)
    return _as_utc(parsed) if parsed is not None else None


def _first(mapping: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    """Return the first present value from a set of event aliases."""
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _number_or_original(value: Any) -> Any:
    """Keep invalid values visible for diagnostics while normalizing numbers."""
    number = _finite_number(value)
    return number if number is not None else _json_safe(value)


def _sample_mapping(
    sample: OverrideSample | Mapping[str, Any],
) -> dict[str, Any] | None:
    """Convert a candidate into the learner's small, JSON-safe vocabulary."""
    if isinstance(sample, OverrideSample):
        result = sample.as_dict()
        # OverrideSample intentionally has only the learner fields.  The
        # metadata aliases below are for event adapters passing a mapping.
        return {str(key): _json_safe(value) for key, value in result.items()}
    if not isinstance(sample, Mapping):
        return None

    source = _first(sample, "source", "actor", default="human")
    context = _first(sample, "context", default=_MISSING)
    intent = _first(sample, "intent", default=_MISSING)
    if intent is _MISSING or not isinstance(intent, str):
        intent = context if isinstance(context, str) else "ambient"
    metadata = _first(
        sample,
        "metadata",
        "context_metadata",
        "source_metadata",
        default={},
    )
    if not isinstance(metadata, Mapping):
        metadata = {}
    if isinstance(context, Mapping):
        # Some event producers use ``context`` as the metadata object rather
        # than as the learner intent string.
        metadata = {**context, **metadata}

    result: dict[str, Any] = {
        "zone": _json_safe(_first(sample, "zone", "room", "area", default="")),
        "baseline": _number_or_original(
            _first(
                sample,
                "baseline",
                "previous_target",
                "previous_brightness",
                default=None,
            ),
        ),
        "selected": _number_or_original(
            _first(
                sample,
                "selected",
                "user_selected_target",
                "selected_target",
                "user_brightness",
                "selected_brightness",
                "chosen_brightness",
                default=None,
            ),
        ),
        "duration_seconds": _number_or_original(
            _first(
                sample,
                "duration_seconds",
                "duration_s",
                "persisted_seconds",
                "hold_seconds",
                "duration",
                default=None,
            ),
        ),
        "source": _json_safe(source),
        "intent": _json_safe(intent),
        "time_bucket": _json_safe(_first(sample, "time_bucket", default="all")),
        "daylight_band": _json_safe(_first(sample, "daylight_band", default="unknown")),
        "day_type": _json_safe(_first(sample, "day_type", default=None)),
        "safety_context": _json_safe(
            _first(
                sample,
                "safety_context",
                "is_safety_context",
                "safety",
                default=False,
            ),
        ),
        "hard_cap": _json_safe(
            _first(
                sample,
                "hard_cap",
                "hard_brightness_cap",
                "brightness_cap",
                "hard_cap_active",
                "has_hard_cap",
                default=None,
            ),
        ),
    }

    # Retain only bounded metadata needed to explain provenance/rejections.
    if metadata:
        result["metadata"] = _json_safe(metadata)
    for name in (
        "sample_id",
        "id",
        "started_at",
        "override_started_at",
        "observed_at",
        "timestamp",
        "superseded",
        "automation",
        "integration",
        "alarm",
        "good_night",
        "good-night",
    ):
        if name in sample:
            result[name] = _json_safe(sample[name])
    if isinstance(context, str):
        result["context"] = context
    return result


def _candidate_reason(sample: Mapping[str, Any]) -> str | None:  # noqa: PLR0911
    """Return a stable rejection reason before the learner sees the sample."""
    if _truthy(sample.get("superseded")):
        return "superseded"

    source_name = _text(sample.get("source"))
    source = _marker_text(source_name)
    human_sources = {_marker_text(value) for value in _HUMAN_SOURCES}
    if source not in human_sources:
        blocked_markers = {_marker_text(value) for value in _BLOCKED_SOURCE_MARKERS}
        if any(marker in source for marker in blocked_markers):
            if "integration" in source or "adaptive" in source:
                return "integration_source"
            return "automation_source"
        return "non_human_source"

    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for key in ("automation", "is_automation", "integration", "is_integration"):
        if _truthy(sample.get(key)) or _truthy(metadata.get(key)):
            return "automation_or_integration_context"

    if _truthy(sample.get("safety_context")) or any(
        _truthy(metadata.get(key))
        for key in ("safety", "is_safety", "safety_context", "emergency")
    ):
        return "safety_context"

    context_values = (
        sample.get("intent"),
        sample.get("context"),
        metadata.get("intent"),
        metadata.get("context"),
    )
    context_markers = {_marker_text(value) for value in context_values}
    if any(marker in _SAFETY_MARKERS for marker in context_markers):
        if "goodnight" in context_markers:
            return "good_night_context"
        if "alarm" in context_markers:
            return "alarm_context"
        return "safety_context"
    if _truthy(sample.get("alarm")) or _truthy(metadata.get("alarm")):
        return "alarm_context"
    if any(
        _truthy(sample.get(key)) or _truthy(metadata.get(key))
        for key in ("good_night", "good-night", "goodnight")
    ):
        return "good_night_context"
    return None


def _fingerprint(sample: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Identify a context whose pending override can be superseded."""
    return (
        _text(sample.get("zone")),
        _text(sample.get("intent"), "ambient"),
        _text(sample.get("time_bucket"), "all"),
        _text(sample.get("daylight_band"), "unknown"),
        _text(sample.get("day_type"), DAY_TYPE_WEEKDAY),
    )


class AdaptiveLightingTraining:
    """Persisted shadow-training session for local preference commissioning.

    The class only records and evaluates preferences.  It intentionally has
    no light entity, service, or switch dependency, so entering ``active`` is
    still not an instruction to actuate anything.
    """

    def __init__(  # noqa: PLR0915
        self,
        hass: Any,
        *,
        storage_key: str = STORAGE_KEY,
        training_duration_days: float = DEFAULT_TRAINING_DURATION_DAYS,
        training_days: float | None = None,
        training_duration: timedelta | float | None = None,
        auto_promote: bool = False,
        minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
        min_samples: int | None = None,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
        min_confidence: float | None = None,
        durability_seconds: float = DEFAULT_DURABILITY_SECONDS,
        minimum_duration_seconds: float | None = None,
        learner: PreferenceLearner | None = None,
        now: Callable[[], datetime] | None = None,
        day_type_resolver: Callable[[datetime], str] | None = None,
        public_holiday_predicate: Callable[[date], bool] | None = None,
        on_change: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Create a session; call :meth:`async_setup` to load and schedule it."""
        if training_days is not None:
            training_duration_days = training_days
        if training_duration is not None:
            training_duration_days = (
                training_duration.total_seconds() / 86400
                if isinstance(training_duration, timedelta)
                else float(training_duration) / 86400
            )
        if minimum_duration_seconds is not None:
            durability_seconds = minimum_duration_seconds
        if min_samples is not None:
            minimum_samples = min_samples
        if min_confidence is not None:
            minimum_confidence = min_confidence

        duration_days = _finite_number(training_duration_days)
        durability = _finite_number(durability_seconds)
        confidence = _finite_number(minimum_confidence)
        if duration_days is None or duration_days <= 0:
            raise ValueError("training duration must be positive")  # noqa: EM101, TRY003
        if (
            isinstance(minimum_samples, bool)
            or not isinstance(minimum_samples, int)
            or minimum_samples <= 0
        ):
            raise ValueError("minimum_samples must be a positive integer")  # noqa: EM101, TRY003
        if durability is None or durability < 0:
            raise ValueError("durability_seconds must be non-negative")  # noqa: EM101, TRY003
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")  # noqa: EM101, TRY003

        self.hass = hass
        self.storage_key = storage_key
        self.training_duration_days = float(duration_days)
        self.training_duration = timedelta(days=self.training_duration_days)
        self.auto_promote = bool(auto_promote)
        self.minimum_samples = minimum_samples
        self.minimum_confidence = float(confidence)
        self.durability_seconds = float(durability)
        self._now_func = now or dt_util.utcnow
        self._day_type_resolver = day_type_resolver
        self._public_holiday_predicate = public_holiday_predicate
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            storage_key,
            private=True,
        )
        self._learner = learner or PreferenceLearner(
            min_duration_seconds=self.durability_seconds,
        )
        self._phase = PHASE_SHADOW_LEARNING
        self._training_started_at: datetime | None = None
        self._training_deadline: datetime | None = None
        self._promotion_reason = "not_started"
        self._last_sample: dict[str, Any] | None = None
        self._last_rejection_reason: str | None = None
        self._accepted_count = 0
        self._rejected_count = 0
        self._superseded_count = 0
        self._day_type_counts: dict[str, int] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_timers: dict[str, Callable[[], None]] = {}
        self._deadline_timer: Callable[[], None] | None = None
        self._listeners: set[Callable[[dict[str, Any]], Any]] = set()
        self._sequence = 0
        self._loaded = False
        self._unloaded = False
        if on_change is not None:
            self._listeners.add(on_change)

    @property
    def learner(self) -> PreferenceLearner:
        """Expose the pure learner for shadow inspection, never actuation."""
        return self._learner

    @property
    def phase(self) -> str:
        """Return the persisted rollout phase."""
        return self._phase

    @property
    def is_active(self) -> bool:
        """Return whether the session passed its promotion gates."""
        return self._phase == PHASE_ACTIVE

    @property
    def training_started_at(self) -> datetime | None:
        """Return the UTC training start time."""
        return self._training_started_at

    @property
    def start(self) -> datetime | None:
        """Short alias for integrations displaying the training start."""
        return self._training_started_at

    @property
    def training_deadline(self) -> datetime | None:
        """Return the UTC deadline after which promotion may be evaluated."""
        return self._training_deadline

    @property
    def deadline(self) -> datetime | None:
        """Short alias for integrations displaying the training deadline."""
        return self._training_deadline

    @property
    def sample_count(self) -> int:
        """Return accepted learner samples."""
        return self._accepted_count

    @property
    def pending_count(self) -> int:
        """Return candidates waiting for durability acceptance."""
        return len(self._pending)

    @property
    def confidence(self) -> float:
        """Return conservative support confidence for the promotion gate."""
        return min(1.0, self._accepted_count / self.minimum_samples)

    @property
    def last_sample(self) -> dict[str, Any] | None:
        """Return the last candidate or accepted sample as JSON-safe data."""
        return _json_safe(self._last_sample)

    @property
    def last_rejection_reason(self) -> str | None:
        """Return the last explicit rejection reason, if any."""
        return self._last_rejection_reason

    @property
    def sample_counts(self) -> dict[str, int]:
        """Return compact counts suitable for a diagnostics sensor."""
        return {
            "accepted": self._accepted_count,
            "rejected": self._rejected_count,
            "superseded": self._superseded_count,
            "pending": len(self._pending),
            "total": self._accepted_count + self._rejected_count,
        }

    @property
    def day_type_counts(self) -> dict[str, int]:
        """Return accepted sample counts grouped by local day type."""
        return dict(sorted(self._day_type_counts.items()))

    def preference_for(
        self,
        *,
        baseline: float,
        zone: str,
        intent: str,
        time_bucket: str,
        daylight_band: str,
        at: datetime | None = None,
    ) -> dict[str, Any]:
        """Return an exact-context learned proposal with transparent support.

        This method remains side-effect free. The caller must still enforce
        rollout phase, policy permissions, device capability, and manual hold.
        """
        moment = _as_utc(at or self._current_time())
        day_type = self._resolve_day_type(moment, {})
        effective_day_type = (
            DAY_TYPE_WEEKEND if day_type == DAY_TYPE_PUBLIC_HOLIDAY else day_type
        )
        learner_bucket = f"{effective_day_type}:{_text(time_bucket, 'all')}"
        count = self._learner.get_sample_count(
            zone,
            intent,
            learner_bucket,
            daylight_band,
        )
        offset = self._learner.get_offset(
            zone,
            intent,
            learner_bucket,
            daylight_band,
        )
        return {
            "target": self._learner.adjusted_target(
                baseline,
                zone,
                intent,
                learner_bucket,
                daylight_band,
            ),
            "offset": offset,
            "samples": count,
            "confidence": min(1.0, count / max(1, self.minimum_samples)),
            "day_type": day_type,
            "effective_day_type": effective_day_type,
            "time_bucket": learner_bucket,
        }

    def add_listener(
        self,
        listener: Callable[[dict[str, Any]], Any],
    ) -> Callable[[], None]:
        """Register a phase/learning callback and return its unsubscribe."""
        self._listeners.add(listener)

        def _remove() -> None:
            self._listeners.discard(listener)

        return _remove

    async_add_listener = add_listener

    async def async_setup(self) -> AdaptiveLightingTraining:
        """Load state, start a new session if needed, and restore timers."""
        if self._loaded:
            return self
        stored = await self._store.async_load()
        if stored is None:
            self._start_new_session(self._current_time())
            self._loaded = True
            await self._persist()
        else:
            self._restore(stored)
            self._loaded = True
            if self._training_deadline is None:
                self._start_new_session(self._current_time())
                await self._persist()
            elif (
                self._phase == PHASE_SHADOW_LEARNING
                and self._current_time() >= self._training_deadline
            ):
                await self.async_evaluate_promotion(now=self._current_time())
        self._schedule_deadline()
        self._schedule_pending()
        return self

    async_initialize = async_setup
    async_load = async_setup

    async def async_unload(self) -> None:
        """Cancel all timers/listeners while preserving durable session state."""
        if self._unloaded:
            return
        self._cancel_timers()
        # State changes have already been written synchronously by each public
        # operation.  A final save covers a caller unloading immediately after
        # a read/restart operation without deleting the training record.
        if self._loaded:
            await self._persist()
        self._unloaded = True

    async_stop = async_unload

    def _current_time(self) -> datetime:
        """Get a timezone-aware UTC time from the injected/testable clock."""
        current = self._now_func()
        return _as_utc(current)

    def _resolve_day_type(
        self,
        at: datetime,
        sample: Mapping[str, Any],
    ) -> str:
        """Resolve local day type without owning a holiday calendar.

        A caller can inject a complete resolver or only a holiday predicate.
        The latter is intentionally a predicate over a local ``date`` so the
        integration does not hard-code a country, calendar, or HA calendar
        entity.  ``public_holiday`` keeps its diagnostic identity but maps to
        ``weekend`` for learner behavior.
        """
        local = dt_util.as_local(_as_utc(at))
        if (
            self._public_holiday_predicate is not None
            and self._public_holiday_predicate(
                local.date(),
            )
        ):
            resolved = DAY_TYPE_PUBLIC_HOLIDAY
        elif self._day_type_resolver is not None:
            resolved = self._day_type_resolver(local)
        else:
            supplied = _text(sample.get("day_type"))
            resolved = supplied or (
                DAY_TYPE_WEEKEND if local.weekday() >= 5 else DAY_TYPE_WEEKDAY
            )

        normalized = _marker_text(resolved)
        aliases = {
            "weekday": DAY_TYPE_WEEKDAY,
            "weekend": DAY_TYPE_WEEKEND,
            "publicholiday": DAY_TYPE_PUBLIC_HOLIDAY,
        }
        if normalized not in aliases:
            raise ValueError
        return aliases[normalized]

    @staticmethod
    def _learner_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
        """Add day type to the learner context while keeping raw diagnostics."""
        learner_sample = dict(sample)
        day_type = sample["day_type"]
        effective_day_type = (
            DAY_TYPE_WEEKEND if day_type == DAY_TYPE_PUBLIC_HOLIDAY else day_type
        )
        raw_time_bucket = _text(sample.get("time_bucket"), "all")
        # PreferenceLearner has a stable four-part key.  Namespacing its time
        # bucket lets this module add day type without changing the shared
        # learner primitive or breaking its persistence schema.
        learner_sample["time_bucket"] = f"{effective_day_type}:{raw_time_bucket}"
        return learner_sample

    def _start_new_session(self, start: datetime) -> None:
        """Initialize a clean shadow session and its seven-day default."""
        self._phase = PHASE_SHADOW_LEARNING
        self._training_started_at = _as_utc(start)
        self._training_deadline = self._training_started_at + self.training_duration
        self._promotion_reason = "training_in_progress"
        self._last_sample = None
        self._last_rejection_reason = None
        self._accepted_count = 0
        self._rejected_count = 0
        self._superseded_count = 0
        self._day_type_counts.clear()
        self._pending.clear()
        self._learner.reset()

    def _restore(self, stored: Mapping[str, Any]) -> None:
        """Restore a validated, JSON-safe record without trusting its shape."""
        data: Mapping[str, Any] = stored
        if isinstance(stored.get("data"), Mapping):
            # This accepts a raw Store envelope too, which is useful for
            # diagnostics fixtures and keeps the module tolerant of callers
            # passing through storage-manager data.
            data = stored["data"]
        if data.get("data_version", DATA_VERSION) != DATA_VERSION:
            raise ValueError("unsupported training data version")  # noqa: EM101, TRY003

        phase = data.get("phase", PHASE_SHADOW_LEARNING)
        self._phase = phase if phase in VALID_PHASES else PHASE_SHADOW_LEARNING
        self._training_started_at = _parse_datetime(
            data.get("training_started_at", data.get("start")),
        )
        self._training_deadline = _parse_datetime(
            data.get("training_deadline", data.get("deadline")),
        )
        self._promotion_reason = _text(
            data.get("promotion_reason"),
            "training_in_progress",
        )
        last_sample = data.get("last_sample")
        self._last_sample = (
            dict(last_sample) if isinstance(last_sample, Mapping) else None
        )
        reason = data.get("last_rejection_reason")
        self._last_rejection_reason = reason if isinstance(reason, str) else None

        counts = data.get("sample_counts", {})
        if not isinstance(counts, Mapping):
            counts = {}
        self._accepted_count = max(0, _stored_int(counts.get("accepted")))
        self._rejected_count = max(0, _stored_int(counts.get("rejected")))
        self._superseded_count = max(0, _stored_int(counts.get("superseded")))
        day_type_counts = data.get("day_type_counts", {})
        self._day_type_counts = (
            {
                day_type: _stored_int(count)
                for day_type, count in day_type_counts.items()
                if day_type in VALID_DAY_TYPES and _stored_int(count) > 0
            }
            if isinstance(day_type_counts, Mapping)
            else {}
        )

        learner_state = data.get("learner")
        if isinstance(learner_state, Mapping):
            self._learner = PreferenceLearner.from_state(learner_state)
        else:
            self._learner = PreferenceLearner(
                min_duration_seconds=self.durability_seconds,
            )
        # Learner state is the source of truth if an older record omitted a
        # count, while the persisted count remains useful for diagnostics.
        self._accepted_count = max(self._accepted_count, self._learner.sample_count)

        pending = data.get("pending", [])
        self._pending = {}
        if isinstance(pending, list):
            for item in pending:
                if not isinstance(item, Mapping):
                    continue
                candidate_id = item.get("id")
                sample = item.get("sample")
                accept_at = _parse_datetime(item.get("accept_at"))
                if (
                    not isinstance(candidate_id, str)
                    or not isinstance(sample, Mapping)
                    or accept_at is None
                ):
                    continue
                item_fingerprint = item.get("fingerprint")
                if not isinstance(item_fingerprint, (list, tuple)):
                    item_fingerprint = _fingerprint(sample)
                self._pending[candidate_id] = {
                    "id": candidate_id,
                    "sample": dict(sample),
                    "accept_at": accept_at,
                    "fingerprint": tuple(item_fingerprint),
                }

    def _export_pending(self) -> list[dict[str, Any]]:
        """Export pending candidates without exposing internal timer handles."""
        return [
            {
                "id": item["id"],
                "sample": _json_safe(item["sample"]),
                "accept_at": item["accept_at"].isoformat(),
                "fingerprint": list(item["fingerprint"]),
            }
            for item in sorted(self._pending.values(), key=lambda value: value["id"])
        ]

    def export_state(self) -> dict[str, Any]:
        """Return the complete JSON-safe state used by the local Store."""
        return {
            "data_version": DATA_VERSION,
            "config": {
                "training_duration_days": self.training_duration_days,
                "auto_promote": self.auto_promote,
                "minimum_samples": self.minimum_samples,
                "minimum_confidence": self.minimum_confidence,
                "durability_seconds": self.durability_seconds,
            },
            "training_started_at": (
                self._training_started_at.isoformat()
                if self._training_started_at is not None
                else None
            ),
            "start": (
                self._training_started_at.isoformat()
                if self._training_started_at is not None
                else None
            ),
            "training_deadline": (
                self._training_deadline.isoformat()
                if self._training_deadline is not None
                else None
            ),
            "deadline": (
                self._training_deadline.isoformat()
                if self._training_deadline is not None
                else None
            ),
            "phase": self._phase,
            "promotion_reason": self._promotion_reason,
            "learner": self._learner.export_state(),
            "sample_counts": self.sample_counts,
            "sample_count": self._accepted_count,
            "day_type_counts": self.day_type_counts,
            "last_sample": _json_safe(self._last_sample),
            "last_rejection_reason": self._last_rejection_reason,
            "pending": self._export_pending(),
        }

    export = export_state

    def summary(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return compact JSON-safe diagnostics for a HA sensor or service."""
        current = _as_utc(now) if now is not None else self._current_time()
        remaining = 0.0
        if self._training_deadline is not None:
            remaining = max(0.0, (self._training_deadline - current).total_seconds())
        return {
            "phase": self._phase,
            "active": self.is_active,
            "training_started_at": (
                self._training_started_at.isoformat()
                if self._training_started_at is not None
                else None
            ),
            "training_deadline": (
                self._training_deadline.isoformat()
                if self._training_deadline is not None
                else None
            ),
            "remaining_seconds": remaining,
            "training_duration_days": self.training_duration_days,
            "auto_promote": self.auto_promote,
            "sample_counts": self.sample_counts,
            "day_type_counts": self.day_type_counts,
            "confidence": self.confidence,
            "minimum_samples": self.minimum_samples,
            "minimum_confidence": self.minimum_confidence,
            "promotion_reason": self._promotion_reason,
            "last_sample": _json_safe(self._last_sample),
            "last_day_type": (
                self._last_sample.get("day_type")
                if isinstance(self._last_sample, Mapping)
                else None
            ),
            "last_rejection_reason": self._last_rejection_reason,
        }

    diagnostics = summary

    def diagnostics_json(self) -> str:
        """Return compact deterministic JSON for logs and diagnostics."""
        return json.dumps(self.summary(), sort_keys=True, separators=(",", ":"))

    async def async_export(self) -> dict[str, Any]:
        """Async convenience wrapper for service handlers."""
        return self.export_state()

    async def async_summary(self) -> dict[str, Any]:
        """Async convenience wrapper for service handlers."""
        return self.summary()

    def _state_signature(self) -> tuple[Any, ...]:
        """Capture only values that should trigger a user callback."""
        return (
            self._phase,
            self._accepted_count,
            self._rejected_count,
            self._superseded_count,
            len(self._pending),
            self._promotion_reason,
            self._last_rejection_reason,
            self._last_sample,
            self._training_started_at,
            self._training_deadline,
            self._learner.export_state(),
            self.day_type_counts,
        )

    async def _persist(self) -> None:
        """Write only JSON-safe data through HA's local Store."""
        await self._store.async_save(self.export_state())

    async def _persist_and_notify(self, before: tuple[Any, ...]) -> None:
        """Persist and notify listeners after a meaningful state change."""
        await self._persist()
        if before != self._state_signature():
            self._notify()

    def _notify(self) -> None:
        """Publish a compact snapshot without allowing one listener to break HA."""
        snapshot = self.summary()
        for listener in tuple(self._listeners):
            try:
                result = listener(snapshot)
                if asyncio.iscoroutine(result):
                    self.hass.async_create_task(result)
            except Exception:
                _LOGGER.exception("Adaptive Lighting training listener failed")

    def _schedule_deadline(self) -> None:
        """Schedule the persisted deadline using HA's cancellable time helper."""
        if (
            self._unloaded
            or self._deadline_timer is not None
            or self._phase != PHASE_SHADOW_LEARNING
            or self._training_deadline is None
        ):
            return
        if self._current_time() >= self._training_deadline:
            self.hass.async_create_task(self.async_evaluate_promotion())
            return
        self._deadline_timer = async_track_point_in_utc_time(
            self.hass,
            self._async_deadline_reached,
            self._training_deadline,
        )

    def _cancel_deadline_timer(self) -> None:
        """Cancel only the phase deadline, leaving durable samples queued."""
        if self._deadline_timer is not None:
            self._deadline_timer()
            self._deadline_timer = None

    async def _async_deadline_reached(self, _when: datetime) -> None:
        """Evaluate gates once the persisted deadline is reached."""
        self._deadline_timer = None
        await self.async_evaluate_promotion(now=self._current_time())

    def _schedule_pending(self) -> None:
        """Restore durability timers after a Home Assistant restart."""
        if self._unloaded:
            return
        for candidate_id, item in tuple(self._pending.items()):
            if candidate_id in self._pending_timers:
                continue
            accept_at = item["accept_at"]
            if self._current_time() >= accept_at:
                self.hass.async_create_task(self._async_accept_pending(candidate_id))
                continue
            self._pending_timers[candidate_id] = async_track_point_in_utc_time(
                self.hass,
                lambda _when, item_id=candidate_id: self._async_pending_reached(
                    item_id,
                ),
                accept_at,
            )

    async def _async_pending_reached(self, candidate_id: str) -> None:
        """Accept a candidate only after its durability timer fires."""
        self._pending_timers.pop(candidate_id, None)
        await self._async_accept_pending(candidate_id)

    async def _async_accept_pending(
        self,
        candidate_id: str,
        *,
        current: datetime | None = None,
    ) -> bool:
        """Commit a durable candidate if it still exists and is not superseded."""
        item = self._pending.get(candidate_id)
        if item is None:
            return False
        current = current or self._current_time()
        if current < item["accept_at"]:
            self._schedule_pending()
            return False

        before = self._state_signature()
        self._pending.pop(candidate_id, None)
        sample = dict(item["sample"])
        started_at = item["accept_at"] - timedelta(seconds=self.durability_seconds)
        if sample.get("day_type") not in VALID_DAY_TYPES:
            sample["day_type"] = self._resolve_day_type(started_at, sample)
        elapsed = max(self.durability_seconds, (current - started_at).total_seconds())
        sample["duration_seconds"] = max(
            _finite_number(sample.get("duration_seconds")) or 0.0,
            elapsed,
        )
        if self._learner.record(self._learner_sample(sample)):
            self._accepted_count += 1
            day_type = _text(sample.get("day_type"))
            self._day_type_counts[day_type] = self._day_type_counts.get(day_type, 0) + 1
            self._last_sample = _json_safe(sample)
            self._last_rejection_reason = None
            if (
                self._phase == PHASE_SHADOW_LEARNING
                and self._training_deadline is not None
                and current >= self._training_deadline
            ):
                self._evaluate_promotion()
            await self._persist_and_notify(before)
            return True

        self._rejected_count += 1
        self._last_sample = _json_safe(sample)
        self._last_rejection_reason = "learner_rejected"
        await self._persist_and_notify(before)
        return False

    def _cancel_timers(self) -> None:
        """Cancel the deadline and every pending durability callback."""
        if self._deadline_timer is not None:
            self._deadline_timer()
            self._deadline_timer = None
        for cancel in self._pending_timers.values():
            cancel()
        self._pending_timers.clear()

    def _new_candidate_id(self, sample: Mapping[str, Any]) -> str:
        """Use an event id when supplied, otherwise create a local id."""
        supplied = _first(sample, "sample_id", "id", default=_MISSING)
        if isinstance(supplied, str) and supplied.strip():
            return supplied.strip()
        self._sequence += 1
        return f"training-{self._sequence}-{uuid4().hex[:8]}"

    def _reject(
        self,
        sample: Mapping[str, Any] | None,
        reason: str,
    ) -> None:
        """Record a visible rejection without passing contaminated data on."""
        self._rejected_count += 1
        self._last_sample = _json_safe(dict(sample)) if sample is not None else None
        self._last_rejection_reason = reason

    async def async_ingest_candidate(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        sample: OverrideSample | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Queue a manual candidate for durable acceptance.

        A ``True``-like result here means "queued for observation", not that
        the learner has already changed.  The learner is updated only by the
        durability callback and only if no newer candidate superseded it.
        """
        if not self._loaded:
            await self.async_setup()
        if self._unloaded:
            return {"queued": False, "accepted": False, "reason": "unloaded"}

        canonical = _sample_mapping(sample)
        before = self._state_signature()
        if canonical is None:
            self._reject(None, "invalid_sample")
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": "invalid_sample"}

        reason = _candidate_reason(canonical)
        if reason is not None:
            if reason == "superseded":
                candidate_id = _first(canonical, "sample_id", "id", default=_MISSING)
                if isinstance(candidate_id, str):
                    self._cancel_pending(candidate_id, "superseded")
            self._reject(canonical, reason)
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": reason}

        zone = _text(canonical.get("zone"))
        if not zone:
            self._reject(canonical, "invalid_zone")
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": "invalid_zone"}

        for field in ("baseline", "selected"):
            value = _finite_number(canonical.get(field))
            if value is None or not 0 <= value <= 100:
                self._reject(canonical, f"invalid_{field}")
                await self._persist_and_notify(before)
                return {
                    "queued": False,
                    "accepted": False,
                    "reason": f"invalid_{field}",
                }

        duration_value = canonical.get("duration_seconds")
        duration = _finite_number(duration_value)
        if duration_value is not None and duration is None:
            self._reject(canonical, "invalid_duration")
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": "invalid_duration"}
        if duration is not None and duration < 0:
            self._reject(canonical, "invalid_duration")
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": "invalid_duration"}

        current = _as_utc(now) if now is not None else self._current_time()
        started_at = (
            _parse_datetime(
                _first(
                    canonical,
                    "started_at",
                    "override_started_at",
                    "observed_at",
                    "timestamp",
                    default=_MISSING,
                ),
            )
            or current
        )
        try:
            canonical["day_type"] = self._resolve_day_type(started_at, canonical)
        except ValueError:
            self._reject(canonical, "invalid_day_type")
            await self._persist_and_notify(before)
            return {"queued": False, "accepted": False, "reason": "invalid_day_type"}
        candidate_id = self._new_candidate_id(canonical)
        fingerprint = _fingerprint(canonical)

        # A newer manual override in the same context supersedes the older
        # candidate.  Supersession is explicit and durable, never silent.
        for old_id, old_item in tuple(self._pending.items()):
            if old_item["fingerprint"] == fingerprint and old_id != candidate_id:
                self._cancel_pending(old_id, "superseded")

        if candidate_id in self._pending:
            self._cancel_pending(candidate_id, "superseded")

        accept_at = started_at + timedelta(seconds=self.durability_seconds)
        self._pending[candidate_id] = {
            "id": candidate_id,
            "sample": canonical,
            "accept_at": accept_at,
            "fingerprint": fingerprint,
        }
        self._last_sample = _json_safe(canonical)
        await self._persist_and_notify(before)
        if accept_at <= current:
            accepted = await self._async_accept_pending(candidate_id, current=current)
            return {
                "queued": False,
                "accepted": accepted,
                "reason": "accepted" if accepted else "learner_rejected",
                "sample_id": candidate_id,
                "accept_at": accept_at.isoformat(),
            }
        # An explicit ``now`` is the deterministic/test clock.  Let callers
        # advance it through ``async_process_due`` instead of mixing it with
        # the event loop's wall-clock timer.
        if now is None:
            self._schedule_pending()
        return {
            "queued": True,
            "accepted": False,
            "reason": "durability_pending",
            "sample_id": candidate_id,
            "accept_at": accept_at.isoformat(),
        }

    async def async_ingest_sample(
        self,
        sample: OverrideSample | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> bool:
        """Queue a candidate and return whether it passed admission checks."""
        result = await self.async_ingest_candidate(sample, now=now)
        return bool(result.get("queued") or result.get("accepted"))

    async_add_sample = async_ingest_sample
    async_record_sample = async_ingest_sample

    def _cancel_pending(self, candidate_id: str, reason: str) -> bool:
        """Cancel one pending candidate and make the reason observable."""
        if candidate_id not in self._pending:
            return False
        self._pending.pop(candidate_id, None)
        cancel = self._pending_timers.pop(candidate_id, None)
        if cancel is not None:
            cancel()
        self._rejected_count += 1
        self._superseded_count += 1
        self._last_rejection_reason = reason
        return True

    async def async_cancel_candidate(self, candidate_id: str) -> bool:
        """Cancel a pending candidate, usually when an override is released."""
        before = self._state_signature()
        changed = self._cancel_pending(candidate_id, "cancelled")
        if changed:
            await self._persist_and_notify(before)
        return changed

    async_cancel_sample = async_cancel_candidate
    cancel_candidate = async_cancel_candidate

    async def async_process_due(self, *, now: datetime | None = None) -> None:
        """Process due candidates/deadline; useful for deterministic HA tests."""
        current = _as_utc(now) if now is not None else self._current_time()
        for candidate_id, item in tuple(self._pending.items()):
            if item["accept_at"] <= current:
                cancel = self._pending_timers.pop(candidate_id, None)
                if cancel is not None:
                    cancel()
                await self._async_accept_pending(candidate_id, current=current)
        if (
            self._phase == PHASE_SHADOW_LEARNING
            and self._training_deadline is not None
            and current >= self._training_deadline
        ):
            await self.async_evaluate_promotion(now=current)

    def _evaluate_promotion(self) -> bool:
        """Evaluate the gates synchronously after a deadline check."""
        if not self.auto_promote:
            self._promotion_reason = "auto_promotion_disabled"
            return False
        sample_ok = self._accepted_count >= self.minimum_samples
        confidence_ok = self.confidence >= self.minimum_confidence
        if sample_ok and confidence_ok:
            self._phase = PHASE_ACTIVE
            self._promotion_reason = "promoted"
            return True
        reasons: list[str] = []
        if not sample_ok:
            reasons.append(
                f"insufficient_samples({self._accepted_count}/{self.minimum_samples})",
            )
        if not confidence_ok:
            reasons.append(
                f"insufficient_confidence({self.confidence:.2f}/{self.minimum_confidence:.2f})",
            )
        self._phase = PHASE_SHADOW_LEARNING
        self._promotion_reason = "; ".join(reasons)
        return False

    async def async_evaluate_promotion(
        self,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Promote only after the deadline and only when both gates pass."""
        if not self._loaded:
            await self.async_setup()
        if self._phase == PHASE_ACTIVE:
            return True
        current = _as_utc(now) if now is not None else self._current_time()
        if self._training_deadline is None or current < self._training_deadline:
            self._promotion_reason = "training_in_progress"
            return False
        before = self._state_signature()
        promoted = self._evaluate_promotion()
        self._cancel_deadline_timer()
        await self._persist_and_notify(before)
        return promoted

    async_promote_if_eligible = async_evaluate_promotion

    async def async_reset(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Reset learner/session state and begin a new shadow period."""
        before = self._state_signature()
        self._cancel_timers()
        self._start_new_session(
            _as_utc(now) if now is not None else self._current_time(),
        )
        if not self._loaded:
            self._loaded = True
        await self._persist_and_notify(before)
        self._schedule_deadline()
        return self.summary(now=now)


def _stored_int(value: Any) -> int:
    """Read a persisted non-negative integer without coercing booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


# Friendly aliases make the adapter easy to discover without creating
# separate stateful implementations that could drift from this one.
TrainingManager = AdaptiveLightingTraining
LocalTrainingSession = AdaptiveLightingTraining
TrainingSession = AdaptiveLightingTraining

__all__ = [
    "DATA_VERSION",
    "DAY_TYPE_PUBLIC_HOLIDAY",
    "DAY_TYPE_WEEKDAY",
    "DAY_TYPE_WEEKEND",
    "DEFAULT_DURABILITY_SECONDS",
    "DEFAULT_MINIMUM_CONFIDENCE",
    "DEFAULT_MINIMUM_SAMPLES",
    "DEFAULT_TRAINING_DURATION_DAYS",
    "PHASE_ACTIVE",
    "PHASE_SHADOW_LEARNING",
    "STORAGE_KEY",
    "STORAGE_VERSION",
    "AdaptiveLightingTraining",
    "LocalTrainingSession",
    "TrainingManager",
    "TrainingSession",
]
