# ruff: noqa: D101, D102, D105, D107, EM101, EM102, FBT003, PERF401, RUF005, TRY003

"""Explainable temporal features and a small online action model.

This module intentionally has no Home Assistant dependency.  It is suitable
for keeping one learner per entity (or for a small local collection of
entities) and is deliberately conservative when the evidence is sparse.

The model is an online logistic learner with L2 shrinkage and exponential
forgetting.  It is not intended to be a general reinforcement-learning
policy: every update is a weighted, inspectable observation and every
prediction reports its support and freshness.  Autonomous proposals are kept
as bounded feature snapshots so a later human correction can provide
counterfactual feedback without retaining an event log.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Any, Self
from zoneinfo import ZoneInfo

__all__ = [
    "EVENT_TYPES",
    "HORIZONS",
    "ActionModel",
    "ActionModelPrediction",
    "ActionProvenance",
    "EncodedFeatures",
    "OnlineActionModel",
    "PendingAction",
    "Prediction",
    "ProposalOutcome",
    "TemporalContext",
    "TemporalFeatureEncoder",
    "TrainingResult",
    "UnsupportedSchemaError",
    "encode_temporal_features",
    "is_training_eligible",
]

SCHEMA_VERSION = 1
EVENT_TYPES = (
    "motion",
    "presence",
    "arrival",
    "opening",
    "media",
    "alarm",
    "manual",
)
HORIZONS = ("short", "medium", "long")
DEFAULT_HORIZON_SECONDS = {
    "short": 15.0 * 60.0,
    "medium": 2.0 * 60.0 * 60.0,
    "long": 24.0 * 60.0 * 60.0,
}
DEFAULT_CATEGORICAL_SCHEMA = {
    "occupancy": ("empty", "occupied", "unknown"),
    "media": ("idle", "playing", "paused"),
    "mode": ("home", "away", "sleep"),
    "lighting": ("off", "on", "unknown"),
}
DEFAULT_MAX_CATEGORIES = 8
DEFAULT_MAX_STATE_FEATURES = 12
DEFAULT_MAX_EXTRA_BUCKETS = 8
DEFAULT_MAX_DWELL_SECONDS = 6.0 * 60.0 * 60.0
_HALF_LIFE_CONSTANT = math.log(2.0)
_UNREGISTERED_CATEGORY_BUCKETS = "context_extra_bucket_"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ACTION_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}")
_PROPOSAL_ID_RE = re.compile(r"proposal-[0-9]{1,12}")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{20}")
_FEATURE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_MAX_PERSISTED_ACTIONS = 1_024
_MAX_PERSISTED_FEATURES = 4_096
_MAX_PERSISTED_PENDING = 1_024
_MAX_PERSISTED_SUPPRESSIONS = 1_024
_MAX_PERSISTED_CATEGORIES = 64
_MAX_PERSISTED_SCHEMA_FIELDS = 64
_MAX_PERSISTED_DURATION_SECONDS = 366.0 * 24.0 * 60.0 * 60.0
_MAX_PERSISTED_SUPPORT = 4_000_000.0
_MAX_PERSISTED_COUNTER = 1_000_000_000
_EVENT_ALIASES = {
    "motions": "motion",
    "presence_change": "presence",
    "presence_changes": "presence",
    "arrivals": "arrival",
    "arrival_event": "arrival",
    "open": "opening",
    "openings": "opening",
    "door_opening": "opening",
    "door_openings": "opening",
    "media_playback": "media",
    "media_events": "media",
    "alarms": "alarm",
    "alarm_event": "alarm",
    "manual_action": "manual",
    "manual_actions": "manual",
}


class UnsupportedSchemaError(ValueError):
    """Raised when persisted state uses a schema this code does not know."""


def _slug(value: str) -> str:
    """Turn user-provided labels into stable, bounded feature components."""
    normalized = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    normalized = normalized or "unknown"
    return f"v_{normalized}" if normalized[0].isdigit() else normalized


def _normalise_text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _duration_seconds(value: timedelta | float, *, name: str) -> float:
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, bool):
        raise TypeError(f"{name} must be a duration")
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{name} must be positive")
    return seconds


def _as_utc(value: datetime, local_timezone: tzinfo | None = None) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=local_timezone or UTC).astimezone(UTC)
    return value.astimezone(UTC)


def _serialise_datetime(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("serialized timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("serialized timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _finite(value: Any, *, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _required_dict(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    return value


def _required_list(value: Any, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return value


def _required_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} is above its maximum")
    return number


def _required_int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{name} has an invalid field set")


def _required_text(
    value: Any,
    *,
    name: str,
    maximum: int = 128,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > maximum or value != value.strip():
        raise ValueError(f"{name} is invalid")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{name} is invalid")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """The small, HA-independent context snapshot used by the encoder.

    ``event_times`` contains the latest timestamp, or a finite iterable of
    timestamps, for each event family.  ``state_dwell`` values are either
    seconds or ``timedelta`` instances.  All mappings are copied at
    construction so a caller cannot mutate a context after it is encoded.
    A naive timestamp is interpreted in the encoder's configured timezone, or
    UTC when no timezone is configured.
    """

    timestamp: datetime
    event_times: Mapping[str, datetime | Iterable[datetime] | None] = field(
        default_factory=dict,
    )
    state_dwell: Mapping[str, timedelta | float | int] = field(default_factory=dict)
    categorical_context: Mapping[str, str] = field(default_factory=dict)
    entity_id: str | None = None
    is_holiday: bool = False
    holiday_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if not isinstance(self.is_holiday, bool):
            raise TypeError("is_holiday must be a bool")
        if self.holiday_name is not None:
            if not isinstance(self.holiday_name, str):
                raise TypeError("holiday_name must be a string or None")
            cleaned_holiday = self.holiday_name.strip()
            object.__setattr__(self, "holiday_name", cleaned_holiday or None)
            if cleaned_holiday:
                object.__setattr__(self, "is_holiday", True)
        object.__setattr__(self, "event_times", dict(self.event_times))
        object.__setattr__(self, "state_dwell", dict(self.state_dwell))
        object.__setattr__(
            self,
            "categorical_context",
            dict(self.categorical_context),
        )
        if self.entity_id is not None:
            if not isinstance(self.entity_id, str):
                raise TypeError("entity_id must be a string or None")
            entity_id = self.entity_id.strip()
            if len(entity_id) > 128:
                raise ValueError("entity_id must be at most 128 characters")
            object.__setattr__(self, "entity_id", entity_id or None)

    @property
    def at(self) -> datetime:
        """Short alias useful at call sites that speak in terms of an event."""
        return self.timestamp

    @property
    def events(self) -> Mapping[str, datetime | Iterable[datetime] | None]:
        """Alias for ``event_times``."""
        return self.event_times

    @property
    def categories(self) -> Mapping[str, str]:
        """Alias for ``categorical_context``."""
        return self.categorical_context


@dataclass(frozen=True, slots=True)
class EncodedFeatures(Mapping[str, float]):
    """Bounded feature values plus diagnostic context metadata."""

    values: Mapping[str, float]
    metadata: Mapping[str, str | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


class TemporalFeatureEncoder:
    """Encode time, recency, dwell, and fixed-schema context features.

    Recency values use exponential half-life decay.  Categories not present in
    a configured schema map to that field's ``unknown`` feature.  Completely
    unregistered fields use deterministic hash buckets, which prevents a
    changing sensor label from growing the feature vocabulary forever.
    """

    def __init__(
        self,
        horizons: Mapping[str, timedelta | float | int] | None = None,
        categorical_schema: Mapping[str, Sequence[str]] | None = None,
        *,
        category_schema: Mapping[str, Sequence[str]] | None = None,
        max_categories_per_field: int = DEFAULT_MAX_CATEGORIES,
        max_state_features: int = DEFAULT_MAX_STATE_FEATURES,
        max_extra_buckets: int = DEFAULT_MAX_EXTRA_BUCKETS,
        state_dwell_schema: Sequence[str] | None = None,
        max_dwell_seconds: timedelta | float = DEFAULT_MAX_DWELL_SECONDS,
        timezone: tzinfo | None = None,
        timezone_name: str | None = None,
    ) -> None:
        if categorical_schema is not None and category_schema is not None:
            raise ValueError("provide only one categorical schema")
        schema_input = (
            categorical_schema or category_schema or DEFAULT_CATEGORICAL_SCHEMA
        )
        if (
            not isinstance(max_categories_per_field, int)
            or not 1 <= max_categories_per_field <= 64
        ):
            raise ValueError("max_categories_per_field must be between 1 and 64")
        for name, value in (
            ("max_state_features", max_state_features),
            ("max_extra_buckets", max_extra_buckets),
        ):
            if not isinstance(value, int) or not 0 <= value <= 256:
                raise ValueError(f"{name} must be between 0 and 256")
        if timezone_name is not None:
            if timezone is not None:
                raise ValueError("provide only one timezone representation")
            timezone = ZoneInfo(timezone_name)
        if timezone_name is None:
            timezone_name = getattr(timezone, "key", None)

        source_horizons = horizons or DEFAULT_HORIZON_SECONDS
        resolved_horizons: dict[str, float] = {}
        for horizon in HORIZONS:
            if horizon not in source_horizons:
                raise ValueError(f"missing horizon: {horizon}")
            resolved_horizons[horizon] = _duration_seconds(
                source_horizons[horizon],
                name=f"{horizon} horizon",
            )

        resolved_schema: dict[str, tuple[str, ...]] = {}
        for field_name, raw_values in sorted(schema_input.items()):
            field_key = _normalise_text(field_name)
            if not field_key:
                raise ValueError("categorical field names must be non-empty")
            if isinstance(raw_values, str):
                raise TypeError("categorical values must be a sequence")
            values = sorted(
                {
                    _normalise_text(value)
                    for value in raw_values
                    if _normalise_text(value)
                },
            )
            # Reserve one slot for unknown so a schema cannot make the model
            # unbounded through an accidental high-cardinality configuration.
            values = values[: max_categories_per_field - 1]
            resolved_schema[field_key] = tuple(values) + ("__unknown__",)

        self.horizons = resolved_horizons
        self.categorical_schema = resolved_schema
        self.max_categories_per_field = max_categories_per_field
        self.max_state_features = max_state_features
        self.max_extra_buckets = max_extra_buckets
        self.state_dwell_schema = tuple(
            sorted(
                {
                    _normalise_text(value)
                    for value in (state_dwell_schema or ())
                    if _normalise_text(value)
                },
            ),
        )
        self.max_dwell_seconds = _duration_seconds(
            max_dwell_seconds,
            name="max_dwell_seconds",
        )
        self.timezone = timezone
        self.timezone_name = timezone_name
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> tuple[str, ...]:
        names = [
            "local_time_sin",
            "local_time_cos",
            "calendar_weekday",
            "calendar_weekend",
            "calendar_holiday",
            "state_dwell_any",
        ]
        for event_type in EVENT_TYPES:
            for horizon in HORIZONS:
                names.append(f"{event_type}_recency_{horizon}")
        for field_name, values in self.categorical_schema.items():
            names.extend(
                f"context_{_slug(field_name)}_{_slug(value)}" for value in values
            )
        names.extend(
            f"{_UNREGISTERED_CATEGORY_BUCKETS}{index}"
            for index in range(self.max_extra_buckets)
        )
        for state_name in self.state_dwell_schema:
            names.append(f"state_dwell_{_slug(state_name)}")
        return tuple(names)

    def _local_time(self, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=self.timezone or UTC)
        if self.timezone is not None:
            return timestamp.astimezone(self.timezone)
        return timestamp

    def _event_timestamp(self, timestamp: datetime) -> datetime:
        return _as_utc(timestamp, self.timezone)

    def normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Return a canonical UTC timestamp using the encoder's local zone."""
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        return self._event_timestamp(timestamp)

    @staticmethod
    def decay(age: timedelta | float, horizon: timedelta | float) -> float:
        """Return a half-life decay in ``[0, 1]`` for a non-negative age."""
        age_seconds = max(
            0.0,
            age.total_seconds() if isinstance(age, timedelta) else float(age),
        )
        horizon_seconds = _duration_seconds(horizon, name="horizon")
        return math.exp(-_HALF_LIFE_CONSTANT * age_seconds / horizon_seconds)

    @staticmethod
    def _canonical_event_type(name: str) -> str | None:
        normalized = _normalise_text(name)
        normalized = _EVENT_ALIASES.get(normalized, normalized)
        return normalized if normalized in EVENT_TYPES else None

    def _latest_event(
        self,
        value: Any,
        *,
        not_after: datetime | None = None,
    ) -> datetime | None:
        if value is None:
            return None
        candidates: list[datetime] = []
        if isinstance(value, datetime):
            candidates.append(value)
        elif not isinstance(value, (str, bytes)):
            try:
                candidates.extend(item for item in value if isinstance(item, datetime))
            except TypeError:
                return None
        if not candidates:
            return None
        if not_after is not None:
            cutoff = self._event_timestamp(not_after)
            candidates = [
                candidate
                for candidate in candidates
                if self._event_timestamp(candidate) <= cutoff
            ]
        return max(candidates, key=self._event_timestamp) if candidates else None

    def _categorical_features(
        self,
        context: TemporalContext,
        features: dict[str, float],
    ) -> None:
        supplied = {
            _normalise_text(key): _normalise_text(value)
            for key, value in context.categorical_context.items()
        }
        for field_name, allowed_values in self.categorical_schema.items():
            value = supplied.get(field_name, "__unknown__")
            if value not in allowed_values:
                value = "__unknown__"
            for allowed_value in allowed_values:
                features[f"context_{_slug(field_name)}_{_slug(allowed_value)}"] = (
                    1.0 if value == allowed_value else 0.0
                )

        if self.max_extra_buckets == 0:
            return
        for field_name in sorted(set(supplied) - set(self.categorical_schema)):
            value = supplied[field_name]
            digest = hashlib.sha256(f"{field_name}={value}".encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.max_extra_buckets
            features[f"{_UNREGISTERED_CATEGORY_BUCKETS}{bucket}"] = 1.0

    def encode(self, context: TemporalContext) -> EncodedFeatures:
        """Encode a context into finite values with a stable feature budget."""
        if not isinstance(context, TemporalContext):
            raise TypeError("context must be a TemporalContext")
        local = self._local_time(context.timestamp)
        seconds_today = (
            local.hour * 3600.0
            + local.minute * 60.0
            + local.second
            + local.microsecond / 1_000_000.0
        )
        angle = 2.0 * math.pi * seconds_today / 86_400.0
        is_weekend = local.weekday() >= 5 or context.is_holiday
        features: dict[str, float] = {
            "local_time_sin": math.sin(angle),
            "local_time_cos": math.cos(angle),
            "calendar_weekday": 0.0 if is_weekend else 1.0,
            "calendar_weekend": 1.0 if is_weekend else 0.0,
            # A separate bit keeps a holiday diagnostically identifiable even
            # though its behavioral branch intentionally follows weekends.
            "calendar_holiday": 1.0 if context.is_holiday else 0.0,
        }

        latest_events: dict[str, datetime] = {}
        now_utc = self._event_timestamp(context.timestamp)
        for raw_name, raw_time in context.event_times.items():
            event_type = self._canonical_event_type(raw_name)
            latest = self._latest_event(raw_time, not_after=now_utc)
            if event_type is not None and latest is not None:
                previous = latest_events.get(event_type)
                if previous is None or self._event_timestamp(
                    latest,
                ) > self._event_timestamp(previous):
                    latest_events[event_type] = latest
        recency_ages: dict[str, float | None] = {}
        for event_type in EVENT_TYPES:
            latest = latest_events.get(event_type)
            age = (
                max(0.0, (now_utc - self._event_timestamp(latest)).total_seconds())
                if latest is not None
                else None
            )
            recency_ages[event_type] = age
            for horizon in HORIZONS:
                features[f"{event_type}_recency_{horizon}"] = (
                    self.decay(age, self.horizons[horizon]) if age is not None else 0.0
                )

        dwell_values: dict[str, float] = {}
        for raw_name, raw_duration in context.state_dwell.items():
            name = _normalise_text(raw_name)
            if not name:
                continue
            if isinstance(raw_duration, timedelta):
                seconds = raw_duration.total_seconds()
            else:
                seconds = _finite(raw_duration, default=0.0) or 0.0
            dwell_values[name] = min(1.0, max(0.0, seconds) / self.max_dwell_seconds)
        features["state_dwell_any"] = max(dwell_values.values(), default=0.0)
        state_names = self.state_dwell_schema or tuple(
            sorted(dwell_values)[: self.max_state_features],
        )
        for state_name in state_names[: self.max_state_features]:
            features[f"state_dwell_{_slug(state_name)}"] = dwell_values.get(
                state_name,
                0.0,
            )

        self._categorical_features(context, features)
        day_type = "weekend" if is_weekend else "weekday"
        metadata: dict[str, str | bool | None] = {
            "behavior_day_type": day_type,
            "calendar_day_type": "holiday" if context.is_holiday else day_type,
            "holiday": context.holiday_name,
            "entity_id": context.entity_id,
        }
        return EncodedFeatures(features, metadata)

    def context_signature(self, context: TemporalContext) -> str:
        """Return a stable, coarse signature for correction suppression.

        Recency and exact clock time are intentionally omitted.  A correction
        at 18:02 should suppress a repeat proposal at 18:10, while the entity,
        day branch, categorical context, and quarter-hour clock bucket still
        keep unrelated contexts separate.
        """
        local = self._local_time(context.timestamp)
        is_weekend = local.weekday() >= 5 or context.is_holiday
        categories = sorted(
            (_normalise_text(key), _normalise_text(value))
            for key, value in context.categorical_context.items()
        )
        dwell_names = sorted(_normalise_text(key) for key in context.state_dwell if key)
        payload = {
            "entity": context.entity_id or "__default__",
            "day": "weekend" if is_weekend else "weekday",
            "holiday": bool(context.is_holiday),
            "clock_bucket": local.hour * 4 + local.minute // 15,
            "categories": categories,
            "dwell_names": dwell_names,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons": dict(self.horizons),
            "categorical_schema": {
                key: list(values[:-1])
                for key, values in self.categorical_schema.items()
            },
            "max_categories_per_field": self.max_categories_per_field,
            "max_state_features": self.max_state_features,
            "max_extra_buckets": self.max_extra_buckets,
            "state_dwell_schema": list(self.state_dwell_schema),
            "max_dwell_seconds": self.max_dwell_seconds,
            "timezone_name": self.timezone_name,
        }

    def is_valid_feature_name(self, name: Any) -> bool:
        """Return whether a persisted feature name belongs to this encoder."""
        if type(name) is not str or _FEATURE_NAME_RE.fullmatch(name) is None:
            return False
        return name in self.feature_names or name.startswith("state_dwell_")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:  # noqa: PLR0912
        raw = _required_dict(data, name="encoder")
        _require_exact_keys(
            raw,
            {
                "horizons",
                "categorical_schema",
                "max_categories_per_field",
                "max_state_features",
                "max_extra_buckets",
                "state_dwell_schema",
                "max_dwell_seconds",
                "timezone_name",
            },
            name="encoder",
        )
        raw_horizons = _required_dict(raw["horizons"], name="encoder.horizons")
        if set(raw_horizons) != set(HORIZONS):
            raise ValueError("encoder.horizons has an invalid field set")
        horizons = {
            horizon: _required_number(
                raw_horizons[horizon],
                name=f"encoder.horizons.{horizon}",
                minimum=math.nextafter(0.0, 1.0),
                maximum=_MAX_PERSISTED_DURATION_SECONDS,
            )
            for horizon in HORIZONS
        }
        max_categories = _required_int(
            raw["max_categories_per_field"],
            name="encoder.max_categories_per_field",
            minimum=1,
            maximum=_MAX_PERSISTED_CATEGORIES,
        )
        max_state_features = _required_int(
            raw["max_state_features"],
            name="encoder.max_state_features",
            maximum=256,
        )
        max_extra_buckets = _required_int(
            raw["max_extra_buckets"],
            name="encoder.max_extra_buckets",
            maximum=256,
        )
        raw_schema = _required_dict(
            raw["categorical_schema"],
            name="encoder.categorical_schema",
        )
        if len(raw_schema) > _MAX_PERSISTED_SCHEMA_FIELDS:
            raise ValueError("encoder categorical schema is too large")
        categorical_schema: dict[str, list[str]] = {}
        for field_name, raw_values in raw_schema.items():
            if type(field_name) is not str or not field_name.strip():
                raise ValueError("encoder categorical field is invalid")
            values = _required_list(
                raw_values,
                name=f"encoder.categorical_schema.{field_name}",
            )
            if len(values) > max_categories - 1:
                raise ValueError("encoder categorical field has too many values")
            normalized_values: list[str] = []
            for value in values:
                if type(value) is not str or not value.strip():
                    raise ValueError("encoder categorical value is invalid")
                normalized = _normalise_text(value)
                if normalized == "__unknown__" or normalized in normalized_values:
                    raise ValueError("encoder categorical values contain a duplicate")
                normalized_values.append(normalized)
            categorical_schema[field_name] = normalized_values
        raw_state_schema = _required_list(
            raw["state_dwell_schema"],
            name="encoder.state_dwell_schema",
        )
        if len(raw_state_schema) > max_state_features:
            raise ValueError("encoder state schema is too large")
        state_dwell_schema: list[str] = []
        for state_name in raw_state_schema:
            if type(state_name) is not str or not state_name.strip():
                raise ValueError("encoder state name is invalid")
            normalized = _normalise_text(state_name)
            if normalized in state_dwell_schema:
                raise ValueError("encoder state schema contains a duplicate")
            state_dwell_schema.append(normalized)
        timezone_name = raw["timezone_name"]
        if timezone_name is not None:
            timezone_name = _required_text(
                timezone_name,
                name="encoder.timezone_name",
                maximum=128,
            )
        return cls(
            horizons=horizons,
            categorical_schema=categorical_schema,
            max_categories_per_field=max_categories,
            max_state_features=max_state_features,
            max_extra_buckets=max_extra_buckets,
            state_dwell_schema=state_dwell_schema,
            max_dwell_seconds=_required_number(
                raw["max_dwell_seconds"],
                name="encoder.max_dwell_seconds",
                minimum=math.nextafter(0.0, 1.0),
                maximum=_MAX_PERSISTED_DURATION_SECONDS,
            ),
            timezone_name=timezone_name,
        )

    def encode_to_dict(self, context: TemporalContext) -> dict[str, float]:
        """Convenience form for integrations that only need feature values."""
        return dict(self.encode(context).values)


def encode_temporal_features(
    context: TemporalContext,
    encoder: TemporalFeatureEncoder | None = None,
) -> EncodedFeatures:
    """Encode ``context`` using a default or supplied encoder."""
    return (encoder or TemporalFeatureEncoder()).encode(context)


class ActionProvenance(StrEnum):
    """Common origins accepted by the training gate."""

    USER = "user"
    MANUAL = "manual"
    AUTOMATION = "automation"
    PATTERN = "pattern"
    AUTONOMOUS_OBSERVATION = "autonomous_observation"
    ALARM = "alarm"
    SAFETY = "safety"


class ProposalOutcome(StrEnum):
    """Feedback labels for a previously registered autonomous proposal."""

    UNCHANGED = "unchanged"
    CORRECTED = "corrected"
    ACCEPTED = "accepted"


_EXCLUDED_PROVENANCE_WORDS = frozenset(
    {"alarm", "safety", "emergency", "evacuation", "security"},
)
_MANUAL_PROVENANCE_WORDS = frozenset(
    {"manual", "user", "human", "physical", "person", "wall_switch"},
)


def _provenance_words(provenance: ActionProvenance | str) -> set[str]:
    value = provenance.value if isinstance(provenance, ActionProvenance) else provenance
    if not isinstance(value, str):
        return set()
    return set(filter(None, re.split(r"[^a-z0-9]+", value.strip().lower())))


def is_training_eligible(provenance: ActionProvenance | str) -> bool:
    """Return false for alarm/safety-origin targets, including compound labels."""
    words = _provenance_words(provenance)
    return bool(words) and not words.intersection(_EXCLUDED_PROVENANCE_WORDS)


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    name: str
    value: float
    weight: float
    contribution: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass(frozen=True, slots=True)
class Prediction:
    """A prediction with calibrated confidence and reasons for refusal."""

    action: str
    probability: float
    confidence: float
    effective_support: float
    freshness: float
    probability_margin: float
    support_factor: float
    freshness_factor: float
    top_contributing_features: tuple[FeatureContribution, ...]
    not_ready: bool
    not_ready_reasons: tuple[str, ...] = ()
    suppressed: bool = False
    suppression_until: datetime | None = None

    @property
    def support(self) -> float:
        return self.effective_support

    @property
    def top_features(self) -> tuple[FeatureContribution, ...]:
        return self.top_contributing_features

    @property
    def ready(self) -> bool:
        return not self.not_ready

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "probability": self.probability,
            "confidence": self.confidence,
            "effective_support": self.effective_support,
            "support": self.support,
            "freshness": self.freshness,
            "probability_margin": self.probability_margin,
            "support_factor": self.support_factor,
            "freshness_factor": self.freshness_factor,
            "top_contributing_features": [
                feature.as_dict() for feature in self.top_contributing_features
            ],
            "not_ready": self.not_ready,
            "not_ready_reasons": list(self.not_ready_reasons),
            "suppressed": self.suppressed,
            "suppression_until": (
                _serialise_datetime(self.suppression_until)
                if self.suppression_until is not None
                else None
            ),
        }


ActionModelPrediction = Prediction


@dataclass(frozen=True, slots=True)
class TrainingResult:
    accepted: bool
    action: str | None
    reason: str
    effective_weight: float = 0.0
    feedback_kind: str | None = None
    proposal_id: str | None = None

    @property
    def updated(self) -> bool:
        return self.accepted


@dataclass(frozen=True, slots=True)
class PendingAction:
    """Bounded state retained between an autonomous proposal and feedback."""

    proposal_id: str
    action: str
    entity_id: str
    created_at: datetime
    context_signature: str
    features: Mapping[str, float]
    predicted_probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(self.features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "action": self.action,
            "entity_id": self.entity_id,
            "created_at": _serialise_datetime(self.created_at),
            "context_signature": self.context_signature,
            "features": dict(sorted(self.features.items())),
            "predicted_probability": self.predicted_probability,
        }


@dataclass(slots=True)
class _ActionState:
    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    positive_support: float = 0.0
    negative_support: float = 0.0
    observation_count: int = 0
    last_update: datetime | None = None


@dataclass(frozen=True, slots=True)
class _Suppression:
    entity_id: str
    action: str
    context_signature: str
    until: datetime
    factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "action": self.action,
            "context_signature": self.context_signature,
            "until": _serialise_datetime(self.until),
            "factor": self.factor,
        }


_OPPOSITE_ACTIONS = {
    "on": "off",
    "off": "on",
    "turn_on": "turn_off",
    "turn_off": "turn_on",
    "open": "close",
    "close": "open",
    "lock": "unlock",
    "unlock": "lock",
    "play": "pause",
    "pause": "play",
    "increase": "decrease",
    "decrease": "increase",
    "raise": "lower",
    "lower": "raise",
}


def _validated_model_config(raw: Any) -> dict[str, Any]:
    config = _required_dict(raw, name="model.config")
    expected = {
        "learning_rate",
        "regularization",
        "forgetting_half_life",
        "freshness_half_life",
        "stale_after",
        "min_effective_support",
        "max_actions",
        "max_features",
        "max_pending_actions",
        "correction_window",
        "correction_suppression",
        "max_suppressions",
    }
    _require_exact_keys(config, expected, name="model.config")
    return {
        "learning_rate": _required_number(
            config["learning_rate"],
            name="model.learning_rate",
            minimum=math.nextafter(0.0, 1.0),
            maximum=1.0,
        ),
        "regularization": _required_number(
            config["regularization"],
            name="model.regularization",
            minimum=0.0,
            maximum=1_000.0,
        ),
        "forgetting_half_life": _required_number(
            config["forgetting_half_life"],
            name="model.forgetting_half_life",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_DURATION_SECONDS,
        ),
        "freshness_half_life": _required_number(
            config["freshness_half_life"],
            name="model.freshness_half_life",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_DURATION_SECONDS,
        ),
        "stale_after": _required_number(
            config["stale_after"],
            name="model.stale_after",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_DURATION_SECONDS,
        ),
        "min_effective_support": _required_number(
            config["min_effective_support"],
            name="model.min_effective_support",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_SUPPORT,
        ),
        "max_actions": _required_int(
            config["max_actions"],
            name="model.max_actions",
            minimum=1,
            maximum=_MAX_PERSISTED_ACTIONS,
        ),
        "max_features": _required_int(
            config["max_features"],
            name="model.max_features",
            minimum=1,
            maximum=_MAX_PERSISTED_FEATURES,
        ),
        "max_pending_actions": _required_int(
            config["max_pending_actions"],
            name="model.max_pending_actions",
            minimum=1,
            maximum=_MAX_PERSISTED_PENDING,
        ),
        "correction_window": _required_number(
            config["correction_window"],
            name="model.correction_window",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_DURATION_SECONDS,
        ),
        "correction_suppression": _required_number(
            config["correction_suppression"],
            name="model.correction_suppression",
            minimum=math.nextafter(0.0, 1.0),
            maximum=_MAX_PERSISTED_DURATION_SECONDS,
        ),
        "max_suppressions": _required_int(
            config["max_suppressions"],
            name="model.max_suppressions",
            minimum=1,
            maximum=_MAX_PERSISTED_SUPPRESSIONS,
        ),
    }


class OnlineActionModel:
    """A bounded, explainable per-action online logistic learner.

    Confidence is deliberately separate from probability.  It is a conservative
    readiness heuristic, not a calibrated probability:

        ``confidence = probability_margin * support_factor * freshness_factor``

    where support is exponentially forgotten evidence.  Thus a high-margin
    prediction from one old observation remains explicitly not ready.
    """

    def __init__(
        self,
        encoder: TemporalFeatureEncoder | None = None,
        *,
        learning_rate: float = 0.18,
        regularization: float = 0.015,
        forgetting_half_life: timedelta | float = timedelta(days=7),
        freshness_half_life: timedelta | float | None = None,
        stale_after: timedelta | float = timedelta(days=21),
        min_effective_support: float = 3.0,
        max_actions: int = 64,
        max_features: int = 128,
        max_pending_actions: int = 64,
        correction_window: timedelta | float = timedelta(minutes=15),
        correction_suppression: timedelta | float = timedelta(minutes=30),
        max_suppressions: int = 64,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be in (0, 1]")
        if regularization < 0.0 or not math.isfinite(regularization):
            raise ValueError("regularization must be non-negative")
        if min_effective_support <= 0.0 or not math.isfinite(min_effective_support):
            raise ValueError("min_effective_support must be positive")
        for name, value in (
            ("max_actions", max_actions),
            ("max_features", max_features),
            ("max_pending_actions", max_pending_actions),
            ("max_suppressions", max_suppressions),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.encoder = encoder or TemporalFeatureEncoder()
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.forgetting_half_life = _duration_seconds(
            forgetting_half_life,
            name="forgetting_half_life",
        )
        self.freshness_half_life = _duration_seconds(
            freshness_half_life or forgetting_half_life,
            name="freshness_half_life",
        )
        self.stale_after = _duration_seconds(stale_after, name="stale_after")
        self.min_effective_support = float(min_effective_support)
        self.max_actions = max_actions
        self.max_features = max_features
        self.max_pending_actions = max_pending_actions
        self.correction_window = _duration_seconds(
            correction_window,
            name="correction_window",
        )
        self.correction_suppression = _duration_seconds(
            correction_suppression,
            name="correction_suppression",
        )
        self.max_suppressions = max_suppressions
        self._actions: dict[str, _ActionState] = {}
        self._pending: dict[str, PendingAction] = {}
        self._suppressions: dict[tuple[str, str, str], _Suppression] = {}
        self._proposal_counter = 0

    @staticmethod
    def _action_name(action: str) -> str:
        if not isinstance(action, str):
            raise TypeError("action must be a string")
        normalized = action.strip().lower()
        if not normalized:
            raise ValueError("action must be non-empty")
        if len(normalized) > 128:
            raise ValueError("action must be at most 128 characters")
        return normalized

    @staticmethod
    def _target(target: bool) -> bool:
        if not isinstance(target, bool):
            raise TypeError("target must be a bool")
        return target

    def _feature_map(
        self,
        context_or_features: TemporalContext | EncodedFeatures,
    ) -> dict[str, float]:
        encoded = (
            context_or_features.values
            if isinstance(context_or_features, EncodedFeatures)
            else self.encoder.encode(context_or_features).values
        )
        # Apply the persistence budget before both model updates and pending
        # proposal snapshots.  A model must always be able to restore state it
        # produced itself, even when configured with a deliberately tiny cap.
        bounded = (
            (name, max(-1.0, min(1.0, float(value))))
            for name, value in sorted(encoded.items())
            if math.isfinite(float(value))
        )
        return dict(list(bounded)[: self.max_features])

    def _decay_state(self, state: _ActionState, at: datetime) -> None:
        if state.last_update is None:
            state.last_update = at
            return
        age = max(0.0, (_as_utc(at) - _as_utc(state.last_update)).total_seconds())
        if age == 0.0:
            return
        factor = TemporalFeatureEncoder.decay(age, self.forgetting_half_life)
        state.bias *= factor
        state.positive_support *= factor
        state.negative_support *= factor
        for name in list(state.weights):
            state.weights[name] *= factor
            if abs(state.weights[name]) < 1e-9:
                del state.weights[name]
        state.last_update = at

    def _decayed_state(
        self,
        state: _ActionState,
        at: datetime,
    ) -> tuple[float, dict[str, float], float, float, float]:
        if state.last_update is None:
            return state.bias, dict(state.weights), 0.0, 0.0, 0.0
        age = max(0.0, (_as_utc(at) - _as_utc(state.last_update)).total_seconds())
        factor = TemporalFeatureEncoder.decay(age, self.forgetting_half_life)
        return (
            state.bias * factor,
            {name: weight * factor for name, weight in state.weights.items()},
            state.positive_support * factor,
            state.negative_support * factor,
            age,
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def _ensure_action(self, action: str) -> _ActionState | None:
        state = self._actions.get(action)
        if state is None:
            if len(self._actions) >= self.max_actions:
                return None
            state = _ActionState()
            self._actions[action] = state
        return state

    def _apply_update(
        self,
        action: str,
        features: Mapping[str, float],
        target: bool,
        at: datetime,
        weight: float,
        *,
        feedback_kind: str | None = None,
        proposal_id: str | None = None,
    ) -> TrainingResult:
        state = self._ensure_action(action)
        if state is None:
            return TrainingResult(False, action, "action_capacity")
        if state.last_update is not None and _as_utc(at) < _as_utc(state.last_update):
            return TrainingResult(False, action, "out_of_order_timestamp")
        self._decay_state(state, at)
        score = state.bias + sum(
            state.weights.get(name, 0.0) * value for name, value in features.items()
        )
        probability = self._sigmoid(score)
        signed_error = (1.0 if target else 0.0) - probability
        step = self.learning_rate * min(weight, 4.0)
        state.bias += step * signed_error - self.regularization * step * state.bias
        state.bias = max(-12.0, min(12.0, state.bias))
        for name, value in sorted(features.items()):
            if name not in state.weights and len(state.weights) >= self.max_features:
                continue
            current = state.weights.get(name, 0.0)
            updated = current + step * (
                signed_error * value - self.regularization * current
            )
            state.weights[name] = max(-8.0, min(8.0, updated))
        if target:
            state.positive_support += weight
        else:
            state.negative_support += weight
        state.observation_count = min(1_000_000, state.observation_count + 1)
        state.last_update = at
        return TrainingResult(
            True,
            action,
            "accepted",
            effective_weight=weight,
            feedback_kind=feedback_kind,
            proposal_id=proposal_id,
        )

    def update(
        self,
        context: TemporalContext,
        action: str,
        target: bool,
        *,
        provenance: ActionProvenance | str,
        weight: float = 1.0,
    ) -> TrainingResult:
        """Apply one explicit, provenance-gated weighted observation."""
        action_name = self._action_name(action)
        if not is_training_eligible(provenance):
            reason = (
                "excluded_provenance"
                if _provenance_words(provenance)
                else "invalid_provenance"
            )
            return TrainingResult(False, action_name, reason)
        target_value = self._target(target)
        if not math.isfinite(weight) or weight <= 0.0 or weight > 4.0:
            raise ValueError("weight must be in (0, 4]")
        features = self._feature_map(context)
        return self._apply_update(
            action_name,
            features,
            target_value,
            self.encoder.normalize_timestamp(context.timestamp),
            float(weight),
        )

    def observe(
        self,
        context: TemporalContext,
        action: str,
        target: bool,
        *,
        provenance: ActionProvenance | str,
        weight: float = 1.0,
    ) -> TrainingResult:
        """Alias for ``update`` using event-oriented terminology."""
        return self.update(
            context,
            action,
            target,
            provenance=provenance,
            weight=weight,
        )

    def predict(self, context: TemporalContext, action: str) -> Prediction:
        """Predict an action and explain why it is or is not ready."""
        action_name = self._action_name(action)
        model_time = self.encoder.normalize_timestamp(context.timestamp)
        self._prune_suppressions(model_time)
        return self._predict_with_features(
            context,
            action_name,
            self._feature_map(context),
            model_time,
        )

    def _predict_with_features(
        self,
        context: TemporalContext,
        action_name: str,
        features: Mapping[str, float],
        model_time: datetime,
    ) -> Prediction:
        state = self._actions.get(action_name)
        reasons: list[str] = []
        if state is None:
            probability = 0.5
            effective_support = 0.0
            freshness = 0.0
            age = 0.0
            contributions: tuple[FeatureContribution, ...] = ()
            reasons.append("unknown_action")
        else:
            bias, weights, positive, negative, age = self._decayed_state(
                state,
                model_time,
            )
            score = bias + sum(
                weights.get(name, 0.0) * value for name, value in features.items()
            )
            probability = self._sigmoid(score)
            effective_support = positive + negative
            freshness = TemporalFeatureEncoder.decay(age, self.freshness_half_life)
            contributions = tuple(
                FeatureContribution(
                    name,
                    features[name],
                    weights[name],
                    features[name] * weights[name],
                )
                for name in sorted(
                    (
                        name
                        for name in features
                        if abs(weights.get(name, 0.0) * features[name]) > 1e-9
                    ),
                    key=lambda name: (-abs(weights[name] * features[name]), name),
                )[:8]
            )
            if effective_support < self.min_effective_support:
                reasons.append("insufficient_effective_support")
            if age > self.stale_after:
                reasons.append("stale_model")

        support_factor = effective_support / (effective_support + 2.0)
        probability_margin = abs(probability - 0.5) * 2.0
        freshness_factor = freshness
        suppression = self._active_suppression(context, action_name)
        if suppression is not None:
            probability *= suppression.factor
            reasons.append("recent_manual_correction")
        not_ready = bool(reasons)
        confidence = (
            probability_margin * support_factor * freshness_factor
            if not not_ready
            else 0.0
        )
        return Prediction(
            action=action_name,
            probability=max(0.0, min(1.0, probability)),
            confidence=max(0.0, min(1.0, confidence)),
            effective_support=effective_support,
            freshness=freshness,
            probability_margin=probability_margin,
            support_factor=support_factor,
            freshness_factor=freshness_factor,
            top_contributing_features=contributions,
            not_ready=not_ready,
            not_ready_reasons=tuple(reasons),
            suppressed=suppression is not None,
            suppression_until=suppression.until if suppression is not None else None,
        )

    def predict_action(self, action: str, context: TemporalContext) -> Prediction:
        """Argument-order alias for adapters that start with the action."""
        return self.predict(context, action)

    def register_proposal(self, context: TemporalContext, action: str) -> str:
        """Keep a bounded snapshot so later feedback can train the same context."""
        action_name = self._action_name(action)
        model_time = self.encoder.normalize_timestamp(context.timestamp)
        self._prune_suppressions(model_time)
        self._proposal_counter += 1
        proposal_id = f"proposal-{self._proposal_counter:08d}"
        features = self._feature_map(self.encoder.encode(context))
        prediction = self._predict_with_features(
            context,
            action_name,
            features,
            model_time,
        )
        pending = PendingAction(
            proposal_id=proposal_id,
            action=action_name,
            entity_id=context.entity_id or "__default__",
            created_at=model_time,
            context_signature=self.encoder.context_signature(context),
            features=features,
            predicted_probability=prediction.probability,
        )
        if len(self._pending) >= self.max_pending_actions:
            oldest_id = min(
                self._pending,
                key=lambda item: (
                    _as_utc(self._pending[item].created_at),
                    item,
                ),
            )
            del self._pending[oldest_id]
        self._pending[proposal_id] = pending
        return proposal_id

    def record_proposal(self, action: str, context: TemporalContext) -> str:
        """Action-first alias for ``register_proposal``."""
        return self.register_proposal(context, action)

    @property
    def pending_proposal_count(self) -> int:
        return len(self._pending)

    @property
    def pending_proposal_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def feature_count(self) -> int:
        return len(self.encoder.feature_names)

    def _active_suppression(
        self,
        context: TemporalContext,
        action: str,
    ) -> _Suppression | None:
        key = (
            context.entity_id or "__default__",
            action,
            self.encoder.context_signature(context),
        )
        return self._suppressions.get(key)

    def _prune_suppressions(self, at: datetime) -> None:
        at_utc = _as_utc(self.encoder.normalize_timestamp(at))
        for key, suppression in list(self._suppressions.items()):
            if _as_utc(suppression.until) <= at_utc:
                del self._suppressions[key]

    @staticmethod
    def _normalise_outcome(outcome: ProposalOutcome | str) -> str:
        value = outcome.value if isinstance(outcome, ProposalOutcome) else outcome
        if not isinstance(value, str):
            raise TypeError("outcome must be a string")
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"unchanged", "no_change", "no_correction", "tolerated"}:
            return "unchanged"
        if normalized in {"corrected", "rejected", "counterfactual_negative"}:
            return "corrected"
        if normalized in {"accepted", "same", "same_action"}:
            return "accepted"
        raise ValueError(f"unknown proposal outcome: {outcome}")

    @staticmethod
    def _opposite(action: str, explicit: str | None = None) -> str | None:
        if explicit is not None:
            return OnlineActionModel._action_name(explicit)
        opposite = _OPPOSITE_ACTIONS.get(action)
        if opposite is not None:
            return opposite
        prefix, separator, leaf = action.rpartition(".")
        opposite_leaf = _OPPOSITE_ACTIONS.get(leaf)
        return f"{prefix}{separator}{opposite_leaf}" if opposite_leaf else None

    def record_feedback(
        self,
        proposal_id: str,
        *,
        outcome: ProposalOutcome | str = ProposalOutcome.UNCHANGED,
        observed_action: str | None = None,
        provenance: ActionProvenance | str,
        context: TemporalContext | None = None,
        at: datetime | None = None,
    ) -> TrainingResult:
        """Train from proposal feedback with explicit counterfactual weighting.

        A manual opposite action inside ``correction_window`` is strong
        negative evidence.  An explicitly unchanged outcome is only a weak
        positive observation.  The pending snapshot is consumed after an
        eligible feedback update, and a correction also installs a temporary
        context-scoped suppression.
        """
        if not isinstance(proposal_id, str):
            raise TypeError("proposal_id must be a string")
        pending = self._pending.get(proposal_id)
        if pending is None:
            return TrainingResult(
                False,
                None,
                "unknown_proposal",
                proposal_id=proposal_id,
            )
        if not is_training_eligible(provenance):
            reason = (
                "excluded_provenance"
                if _provenance_words(provenance)
                else "invalid_provenance"
            )
            return TrainingResult(
                False,
                pending.action,
                reason,
                proposal_id=proposal_id,
            )

        outcome_name = self._normalise_outcome(outcome)
        feedback_time = context.timestamp if context is not None else at
        if feedback_time is None:
            feedback_time = pending.created_at
        if not isinstance(feedback_time, datetime):
            raise TypeError("feedback time must be a datetime")
        feedback_time = self.encoder.normalize_timestamp(feedback_time)
        age = max(
            0.0,
            (_as_utc(feedback_time) - _as_utc(pending.created_at)).total_seconds(),
        )
        manual = bool(
            _provenance_words(provenance).intersection(_MANUAL_PROVENANCE_WORDS),
        )
        opposite = observed_action is not None and self._opposite(
            pending.action,
        ) == self._action_name(observed_action)
        counterfactual = opposite and manual and age <= self.correction_window
        if outcome_name == "corrected" and not manual and not counterfactual:
            return TrainingResult(
                False,
                pending.action,
                "corrected_requires_manual_provenance",
                proposal_id=proposal_id,
            )
        negative = counterfactual or outcome_name == "corrected"
        if negative:
            feedback_kind = (
                "strong_negative_correction"
                if counterfactual
                else "negative_correction"
            )
            weight = 2.5 if counterfactual else 1.5
            target = False
        else:
            feedback_kind = (
                "weak_positive_unchanged"
                if outcome_name == "unchanged"
                else "accepted_action"
            )
            weight = 0.25 if outcome_name == "unchanged" else 0.5
            target = True

        state = self._actions.get(pending.action)
        learn_at = feedback_time
        if (
            state is not None
            and state.last_update is not None
            and _as_utc(learn_at) < _as_utc(state.last_update)
        ):
            learn_at = state.last_update
        result = self._apply_update(
            pending.action,
            pending.features,
            target,
            learn_at,
            weight,
            feedback_kind=feedback_kind,
            proposal_id=proposal_id,
        )
        if not result.accepted:
            return result
        del self._pending[proposal_id]
        if counterfactual:
            key = (pending.entity_id, pending.action, pending.context_signature)
            suppression = _Suppression(
                entity_id=pending.entity_id,
                action=pending.action,
                context_signature=pending.context_signature,
                until=feedback_time + timedelta(seconds=self.correction_suppression),
                factor=0.05,
            )
            self._suppressions[key] = suppression
            if len(self._suppressions) > self.max_suppressions:
                oldest_key = min(
                    self._suppressions,
                    key=lambda item: (
                        _as_utc(self._suppressions[item].until),
                        item,
                    ),
                )
                del self._suppressions[oldest_key]
        return result

    def feedback(
        self,
        proposal_id: str,
        *,
        outcome: ProposalOutcome | str = ProposalOutcome.UNCHANGED,
        observed_action: str | None = None,
        provenance: ActionProvenance | str,
        context: TemporalContext | None = None,
        at: datetime | None = None,
    ) -> TrainingResult:
        """Alias for ``record_feedback``."""
        return self.record_feedback(
            proposal_id,
            outcome=outcome,
            observed_action=observed_action,
            provenance=provenance,
            context=context,
            at=at,
        )

    def settle_unchanged(
        self,
        at: datetime,
        *,
        outcome_window: timedelta | float = timedelta(hours=1),
        provenance: ActionProvenance | str = ActionProvenance.AUTONOMOUS_OBSERVATION,
    ) -> tuple[TrainingResult, ...]:
        """Weakly reinforce proposals whose observation window elapsed."""
        window = _duration_seconds(outcome_window, name="outcome_window")
        model_time = self.encoder.normalize_timestamp(at)
        eligible_ids = [
            pending.proposal_id
            for pending in self._pending.values()
            if (_as_utc(model_time) - _as_utc(pending.created_at)).total_seconds()
            >= window
        ]
        return tuple(
            self.record_feedback(
                proposal_id,
                outcome=ProposalOutcome.UNCHANGED,
                provenance=provenance,
                at=model_time,
            )
            for proposal_id in sorted(eligible_ids)
        )

    def storage_stats(self) -> dict[str, int]:
        """Expose bounded storage counters for diagnostics and tests."""
        return {
            "actions": len(self._actions),
            "weights": sum(len(state.weights) for state in self._actions.values()),
            "pending_proposals": len(self._pending),
            "suppressions": len(self._suppressions),
            "max_actions": self.max_actions,
            "max_features": self.max_features,
            "max_pending_actions": self.max_pending_actions,
            "max_suppressions": self.max_suppressions,
        }

    def to_dict(self) -> dict[str, Any]:
        actions = {}
        for action, state in sorted(self._actions.items()):
            actions[action] = {
                "bias": state.bias,
                "weights": dict(sorted(state.weights.items())),
                "positive_support": state.positive_support,
                "negative_support": state.negative_support,
                "observation_count": state.observation_count,
                "last_update": (
                    _serialise_datetime(state.last_update)
                    if state.last_update is not None
                    else None
                ),
            }
        suppressions = [
            suppression.to_dict()
            for suppression in sorted(
                self._suppressions.values(),
                key=lambda item: (item.entity_id, item.action, item.context_signature),
            )
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "encoder": self.encoder.to_dict(),
            "config": {
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "forgetting_half_life": self.forgetting_half_life,
                "freshness_half_life": self.freshness_half_life,
                "stale_after": self.stale_after,
                "min_effective_support": self.min_effective_support,
                "max_actions": self.max_actions,
                "max_features": self.max_features,
                "max_pending_actions": self.max_pending_actions,
                "correction_window": self.correction_window,
                "correction_suppression": self.correction_suppression,
                "max_suppressions": self.max_suppressions,
            },
            "actions": actions,
            "pending": [
                pending.to_dict()
                for pending in sorted(
                    self._pending.values(),
                    key=lambda item: item.proposal_id,
                )
            ],
            "suppressions": suppressions,
            "proposal_counter": self._proposal_counter,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:  # noqa: PLR0912, PLR0915
        raw = _required_dict(data, name="temporal model")
        if type(raw.get("schema_version")) is not int:
            raise UnsupportedSchemaError("unsupported temporal model schema")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise UnsupportedSchemaError("unsupported temporal model schema")
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "encoder",
                "config",
                "actions",
                "pending",
                "suppressions",
                "proposal_counter",
            },
            name="temporal model",
        )
        config = _validated_model_config(raw["config"])
        raw_actions = _required_dict(raw["actions"], name="model.actions")
        raw_pending = _required_list(raw["pending"], name="model.pending")
        raw_suppressions = _required_list(
            raw["suppressions"],
            name="model.suppressions",
        )
        if len(raw_actions) > config["max_actions"]:
            raise ValueError("model.actions exceeds max_actions")
        if len(raw_pending) > config["max_pending_actions"]:
            raise ValueError("model.pending exceeds max_pending_actions")
        if len(raw_suppressions) > config["max_suppressions"]:
            raise ValueError("model.suppressions exceeds max_suppressions")
        encoder = TemporalFeatureEncoder.from_dict(raw["encoder"])
        model = cls(encoder=encoder, **config)

        parsed_actions: dict[str, _ActionState] = {}
        for raw_action, raw_state_value in raw_actions.items():
            action = _required_text(
                raw_action,
                name="model action id",
                pattern=_ACTION_ID_RE,
            )
            if action != action.lower() or action in parsed_actions:
                raise ValueError("model action ids contain a duplicate or invalid case")
            raw_state = _required_dict(
                raw_state_value,
                name=f"model.actions.{action}",
            )
            _require_exact_keys(
                raw_state,
                {
                    "bias",
                    "weights",
                    "positive_support",
                    "negative_support",
                    "observation_count",
                    "last_update",
                },
                name=f"model.actions.{action}",
            )
            raw_weights = _required_dict(
                raw_state["weights"],
                name=f"model.actions.{action}.weights",
            )
            if len(raw_weights) > model.max_features:
                raise ValueError("action weights exceed max_features")
            weights: dict[str, float] = {}
            for feature_name, raw_weight in raw_weights.items():
                if not encoder.is_valid_feature_name(feature_name):
                    raise ValueError("action contains an invalid feature name")
                if feature_name in weights:
                    raise ValueError("action weights contain a duplicate feature")
                weights[feature_name] = _required_number(
                    raw_weight,
                    name=f"weight.{feature_name}",
                    minimum=-8.0,
                    maximum=8.0,
                )
            last_update = _parse_datetime(raw_state["last_update"])
            positive_support = _required_number(
                raw_state["positive_support"],
                name="positive_support",
                minimum=0.0,
                maximum=_MAX_PERSISTED_SUPPORT,
            )
            negative_support = _required_number(
                raw_state["negative_support"],
                name="negative_support",
                minimum=0.0,
                maximum=_MAX_PERSISTED_SUPPORT,
            )
            if positive_support + negative_support <= 0.0:
                raise ValueError("action has no support")
            state = _ActionState(
                bias=_required_number(
                    raw_state["bias"],
                    name="action.bias",
                    minimum=-12.0,
                    maximum=12.0,
                ),
                weights=weights,
                positive_support=positive_support,
                negative_support=negative_support,
                observation_count=_required_int(
                    raw_state["observation_count"],
                    name="observation_count",
                    minimum=1,
                    maximum=1_000_000,
                ),
                last_update=last_update,
            )
            parsed_actions[action] = state

        parsed_pending: dict[str, PendingAction] = {}
        for raw_pending_value in raw_pending:
            raw_item = _required_dict(raw_pending_value, name="model pending item")
            _require_exact_keys(
                raw_item,
                {
                    "proposal_id",
                    "action",
                    "entity_id",
                    "created_at",
                    "context_signature",
                    "features",
                    "predicted_probability",
                },
                name="model pending item",
            )
            proposal_id = _required_text(
                raw_item["proposal_id"],
                name="pending proposal id",
                pattern=_PROPOSAL_ID_RE,
            )
            if proposal_id in parsed_pending:
                raise ValueError("pending proposal ids contain a duplicate")
            action = _required_text(
                raw_item["action"],
                name="pending action id",
                pattern=_ACTION_ID_RE,
            )
            entity_id = _required_text(
                raw_item["entity_id"],
                name="pending entity id",
            )
            context_signature = _required_text(
                raw_item["context_signature"],
                name="pending context signature",
                maximum=20,
                pattern=_SIGNATURE_RE,
            )
            raw_features = _required_dict(
                raw_item["features"],
                name="pending features",
            )
            if len(raw_features) > model.max_features:
                raise ValueError("pending features exceed max_features")
            features: dict[str, float] = {}
            for feature_name, raw_feature_value in raw_features.items():
                if not encoder.is_valid_feature_name(feature_name):
                    raise ValueError("pending item contains an invalid feature name")
                features[feature_name] = _required_number(
                    raw_feature_value,
                    name=f"pending feature.{feature_name}",
                    minimum=-1.0,
                    maximum=1.0,
                )
            parsed_pending[proposal_id] = PendingAction(
                proposal_id=proposal_id,
                action=action,
                entity_id=entity_id,
                created_at=_parse_datetime(raw_item["created_at"]),
                context_signature=context_signature,
                features=features,
                predicted_probability=_required_number(
                    raw_item["predicted_probability"],
                    name="pending predicted_probability",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

        parsed_suppressions: dict[tuple[str, str, str], _Suppression] = {}
        for raw_suppression_value in raw_suppressions:
            raw_item = _required_dict(
                raw_suppression_value,
                name="model suppression item",
            )
            _require_exact_keys(
                raw_item,
                {"entity_id", "action", "context_signature", "until", "factor"},
                name="model suppression item",
            )
            entity_id = _required_text(
                raw_item["entity_id"],
                name="suppression entity id",
            )
            action = _required_text(
                raw_item["action"],
                name="suppression action id",
                pattern=_ACTION_ID_RE,
            )
            context_signature = _required_text(
                raw_item["context_signature"],
                name="suppression context signature",
                maximum=20,
                pattern=_SIGNATURE_RE,
            )
            key = (entity_id, action, context_signature)
            if key in parsed_suppressions:
                raise ValueError("suppressions contain a duplicate key")
            parsed_suppressions[key] = _Suppression(
                entity_id=entity_id,
                action=action,
                context_signature=context_signature,
                until=_parse_datetime(raw_item["until"]),
                factor=_required_number(
                    raw_item["factor"],
                    name="suppression.factor",
                    minimum=math.nextafter(0.0, 1.0),
                    maximum=1.0,
                ),
            )

        proposal_counter = _required_int(
            raw["proposal_counter"],
            name="proposal_counter",
            maximum=_MAX_PERSISTED_COUNTER,
        )
        proposal_numbers = [
            int(proposal_id.removeprefix("proposal-")) for proposal_id in parsed_pending
        ]
        if proposal_numbers and proposal_counter < max(proposal_numbers):
            raise ValueError("proposal_counter is behind a pending proposal")

        # Assign only after every persisted value has passed validation.  A
        # malformed payload therefore cannot leave a partially loaded model.
        model._actions = parsed_actions
        model._pending = parsed_pending
        model._suppressions = parsed_suppressions
        model._proposal_counter = proposal_counter
        return model

    @classmethod
    def from_dict_or_reset(
        cls,
        data: Mapping[str, Any],
        *,
        encoder: TemporalFeatureEncoder | None = None,
        **model_kwargs: Any,
    ) -> tuple[Self, bool]:
        """Load state, or explicitly return a fresh model for unknown schemas.

        The boolean is ``True`` only when an unsupported schema caused the
        reset.  Malformed current-schema data still raises and is never
        silently discarded.
        """
        try:
            return cls.from_dict(data), False
        except UnsupportedSchemaError:
            return cls(encoder=encoder, **model_kwargs), True

    def dumps(self) -> str:
        """Serialize deterministically to JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    to_json = dumps

    @classmethod
    def loads(cls, payload: str) -> Self:
        return cls.from_dict(json.loads(payload))

    from_json = loads


ActionModel = OnlineActionModel
