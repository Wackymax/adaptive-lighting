"""Small, local zone-transition prediction primitives.

The predictor is intentionally a transparent count table with time buckets,
not a reinforcement-learning policy.  In shadow mode we want every prediction
to be explainable, expiry-bounded, and easy to reset when the house changes;
there is no need for an opaque model to choose a light level.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

_SCHEMA_VERSION = 1
_DAY_TYPES = frozenset({"public_holiday", "weekday", "weekend"})


def _zone(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    """Return a positive integer without coercing floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _as_datetime(value: datetime | float | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    timestamp = _number(value)
    if timestamp is None:
        raise ValueError
    return datetime.fromtimestamp(timestamp, UTC)


def _serializable_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class TransitionPrediction:
    """A bounded prediction returned by :class:`SequencePredictor`."""

    __slots__ = (
        "confidence",
        "day_type",
        "day_type_behavior",
        "expires_at",
        "from_zone",
        "observations",
        "prelight_brightness",
        "time_bucket",
        "to_zone",
    )

    def __init__(
        self,
        from_zone: str,
        to_zone: str,
        confidence: float,
        expires_at: datetime,
        prelight_brightness: float,
        time_bucket: str,
        observations: int,
        day_type: str = "weekday",
        day_type_behavior: str | None = None,
    ) -> None:
        """Create an immutable-by-convention prediction result."""
        self.from_zone = from_zone
        self.to_zone = to_zone
        self.confidence = confidence
        self.expires_at = expires_at
        self.prelight_brightness = prelight_brightness
        self.time_bucket = time_bucket
        self.observations = observations
        self.day_type = day_type
        self.day_type_behavior = day_type_behavior or day_type

    @property
    def brightness(self) -> float:
        """Alias used by simple actuation adapters."""
        return self.prelight_brightness

    @property
    def behavior_day_type(self) -> str:
        """Return the day type whose behavior table supplied the prediction."""
        return self.day_type_behavior

    @property
    def is_expired(self) -> bool:
        """Return whether the prediction has expired at the current UTC time."""
        return datetime.now(UTC) >= self.expires_at

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe prediction explanation."""
        return {
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "confidence": self.confidence,
            "day_type": self.day_type,
            "day_type_behavior": self.day_type_behavior,
            "behavior_day_type": self.behavior_day_type,
            "expires_at": _serializable_time(self.expires_at),
            "prelight_brightness": self.prelight_brightness,
            "time_bucket": self.time_bucket,
            "observations": self.observations,
        }


SequencePrediction = TransitionPrediction
Prediction = TransitionPrediction


class SequencePredictor:
    """Learn first-order zone transitions with transparent time-bucket priors."""

    def __init__(
        self,
        prelight_brightness_cap: float = 10.0,
        expiry_seconds: float = 60.0,
        bucket_minutes: int = 60,
        min_confidence: float = 0.0,
        prior_strength: float = 1.0,
        max_transition_gap_seconds: float = 900.0,
        min_observations: int = 3,
        *,
        prelight_cap: float | None = None,
        expiry_s: float | None = None,
    ) -> None:
        """Configure the cap, expiry, priors, and event adjacency window."""
        if prelight_cap is not None:
            prelight_brightness_cap = prelight_cap
        if expiry_s is not None:
            expiry_seconds = expiry_s
        cap = _number(prelight_brightness_cap)
        expiry = _number(expiry_seconds)
        prior = _number(prior_strength)
        gap = _number(max_transition_gap_seconds)
        confidence = _number(min_confidence)
        if cap is None or not 0 <= cap <= 100:
            raise ValueError
        if expiry is None or expiry <= 0:
            raise ValueError
        if not isinstance(bucket_minutes, int) or not 1 <= bucket_minutes <= 1440:
            raise ValueError
        if confidence is None or not 0 <= confidence <= 1:
            raise ValueError
        if prior is None or prior < 0:
            raise ValueError
        if gap is None or gap <= 0:
            raise ValueError
        if _positive_int(min_observations) is None:
            raise ValueError

        self.prelight_brightness_cap = cap
        self.expiry_seconds = expiry
        self.bucket_minutes = bucket_minutes
        self.min_confidence = confidence
        self.prior_strength = prior
        self.max_transition_gap_seconds = gap
        self.min_observations = min_observations
        self._bucket_counts: defaultdict[
            tuple[str, str],
            defaultdict[str, int],
        ] = defaultdict(lambda: defaultdict(int))
        self._global_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )
        self._day_type_counts: defaultdict[
            tuple[str, str, str],
            defaultdict[str, int],
        ] = defaultdict(lambda: defaultdict(int))
        self._last_zone: str | None = None
        self._last_at: datetime | None = None

    @staticmethod
    def _resolve_day_type(
        moment: datetime,
        day_type: str | None,
        public_holiday: bool,
        holiday: bool | None,
    ) -> tuple[str, str]:
        if holiday is not None:
            if not isinstance(holiday, bool):
                raise TypeError
            public_holiday = holiday
        if not isinstance(public_holiday, bool):
            raise TypeError
        normalized = _zone(day_type) if day_type is not None else ""
        if normalized and normalized not in _DAY_TYPES:
            raise ValueError
        if public_holiday:
            if normalized and normalized != "public_holiday":
                raise ValueError
            normalized = "public_holiday"
        if not normalized:
            normalized = "weekend" if moment.weekday() >= 5 else "weekday"
        behavior = "weekend" if normalized == "public_holiday" else normalized
        return normalized, behavior

    def _bucket_context(
        self,
        at: datetime | float | None,
        time_bucket: str | None,
        day_type: str | None,
        public_holiday: bool,
        holiday: bool | None,
    ) -> tuple[datetime, str, str, str, str | None]:
        moment = _as_datetime(at)
        label, behavior = self._resolve_day_type(
            moment,
            day_type,
            public_holiday,
            holiday,
        )
        explicit_day_context = (
            day_type is not None or public_holiday or holiday is not None
        )
        if time_bucket is None:
            slot = (moment.hour * 60 + moment.minute) // self.bucket_minutes
            return moment, f"{behavior}:{slot}", label, behavior, label

        bucket = _zone(time_bucket)
        if not bucket:
            return moment, "", label, behavior, None
        if bucket.startswith(("weekday:", "weekend:")):
            provenance = label if explicit_day_context else None
            return moment, bucket, label, behavior, provenance
        if explicit_day_context:
            return moment, f"{behavior}:{bucket}", label, behavior, label
        # Semantic buckets predate day-type support. Preserve their exact key
        # unless the caller explicitly supplies a day type or holiday flag.
        return moment, bucket, label, behavior, None

    def time_bucket(
        self,
        at: datetime | float | None = None,
        *,
        day_type: str | None = None,
        public_holiday: bool = False,
        holiday: bool | None = None,
    ) -> str:
        """Return a deterministic day-type and local-time bucket identifier.

        Public holidays retain their label but intentionally use weekend
        behavior. The caller supplies holiday knowledge; no calendar lookup is
        performed by this pure module.
        """
        return self._bucket_context(
            at,
            None,
            day_type,
            public_holiday,
            holiday,
        )[1]

    def _record(
        self,
        from_zone: str,
        to_zone: str,
        time_bucket: str,
        day_type: str | None = None,
    ) -> bool:
        if not from_zone or not to_zone or not time_bucket or from_zone == to_zone:
            return False
        self._bucket_counts[(from_zone, time_bucket)][to_zone] += 1
        self._global_counts[from_zone][to_zone] += 1
        if day_type is not None:
            self._day_type_counts[(from_zone, time_bucket, to_zone)][day_type] += 1
        return True

    def record_transition(
        self,
        from_zone: str,
        to_zone: str,
        at: datetime | float | None = None,
        *,
        time_bucket: str | None = None,
        timestamp: datetime | float | None = None,
        day_type: str | None = None,
        public_holiday: bool = False,
        holiday: bool | None = None,
    ) -> bool:
        """Record one observed transition and return whether it was accepted."""
        source = _zone(from_zone)
        target = _zone(to_zone)
        if timestamp is not None:
            at = timestamp
        _moment, bucket, _label, _behavior, provenance = self._bucket_context(
            at,
            time_bucket,
            day_type,
            public_holiday,
            holiday,
        )
        return self._record(source, target, bucket, provenance)

    # ``learn`` reads naturally in a recorder adapter.
    learn = record_transition

    def observe(
        self,
        zone: str,
        at: datetime | float | None = None,
        *,
        timestamp: datetime | float | None = None,
        day_type: str | None = None,
        public_holiday: bool = False,
        holiday: bool | None = None,
    ) -> bool:
        """Observe a zone event and learn a transition from the prior event."""
        current = _zone(zone)
        if not current:
            return False
        if timestamp is not None:
            at = timestamp
        moment = _as_datetime(at)
        learned = False
        if self._last_zone is not None and self._last_at is not None:
            elapsed = (moment - self._last_at).total_seconds()
            if 0 <= elapsed <= self.max_transition_gap_seconds:
                learned = self.record_transition(
                    self._last_zone,
                    current,
                    at=moment,
                    day_type=day_type,
                    public_holiday=public_holiday,
                    holiday=holiday,
                )
        self._last_zone = current
        self._last_at = moment
        return learned

    def cap_brightness(self, requested_brightness: float | None) -> float:
        """Clamp a predicted pre-light request to the configured low cap."""
        requested = (
            self.prelight_brightness_cap
            if requested_brightness is None
            else _number(
                requested_brightness,
            )
        )
        if requested is None:
            raise ValueError
        return max(0.0, min(self.prelight_brightness_cap, requested))

    def predict(
        self,
        from_zone: str,
        at: datetime | float | None = None,
        requested_brightness: float | None = None,
        *,
        brightness: float | None = None,
        time_bucket: str | None = None,
        timestamp: datetime | float | None = None,
        day_type: str | None = None,
        public_holiday: bool = False,
        holiday: bool | None = None,
    ) -> TransitionPrediction | None:
        """Predict the next zone, with confidence, expiry, and a low cap.

        A bucket count is smoothed toward the all-time source-zone prior.  The
        result is a prediction of *where* to pre-light, never a recommendation
        to use the normal task or alarm brightness.
        """
        source = _zone(from_zone)
        if not source:
            return None
        if timestamp is not None:
            at = timestamp
        moment, bucket, day_label, day_behavior, _provenance = self._bucket_context(
            at,
            time_bucket,
            day_type,
            public_holiday,
            holiday,
        )
        if not bucket:
            return None
        counts = self._bucket_counts.get((source, bucket), {})
        if (
            not counts
            and time_bucket is None
            and day_type is None
            and not public_holiday
            and holiday is None
        ):
            slot = (moment.hour * 60 + moment.minute) // self.bucket_minutes
            legacy_bucket = f"{moment.weekday()}:{slot}"
            counts = self._bucket_counts.get((source, legacy_bucket), {})
        global_counts = self._global_counts.get(source, {})
        if not global_counts:
            return None

        global_total = sum(global_counts.values())
        bucket_total = sum(counts.values())
        candidates = set(global_counts) | set(counts)
        scored: list[tuple[float, str]] = []
        for target in candidates:
            global_probability = global_counts.get(target, 0) / global_total
            if bucket_total == 0:
                score = global_probability
            else:
                score = (
                    counts.get(target, 0) + self.prior_strength * global_probability
                ) / (bucket_total + self.prior_strength)
            scored.append((score, target))
        confidence, target = max(scored, key=lambda item: (item[0], item[1]))
        target_observations = global_counts[target]
        if (
            confidence < self.min_confidence
            or target_observations < self.min_observations
        ):
            return None

        return TransitionPrediction(
            from_zone=source,
            to_zone=target,
            confidence=max(0.0, min(1.0, confidence)),
            expires_at=moment + timedelta(seconds=self.expiry_seconds),
            prelight_brightness=self.cap_brightness(
                brightness if brightness is not None else requested_brightness,
            ),
            time_bucket=bucket,
            observations=target_observations,
            day_type=day_label,
            day_type_behavior=day_behavior,
        )

    predict_next = predict

    def export_state(self) -> dict[str, object]:
        """Export deterministic counts and configuration using JSON scalars."""
        entries: list[dict[str, object]] = []
        for (source, bucket), counts in sorted(self._bucket_counts.items()):
            for target, count in sorted(counts.items()):
                entry: dict[str, object] = {
                    "from_zone": source,
                    "to_zone": target,
                    "time_bucket": bucket,
                    "count": count,
                }
                provenance = self._day_type_counts.get(
                    (source, bucket, target),
                )
                if provenance:
                    entry["day_type_counts"] = dict(sorted(provenance.items()))
                entries.append(entry)
        last_event: dict[str, object] | None = None
        if self._last_zone is not None and self._last_at is not None:
            last_event = {
                "zone": self._last_zone,
                "at": _serializable_time(self._last_at),
            }
        return {
            "version": _SCHEMA_VERSION,
            "config": {
                "prelight_brightness_cap": self.prelight_brightness_cap,
                "expiry_seconds": self.expiry_seconds,
                "bucket_minutes": self.bucket_minutes,
                "min_confidence": self.min_confidence,
                "min_observations": self.min_observations,
                "prior_strength": self.prior_strength,
                "max_transition_gap_seconds": self.max_transition_gap_seconds,
            },
            "entries": entries,
            "last_event": last_event,
        }

    to_dict = export_state
    export = export_state

    def to_json(self) -> str:
        """Export state as stable JSON."""
        return json.dumps(self.export_state(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> SequencePredictor:
        """Create a predictor from a JSON string."""
        return cls.from_state(json.loads(payload))

    def import_json(self, payload: str, *, replace: bool = True) -> None:
        """Import a predictor from a JSON string."""
        self.import_state(json.loads(payload), replace=replace)

    def _import_configuration(self, config: Any) -> SequencePredictor:
        if not isinstance(config, Mapping):
            raise TypeError
        return type(self)(
            prelight_brightness_cap=config.get(
                "prelight_brightness_cap",
                self.prelight_brightness_cap,
            ),
            expiry_seconds=config.get("expiry_seconds", self.expiry_seconds),
            bucket_minutes=config.get("bucket_minutes", self.bucket_minutes),
            min_confidence=config.get("min_confidence", self.min_confidence),
            prior_strength=config.get("prior_strength", self.prior_strength),
            max_transition_gap_seconds=config.get(
                "max_transition_gap_seconds",
                self.max_transition_gap_seconds,
            ),
            min_observations=config.get(
                "min_observations",
                self.min_observations,
            ),
        )

    @staticmethod
    def _import_day_type_counts(
        entry: Mapping[str, Any],
        count: int,
        bucket: str,
    ) -> dict[str, int]:
        raw_counts = entry.get("day_type_counts")
        if raw_counts is None:
            return {}
        if not isinstance(raw_counts, Mapping):
            raise TypeError
        imported: dict[str, int] = {}
        for raw_day_type, raw_count in raw_counts.items():
            day_type = _zone(raw_day_type)
            day_count = _positive_int(raw_count)
            behavior = "weekend" if day_type == "public_holiday" else day_type
            if (
                day_type not in _DAY_TYPES
                or day_count is None
                or day_type in imported
                or not bucket.startswith(f"{behavior}:")
            ):
                raise ValueError
            imported[day_type] = day_count
        if sum(imported.values()) != count:
            raise ValueError
        return imported

    def _import_counts(
        self,
        entries: list[Any],
        replace: bool,
    ) -> tuple[
        dict[tuple[str, str], dict[str, int]],
        dict[tuple[str, str, str], dict[str, int]],
    ]:
        imported = (
            {}
            if replace
            else {key: dict(counts) for key, counts in self._bucket_counts.items()}
        )
        day_type_counts = (
            {}
            if replace
            else {key: dict(counts) for key, counts in self._day_type_counts.items()}
        )
        seen = {
            (source, bucket, target)
            for (source, bucket), counts in imported.items()
            for target in counts
        }
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError
            source = _zone(entry.get("from_zone"))
            target = _zone(entry.get("to_zone"))
            bucket = _zone(entry.get("time_bucket"))
            count = _positive_int(entry.get("count"))
            transition = (source, bucket, target)
            if (
                not source
                or not target
                or not bucket
                or source == target
                or count is None
                or transition in seen
            ):
                raise ValueError
            imported.setdefault((source, bucket), {})[target] = count
            provenance = self._import_day_type_counts(entry, count, bucket)
            if provenance:
                day_type_counts[transition] = provenance
            seen.add(transition)
        return imported, day_type_counts

    def _import_cursor(
        self,
        payload: Mapping[str, Any],
        replace: bool,
    ) -> tuple[str | None, datetime | None]:
        last_zone = None if replace else self._last_zone
        last_at = None if replace else self._last_at
        if "last_event" not in payload:
            return last_zone, last_at
        last_event = payload["last_event"]
        if last_event is None:
            return None, None
        if not isinstance(last_event, Mapping):
            raise TypeError
        zone = _zone(last_event.get("zone"))
        at = last_event.get("at")
        if not zone or not isinstance(at, str):
            raise ValueError
        try:
            parsed_at = _as_datetime(datetime.fromisoformat(at))
        except ValueError as err:
            raise ValueError from err
        return zone, parsed_at

    @staticmethod
    def _global_counts_from_import(
        imported: Mapping[tuple[str, str], Mapping[str, int]],
    ) -> dict[str, dict[str, int]]:
        global_counts: dict[str, dict[str, int]] = {}
        for (source, _bucket), counts in imported.items():
            source_counts = global_counts.setdefault(source, {})
            for target, count in counts.items():
                source_counts[target] = source_counts.get(target, 0) + count
        return global_counts

    def import_state(self, payload: Mapping[str, Any], *, replace: bool = True) -> None:
        """Atomically import state previously returned by :meth:`export_state`."""
        if not isinstance(payload, Mapping):
            raise TypeError
        version = payload.get("version", _SCHEMA_VERSION)
        if version != _SCHEMA_VERSION:
            raise ValueError
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise TypeError
        configured = self._import_configuration(payload.get("config", {}))
        imported, day_type_counts = self._import_counts(entries, replace)
        last_zone, last_at = self._import_cursor(payload, replace)
        global_counts = self._global_counts_from_import(imported)

        self.prelight_brightness_cap = configured.prelight_brightness_cap
        self.expiry_seconds = configured.expiry_seconds
        self.bucket_minutes = configured.bucket_minutes
        self.min_confidence = configured.min_confidence
        self.prior_strength = configured.prior_strength
        self.max_transition_gap_seconds = configured.max_transition_gap_seconds
        self.min_observations = configured.min_observations
        self._bucket_counts = defaultdict(lambda: defaultdict(int))
        for key, counts in imported.items():
            self._bucket_counts[key].update(counts)
        self._global_counts = defaultdict(lambda: defaultdict(int))
        for source, counts in global_counts.items():
            self._global_counts[source].update(counts)
        self._day_type_counts = defaultdict(lambda: defaultdict(int))
        for transition, counts in day_type_counts.items():
            self._day_type_counts[transition].update(counts)
        self._last_zone = last_zone
        self._last_at = last_at

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> SequencePredictor:
        """Create a predictor from JSON-safe exported state."""
        config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(config, Mapping):
            config = {}
        predictor = cls(
            prelight_brightness_cap=config.get("prelight_brightness_cap", 10.0),
            expiry_seconds=config.get("expiry_seconds", 60.0),
            bucket_minutes=config.get("bucket_minutes", 60),
            min_confidence=config.get("min_confidence", 0.0),
            prior_strength=config.get("prior_strength", 1.0),
            min_observations=config.get("min_observations", 3),
            max_transition_gap_seconds=config.get(
                "max_transition_gap_seconds",
                900.0,
            ),
        )
        predictor.import_state(payload)
        return predictor

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SequencePredictor:
        """Create a predictor from a JSON-compatible mapping."""
        return cls.from_state(payload)

    def reset(self, from_zone: str | None = None, to_zone: str | None = None) -> int:
        """Reset all transitions or transitions touching the selected zones."""
        source_filter = _zone(from_zone) if from_zone is not None else None
        target_filter = _zone(to_zone) if to_zone is not None else None
        # Any reset starts a new observation sequence. Keeping this cursor could
        # immediately recreate a transition that the caller just removed.
        self._last_zone = None
        self._last_at = None
        if source_filter is None and target_filter is None:
            removed = sum(
                sum(counts.values()) for counts in self._bucket_counts.values()
            )
            self._bucket_counts.clear()
            self._global_counts.clear()
            self._day_type_counts.clear()
            return removed

        removed = 0
        for key in list(self._bucket_counts):
            source, _bucket = key
            if source_filter is not None and source != source_filter:
                continue
            counts = self._bucket_counts[key]
            for target in list(counts):
                if target_filter is not None and target != target_filter:
                    continue
                removed_count = counts.pop(target)
                removed += removed_count
                self._day_type_counts.pop((source, _bucket, target), None)
                self._global_counts[source][target] -= removed_count
                if self._global_counts[source][target] <= 0:
                    del self._global_counts[source][target]
            if not counts:
                del self._bucket_counts[key]
        for source in list(self._global_counts):
            if not self._global_counts[source]:
                del self._global_counts[source]
        return removed

    clear = reset
