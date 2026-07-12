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
from datetime import datetime, timedelta, timezone
from typing import Any

_SCHEMA_VERSION = 1


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


def _as_datetime(value: datetime | float | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    timestamp = _number(value)
    if timestamp is None:
        raise ValueError
    return datetime.fromtimestamp(timestamp, timezone.utc)


def _serializable_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


class TransitionPrediction:
    """A bounded prediction returned by :class:`SequencePredictor`."""

    __slots__ = (
        "confidence",
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
    ) -> None:
        """Create an immutable-by-convention prediction result."""
        self.from_zone = from_zone
        self.to_zone = to_zone
        self.confidence = confidence
        self.expires_at = expires_at
        self.prelight_brightness = prelight_brightness
        self.time_bucket = time_bucket
        self.observations = observations

    @property
    def brightness(self) -> float:
        """Alias used by simple actuation adapters."""
        return self.prelight_brightness

    @property
    def is_expired(self) -> bool:
        """Return whether the prediction has expired at the current UTC time."""
        return datetime.now(timezone.utc) >= self.expires_at

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe prediction explanation."""
        return {
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "confidence": self.confidence,
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

        self.prelight_brightness_cap = cap
        self.expiry_seconds = expiry
        self.bucket_minutes = bucket_minutes
        self.min_confidence = confidence
        self.prior_strength = prior
        self.max_transition_gap_seconds = gap
        self._bucket_counts: defaultdict[
            tuple[str, str], defaultdict[str, int],
        ] = defaultdict(lambda: defaultdict(int))
        self._global_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )
        self._last_zone: str | None = None
        self._last_at: datetime | None = None

    def time_bucket(self, at: datetime | float | None = None) -> str:
        """Return a deterministic local-time bucket identifier.

        Weekday is included, so Monday morning and Saturday morning can learn
        different priors.  Callers may also pass an explicit string to
        :meth:`record_transition` for semantic buckets such as ``"evening"``.
        """
        moment = _as_datetime(at)
        slot = (moment.hour * 60 + moment.minute) // self.bucket_minutes
        return f"{moment.weekday()}:{slot}"

    def _record(self, from_zone: str, to_zone: str, time_bucket: str) -> bool:
        if not from_zone or not to_zone or not time_bucket or from_zone == to_zone:
            return False
        self._bucket_counts[(from_zone, time_bucket)][to_zone] += 1
        self._global_counts[from_zone][to_zone] += 1
        return True

    def record_transition(
        self,
        from_zone: str,
        to_zone: str,
        at: datetime | float | None = None,
        *,
        time_bucket: str | None = None,
        timestamp: datetime | float | None = None,
    ) -> bool:
        """Record one observed transition and return whether it was accepted."""
        source = _zone(from_zone)
        target = _zone(to_zone)
        if timestamp is not None:
            at = timestamp
        bucket = _zone(time_bucket) if time_bucket is not None else self.time_bucket(at)
        return self._record(source, target, bucket)

    # ``learn`` reads naturally in a recorder adapter.
    learn = record_transition

    def observe(
        self,
        zone: str,
        at: datetime | float | None = None,
        *,
        timestamp: datetime | float | None = None,
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
                learned = self.record_transition(self._last_zone, current, at=moment)
        self._last_zone = current
        self._last_at = moment
        return learned

    def cap_brightness(self, requested_brightness: float | None) -> float:
        """Clamp a predicted pre-light request to the configured low cap."""
        requested = self.prelight_brightness_cap if requested_brightness is None else _number(
            requested_brightness,
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
        moment = _as_datetime(at)
        bucket = _zone(time_bucket) if time_bucket is not None else self.time_bucket(moment)
        if not bucket:
            return None
        counts = self._bucket_counts.get((source, bucket), {})
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
        if confidence < self.min_confidence:
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
            observations=global_total,
        )

    predict_next = predict

    def export_state(self) -> dict[str, object]:
        """Export deterministic counts and configuration using JSON scalars."""
        entries = []
        for (source, bucket), counts in sorted(self._bucket_counts.items()):
            for target, count in sorted(counts.items()):
                entries.append(
                    {
                        "from_zone": source,
                        "to_zone": target,
                        "time_bucket": bucket,
                        "count": count,
                    },
                )
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
            )
            self.prelight_brightness_cap = configured.prelight_brightness_cap
            self.expiry_seconds = configured.expiry_seconds
            self.bucket_minutes = configured.bucket_minutes
            self.min_confidence = configured.min_confidence
            self.prior_strength = configured.prior_strength
            self.max_transition_gap_seconds = configured.max_transition_gap_seconds
        if replace:
            self._bucket_counts.clear()
            self._global_counts.clear()
            self._last_zone = None
            self._last_at = None

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            source = _zone(entry.get("from_zone"))
            target = _zone(entry.get("to_zone"))
            bucket = _zone(entry.get("time_bucket"))
            count = _number(entry.get("count", 0))
            if not source or not target or not bucket or count is None or count < 1:
                continue
            whole_count = int(count)
            self._bucket_counts[(source, bucket)][target] += whole_count
            self._global_counts[source][target] += whole_count

        last_event = payload.get("last_event")
        if isinstance(last_event, Mapping):
            zone = _zone(last_event.get("zone"))
            at = last_event.get("at")
            if zone and isinstance(at, str):
                try:
                    self._last_zone = zone
                    self._last_at = _as_datetime(datetime.fromisoformat(at))
                except ValueError:
                    self._last_zone = None
                    self._last_at = None

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
        if source_filter is None and target_filter is None:
            removed = sum(
                sum(counts.values()) for counts in self._bucket_counts.values()
            )
            self._bucket_counts.clear()
            self._global_counts.clear()
            self._last_zone = None
            self._last_at = None
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
