"""Bounded, area-local environmental context for behavior learning.

This module has no Home Assistant dependency and never actuates a device.  It
turns current humidity and temperature readings into coarse, explainable
context.  The public ``showering`` contract intentionally remains true through
the recovery phase so a future extractor fan can finish removing moisture.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class ShowerState(StrEnum):
    """Conservative shower activity state exposed to context consumers."""

    UNKNOWN = "unknown"
    CLEAR = "clear"
    ACTIVE = "active"
    RECOVERING = "recovering"


class EnvironmentalTrend(StrEnum):
    """Direction of one environmental signal over the evidence window."""

    UNKNOWN = "unknown"
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"


@dataclass(frozen=True, slots=True)
class EnvironmentalSample:
    """One available native sensor reading supplied by the HA adapter."""

    entity_id: str
    area: str
    kind: str
    value: object
    observed_at: dt.datetime


@dataclass(frozen=True, slots=True)
class AreaEnvironmentalFeatures:
    """Bounded diagnostic and categorical context for one area."""

    area: str
    observed_at: dt.datetime | None
    humidity_current: float | None
    humidity_baseline: float | None
    humidity_rise: float | None
    humidity_rate_per_minute: float | None
    humidity_trend: EnvironmentalTrend
    temperature_current: float | None
    temperature_baseline: float | None
    temperature_delta: float | None
    temperature_trend: EnvironmentalTrend
    shower_state: ShowerState
    evidence: tuple[str, ...]

    @property
    def showering(self) -> bool | None:
        """Return the stable extractor-facing signal, preserving unknown."""
        if self.shower_state is ShowerState.UNKNOWN:
            return None
        return self.shower_state in {ShowerState.ACTIVE, ShowerState.RECOVERING}

    def as_dict(self) -> dict[str, object]:
        """Return a compact JSON-safe projection for HA state attributes."""
        return {
            "area": self.area,
            "observed_at": (
                self.observed_at.isoformat() if self.observed_at is not None else None
            ),
            "humidity_current": self.humidity_current,
            "humidity_baseline": self.humidity_baseline,
            "humidity_rise": self.humidity_rise,
            "humidity_rate_per_minute": self.humidity_rate_per_minute,
            "humidity_trend": self.humidity_trend.value,
            "temperature_current": self.temperature_current,
            "temperature_baseline": self.temperature_baseline,
            "temperature_delta": self.temperature_delta,
            "temperature_trend": self.temperature_trend.value,
            "shower_state": self.shower_state.value,
            "showering": self.showering,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class _Point:
    at: dt.datetime
    value: float


@dataclass(frozen=True, slots=True)
class _Metrics:
    observed_at: dt.datetime | None = None
    current: float | None = None
    baseline: float | None = None
    delta: float | None = None
    rate: float | None = None
    trend: EnvironmentalTrend = EnvironmentalTrend.UNKNOWN
    samples: int = 0
    span_seconds: float = 0.0


@dataclass(slots=True)
class _ShowerMemory:
    state: ShowerState = ShowerState.UNKNOWN
    started_at: dt.datetime | None = None
    last_evidence_at: dt.datetime | None = None
    baseline: float | None = None


class EnvironmentalSignalTracker:
    """Keep bounded sensor windows and derive conservative shower context.

    Defaults are a locally validated prior, not a universal physiology rule.
    A shower requires a five percentage-point humidity rise within the evidence
    window plus recent room activity or a corroborating temperature rise.  High
    but flat humidity is explicitly not sufficient.
    """

    def __init__(
        self,
        *,
        evidence_window: dt.timedelta = dt.timedelta(minutes=15),
        minimum_span: dt.timedelta = dt.timedelta(minutes=5),
        history_window: dt.timedelta = dt.timedelta(minutes=45),
        maximum_sample_age: dt.timedelta = dt.timedelta(minutes=30),
        active_hold: dt.timedelta = dt.timedelta(minutes=15),
        maximum_event: dt.timedelta = dt.timedelta(minutes=75),
        humidity_rise_threshold: float = 5.0,
        humidity_rate_threshold: float = 0.25,
        temperature_rise_threshold: float = 0.5,
        recovery_margin: float = 2.0,
        max_samples_per_entity: int = 64,
    ) -> None:
        """Initialize bounded windows and locally calibrated safety thresholds."""
        if max_samples_per_entity < 3:
            message = "max_samples_per_entity must be at least three"
            raise ValueError(message)
        for name, value in (
            ("evidence_window", evidence_window),
            ("minimum_span", minimum_span),
            ("history_window", history_window),
            ("maximum_sample_age", maximum_sample_age),
            ("active_hold", active_hold),
            ("maximum_event", maximum_event),
        ):
            if value <= dt.timedelta(0):
                message = f"{name} must be positive"
                raise ValueError(message)
        if minimum_span >= evidence_window or evidence_window > history_window:
            message = "environmental windows are inconsistent"
            raise ValueError(message)
        if active_hold >= maximum_event:
            message = "active_hold must be shorter than maximum_event"
            raise ValueError(message)

        self.evidence_window = evidence_window
        self.minimum_span = minimum_span
        self.history_window = history_window
        self.maximum_sample_age = maximum_sample_age
        self.active_hold = active_hold
        self.maximum_event = maximum_event
        self.humidity_rise_threshold = humidity_rise_threshold
        self.humidity_rate_threshold = humidity_rate_threshold
        self.temperature_rise_threshold = temperature_rise_threshold
        self.recovery_margin = recovery_margin
        self.max_samples_per_entity = max_samples_per_entity
        self._history: dict[str, deque[_Point]] = {}
        self._entity_meta: dict[str, tuple[str, str]] = {}
        self._shower: dict[str, _ShowerMemory] = {}

    @property
    def history_lengths(self) -> dict[str, int]:
        """Expose bounded counts for diagnostics and tests, never raw history."""
        return {entity_id: len(points) for entity_id, points in self._history.items()}

    def update(
        self,
        *,
        now: dt.datetime,
        samples: Sequence[EnvironmentalSample],
        current_entity_areas: Mapping[str, str],
        area_activity: Mapping[str, bool | None] | None = None,
        area_opening: Mapping[str, bool | None] | None = None,
    ) -> dict[str, AreaEnvironmentalFeatures]:
        """Reconcile inventory, ingest current samples, and return area features."""
        now = _utc(now)
        area_activity = area_activity or {}
        area_opening = area_opening or {}
        self._reconcile(current_entity_areas)

        for sample in samples:
            area = current_entity_areas.get(sample.entity_id)
            kind = sample.kind.strip().casefold()
            value = _number(sample.value, humidity=kind == "humidity")
            if area is None or area != sample.area or kind not in {"humidity", "temperature"}:
                continue
            observed_at = _utc(sample.observed_at)
            if value is None or observed_at > now + dt.timedelta(minutes=1):
                continue
            self._ingest(sample.entity_id, area, kind, observed_at, value)

        cutoff = now - self.history_window
        for points in self._history.values():
            while points and points[0].at < cutoff:
                points.popleft()

        areas = sorted(set(current_entity_areas.values()))[:128]
        self._shower = {area: self._shower.get(area, _ShowerMemory()) for area in areas}
        return {
            area: self._area_features(
                area,
                now,
                activity=area_activity.get(area),
                opening=area_opening.get(area),
            )
            for area in areas
        }

    def _reconcile(self, current_entity_areas: Mapping[str, str]) -> None:
        for entity_id in tuple(self._entity_meta):
            current_area = current_entity_areas.get(entity_id)
            previous_area, _ = self._entity_meta[entity_id]
            if current_area is None or current_area != previous_area:
                self._entity_meta.pop(entity_id, None)
                self._history.pop(entity_id, None)

    def _ingest(
        self,
        entity_id: str,
        area: str,
        kind: str,
        observed_at: dt.datetime,
        value: float,
    ) -> None:
        previous = self._entity_meta.get(entity_id)
        if previous is not None and previous != (area, kind):
            self._history.pop(entity_id, None)
        self._entity_meta[entity_id] = (area, kind)
        points = self._history.setdefault(entity_id, deque())
        if points and observed_at < points[-1].at:
            return
        if points and observed_at == points[-1].at:
            if value != points[-1].value:
                points[-1] = _Point(observed_at, value)
            return
        points.append(_Point(observed_at, value))
        while len(points) > self.max_samples_per_entity:
            points.popleft()

    def _metrics(self, area: str, kind: str, now: dt.datetime) -> _Metrics:
        candidates: list[tuple[dt.datetime, int, _Metrics]] = []
        for entity_id, (entity_area, entity_kind) in self._entity_meta.items():
            if (entity_area, entity_kind) != (area, kind):
                continue
            points = self._history.get(entity_id, ())
            if not points:
                continue
            latest = points[-1]
            if now - latest.at > self.maximum_sample_age:
                continue
            window_start = latest.at - self.evidence_window
            evidence = [point for point in points if point.at >= window_start]
            baseline_points = [
                point
                for point in evidence
                if latest.at - point.at >= self.minimum_span
            ]
            if not baseline_points:
                metric = _Metrics(observed_at=latest.at, current=latest.value, samples=len(evidence))
            else:
                baseline = median(point.value for point in baseline_points)
                baseline_at = baseline_points[len(baseline_points) // 2].at
                delta = latest.value - baseline
                minutes = max((latest.at - baseline_at).total_seconds() / 60.0, 1 / 60)
                stable_delta = 1.0 if kind == "humidity" else 0.3
                trend = (
                    EnvironmentalTrend.RISING
                    if delta > stable_delta
                    else EnvironmentalTrend.FALLING
                    if delta < -stable_delta
                    else EnvironmentalTrend.STABLE
                )
                metric = _Metrics(
                    observed_at=latest.at,
                    current=latest.value,
                    baseline=baseline,
                    delta=delta,
                    rate=delta / minutes,
                    trend=trend,
                    samples=len(evidence),
                    span_seconds=(latest.at - evidence[0].at).total_seconds(),
                )
            candidates.append((latest.at, len(evidence), metric))
        return max(candidates, default=(dt.datetime.min.replace(tzinfo=dt.UTC), 0, _Metrics()))[2]

    def _area_features(
        self,
        area: str,
        now: dt.datetime,
        *,
        activity: bool | None,
        opening: bool | None,
    ) -> AreaEnvironmentalFeatures:
        humidity = self._metrics(area, "humidity", now)
        temperature = self._metrics(area, "temperature", now)
        memory = self._shower[area]
        evidence: list[str] = []
        ready = (
            humidity.samples >= 3
            and humidity.span_seconds >= self.minimum_span.total_seconds()
            and humidity.delta is not None
            and humidity.rate is not None
        )
        ramp = bool(
            ready
            and humidity.delta >= self.humidity_rise_threshold
            and humidity.rate >= self.humidity_rate_threshold,
        )
        temperature_rise = bool(
            temperature.delta is not None
            and temperature.delta >= self.temperature_rise_threshold,
        )
        if ramp:
            evidence.append("humidity_ramp")
        if activity is True:
            evidence.append("recent_area_activity")
        if temperature_rise:
            evidence.append("temperature_rise")
        if opening is True:
            evidence.append("opening_active")
        trigger = ramp and (activity is True or temperature_rise)
        state = self._advance_shower(memory, now, humidity, trigger)
        if not ready and state is ShowerState.UNKNOWN:
            evidence.append("insufficient_history")
        elif ramp and not trigger:
            evidence.append("missing_corroboration")
        elif ready and not ramp and state is ShowerState.CLEAR:
            evidence.append("no_humidity_ramp")

        observed_at = max(
            (value for value in (humidity.observed_at, temperature.observed_at) if value),
            default=None,
        )
        return AreaEnvironmentalFeatures(
            area=area,
            observed_at=observed_at,
            humidity_current=humidity.current,
            humidity_baseline=humidity.baseline,
            humidity_rise=humidity.delta,
            humidity_rate_per_minute=humidity.rate,
            humidity_trend=humidity.trend,
            temperature_current=temperature.current,
            temperature_baseline=temperature.baseline,
            temperature_delta=temperature.delta,
            temperature_trend=temperature.trend,
            shower_state=state,
            evidence=tuple(evidence),
        )

    def _advance_shower(
        self,
        memory: _ShowerMemory,
        now: dt.datetime,
        humidity: _Metrics,
        trigger: bool,
    ) -> ShowerState:
        if trigger:
            if memory.started_at is None:
                memory.started_at = now
                memory.baseline = humidity.baseline
            memory.last_evidence_at = now
            memory.state = ShowerState.ACTIVE
            return memory.state

        if memory.started_at is not None:
            elapsed = now - memory.started_at
            if elapsed >= self.maximum_event:
                memory.state = ShowerState.UNKNOWN
                memory.started_at = None
                memory.last_evidence_at = None
                memory.baseline = None
                return memory.state
            if memory.last_evidence_at and now - memory.last_evidence_at <= self.active_hold:
                memory.state = ShowerState.ACTIVE
                return memory.state
            recovered = bool(
                humidity.current is not None
                and memory.baseline is not None
                and humidity.current <= memory.baseline + self.recovery_margin,
            )
            if recovered:
                memory.state = ShowerState.CLEAR
                memory.started_at = None
                memory.last_evidence_at = None
                memory.baseline = None
            else:
                memory.state = ShowerState.RECOVERING
            return memory.state

        memory.state = (
            ShowerState.CLEAR
            if humidity.baseline is not None and humidity.delta is not None
            else ShowerState.UNKNOWN
        )
        return memory.state


def _utc(value: dt.datetime) -> dt.datetime:
    """Normalize a datetime without depending on Home Assistant utilities."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _number(value: object, *, humidity: bool) -> float | None:
    """Return one finite sensor value and reject impossible relative humidity."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (humidity and not 0.0 <= number <= 100.0):
        return None
    return number


__all__ = [
    "AreaEnvironmentalFeatures",
    "EnvironmentalSample",
    "EnvironmentalSignalTracker",
    "EnvironmentalTrend",
    "ShowerState",
]
