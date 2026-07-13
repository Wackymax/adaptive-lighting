"""Deterministic tests for the pure environmental signal tracker."""

from __future__ import annotations

import datetime as dt
import math

from custom_components.adaptive_lighting.environment import (
    EnvironmentalSample,
    EnvironmentalSignalTracker,
    EnvironmentalTrend,
    ShowerState,
)

UTC = dt.UTC
START = dt.datetime(2026, 7, 13, 6, tzinfo=UTC)
HUMIDITY = "sensor.bathroom_humidity"
TEMPERATURE = "sensor.bathroom_temperature"
AREA = "Main Bathroom"
INVENTORY = {HUMIDITY: AREA, TEMPERATURE: AREA}


def _sample(
    entity_id: str,
    kind: str,
    minute: int,
    value: object,
    *,
    area: str = AREA,
) -> EnvironmentalSample:
    return EnvironmentalSample(entity_id, area, kind, value, START + dt.timedelta(minutes=minute))


def _update(
    tracker: EnvironmentalSignalTracker,
    minute: int,
    humidity: object,
    temperature: object,
    *,
    activity: bool | None,
    inventory=INVENTORY,
):
    samples = (
        _sample(HUMIDITY, "humidity", minute, humidity),
        _sample(TEMPERATURE, "temperature", minute, temperature),
    )
    return tracker.update(
        now=START + dt.timedelta(minutes=minute),
        samples=samples,
        current_entity_areas=inventory,
        area_activity={AREA: activity},
        area_opening={AREA: False},
    )[AREA]


def test_true_shower_ramp_uses_activity_and_exposes_extractor_signal() -> None:
    tracker = EnvironmentalSignalTracker()
    for minute, humidity, temperature in (
        (0, 68.0, 14.0),
        (5, 69.0, 14.1),
        (10, 72.0, 14.2),
        (15, 78.0, 14.3),
    ):
        result = _update(tracker, minute, humidity, temperature, activity=True)

    assert result.shower_state is ShowerState.ACTIVE
    assert result.showering is True
    assert result.humidity_rise >= 5.0
    assert result.humidity_rate_per_minute >= 0.25
    assert result.humidity_trend is EnvironmentalTrend.RISING
    assert "humidity_ramp" in result.evidence
    assert "recent_area_activity" in result.evidence


def test_temperature_rise_can_corroborate_when_activity_sensor_misses() -> None:
    tracker = EnvironmentalSignalTracker()
    for minute, humidity, temperature in (
        (0, 66.0, 14.0),
        (5, 67.0, 14.1),
        (10, 71.0, 14.3),
        (15, 77.0, 15.0),
    ):
        result = _update(tracker, minute, humidity, temperature, activity=False)

    assert result.shower_state is ShowerState.ACTIVE
    assert "temperature_rise" in result.evidence


def test_high_but_flat_humidity_is_clear_not_a_shower() -> None:
    tracker = EnvironmentalSignalTracker()
    for minute, humidity in ((0, 82.0), (5, 82.4), (10, 82.1), (15, 82.3)):
        result = _update(tracker, minute, humidity, 16.0, activity=True)

    assert result.humidity_current > 80.0
    assert result.humidity_trend is EnvironmentalTrend.STABLE
    assert result.shower_state is ShowerState.CLEAR
    assert result.showering is False


def test_cold_start_and_uncorroborated_ramp_remain_unknown_or_clear() -> None:
    tracker = EnvironmentalSignalTracker()
    result = _update(tracker, 0, 68.0, 14.0, activity=None)
    assert result.shower_state is ShowerState.UNKNOWN
    assert result.showering is None

    for minute, humidity in ((5, 70.0), (10, 74.0), (15, 79.0)):
        result = _update(tracker, minute, humidity, 14.0, activity=None)
    assert result.shower_state is ShowerState.CLEAR
    assert "missing_corroboration" in result.evidence


def test_hysteresis_keeps_extractor_signal_on_through_recovery() -> None:
    tracker = EnvironmentalSignalTracker(
        active_hold=dt.timedelta(minutes=5),
        maximum_event=dt.timedelta(minutes=40),
    )
    for minute, humidity in ((0, 68.0), (5, 69.0), (10, 73.0), (15, 79.0)):
        result = _update(tracker, minute, humidity, 15.0, activity=True)
    assert result.shower_state is ShowerState.ACTIVE

    result = _update(tracker, 21, 77.0, 15.0, activity=False)
    assert result.shower_state is ShowerState.RECOVERING
    assert result.showering is True

    result = _update(tracker, 30, 69.5, 14.8, activity=False)
    assert result.shower_state is ShowerState.CLEAR
    assert result.showering is False


def test_removed_and_moved_entities_drop_area_history() -> None:
    tracker = EnvironmentalSignalTracker()
    _update(tracker, 0, 68.0, 14.0, activity=False)
    _update(tracker, 5, 69.0, 14.1, activity=False)
    assert tracker.history_lengths == {HUMIDITY: 2, TEMPERATURE: 2}

    guest = "Guest Bathroom"
    moved_inventory = {HUMIDITY: guest, TEMPERATURE: guest}
    moved = tracker.update(
        now=START + dt.timedelta(minutes=10),
        samples=(
            _sample(HUMIDITY, "humidity", 10, 70.0, area=guest),
            _sample(TEMPERATURE, "temperature", 10, 14.2, area=guest),
        ),
        current_entity_areas=moved_inventory,
    )
    assert AREA not in moved
    assert moved[guest].humidity_baseline is None
    assert tracker.history_lengths == {HUMIDITY: 1, TEMPERATURE: 1}

    assert tracker.update(
        now=START + dt.timedelta(minutes=11),
        samples=(),
        current_entity_areas={},
    ) == {}
    assert tracker.history_lengths == {}


def test_bad_duplicate_and_stale_values_do_not_create_evidence() -> None:
    tracker = EnvironmentalSignalTracker(maximum_sample_age=dt.timedelta(minutes=10))
    result = tracker.update(
        now=START,
        samples=(
            _sample(HUMIDITY, "humidity", 0, math.nan),
            _sample(TEMPERATURE, "temperature", 0, math.inf),
        ),
        current_entity_areas=INVENTORY,
    )[AREA]
    assert result.shower_state is ShowerState.UNKNOWN
    assert tracker.history_lengths == {}

    sample = _sample(HUMIDITY, "humidity", 0, 70.0)
    tracker.update(now=START, samples=(sample, sample), current_entity_areas=INVENTORY)
    assert tracker.history_lengths == {HUMIDITY: 1}

    stale = tracker.update(
        now=START + dt.timedelta(minutes=20),
        samples=(),
        current_entity_areas=INVENTORY,
    )[AREA]
    assert stale.shower_state is ShowerState.UNKNOWN


def test_history_is_bounded_per_entity() -> None:
    tracker = EnvironmentalSignalTracker(max_samples_per_entity=3)
    for minute in range(6):
        tracker.update(
            now=START + dt.timedelta(minutes=minute),
            samples=(_sample(HUMIDITY, "humidity", minute, 60.0 + minute),),
            current_entity_areas={HUMIDITY: AREA},
        )
    assert tracker.history_lengths == {HUMIDITY: 3}
