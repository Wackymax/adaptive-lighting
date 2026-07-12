"""Tests for the pure shadow-mode learning and prediction primitives."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "custom_components"
        / "adaptive_lighting",
    ),
)

from learning import (
    OverrideSample,
    PreferenceLearner,
)
from prediction import SequencePredictor


def sample(**overrides: object) -> OverrideSample:
    values: dict[str, object] = {
        "zone": "living",
        "baseline": 30,
        "selected": 50,
        "duration_seconds": 60,
        "source": "human",
        "intent": "ambient",
        "time_bucket": "evening",
        "daylight_band": "dark",
    }
    values.update(overrides)
    return OverrideSample(**values)


def test_preference_learner_accepts_durable_human_override() -> None:
    learner = PreferenceLearner(
        learning_rate=0.5, max_offset=20, min_duration_seconds=30,
    )

    assert learner.record(sample()) is True
    assert learner.sample_count == 1
    assert learner.get_offset("living", "ambient", "evening", "dark") == 10
    assert learner.adjusted_target(30, "living", "ambient", "evening", "dark") == 40


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "automation"},
        {"source": "adaptive_lighting"},
        {"source": "script"},
        {"source": "manual_automation"},
        {"source": "physical_automation"},
        {"duration_seconds": 29},
        {"safety_context": True},
        {"intent": "alarm"},
        {"intent": "emergency"},
        {"intent": "evacuation"},
        {"intent": "fire"},
        {"intent": "good_night"},
        {"intent": "grid_emergency"},
        {"intent": "night_path"},
        {"intent": "night_safety"},
        {"intent": "safety"},
        {"intent": "security"},
        {"intent": "sleep"},
    ],
)
def test_preference_learner_rejects_non_preference_samples(
    changes: dict[str, object],
) -> None:
    learner = PreferenceLearner()

    assert learner.record(sample(**changes)) is False
    assert learner.sample_count == 0
    assert learner.get_offset("living", "ambient", "evening", "dark") == 0


def test_preference_learner_uses_bounded_exponential_average() -> None:
    learner = PreferenceLearner(
        learning_rate=0.5, max_offset=10, min_duration_seconds=0,
    )

    assert learner.record(sample(selected=100)) is True
    assert learner.get_offset("living", "ambient", "evening", "dark") == 5
    assert learner.record(sample(selected=0)) is True
    assert learner.get_offset("living", "ambient", "evening", "dark") == -2.5
    assert learner.record(sample(selected=100)) is True
    assert learner.get_offset("living", "ambient", "evening", "dark") == 3.75


def test_preference_learner_keeps_contexts_separate_and_clamps_targets() -> None:
    learner = PreferenceLearner(learning_rate=1, max_offset=20, min_duration_seconds=0)
    assert learner.record(sample(intent="task", selected=80)) is True

    assert learner.get_offset("living", "ambient", "evening", "dark") == 0
    assert (
        learner.adjusted_target(
            95,
            "living",
            "task",
            "evening",
            "dark",
        )
        == 100
    )


def test_preference_learner_never_learns_upward_through_hard_cap() -> None:
    learner = PreferenceLearner(learning_rate=1, min_duration_seconds=0)

    assert learner.record(sample(baseline=30, selected=40, hard_cap=40)) is False
    assert learner.record(sample(baseline=30, selected=40, hard_cap=True)) is False
    assert learner.record(sample(baseline=30, selected=20, hard_cap=40)) is True
    assert learner.get_offset("living", "ambient", "evening", "dark") == -10

    upward = PreferenceLearner(learning_rate=1, min_duration_seconds=0)
    assert upward.record(sample(baseline=30, selected=50)) is True
    assert (
        upward.adjusted_target(
            30,
            "living",
            "ambient",
            "evening",
            "dark",
            hard_cap=35,
        )
        == 35
    )


def test_preference_learner_accepts_json_mapping_aliases() -> None:
    learner = PreferenceLearner(min_duration_seconds=10)

    assert (
        learner.record(
            {
                "room": "kitchen",
                "previous_target": 20,
                "selected_target": 35,
                "duration_s": 15,
                "source": "physical",
            },
        )
        is True
    )
    assert learner.get_offset("kitchen") == 3.75


def test_preference_learner_export_import_and_reset_are_json_safe() -> None:
    learner = PreferenceLearner(
        learning_rate=0.5, max_offset=12, min_duration_seconds=0,
    )
    learner.record(sample())
    learner.record(sample(zone="kitchen", selected=10))

    exported = learner.export_state()
    json.dumps(exported)
    restored = PreferenceLearner.from_state(json.loads(learner.to_json()))
    assert restored.export_state() == exported
    assert restored.get_offset("living", "ambient", "evening", "dark") == 6
    imported = PreferenceLearner()
    imported.import_json(learner.to_json())
    assert imported.export_state() == exported
    assert restored.reset("living") == 1
    assert restored.get_offset("living", "ambient", "evening", "dark") == 0
    assert restored.reset() == 1


@pytest.mark.parametrize(
    "entries",
    [
        [
            {
                "zone": "living",
                "intent": "ambient",
                "time_bucket": "evening",
                "daylight_band": "dark",
                "offset": 2,
                "count": 0,
            },
        ],
        [
            {
                "zone": "living",
                "intent": "ambient",
                "time_bucket": "evening",
                "daylight_band": "dark",
                "offset": 2,
                "count": 1.5,
            },
        ],
        [
            {
                "zone": "living",
                "intent": "sleep",
                "time_bucket": "evening",
                "daylight_band": "dark",
                "offset": 2,
                "count": 1,
            },
        ],
        [
            {
                "zone": "living",
                "intent": "ambient",
                "time_bucket": "evening",
                "daylight_band": "dark",
                "offset": 2,
                "count": 1,
            },
            {
                "zone": "living",
                "intent": "ambient",
                "time_bucket": "evening",
                "daylight_band": "dark",
                "offset": 3,
                "count": 2,
            },
        ],
    ],
)
def test_preference_import_is_atomic_and_rejects_invalid_counts_and_duplicates(
    entries: list[dict[str, object]],
) -> None:
    learner = PreferenceLearner(learning_rate=0.5, max_offset=20)
    assert learner.record(sample()) is True
    before = learner.export_state()
    payload = {
        "version": 1,
        "config": {
            "learning_rate": 1,
            "max_offset": 5,
            "min_duration_seconds": 0,
        },
        "entries": entries,
    }

    with pytest.raises((TypeError, ValueError)):
        learner.import_state(payload)

    assert learner.export_state() == before


def test_sequence_predictor_learns_explicit_time_bucket_transition() -> None:
    predictor = SequencePredictor(
        prelight_brightness_cap=8,
        expiry_seconds=45,
        min_observations=1,
        prior_strength=0,
    )
    assert predictor.record_transition("garage", "living", time_bucket="evening")
    assert predictor.record_transition("garage", "living", time_bucket="evening")
    assert predictor.record_transition("garage", "kitchen", time_bucket="morning")

    at = datetime(2026, 7, 12, 18, tzinfo=UTC)
    prediction = predictor.predict(
        "garage",
        at=at,
        requested_brightness=80,
        time_bucket="evening",
    )
    assert prediction is not None
    assert prediction.to_zone == "living"
    assert prediction.confidence == 1
    assert prediction.expires_at == datetime(2026, 7, 12, 18, 0, 45, tzinfo=UTC)
    assert prediction.prelight_brightness == 8


def test_sequence_predictor_uses_global_prior_for_unseen_bucket() -> None:
    predictor = SequencePredictor(min_observations=1, prior_strength=1)
    predictor.record_transition("kitchen", "living", time_bucket="morning")
    predictor.record_transition("kitchen", "living", time_bucket="morning")
    predictor.record_transition("kitchen", "office", time_bucket="evening")

    prediction = predictor.predict(
        "kitchen",
        at=datetime(2026, 7, 12, 3, tzinfo=UTC),
    )
    assert prediction is not None
    assert prediction.to_zone == "living"
    assert 0 < prediction.confidence < 1


def test_sequence_predictor_separates_weekday_and_weekend_buckets() -> None:
    predictor = SequencePredictor(min_observations=1, prior_strength=0)
    monday = datetime(2026, 7, 13, 8, tzinfo=UTC)
    saturday = datetime(2026, 7, 18, 8, tzinfo=UTC)
    predictor.record_transition("kitchen", "office", at=monday)
    predictor.record_transition("kitchen", "garden", at=saturday)

    weekday = predictor.predict("kitchen", at=monday)
    weekend = predictor.predict("kitchen", at=saturday)

    assert weekday is not None
    assert weekday.to_zone == "office"
    assert weekday.time_bucket == "weekday:8"
    assert weekday.day_type == "weekday"
    assert weekday.day_type_behavior == "weekday"
    assert weekend is not None
    assert weekend.to_zone == "garden"
    assert weekend.time_bucket == "weekend:8"
    assert weekend.day_type == "weekend"
    assert weekend.day_type_behavior == "weekend"


def test_monday_public_holiday_uses_weekend_behavior_and_keeps_provenance() -> None:
    predictor = SequencePredictor(min_observations=1, prior_strength=0)
    saturday = datetime(2026, 7, 18, 9, tzinfo=UTC)
    monday_holiday = datetime(2026, 7, 13, 9, tzinfo=UTC)
    predictor.record_transition("bedroom", "kitchen", at=saturday)
    assert predictor.observe("bedroom", monday_holiday, holiday=True) is False
    assert predictor.observe(
        "kitchen",
        monday_holiday.replace(minute=1),
        day_type="public_holiday",
    ) is True

    prediction = predictor.predict(
        "bedroom",
        at=monday_holiday,
        public_holiday=True,
    )

    assert prediction is not None
    assert prediction.to_zone == "kitchen"
    assert prediction.time_bucket == "weekend:9"
    assert prediction.day_type == "public_holiday"
    assert prediction.day_type_behavior == "weekend"
    assert prediction.behavior_day_type == "weekend"
    assert prediction.as_dict()["day_type"] == "public_holiday"
    entry = predictor.export_state()["entries"][0]
    assert entry["day_type_counts"] == {"public_holiday": 1, "weekend": 1}

    restored = SequencePredictor.from_state(
        json.loads(json.dumps(predictor.export_state())),
    )
    assert restored.export_state() == predictor.export_state()


def test_sequence_predictor_imports_legacy_bucket_without_day_type_metadata() -> None:
    payload = {
        "version": 1,
        "config": {"min_observations": 1},
        "entries": [
            {
                "from_zone": "bedroom",
                "to_zone": "kitchen",
                "time_bucket": "0:8",
                "count": 1,
            },
            {
                "from_zone": "bedroom",
                "to_zone": "office",
                "time_bucket": "1:8",
                "count": 1,
            },
        ],
        "last_event": None,
    }

    predictor = SequencePredictor.from_state(payload)
    monday = predictor.predict(
        "bedroom",
        at=datetime(2026, 7, 13, 8, tzinfo=UTC),
    )
    tuesday = predictor.predict(
        "bedroom",
        at=datetime(2026, 7, 14, 8, tzinfo=UTC),
    )

    assert monday is not None
    assert monday.to_zone == "kitchen"
    assert tuesday is not None
    assert tuesday.to_zone == "office"
    assert "day_type_counts" not in predictor.export_state()["entries"][0]


def test_sequence_predictor_observe_learns_only_reasonable_adjacent_events() -> None:
    predictor = SequencePredictor(
        max_transition_gap_seconds=60,
        min_observations=1,
    )
    first = datetime(2026, 7, 12, 8, tzinfo=UTC)
    assert predictor.observe("bedroom", first) is False
    assert predictor.observe("kitchen", first.replace(minute=1)) is True
    assert predictor.observe("kitchen", first.replace(minute=1, second=30)) is False
    assert predictor.observe("office", first.replace(minute=5)) is False

    assert predictor.predict("bedroom", at=first) is not None
    assert predictor.predict("kitchen", at=first) is None


def test_sequence_predictor_rejects_invalid_and_self_transitions() -> None:
    predictor = SequencePredictor()

    assert predictor.record_transition("living", "living") is False
    assert predictor.record_transition("", "kitchen") is False
    assert predictor.predict("living") is None


def test_sequence_predictor_requires_conservative_default_support() -> None:
    predictor = SequencePredictor()
    predictor.record_transition("garage", "living", time_bucket="arrival")
    assert predictor.predict("garage", time_bucket="arrival") is None
    predictor.record_transition("garage", "living", time_bucket="arrival")
    assert predictor.predict("garage", time_bucket="arrival") is None
    predictor.record_transition("garage", "living", time_bucket="arrival")

    prediction = predictor.predict("garage", time_bucket="arrival")
    assert prediction is not None
    assert prediction.observations == 3


def test_selective_predictor_reset_clears_observation_cursor() -> None:
    predictor = SequencePredictor(
        max_transition_gap_seconds=60,
        min_observations=1,
    )
    first = datetime(2026, 7, 12, 8, tzinfo=UTC)
    assert predictor.observe("garage", first) is False
    assert predictor.reset(from_zone="garage") == 0
    assert predictor.observe("living", first.replace(second=30)) is False

    assert predictor.export_state()["entries"] == []


def test_sequence_predictor_export_import_and_reset() -> None:
    predictor = SequencePredictor(
        prelight_cap=7,
        expiry_s=30,
        bucket_minutes=30,
        min_observations=1,
    )
    predictor.record_transition("garage", "living", time_bucket="arrival")
    exported = predictor.export_state()
    json.dumps(exported)

    restored = SequencePredictor.from_state(json.loads(predictor.to_json()))
    assert restored.export_state() == exported
    prediction = restored.predict("garage", time_bucket="arrival")
    assert prediction is not None
    imported = SequencePredictor()
    imported.import_json(predictor.to_json())
    assert imported.export_state() == exported
    assert restored.reset(from_zone="garage") == 1
    assert restored.predict("garage") is None


@pytest.mark.parametrize(
    "entries",
    [
        [
            {
                "from_zone": "garage",
                "to_zone": "living",
                "time_bucket": "arrival",
                "count": 0,
            },
        ],
        [
            {
                "from_zone": "garage",
                "to_zone": "living",
                "time_bucket": "arrival",
                "count": 1.5,
            },
        ],
        [
            {
                "from_zone": "garage",
                "to_zone": "garage",
                "time_bucket": "arrival",
                "count": 1,
            },
        ],
        [
            {
                "from_zone": "garage",
                "to_zone": "living",
                "time_bucket": "arrival",
                "count": 1,
            },
            {
                "from_zone": "garage",
                "to_zone": "living",
                "time_bucket": "arrival",
                "count": 2,
            },
        ],
    ],
)
def test_predictor_import_is_atomic_and_rejects_invalid_transition_state(
    entries: list[dict[str, object]],
) -> None:
    predictor = SequencePredictor(min_observations=1)
    predictor.record_transition("kitchen", "living", time_bucket="morning")
    predictor.observe(
        "office",
        datetime(2026, 7, 12, 8, tzinfo=UTC),
    )
    before = predictor.export_state()
    payload = {
        "version": 1,
        "config": {
            "prelight_brightness_cap": 3,
            "expiry_seconds": 30,
            "bucket_minutes": 30,
            "min_confidence": 0.9,
            "min_observations": 2,
            "prior_strength": 1,
            "max_transition_gap_seconds": 60,
        },
        "entries": entries,
        "last_event": None,
    }

    with pytest.raises((TypeError, ValueError)):
        predictor.import_state(payload)

    assert predictor.export_state() == before
