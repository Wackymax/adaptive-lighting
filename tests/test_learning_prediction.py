"""Tests for the pure shadow-mode learning and prediction primitives."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "custom_components" / "adaptive_lighting"))

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
    learner = PreferenceLearner(learning_rate=0.5, max_offset=20, min_duration_seconds=30)

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
        {"duration_seconds": 29},
        {"safety_context": True},
        {"intent": "good_night"},
        {"intent": "alarm"},
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
    learner = PreferenceLearner(learning_rate=0.5, max_offset=10, min_duration_seconds=0)

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
    assert learner.adjusted_target(
        95,
        "living",
        "task",
        "evening",
        "dark",
    ) == 100


def test_preference_learner_accepts_json_mapping_aliases() -> None:
    learner = PreferenceLearner(min_duration_seconds=10)

    assert learner.record(
        {
            "room": "kitchen",
            "previous_target": 20,
            "selected_target": 35,
            "duration_s": 15,
            "source": "physical",
        },
    ) is True
    assert learner.get_offset("kitchen") == 3.75


def test_preference_learner_export_import_and_reset_are_json_safe() -> None:
    learner = PreferenceLearner(learning_rate=0.5, max_offset=12, min_duration_seconds=0)
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


def test_sequence_predictor_learns_explicit_time_bucket_transition() -> None:
    predictor = SequencePredictor(
        prelight_brightness_cap=8,
        expiry_seconds=45,
        prior_strength=0,
    )
    assert predictor.record_transition("garage", "living", time_bucket="evening")
    assert predictor.record_transition("garage", "living", time_bucket="evening")
    assert predictor.record_transition("garage", "kitchen", time_bucket="morning")

    at = datetime(2026, 7, 12, 18, tzinfo=timezone.utc)
    prediction = predictor.predict(
        "garage",
        at=at,
        requested_brightness=80,
        time_bucket="evening",
    )
    assert prediction is not None
    assert prediction.to_zone == "living"
    assert prediction.confidence == 1
    assert prediction.expires_at == datetime(2026, 7, 12, 18, 0, 45, tzinfo=timezone.utc)
    assert prediction.prelight_brightness == 8


def test_sequence_predictor_uses_global_prior_for_unseen_bucket() -> None:
    predictor = SequencePredictor(prior_strength=1)
    predictor.record_transition("kitchen", "living", time_bucket="morning")
    predictor.record_transition("kitchen", "living", time_bucket="morning")
    predictor.record_transition("kitchen", "office", time_bucket="evening")

    prediction = predictor.predict(
        "kitchen",
        at=datetime(2026, 7, 12, 3, tzinfo=timezone.utc),
    )
    assert prediction is not None
    assert prediction.to_zone == "living"
    assert 0 < prediction.confidence < 1


def test_sequence_predictor_observe_learns_only_reasonable_adjacent_events() -> None:
    predictor = SequencePredictor(max_transition_gap_seconds=60)
    first = datetime(2026, 7, 12, 8, tzinfo=timezone.utc)
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


def test_sequence_predictor_export_import_and_reset() -> None:
    predictor = SequencePredictor(prelight_cap=7, expiry_s=30, bucket_minutes=30)
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
