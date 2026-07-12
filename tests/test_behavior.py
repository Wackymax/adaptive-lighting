"""Tests for the pure semantic behavior normalizer."""

import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "custom_components" / "adaptive_lighting",
    ),
)

from behavior import (
    BehaviorObservation,
    NormalizationResult,
    normalize_behavior_event,
    normalize_good_night_routine,
)

BASE_TIME = datetime(2026, 7, 13, 18, 0, tzinfo=UTC)


def event(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "entity_id": "light.living_room",
        "domain": "light",
        "action": "on",
        "timestamp": BASE_TIME,
        "area": "Living Room",
        "semantic_routine": "ambient",
        "day_type": "weekday",
        "home_away": "home",
        "arrival": "none",
        "media_type": "none",
        "media_app": "none",
        "weather_band": "clear",
        "daylight_band": "dark",
        "preceding_event": "arrival",
        "source": "physical",
        "provenance": "user",
    }
    result.update(overrides)
    return result


def accepted(**overrides: object) -> BehaviorObservation:
    result = normalize_behavior_event(event(**overrides))
    assert result.accepted
    assert result.observation is not None
    return result.observation


def test_human_dimmable_and_non_dimmable_on_off_are_first_class() -> None:
    dimmable = accepted(
        supports_brightness=True,
        brightness=40,
        action="on",
    )
    non_dimmable = accepted(
        entity_id="light.hallway",
        area="Hallway",
        supports_brightness=False,
        action="off",
    )

    assert dimmable.action == "on"
    assert dimmable.supports_brightness
    assert dimmable.is_dimmable
    assert non_dimmable.action == "off"
    assert not non_dimmable.supports_brightness
    assert not non_dimmable.is_dimmable


def test_switch_requires_explicit_light_like_flag() -> None:
    rejected = normalize_behavior_event(
        event(entity_id="switch.lamp", domain="switch"),
    )
    accepted_switch = normalize_behavior_event(
        event(
            entity_id="switch.lamp",
            domain="switch",
            explicit_light_switch=True,
        ),
    )

    assert not rejected.accepted
    assert rejected.reason == "switch_requires_explicit_light_flag"
    assert accepted_switch.accepted
    assert accepted_switch.observation is not None
    assert accepted_switch.observation.explicit_light_switch


def test_good_night_scene_and_automation_are_explicitly_learnable() -> None:
    scene = normalize_behavior_event(
        event(
            action="off",
            source="scene.good_night",
            semantic_routine="good_night",
        ),
    )
    automation = normalize_behavior_event(
        event(
            action="off",
            source="automation.good_night",
            routine="good_night",
        ),
    )

    assert scene.accepted
    assert scene.reason == "accepted_good_night_routine"
    assert automation.accepted
    assert scene.observation is not None
    assert scene.observation.semantic_routine == "good_night"


def test_good_night_routine_normalizes_multiple_targets_and_defaults_off() -> None:
    observations = normalize_good_night_routine(
        [
            "light.living_room",
            {"entity_id": "light.kitchen", "action": "off"},
            {
                "entity_id": "switch.office_lamp",
                "domain": "switch",
                "explicit_light_switch": True,
            },
        ],
        {
            "timestamp": BASE_TIME,
            "area": "downstairs",
            "source": "scene.good_night",
            "day_type": "weekday",
        },
    )

    assert len(observations) == 3
    assert all(item.action == "off" for item in observations)
    assert all(item.semantic_routine == "good_night" for item in observations)
    assert observations[-1].explicit_light_switch


@pytest.mark.parametrize(
    "label",
    [
        "fire_alarm",
        "alarm.armed",
        "security_event",
        "emergency_shutdown",
        "night_safety_override",
    ],
)
def test_compound_alarm_safety_and_emergency_labels_always_reject(label: str) -> None:
    result = normalize_behavior_event(
        event(source=label, semantic_routine="good_night", action="off"),
    )
    assert not result.accepted
    assert result.reason == "safety_override"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"domain": "cover", "entity_id": "cover.blinds"}, "blocked_actuator_domain"),
        ({"domain": "door", "entity_id": "door.front"}, "blocked_actuator_domain"),
        ({"domain": "garage", "entity_id": "garage.door"}, "blocked_actuator_domain"),
        ({"domain": "lock", "entity_id": "lock.front"}, "blocked_actuator_domain"),
        ({"domain": "valve", "entity_id": "valve.water"}, "blocked_actuator_domain"),
        ({"domain": "window", "entity_id": "window.office"}, "blocked_actuator_domain"),
        ({"domain": "light", "entity_id": "cover.spoofed"}, "entity_domain_mismatch"),
        ({"available": False}, "unavailable_or_unknown_entity"),
        ({"entity_state": "unknown"}, "unavailable_or_unknown_entity"),
        ({"source": "adaptive_lighting"}, "non_human_or_automation_source"),
        ({"source": "automation"}, "non_human_or_automation_source"),
        ({"source": "script"}, "non_human_or_automation_source"),
    ],
)
def test_non_light_or_non_attributable_events_reject(
    changes: dict[str, object],
    reason: str,
) -> None:
    result = normalize_behavior_event(event(**changes))
    assert not result.accepted
    assert result.reason == reason


@pytest.mark.parametrize(
    "changes",
    [
        {"action": "toggle"},
        {"action": "on", "timestamp": "2026-07-13T18:00:00"},
        {"action": "off", "timestamp": "not-a-time"},
        {"action": "on", "timestamp": float("nan")},
        {"action": "off", "timestamp": float("inf")},
    ],
)
def test_invalid_actions_and_timestamps_reject(changes: dict[str, object]) -> None:
    result = normalize_behavior_event(event(**changes))
    assert not result.accepted
    assert result.reason in {"invalid_action", "invalid_timestamp"}


def test_public_holiday_keeps_provenance_and_uses_weekend_behavior() -> None:
    observation = accepted(
        day_type="public_holiday",
        holiday_name="Freedom Day",
    )

    assert observation.day_type == "public_holiday"
    assert observation.behavior_day_type == "weekend"
    assert observation.holiday_provenance == "freedom_day"


def test_all_semantic_fields_are_normalized_and_bounded() -> None:
    observation = accepted(
        area="Open Plan / Kitchen",
        semantic_routine="Movie Night",
        home_away="away",
        arrival=True,
        media_type="TV / Movie",
        media_app="My Favorite App!!!" + "x" * 100,
        weather_band="Heavy Rain",
        daylight_band="Very Dark",
        preceding_event="Front Door Open",
        capabilities=["brightness", "rgb", "custom capability"],
    )

    assert observation.area == "open_plan_kitchen"
    assert observation.zone == observation.area
    assert observation.semantic_routine == "movie_night"
    assert observation.home_away == "away"
    assert observation.arrival == "recent_arrival"
    assert observation.media_type == "tv_movie"
    assert len(observation.media_app) <= 48
    assert observation.weather_band == "heavy_rain"
    assert observation.daylight_band == "very_dark"
    assert observation.preceding_event == "front_door_open"
    assert observation.capabilities == ("brightness", "rgb", "custom_capability")


def test_observation_and_result_are_immutable_and_normalizer_is_stateless() -> None:
    first = normalize_behavior_event(event())
    second = normalize_behavior_event(event())

    assert isinstance(first, NormalizationResult)
    assert isinstance(first.observation, BehaviorObservation)
    assert dataclasses.is_dataclass(first.observation)
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.observation.action = "off"  # type: ignore[misc]
    assert not hasattr(normalize_behavior_event, "learner")
    assert not hasattr(first.observation, "predict")


def test_routine_omits_invalid_targets_without_mutating_context() -> None:
    context = {
        "timestamp": BASE_TIME,
        "source": "automation.good_night",
        "area": "home",
    }
    observations = normalize_good_night_routine(
        [
            {"entity_id": "light.good", "action": "off"},
            {"entity_id": "cover.bad", "domain": "cover", "action": "off"},
            {"entity_id": "switch.ambiguous", "domain": "switch", "action": "off"},
        ],
        context,
    )

    assert [item.entity_id for item in observations] == ["light.good"]
    assert context == {
        "timestamp": BASE_TIME,
        "source": "automation.good_night",
        "area": "home",
    }
