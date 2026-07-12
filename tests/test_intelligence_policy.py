"""Tests for the Home-Assistant-independent adaptive-lighting policy core."""

import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

# The integration package's existing __init__.py is the Home Assistant adapter.
# Install a tiny namespace package for this pure-core test so collection does
# not need Home Assistant at all; the modules under test still use their normal
# package-relative imports.
# ruff: noqa: E402
_PACKAGE = "custom_components.adaptive_lighting"
_PACKAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "adaptive_lighting"
)
if (package := sys.modules.get(_PACKAGE)) is None:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_PACKAGE_PATH)]
    sys.modules[_PACKAGE] = package
elif str(_PACKAGE_PATH) not in package.__path__:
    # The full HA suite may have created this namespace from its symlinked core
    # tree already. Keep the fork's pure modules available in both layouts.
    package.__path__.append(str(_PACKAGE_PATH))

from custom_components.adaptive_lighting.context import (
    ContextSignal,
    ContextSnapshot,
    signal,
    unavailable,
)
from custom_components.adaptive_lighting.explain import explain_decision
from custom_components.adaptive_lighting.intent import Intent, resolve_intent
from custom_components.adaptive_lighting.policy import PolicyConfig, decide


def active(value: bool = True, source: str = "test"):
    return signal(value, source=source)


def test_priority_emergency_beats_every_other_intent() -> None:
    snapshot = ContextSnapshot(
        emergency=active(source="alarm"),
        manual=active(source="switch"),
        sleep=active(source="sleep-mode"),
        night_path=active(source="path"),
        task=active(source="work"),
        video=active(source="media"),
        arrival=active(source="arrival"),
        vacant=active(source="vacancy"),
    )

    decision = decide(snapshot)

    assert decision.intent is Intent.EMERGENCY
    assert decision.priority > decision.rejected_alternatives[0].priority
    assert any(item.intent is Intent.MANUAL for item in decision.rejected_alternatives)
    assert decision.companion_on is True
    assert decision.can_adjust is True
    assert decision.can_turn_on is True
    assert decision.can_turn_off is False


@pytest.mark.parametrize(
    ("field", "intent"),
    [
        ("manual", Intent.MANUAL),
        ("sleep", Intent.SLEEP),
        ("night_path", Intent.NIGHT_PATH),
        ("task", Intent.TASK),
        ("video", Intent.VIDEO),
        ("arrival", Intent.ARRIVAL),
    ],
)
def test_explicit_intents_have_expected_priority(field: str, intent: Intent) -> None:
    snapshot = ContextSnapshot(**{field: active()})

    assert resolve_intent(snapshot).intent is intent


def test_manual_control_is_never_fought() -> None:
    decision = decide(
        ContextSnapshot(
            manual_hold=active(source="physical-switch"),
            requested_brightness=signal(72, source="user"),
            occupancy=active(source="presence"),
        ),
    )

    assert decision.intent is Intent.MANUAL
    assert decision.brightness_target == 72
    assert decision.companion_on is None
    assert decision.can_adjust is False
    assert decision.can_turn_on is False
    assert decision.can_turn_off is False
    assert decision.should_apply is False
    assert any("will not fight" in reason for reason in decision.reasons)


def test_unavailable_inputs_degrade_without_turning_companion_on_or_off() -> None:
    decision = decide(ContextSnapshot())
    explanation = explain_decision(decision)

    assert decision.intent is Intent.AMBIENT
    assert decision.brightness_target == 30
    assert decision.companion_on is None
    assert decision.confidence == 0.25
    assert decision.can_adjust is False
    assert decision.can_turn_on is False
    assert decision.can_turn_off is False
    assert decision.should_apply is False
    assert all(item.available is False for item in decision.input_provenance)
    assert explanation.can_adjust is False
    assert explanation.can_turn_on is False
    assert explanation.can_turn_off is False
    assert "no automatic action is authorized" in explanation.summary


def test_known_vacancy_can_turn_companion_off_but_missing_occupancy_cannot() -> None:
    vacant = decide(ContextSnapshot(occupancy=active(False, source="presence")))
    unknown = decide(ContextSnapshot(occupancy=unavailable(source="sensor-down")))

    assert vacant.intent is Intent.VACANT
    assert vacant.companion_on is False
    assert vacant.can_adjust is False
    assert vacant.can_turn_on is False
    assert vacant.can_turn_off is True
    assert vacant.should_apply is True
    assert unknown.companion_on is None
    assert unknown.can_turn_off is False


def test_ambient_permissions_separate_adjustment_from_turn_on() -> None:
    unknown_occupancy = decide(ContextSnapshot(ambient=active(source="scene")))
    confirmed_occupancy = decide(
        ContextSnapshot(
            ambient=active(source="scene"),
            occupancy=active(source="presence"),
        ),
    )
    low_confidence = decide(
        ContextSnapshot(
            ambient=signal(True, source="weak-scene", confidence=0.2),
            occupancy=active(source="presence"),
        ),
    )

    assert unknown_occupancy.can_adjust is True
    assert unknown_occupancy.can_turn_on is False
    assert unknown_occupancy.companion_on is None
    assert confirmed_occupancy.can_adjust is True
    assert confirmed_occupancy.can_turn_on is True
    assert confirmed_occupancy.companion_on is True
    assert low_confidence.can_adjust is False
    assert low_confidence.can_turn_on is False
    assert low_confidence.should_apply is False


@pytest.mark.parametrize(("raw", "expected"), [("on", True), ("off", False)])
def test_raw_ha_boolean_strings_are_normalized(raw: str, expected: bool) -> None:
    direct = ContextSnapshot(occupancy=raw)  # type: ignore[arg-type]
    wrapped = ContextSnapshot(occupancy=signal(raw, source="ha-state"))

    assert direct.occupancy.value is expected
    assert wrapped.occupancy.value is expected
    assert direct.occupancy.usable() is True
    assert wrapped.occupancy.usable() is True
    if expected:
        task = decide(ContextSnapshot(task=raw))  # type: ignore[arg-type]
        assert task.intent is Intent.TASK
        assert task.can_turn_on is True
    else:
        vacancy = decide(direct)
        assert vacancy.intent is Intent.VACANT
        assert vacancy.can_turn_off is True


def test_ambiguous_boolean_string_is_rejected_and_cannot_act() -> None:
    snapshot = ContextSnapshot(
        occupancy="unknown",  # type: ignore[arg-type]
        ambient="unknown",  # type: ignore[arg-type]
    )
    decision = decide(snapshot)

    assert snapshot.occupancy.usable() is False
    assert snapshot.ambient.usable() is False
    assert "invalid boolean value" in snapshot.occupancy.detail
    assert decision.can_adjust is False
    assert decision.can_turn_on is False
    assert decision.can_turn_off is False


@pytest.mark.parametrize("max_age", [float("inf"), float("nan"), -1.0, "invalid"])
def test_invalid_max_age_fails_closed(max_age: object) -> None:
    value = ContextSignal(
        value=True,
        source="sensor",
        age_seconds=0,
        max_age_seconds=max_age,  # type: ignore[arg-type]
    )

    assert value.max_age_seconds is not None
    assert value.fresh is False
    assert value.usable() is False
    assert value.available is False
    assert "invalid freshness metadata" in value.detail


def test_ambiguous_semantic_aliases_do_not_authorize_actions() -> None:
    away = decide(ContextSnapshot(semantic_intent="away"))  # type: ignore[arg-type]
    prelight = decide(ContextSnapshot(intent_hint="prelight"))  # type: ignore[arg-type]
    night = decide(ContextSnapshot(intent_hint="night"))  # type: ignore[arg-type]

    assert away.intent is Intent.AMBIENT
    assert away.should_apply is False
    assert prelight.intent is Intent.AMBIENT
    assert prelight.should_apply is False
    assert night.intent is Intent.NIGHT_PATH
    assert night.can_adjust is True
    assert night.can_turn_on is True


def test_sleep_and_night_path_targets_never_exceed_caps() -> None:
    config = PolicyConfig(
        sleep_brightness=90,
        sleep_cap=4,
        night_path_brightness=90,
        night_path_cap=6,
    )

    sleep = decide(ContextSnapshot(sleep=active()), config)
    night_path = decide(ContextSnapshot(night_path=active()), config)

    assert sleep.brightness_target == 4
    assert night_path.brightness_target == 6


@pytest.mark.parametrize("brightness", [-50, 0, 50, 150, float("inf")])
def test_general_targets_are_bounded(brightness: float) -> None:
    decision = decide(
        ContextSnapshot(ambient_brightness=signal(brightness, source="room-model")),
        PolicyConfig(min_brightness=5, max_brightness=80),
    )

    assert decision.brightness_target is not None
    assert 5 <= decision.brightness_target <= 80


def test_explanation_contains_reasons_rejected_intents_and_provenance() -> None:
    decision = decide(
        ContextSnapshot(
            task=active(source="desk-automation"),
            video=active(source="media-player"),
        ),
    )
    explanation = explain_decision(decision)

    assert explanation.intent is Intent.TASK
    assert explanation.reasons == decision.reasons
    assert any(item.intent is Intent.VIDEO for item in explanation.rejected_alternatives)
    assert any(item.source == "desk-automation" for item in explanation.input_provenance)
    assert "task" in explanation.as_text()


def test_snapshots_decisions_and_explanations_are_immutable_and_deterministic() -> None:
    snapshot = ContextSnapshot(ambient=active(source="scene"))
    first = decide(snapshot)
    second = decide(snapshot)

    assert first == second
    assert explain_decision(first) == explain_decision(second)
    with pytest.raises(FrozenInstanceError):
        snapshot.ambient = active()  # type: ignore[misc]
