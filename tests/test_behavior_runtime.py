# ruff: noqa: COM812

"""Focused Home Assistant tests for the continuous behavior runtime."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from homeassistant.components.adaptive_lighting.behavior_runtime import (
    MAX_TRACKED_CONTEXTS,
    BehaviorRuntimeAdapter,
    CandidateRecord,
)
from homeassistant.const import EVENT_CALL_SERVICE, EVENT_STATE_CHANGED
from homeassistant.core import Context, Event, State
from homeassistant.util import dt as dt_util

BASE_TIME = datetime(2026, 7, 12, 18, 0, tzinfo=UTC)
_ACTIVE_RUNTIMES: set[BehaviorRuntimeAdapter] = set()


@pytest.fixture(autouse=True)
async def _unload_runtime_adapters():
    """Unload every adapter even when a test fails before its own cleanup."""
    yield
    for runtime in tuple(_ACTIVE_RUNTIMES):
        await runtime.async_stop()
    _ACTIVE_RUNTIMES.clear()


def _call_event(
    entity_id: str,
    action: str,
    *,
    at: datetime = BASE_TIME,
    context: Context | None = None,
    source: str | None = None,
    provenance: str | None = None,
    routine: str | None = None,
) -> Event:
    data: dict[str, object] = {
        "domain": entity_id.split(".", 1)[0],
        "service": f"turn_{action}",
        "service_data": {"entity_id": entity_id},
    }
    if source is not None:
        data["source"] = source
    if provenance is not None:
        data["provenance"] = provenance
    if routine is not None:
        data["routine"] = routine
    data["timestamp"] = at
    return Event(EVENT_CALL_SERVICE, data, context=context)


def _state_event(
    entity_id: str,
    state: str,
    *,
    at: datetime = BASE_TIME,
    context: Context | None = None,
    source: str | None = None,
    old_state: str | None = None,
) -> Event:
    attributes = {"source": source} if source is not None else {}
    data = {
        "entity_id": entity_id,
        "new_state": State(entity_id, state, attributes=attributes, context=context),
        "timestamp": at,
    }
    if old_state is not None:
        data["old_state"] = State(entity_id, old_state, context=context)
    return Event(
        EVENT_STATE_CHANGED,
        data,
        context=context,
    )


def _context(at: datetime = BASE_TIME, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "timestamp": at,
        "home_away": "home",
        "occupancy": "occupied",
        "semantic_routine": "ambient",
        "event_times": {},
        "state_dwell": {},
    }
    value.update(changes)
    return value


async def _runtime(
    hass,
    candidates: list[CandidateRecord],
    context: dict[str, object] | None = None,
    **kwargs: object,
) -> BehaviorRuntimeAdapter:
    runtime = BehaviorRuntimeAdapter(
        hass,
        candidate_provider=lambda: tuple(candidates),
        context_provider=lambda: dict(context or _context()),
        storage_key=f"adaptive_lighting_behavior_test_{uuid4().hex}",
        **kwargs,
    )
    await runtime.async_start()
    _ACTIVE_RUNTIMES.add(runtime)
    return runtime


async def _transition(
    runtime: BehaviorRuntimeAdapter,
    entity_id: str,
    action: str,
    *,
    at: datetime = BASE_TIME,
    context: Context | None = None,
    source: str | None = None,
    provenance: str | None = None,
    routine: str | None = None,
    old_state: str | None = None,
) -> None:
    """Emit the service intent followed by the resulting state transition."""
    transition_context = context or Context(user_id="person")
    await runtime.async_process_call_service(
        _call_event(
            entity_id,
            action,
            at=at,
            context=transition_context,
            source=source,
            provenance=provenance,
            routine=routine,
        ),
    )
    await runtime.async_process_state_change(
        _state_event(
            entity_id,
            action,
            at=at,
            context=transition_context,
            old_state=old_state,
        ),
    )


def _register_light_services(hass) -> None:
    """Provide minimal service targets for active adapter tests."""

    async def _set_state(call, state: str) -> None:
        hass.states.async_set(
            call.data["entity_id"],
            state,
            context=call.context,
        )

    async def _turn_on(call) -> None:
        await _set_state(call, "on")

    async def _turn_off(call) -> None:
        await _set_state(call, "off")

    hass.services.async_register("light", "turn_on", _turn_on)
    hass.services.async_register("light", "turn_off", _turn_off)


def _light(entity_id: str = "light.living", **changes: object) -> CandidateRecord:
    value: dict[str, object] = {
        "entity_id": entity_id,
        "area": "living",
        "domain": "light",
        "supports_brightness": False,
        "available": True,
    }
    value.update(changes)
    return CandidateRecord(**value)


async def test_shadow_learning_has_no_service_calls_and_counts_human_actions(
    hass,
) -> None:
    calls: list[Event] = []
    accepted = []
    hass.bus.async_listen(EVENT_CALL_SERVICE, calls.append)
    runtime = await _runtime(
        hass,
        [_light()],
        on_accepted_observation=accepted.append,
    )

    first_context = Context(user_id="person")
    await runtime.async_process_call_service(
        _call_event("light.living", "on", context=first_context),
    )
    assert not accepted
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=first_context),
    )
    assert len(accepted) == 1
    await _transition(
        runtime,
        "light.living",
        "off",
        at=BASE_TIME + timedelta(minutes=1),
        context=Context(user_id="person"),
    )

    proposals = await runtime.async_evaluate()
    assert not calls
    assert len(accepted) == 2
    assert runtime.diagnostics["accepted_observations"] == 2
    assert proposals
    assert not proposals[0].active
    assert proposals[0].reason == "stale_context"
    await runtime.async_stop()


async def test_non_dimmable_active_action_uses_small_turn_on_service(hass) -> None:
    _register_light_services(hass)
    service_events: list[Event] = []
    hass.bus.async_listen(EVENT_CALL_SERVICE, service_events.append)
    context = _context()
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    for index in range(24):
        await _transition(
            runtime,
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=index),
            context=Context(user_id=f"person-{index}"),
        )
    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    context["timestamp"] = BASE_TIME + timedelta(minutes=24)
    proposals = await runtime.async_evaluate(now=BASE_TIME + timedelta(minutes=24))
    await hass.async_block_till_done()
    assert proposals
    assert proposals[0].active
    assert proposals[0].ready
    assert proposals[0].executed
    assert proposals[0].reason == "executed"
    own_calls = [
        event
        for event in service_events
        if event.data.get("domain") == "light"
        and event.data.get("service") == "turn_on"
    ]
    assert own_calls
    assert own_calls[-1].data["service_data"] == {"entity_id": "light.living"}
    await runtime.async_stop()


async def test_failed_actuation_never_registers_or_settles_pending(hass) -> None:
    context = _context()
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    for index in range(3):
        await _transition(
            runtime,
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=index),
            context=Context(user_id=f"person-{index}"),
        )
    runtime._states["light.living"] = "off"
    context["timestamp"] = BASE_TIME + timedelta(minutes=3)

    async def _fail_service(_call) -> None:
        message = "expected service failure"
        raise RuntimeError(message)

    hass.services.async_register("light", "turn_on", _fail_service)
    proposals = await runtime.async_evaluate(now=BASE_TIME + timedelta(minutes=3))

    assert proposals
    assert proposals[0].reason == "actuation_failed"
    assert runtime.models["light.living"].pending_proposal_count == 0
    assert (
        runtime.models["light.living"].settle_unchanged(
            BASE_TIME + timedelta(hours=2),
        )
        == ()
    )
    await runtime.async_stop()


async def test_success_without_state_confirmation_has_no_pending_reward(hass) -> None:
    context = _context()
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    await _transition(
        runtime,
        "light.living",
        "on",
        context=Context(user_id="person"),
    )
    runtime._states["light.living"] = "off"
    context["timestamp"] = BASE_TIME + timedelta(minutes=1)
    hass.services.async_register("light", "turn_on", lambda _call: None)

    proposals = await runtime.async_evaluate(now=BASE_TIME + timedelta(minutes=1))

    assert proposals
    assert proposals[0].reason == "actuation_unconfirmed"
    assert runtime.models["light.living"].pending_proposal_count == 0
    assert (
        runtime.models["light.living"].settle_unchanged(
            BASE_TIME + timedelta(hours=2),
        )
        == ()
    )
    await runtime.async_stop()


async def test_quick_reversal_suppresses_pending_proposal(hass) -> None:
    _register_light_services(hass)
    proposal_at = dt_util.utcnow()
    training_start = proposal_at - timedelta(minutes=6)
    context = _context(proposal_at)
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    for index in range(6):
        await _transition(
            runtime,
            "light.living",
            "on",
            at=training_start + timedelta(minutes=index),
            context=Context(user_id=f"person-{index}"),
        )
    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    proposals = await runtime.async_evaluate(now=proposal_at)
    await hass.async_block_till_done()
    assert proposals
    assert runtime.models["light.living"].pending_proposal_count == 1, proposals
    repeated = await runtime.async_evaluate(now=proposal_at + timedelta(seconds=1))
    assert repeated
    assert repeated[0].reason == "pending_proposal"
    assert not repeated[0].active

    own_state_at = runtime._state_event_times["light.living"]
    correction_at = own_state_at + timedelta(minutes=1)
    context["timestamp"] = correction_at
    await _transition(
        runtime,
        "light.living",
        "off",
        at=correction_at,
        context=Context(user_id="corrector"),
        old_state="on",
    )
    assert runtime.models["light.living"].pending_proposal_count == 0
    assert runtime.models["light.living"].storage_stats()["suppressions"] == 1
    assert runtime.diagnostics["corrections"] == 1
    assert runtime.diagnostics["active_suppressions"] == 1
    await runtime.async_stop()


async def test_late_opposite_feedback_is_negative_without_suppression(hass) -> None:
    _register_light_services(hass)
    proposal_at = dt_util.utcnow()
    training_start = proposal_at - timedelta(minutes=4)
    context = _context(proposal_at)
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    for index in range(4):
        await _transition(
            runtime,
            "light.living",
            "on",
            at=training_start + timedelta(minutes=index),
            context=Context(user_id=f"person-{index}"),
        )
    runtime._states["light.living"] = "off"
    await runtime.async_evaluate(now=proposal_at)
    await hass.async_block_till_done()
    model = runtime.models["light.living"]
    negative_before = model.to_dict()["actions"]["on"]["negative_support"]

    own_state_at = runtime._state_event_times["light.living"]
    correction_at = own_state_at + timedelta(minutes=20)
    context["timestamp"] = correction_at
    await _transition(
        runtime,
        "light.living",
        "off",
        at=correction_at,
        context=Context(user_id="late-corrector"),
        old_state="on",
    )
    negative_after = model.to_dict()["actions"]["on"]["negative_support"]

    assert model.pending_proposal_count == 0
    assert negative_after - negative_before > 1.5
    assert model.storage_stats()["suppressions"] == 0
    assert runtime.diagnostics["corrections"] == 1
    assert runtime.diagnostics["active_suppressions"] == 0
    await runtime.async_stop()


async def test_own_state_change_does_not_consume_pending_proposal(hass) -> None:
    _register_light_services(hass)
    context = _context()
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=True.__bool__,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    for index in range(6):
        await _transition(
            runtime,
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=index),
            context=Context(user_id=f"person-{index}"),
        )
    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    await runtime.async_evaluate(now=BASE_TIME + timedelta(minutes=7))
    await hass.async_block_till_done()
    assert runtime.models["light.living"].pending_proposal_count == 1, (
        runtime.diagnostics
    )
    own_context = next(iter(runtime._own_contexts))
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=8),
            context=Context(id=own_context),
        ),
    )
    assert runtime.models["light.living"].pending_proposal_count == 1
    await runtime.async_stop()


async def test_same_power_state_attribute_change_does_not_learn_or_resolve(
    hass,
) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    await _transition(
        runtime,
        "light.living",
        "on",
        context=Context(user_id="person"),
    )
    model = runtime.models["light.living"]
    temporal = runtime._temporal_context(
        _context(),
        "light.living",
        "living",
        "off",
    )
    model.register_proposal(temporal, "off")

    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "on",
            old_state="on",
            at=BASE_TIME + timedelta(minutes=1),
            context=Context(user_id="person"),
        ),
    )

    assert len(accepted) == 1
    assert model.pending_proposal_count == 1
    await runtime.async_stop()


async def test_unrelated_automation_and_compound_alarm_are_rejected(hass) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    automation_context = Context()
    await runtime.async_process_call_service(
        _call_event(
            "light.living",
            "on",
            source="automation",
            context=automation_context,
        ),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=automation_context),
    )
    alarm_context = Context()
    await runtime.async_process_call_service(
        _call_event(
            "light.living",
            "off",
            source="automation+alarm",
            provenance="security_alarm",
            context=alarm_context,
        ),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "off", context=alarm_context),
    )
    assert not accepted
    assert runtime.diagnostics["accepted_observations"] == 0
    await runtime.async_stop()


async def test_parent_bearing_unlabeled_service_call_fails_closed(hass) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    call_context = Context(id="automation-child", parent_id="automation-parent")
    await runtime.async_process_call_service(
        _call_event("light.living", "on", context=call_context),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=call_context),
    )

    assert not accepted
    assert runtime.diagnostics["accepted_observations"] == 0
    await runtime.async_stop()


async def test_automation_and_safety_labels_outrank_human_labels(hass) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "on",
            source="human automation",
            context=Context(id="conflicting-automation"),
        ),
    )
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "off",
            source="physical safety",
            context=Context(id="conflicting-safety"),
            at=BASE_TIME + timedelta(minutes=1),
        ),
    )

    assert not accepted
    await runtime.async_stop()


async def test_good_night_intent_propagates_to_child_light_context(hass) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    scene_context = Context(id="scene-good-night")
    await runtime.async_process_call_service(
        _call_event(
            "scene.good_night",
            "on",
            context=scene_context,
        ),
    )
    child_context = Context(id="child-light-call", parent_id=scene_context.id)
    await runtime.async_process_call_service(
        _call_event(
            "light.living",
            "off",
            source="automation",
            context=child_context,
        ),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "off", context=child_context),
    )
    assert len(accepted) == 1
    assert accepted[0].semantic_routine == "good_night"
    await runtime.async_stop()


async def test_live_provider_safety_blocks_state_learning_but_not_good_night_label(
    hass,
) -> None:
    accepted = []
    context = _context(safety=True)
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        on_accepted_observation=accepted.append,
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=Context(id="physical-safety")),
    )
    context.update(safety=False, security_state="problem")
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "off",
            at=BASE_TIME + timedelta(minutes=1),
            context=Context(id="physical-security-problem"),
        ),
    )
    context.update(security_state="ok", semantic_routine="good_night")
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=2),
            context=Context(id="physical-good-night-label"),
        ),
    )

    assert len(accepted) == 1
    assert runtime.diagnostics["accepted_observations"] == 1
    await runtime.async_stop()


async def test_safety_good_night_context_does_not_contaminate_children(hass) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    scene_context = Context(id="unsafe-good-night")
    await runtime.async_process_call_service(
        _call_event(
            "scene.good_night",
            "on",
            source="safety alarm",
            context=scene_context,
        ),
    )
    child_context = Context(id="unsafe-child", parent_id=scene_context.id)
    await runtime.async_process_call_service(
        _call_event("light.living", "off", context=child_context),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "off", context=child_context),
    )

    assert not accepted
    assert scene_context.id not in runtime._good_night_contexts
    assert scene_context.id in runtime._safety_contexts
    await runtime.async_stop()


async def test_good_night_and_safety_context_tracking_is_bounded(hass) -> None:
    runtime = await _runtime(hass, [_light()])
    for index in range(MAX_TRACKED_CONTEXTS + 8):
        await runtime.async_process_call_service(
            _call_event(
                "scene.good_night",
                "on",
                context=Context(id=f"good-night-{index}"),
            ),
        )
        await runtime.async_process_call_service(
            _call_event(
                "light.living",
                "off",
                source="safety alarm",
                context=Context(id=f"safety-{index}"),
            ),
        )

    assert len(runtime._good_night_contexts) <= MAX_TRACKED_CONTEXTS
    assert len(runtime._safety_contexts) <= MAX_TRACKED_CONTEXTS
    await runtime.async_stop()


async def test_bare_physical_context_learns_but_parent_context_is_rejected(
    hass,
) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=Context(id="physical-device")),
    )
    assert len(accepted) == 1
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "off",
            context=Context(id="child-automation", parent_id="automation-parent"),
        ),
    )
    assert len(accepted) == 1
    await runtime.async_stop()


async def test_physical_state_uses_live_provider_context_for_learning(hass) -> None:
    accepted = []
    provider_context = _context(
        occupancy="occupied",
        media="playing",
        home_away="home",
        weather_band="rain",
        event_times={"motion": BASE_TIME - timedelta(minutes=1)},
        state_dwell={"occupancy": 120},
    )
    runtime = await _runtime(
        hass,
        [_light()],
        context=provider_context,
        on_accepted_observation=accepted.append,
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=Context(id="physical-device")),
    )

    assert len(accepted) == 1
    action_state = runtime.models["light.living"].to_dict()["actions"]["on"]
    weights = action_state["weights"]
    assert weights["context_occupancy_occupied"] != 0
    assert weights["context_media_playing"] != 0
    assert weights["motion_recency_short"] != 0
    assert weights["state_dwell_any"] != 0
    assert any(name.startswith("context_extra_bucket_") for name in weights)
    await runtime.async_stop()


async def test_unrelated_physical_state_falls_through_stale_service_attribution(
    hass,
) -> None:
    accepted = []
    runtime = await _runtime(hass, [_light()], on_accepted_observation=accepted.append)
    await runtime.async_process_call_service(
        _call_event(
            "light.living",
            "on",
            context=Context(id="stale-human-call", user_id="person"),
        ),
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "off", context=Context(id="physical-later")),
    )

    assert len(accepted) == 1
    assert accepted[0].action == "off"
    assert accepted[0].source == "physical"
    await runtime.async_stop()


async def test_good_night_is_learnable_but_unknown_occupancy_does_not_allow_off(
    hass,
) -> None:
    accepted = []
    context = _context(occupancy="unknown")
    runtime = await _runtime(
        hass, [_light()], context=context, on_accepted_observation=accepted.append
    )
    await runtime.async_record_good_night(
        ("light.living",),
        context={"timestamp": BASE_TIME, "source": "scene.good_night"},
    )
    assert len(accepted) == 1
    hass.states.async_set("light.living", "on")
    await hass.async_block_till_done()
    proposals = await runtime.async_evaluate(now=BASE_TIME)
    assert proposals
    assert proposals[0].reason == "off_requires_good_night_away_or_empty_dwell"
    await runtime.async_stop()


async def test_on_requires_home_and_occupancy_or_fresh_arrival(hass) -> None:
    context = _context(home_away="away", occupancy="occupied")
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    proposals = await runtime.async_evaluate(now=BASE_TIME)
    assert proposals
    assert proposals[0].reason == "household_not_home"

    context.update(home_away="home", occupancy="unknown", semantic_routine="home")
    proposals = await runtime.async_evaluate(now=BASE_TIME)
    assert proposals
    assert proposals[0].reason == "no_occupancy_or_fresh_arrival"
    context.update(recent_arrival=True)
    proposals = await runtime.async_evaluate(now=BASE_TIME)
    assert proposals
    assert proposals[0].reason == "ready_shadow"
    await runtime.async_stop()


async def test_missing_or_invalid_context_timestamp_fails_freshness_closed(
    hass,
) -> None:
    context = _context()
    context.pop("timestamp")
    runtime = await _runtime(
        hass,
        [_light()],
        context=context,
        actuation_enabled=lambda: True,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    runtime._states["light.living"] = "off"

    missing = await runtime.async_evaluate(now=BASE_TIME)
    context["timestamp"] = "not-a-timestamp"
    invalid = await runtime.async_evaluate(now=BASE_TIME)

    assert missing
    assert missing[0].reason == "stale_context"
    assert invalid
    assert invalid[0].reason == "stale_context"
    assert runtime.models["light.living"].pending_proposal_count == 0
    await runtime.async_stop()


async def test_candidate_manual_hold_blocks_actuation_but_not_learning(hass) -> None:
    _register_light_services(hass)
    accepted = []
    context = _context()
    runtime = await _runtime(
        hass,
        [_light(manual_hold=True)],
        context=context,
        actuation_enabled=lambda: True,
        on_accepted_observation=accepted.append,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    await runtime.async_process_state_change(
        _state_event("light.living", "on", context=Context(id="physical-device")),
    )

    proposals = await runtime.async_evaluate(now=BASE_TIME)

    assert len(accepted) == 1
    assert proposals
    assert proposals[0].reason == "manual_hold_entity"
    assert not proposals[0].active
    assert runtime.models["light.living"].pending_proposal_count == 0
    await runtime.async_stop()


async def test_area_occupancy_cannot_authorize_another_area_light(hass) -> None:
    context = _context(
        occupancy="unknown",
        area_context={
            "kitchen": {
                "occupancy": "occupied",
                "event_times": {"motion": BASE_TIME},
            },
            "bedroom": {
                "occupancy": "empty",
                "event_times": {},
            },
        },
    )
    runtime = await _runtime(
        hass,
        [
            _light("light.kitchen", area="kitchen"),
            _light("light.bedroom", area="bedroom"),
        ],
        context=context,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    await runtime.async_process_state_change(
        _state_event("light.kitchen", "on", context=Context(id="kitchen-device")),
    )
    await runtime.async_process_state_change(
        _state_event("light.bedroom", "on", context=Context(id="bedroom-device")),
    )
    runtime._states.update({"light.kitchen": "off", "light.bedroom": "off"})

    proposals = await runtime.async_evaluate(now=BASE_TIME)
    by_entity = {proposal.entity_id: proposal for proposal in proposals}

    assert by_entity["light.kitchen"].reason == "ready_shadow"
    assert by_entity["light.bedroom"].reason == "no_occupancy_or_fresh_arrival"
    await runtime.async_stop()


async def test_area_context_deep_merges_global_temporal_features(hass) -> None:
    context = _context(
        occupancy="unknown",
        media="playing",
        weather_band="rain",
        event_times={"arrival": BASE_TIME - timedelta(minutes=2)},
        state_dwell={"home": 600},
        categorical_context={"calendar": "workday", "security": "secure"},
        area_context={
            "kitchen": {
                "occupancy": "occupied",
                "event_times": {"motion": BASE_TIME - timedelta(minutes=1)},
                "state_dwell": {"occupancy": 120},
                "categorical_context": {"local_mode": "cooking"},
            },
        },
    )
    runtime = await _runtime(hass, [_light("light.kitchen", area="kitchen")])

    merged = runtime._context_for_area(context, "kitchen")
    temporal = runtime._temporal_context(
        merged,
        "light.kitchen",
        "kitchen",
        "on",
    )
    encoded = runtime.models["light.kitchen"].encoder.encode(temporal).values

    assert merged["occupancy"] == "occupied"
    assert temporal.event_times == {
        "arrival": BASE_TIME - timedelta(minutes=2),
        "motion": BASE_TIME - timedelta(minutes=1),
    }
    assert temporal.state_dwell == {"home": 600, "occupancy": 120}
    assert temporal.categorical_context["media"] == "playing"
    assert temporal.categorical_context["weather"] == "rain"
    assert temporal.categorical_context["calendar"] == "workday"
    assert temporal.categorical_context["security"] == "secure"
    assert temporal.categorical_context["local_mode"] == "cooking"
    assert encoded["arrival_recency_short"] > 0
    assert encoded["motion_recency_short"] > 0
    await runtime.async_stop()


async def test_candidate_reconciliation_rejects_cover_unavailable_and_moved_area(
    hass,
) -> None:
    candidates = [
        _light(),
        CandidateRecord("cover.garage", domain="cover"),
        CandidateRecord(
            "switch.domain_mismatch",
            domain="light",
            explicit_light_switch=True,
        ),
    ]
    runtime = await _runtime(hass, candidates)
    assert set(runtime.models) == {"light.living"}
    candidates[0] = _light(available=False)
    await runtime.async_evaluate()
    assert runtime.diagnostics["available_candidates"] == 0
    candidates[0] = _light(area="kitchen", available=True)
    await runtime.async_evaluate()
    assert runtime.diagnostics["sample_counts"]["light.living"] == 0
    candidates.clear()
    await runtime.async_evaluate()
    assert runtime.diagnostics["candidates"] == 0
    await runtime.async_stop()


async def test_state_order_hass_preference_and_readd_clear_stale_cache(hass) -> None:
    accepted = []
    candidates = [_light()]
    runtime = await _runtime(
        hass,
        candidates,
        on_accepted_observation=accepted.append,
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "on",
            at=BASE_TIME + timedelta(minutes=10),
            context=Context(id="newer-state"),
        ),
    )
    await runtime.async_process_state_change(
        _state_event(
            "light.living",
            "off",
            at=BASE_TIME,
            context=Context(id="older-state"),
        ),
    )
    assert len(accepted) == 1
    assert runtime._states["light.living"] == "on"

    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    runtime._states["light.living"] = "on"
    proposals = await runtime.async_evaluate(now=BASE_TIME)
    assert proposals
    assert proposals[0].action == "on"

    candidates.clear()
    await runtime.async_evaluate(now=BASE_TIME)
    hass.states.async_remove("light.living")
    await hass.async_block_till_done()
    candidates.append(_light())
    readded = await runtime.async_evaluate(now=BASE_TIME)
    assert readded == ()
    assert "light.living" not in runtime._states
    await runtime.async_stop()


async def test_persistence_roundtrip_and_corrupt_reset(hass) -> None:
    key = f"adaptive_lighting_behavior_persistence_{uuid4().hex}"
    candidates = [_light()]
    runtime = BehaviorRuntimeAdapter(
        hass,
        candidate_provider=lambda: tuple(candidates),
        context_provider=lambda: _context(),
        storage_key=key,
    )
    _ACTIVE_RUNTIMES.add(runtime)
    await runtime.async_start()
    await _transition(
        runtime,
        "light.living",
        "on",
        context=Context(user_id="person"),
    )
    exported = await runtime._store.async_load()
    json.dumps(exported)
    await runtime.async_stop()

    restored = BehaviorRuntimeAdapter(
        hass,
        candidate_provider=lambda: tuple(candidates),
        context_provider=lambda: _context(),
        storage_key=key,
    )
    await restored.async_start()
    assert restored.diagnostics["accepted_observations"] == 1
    await restored.async_stop()
    await restored._store.async_save({"data_version": 999})

    reset = BehaviorRuntimeAdapter(
        hass,
        candidate_provider=lambda: tuple(candidates),
        context_provider=lambda: _context(),
        storage_key=key,
    )
    await reset.async_start()
    assert reset.diagnostics["last_load_reset_reason"] == "unsupported_schema"
    assert reset.diagnostics["accepted_observations"] == 0
    await reset.async_stop()


async def test_stop_during_context_await_prevents_actuation_and_pending(hass) -> None:
    _register_light_services(hass)
    hass.states.async_set("light.living", "off")
    await hass.async_block_till_done()
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    async def _slow_context_provider():
        provider_started.set()
        await release_provider.wait()
        return _context()

    runtime = BehaviorRuntimeAdapter(
        hass,
        candidate_provider=lambda: (_light(),),
        context_provider=_slow_context_provider,
        actuation_enabled=lambda: True,
        storage_key=f"adaptive_lighting_behavior_stop_{uuid4().hex}",
        min_probability=0.0,
        min_confidence=0.0,
        min_effective_support=0.1,
        min_freshness=0.0,
    )
    _ACTIVE_RUNTIMES.add(runtime)
    await runtime.async_start()
    temporal = runtime._temporal_context(
        _context(),
        "light.living",
        "living",
        "on",
    )
    runtime.models["light.living"].update(
        temporal,
        "on",
        True,
        provenance="user",
    )

    evaluation = asyncio.create_task(runtime.async_evaluate(now=BASE_TIME))
    await provider_started.wait()
    await runtime.async_stop()
    release_provider.set()

    assert await evaluation == ()
    assert runtime.models["light.living"].pending_proposal_count == 0


async def test_listener_unload_and_bounded_entity_count(hass) -> None:
    candidates = [_light(f"light.{index}") for index in range(4)]
    runtime = await _runtime(hass, candidates, max_entities=2)
    assert len(runtime.models) == 2
    candidates.clear()
    await runtime.async_evaluate()
    assert runtime.diagnostics["candidates"] == 0
    await runtime.async_stop()
    hass.bus.async_fire(
        EVENT_STATE_CHANGED,
        _state_event("light.0", "on").data,
    )
    await hass.async_block_till_done()
    assert runtime.diagnostics["accepted_observations"] == 0
