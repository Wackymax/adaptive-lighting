"""Focused tests for the local Home Assistant training session."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from homeassistant.components.adaptive_lighting.training import (
    DAY_TYPE_PUBLIC_HOLIDAY,
    DAY_TYPE_WEEKDAY,
    DAY_TYPE_WEEKEND,
    MAX_BEHAVIOR_SAMPLES_PER_SESSION,
    PHASE_ACTIVE,
    PHASE_SHADOW_LEARNING,
    AdaptiveLightingTraining,
)


def candidate(**changes: object) -> dict[str, object]:
    """Build a valid manual preference event for the training adapter."""
    value: dict[str, object] = {
        "sample_id": "sample-1",
        "zone": "living",
        "baseline": 30,
        "selected": 50,
        "source": "manual",
        "context": "ambient",
        "time_bucket": "evening",
        "daylight_band": "dark",
        "duration_seconds": 0,
    }
    value.update(changes)
    return value


async def make_training(hass, **kwargs: object) -> AdaptiveLightingTraining:
    """Create a loaded manager with a unique local Store key per test."""
    training = AdaptiveLightingTraining(
        hass,
        storage_key=f"adaptive_lighting_training_{id(kwargs)}",
        **kwargs,
    )
    await training.async_setup()
    return training


async def test_persistence_roundtrip_keeps_session_and_learner(hass) -> None:
    """The HA Store roundtrip retains versioned session and learner state."""
    training = await make_training(hass, durability_seconds=0)
    assert await training.async_ingest_sample(candidate()) is True
    exported = training.export_state()
    json.dumps(exported)

    await training.async_unload()
    restored = AdaptiveLightingTraining(
        hass,
        storage_key=training.storage_key,
        durability_seconds=0,
    )
    await restored.async_setup()

    assert restored.phase == PHASE_SHADOW_LEARNING
    assert restored.training_started_at == training.training_started_at
    assert restored.training_deadline == training.training_deadline
    assert restored.sample_counts == training.sample_counts
    assert restored.learner.export_state() == training.learner.export_state()
    assert restored.export_state()["data_version"] == 1
    stored = await restored._store.async_load()
    assert stored is not None
    assert stored["phase"] == PHASE_SHADOW_LEARNING
    assert stored["start"] == stored["training_started_at"]
    assert stored["deadline"] == stored["training_deadline"]

    await restored.async_unload()


async def test_default_training_deadline_is_seven_days(hass) -> None:
    """A fresh session persists a seven-day training window."""
    training = await make_training(hass)

    assert training.training_duration == timedelta(days=7)
    assert training.training_deadline - training.training_started_at == timedelta(
        days=7,
    )

    await training.async_unload()


async def test_local_day_type_and_public_holiday_use_weekend_learning(hass) -> None:
    """Weekday, weekend, and an injected Monday holiday stay distinguishable."""
    holiday = datetime(2026, 7, 13, 12, tzinfo=UTC).date()

    def is_public_holiday(day) -> bool:
        return day == holiday

    training = await make_training(
        hass,
        durability_seconds=0,
        public_holiday_predicate=is_public_holiday,
    )
    samples = (
        ("weekday", datetime(2026, 7, 10, 12, tzinfo=UTC), DAY_TYPE_WEEKDAY),
        ("weekend", datetime(2026, 7, 11, 12, tzinfo=UTC), DAY_TYPE_WEEKEND),
        ("holiday", datetime(2026, 7, 13, 12, tzinfo=UTC), DAY_TYPE_PUBLIC_HOLIDAY),
    )

    for sample_id, observed_at, expected_day_type in samples:
        assert await training.async_ingest_sample(
            candidate(
                sample_id=sample_id,
                observed_at=observed_at.isoformat(),
            ),
            now=observed_at,
        )
        assert training.last_sample["day_type"] == expected_day_type

    assert training.day_type_counts == {
        DAY_TYPE_PUBLIC_HOLIDAY: 1,
        DAY_TYPE_WEEKDAY: 1,
        DAY_TYPE_WEEKEND: 1,
    }
    assert training.summary()["last_day_type"] == DAY_TYPE_PUBLIC_HOLIDAY
    learner_buckets = {
        entry["time_bucket"] for entry in training.learner.export_state()["entries"]
    }
    assert learner_buckets == {"weekday:evening", "weekend:evening"}
    assert training.export_state()["last_sample"]["day_type"] == DAY_TYPE_PUBLIC_HOLIDAY

    await training.async_unload()


async def test_deadline_with_insufficient_data_stays_shadow(hass) -> None:
    """Auto-promotion must not bypass the minimum sample gate."""
    training = await make_training(
        hass,
        auto_promote=True,
        minimum_samples=2,
        minimum_confidence=0.5,
        durability_seconds=0,
    )

    promoted = await training.async_evaluate_promotion(now=training.deadline)

    assert promoted is False
    assert training.phase == PHASE_SHADOW_LEARNING
    assert "insufficient_samples" in training.summary()["promotion_reason"]

    await training.async_unload()


async def test_qualified_training_promotes_after_deadline(hass) -> None:
    """Both sample and confidence gates are required for promotion."""
    training = await make_training(
        hass,
        auto_promote=True,
        minimum_samples=2,
        minimum_confidence=0.75,
        durability_seconds=0,
    )
    assert await training.async_ingest_sample(candidate(sample_id="living-1"))
    assert await training.async_ingest_sample(
        candidate(sample_id="kitchen-1", zone="kitchen", selected=40),
    )
    assert training.phase == PHASE_SHADOW_LEARNING

    promoted = await training.async_evaluate_promotion(now=training.deadline)

    assert promoted is True
    assert training.phase == PHASE_ACTIVE
    assert training.summary()["promotion_reason"] == "promoted"

    await training.async_unload()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"source": "automation"}, "automation_source"),
        ({"source": "adaptive_lighting"}, "integration_source"),
        ({"context": {"safety": True}}, "safety_context"),
        ({"context": "alarm"}, "alarm_context"),
        ({"context": "good-night"}, "good_night_context"),
    ],
)
async def test_rejects_contaminated_preference_candidates(
    hass,
    changes: dict[str, object],
    reason: str,
) -> None:
    """Automation, integration, and safety contexts never reach the learner."""
    training = await make_training(hass, durability_seconds=0)

    result = await training.async_ingest_candidate(candidate(**changes))

    assert result["queued"] is False
    assert result["accepted"] is False
    assert result["reason"] == reason
    assert training.sample_count == 0
    assert training.last_rejection_reason == reason
    assert training.sample_counts["rejected"] == 1

    await training.async_unload()


async def test_durability_waits_and_new_override_supersedes_old(hass) -> None:
    """A candidate is accepted only after its timer and only if still current."""
    training = await make_training(hass, durability_seconds=10)
    now = training.training_started_at
    first = candidate(sample_id="first", selected=40)
    second = candidate(sample_id="second", selected=60)

    assert await training.async_ingest_sample(first, now=now)
    assert training.sample_count == 0
    assert training.pending_count == 1
    assert await training.async_ingest_sample(second, now=now)
    assert training.sample_count == 0
    assert training.pending_count == 1
    assert training.sample_counts["superseded"] == 1

    await training.async_process_due(now=now + timedelta(seconds=9))
    assert training.sample_count == 0
    await training.async_process_due(now=now + timedelta(seconds=10))
    assert training.sample_count == 1
    assert training.pending_count == 0
    assert training.learner.export_state()["entries"][0]["offset"] == 5

    await training.async_unload()


async def test_unload_cancels_deadline_and_durability_callbacks(hass) -> None:
    """Unload removes all scheduled callbacks while preserving local data."""
    training = await make_training(hass, durability_seconds=3600)
    await training.async_ingest_sample(candidate())
    assert training._deadline_timer is not None
    assert training._pending_timers

    await training.async_unload()

    assert training._deadline_timer is None
    assert training._pending_timers == {}
    assert training.pending_count == 1


async def test_restart_continuity_restores_pending_durability(hass) -> None:
    """Restarting HA-facing state does not turn a pending override into loss."""
    training = await make_training(hass, durability_seconds=3600)
    now = training.training_started_at
    await training.async_ingest_sample(candidate(), now=now)
    storage_key = training.storage_key
    await training.async_unload()

    restored = AdaptiveLightingTraining(
        hass,
        storage_key=storage_key,
        durability_seconds=3600,
    )
    await restored.async_setup()
    assert restored.pending_count == 1
    assert restored.sample_count == 0

    await restored.async_process_due(now=now + timedelta(hours=1))

    assert restored.pending_count == 0
    assert restored.sample_count == 1
    await restored.async_unload()


async def test_reset_starts_a_new_shadow_session(hass) -> None:
    """Reset clears learner/counters and creates a fresh persisted window."""
    training = await make_training(
        hass,
        auto_promote=True,
        minimum_samples=1,
        minimum_confidence=0,
        durability_seconds=0,
    )
    assert await training.async_ingest_sample(candidate())
    await training.async_evaluate_promotion(now=training.deadline)
    assert training.phase == PHASE_ACTIVE

    reset_at = datetime(2026, 7, 12, 12, tzinfo=UTC)
    summary = await training.async_reset(now=reset_at)

    assert training.phase == PHASE_SHADOW_LEARNING
    assert training.sample_count == 0
    assert training.pending_count == 0
    assert training.training_started_at == reset_at
    assert training.training_deadline == reset_at + timedelta(days=7)
    assert summary["promotion_reason"] == "training_in_progress"
    assert training.learner.export_state()["entries"] == []

    await training.async_unload()


async def test_listener_receives_learning_and_phase_changes(hass) -> None:
    """Callbacks expose compact snapshots when learning or phase changes."""
    changes: list[dict[str, object]] = []
    training = await make_training(
        hass,
        on_change=changes.append,
        auto_promote=True,
        minimum_samples=1,
        minimum_confidence=0,
        durability_seconds=0,
    )

    await training.async_ingest_sample(candidate())
    await training.async_evaluate_promotion(now=training.deadline)

    assert any(change["sample_counts"]["accepted"] == 1 for change in changes)
    assert changes[-1]["phase"] == PHASE_ACTIVE
    json.dumps(changes[-1])

    await training.async_unload()


async def test_on_off_behavior_counts_for_commissioning_and_deduplicates(hass) -> None:
    """Non-dimmable light choices contribute without inventing brightness."""
    training = await make_training(hass, minimum_samples=1)
    observed_at = training.training_started_at
    observation = {
        "entity_id": "light.hallway",
        "action": "off",
        "timestamp": observed_at,
        "source": "physical",
        "provenance": "human",
        "semantic_routine": "ambient",
        "day_type": DAY_TYPE_WEEKDAY,
    }

    assert await training.async_record_behavior_observation(
        observation,
        now=observed_at,
    )
    assert not await training.async_record_behavior_observation(
        observation,
        now=observed_at,
    )
    assert training.sample_counts["accepted"] == 1
    assert training.sample_counts["behavior_accepted"] == 1
    assert training.last_sample["kind"] == "on_off_behavior"
    assert training.learner.sample_count == 0

    await training.async_unload()


async def test_good_night_behavior_counts_but_alarm_behavior_does_not(hass) -> None:
    """Good Night is a routine preference; safety actions are never targets."""
    training = await make_training(hass)
    observed_at = training.training_started_at
    common = {
        "entity_id": "light.hallway",
        "action": "off",
        "timestamp": observed_at,
        "source": "automation.good_night",
        "provenance": "automation",
        "semantic_routine": "good_night",
        "day_type": DAY_TYPE_WEEKDAY,
    }

    assert await training.async_record_behavior_observation(common, now=observed_at)
    assert not await training.async_record_behavior_observation(
        {
            **common,
            "timestamp": observed_at + timedelta(seconds=1),
            "source": "fire_alarm",
            "provenance": "security_event",
        },
        now=observed_at + timedelta(seconds=1),
    )
    assert training.sample_counts["behavior_accepted"] == 1

    await training.async_unload()


async def test_behavior_commissioning_rejects_out_of_session_and_future_events(
    hass,
) -> None:
    """Old/future observations cannot inflate the persisted promotion gate."""
    training = await make_training(hass)
    started = training.training_started_at
    common = {
        "entity_id": "light.hallway",
        "action": "on",
        "source": "physical",
        "provenance": "human",
        "semantic_routine": "ambient",
    }

    assert not await training.async_record_behavior_observation(
        {**common, "timestamp": started - timedelta(seconds=1)},
        now=started,
    )
    assert not await training.async_record_behavior_observation(
        {**common, "timestamp": started + timedelta(seconds=31)},
        now=started,
    )
    assert training.sample_count == 0

    await training.async_unload()


async def test_behavior_commissioning_ledger_fails_closed_at_bound(hass) -> None:
    """A full fingerprint ledger refuses new counts instead of evicting IDs."""
    training = await make_training(hass)
    started = training.training_started_at
    training._behavior_sample_ids = [
        f"{index:032x}" for index in range(MAX_BEHAVIOR_SAMPLES_PER_SESSION)
    ]

    assert not await training.async_record_behavior_observation(
        {
            "entity_id": "light.hallway",
            "action": "on",
            "timestamp": started,
            "source": "physical",
            "provenance": "human",
            "semantic_routine": "ambient",
        },
        now=started,
    )
    assert training.sample_count == 0

    await training.async_unload()
