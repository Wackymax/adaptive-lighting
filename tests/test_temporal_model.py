"""Tests for the pure-Python temporal feature and action model layer."""

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "custom_components" / "adaptive_lighting",
    ),
)

from temporal_model import (
    ActionProvenance,
    OnlineActionModel,
    ProposalOutcome,
    TemporalContext,
    TemporalFeatureEncoder,
    UnsupportedSchemaError,
    is_training_eligible,
)

BASE_TIME = datetime(2026, 1, 12, 18, 0, tzinfo=UTC)


def context(
    at: datetime = BASE_TIME,
    *,
    entity_id: str | None = "living-room",
    events: dict[str, datetime | list[datetime] | None] | None = None,
    dwell: dict[str, float] | None = None,
    categories: dict[str, str] | None = None,
    holiday_name: str | None = None,
) -> TemporalContext:
    return TemporalContext(
        timestamp=at,
        event_times=events or {},
        state_dwell=dwell or {},
        categorical_context=categories or {"occupancy": "occupied"},
        entity_id=entity_id,
        holiday_name=holiday_name,
    )


def train(
    model: OnlineActionModel,
    action: str,
    target: bool,
    count: int,
    *,
    start: datetime = BASE_TIME,
    entity_id: str = "living-room",
) -> None:
    for index in range(count):
        result = model.update(
            context(start + timedelta(minutes=index), entity_id=entity_id),
            action,
            target,
            provenance=ActionProvenance.USER,
        )
        assert result.accepted


def test_encoder_is_pure_python_and_cyclic_at_midnight_boundary() -> None:
    encoder = TemporalFeatureEncoder()
    before_midnight = encoder.encode(
        context(datetime(2026, 1, 12, 23, 59, tzinfo=UTC)),
    )
    after_midnight = encoder.encode(
        context(datetime(2026, 1, 13, 0, 1, tzinfo=UTC)),
    )

    distance = (
        sum(
            (before_midnight[name] - after_midnight[name]) ** 2
            for name in ("local_time_sin", "local_time_cos")
        )
        ** 0.5
    )
    assert distance < 0.02
    assert "homeassistant" not in sys.modules


def test_recency_horizons_decay_at_different_rates() -> None:
    encoder = TemporalFeatureEncoder(
        horizons={"short": 10.0, "medium": 100.0, "long": 1_000.0},
    )
    sample = encoder.encode(
        context(
            events={
                "motion": BASE_TIME - timedelta(seconds=100),
            },
        ),
    )

    assert sample["motion_recency_short"] < sample["motion_recency_medium"]
    assert sample["motion_recency_medium"] < sample["motion_recency_long"]
    assert sample["motion_recency_short"] == pytest.approx(2**-10)
    assert sample["motion_recency_long"] == pytest.approx(2**-0.1)


def test_future_event_timestamps_are_ignored() -> None:
    encoder = TemporalFeatureEncoder()
    encoded = encoder.encode(
        context(
            events={"motion": BASE_TIME + timedelta(minutes=5)},
        ),
    )

    assert all(
        encoded[f"motion_recency_{horizon}"] == 0
        for horizon in ("short", "medium", "long")
    )


def test_model_normalizes_naive_timestamps_in_encoder_timezone() -> None:
    encoder = TemporalFeatureEncoder(timezone=ZoneInfo("Africa/Johannesburg"))
    naive_model = OnlineActionModel(encoder=encoder, min_effective_support=1)
    utc_model = OnlineActionModel(
        encoder=TemporalFeatureEncoder(timezone=ZoneInfo("Africa/Johannesburg")),
        min_effective_support=1,
    )
    naive = TemporalContext(
        timestamp=datetime(2026, 1, 12, 18, 0),
        categorical_context={"occupancy": "occupied"},
        entity_id="living-room",
    )
    equivalent_utc = context(datetime(2026, 1, 12, 16, 0, tzinfo=UTC))

    naive_model.update(naive, "turn_on", True, provenance="user")
    utc_model.update(equivalent_utc, "turn_on", True, provenance="user")

    assert naive_model.dumps() == utc_model.dumps()
    assert naive_model.to_dict()["actions"]["turn_on"]["last_update"] == (
        "2026-01-12T16:00:00Z"
    )


def test_all_requested_event_families_have_three_recency_horizons() -> None:
    encoder = TemporalFeatureEncoder()
    encoded = encoder.encode(
        context(
            events={
                event: BASE_TIME - timedelta(minutes=5)
                for event in (
                    "motion",
                    "presence",
                    "arrival",
                    "opening",
                    "media",
                    "alarm",
                    "manual",
                )
            },
            dwell={"light_on": 600},
        ),
    )

    for event in (
        "motion",
        "presence",
        "arrival",
        "opening",
        "media",
        "alarm",
        "manual",
    ):
        assert all(
            encoded[f"{event}_recency_{horizon}"] > 0
            for horizon in ("short", "medium", "long")
        )
    assert 0 < encoded["state_dwell_light_on"] < 1
    assert encoded["state_dwell_any"] == encoded["state_dwell_light_on"]


def test_public_holiday_uses_weekend_behavior_but_is_identifiable() -> None:
    encoder = TemporalFeatureEncoder()
    holiday = encoder.encode(
        context(
            datetime(2026, 1, 12, 10, tzinfo=UTC),  # Monday
            holiday_name="Freedom Day",
        ),
    )
    weekday = encoder.encode(context(datetime(2026, 1, 12, 10, tzinfo=UTC)))

    assert holiday.metadata["behavior_day_type"] == "weekend"
    assert holiday.metadata["calendar_day_type"] == "holiday"
    assert holiday.metadata["holiday"] == "Freedom Day"
    assert holiday["calendar_weekend"] == 1
    assert holiday["calendar_weekday"] == 0
    assert holiday["calendar_holiday"] == 1
    assert weekday["calendar_weekday"] == 1


def test_categorical_context_is_stable_and_bounded() -> None:
    encoder = TemporalFeatureEncoder(
        categorical_schema={"room": [f"room-{index}" for index in range(50)]},
        max_categories_per_field=4,
        max_extra_buckets=3,
        max_state_features=2,
    )
    encoded = encoder.encode(
        context(
            categories={
                "room": "room-49",
                **{f"unregistered-{index}": str(index) for index in range(100)},
            },
            dwell={"a": 1, "b": 2, "c": 3},
        ),
    )
    encoded_again = encoder.encode(
        context(
            categories={
                "room": "room-49",
                **{f"unregistered-{index}": str(index) for index in range(100)},
            },
            dwell={"a": 1, "b": 2, "c": 3},
        ),
    )

    assert len(encoded) == len(encoded_again)
    assert dict(encoded) == dict(encoded_again)
    assert len(encoded) <= 5 + 21 + 4 + 3 + 3
    assert (
        sum(
            value
            for name, value in encoded.items()
            if name.startswith("context_extra_bucket_")
        )
        <= 3
    )


def test_sparse_data_refuses_prediction_and_explains_why() -> None:
    model = OnlineActionModel(min_effective_support=3)
    initial = model.predict(context(), "turn_on")
    assert initial.probability == 0.5
    assert initial.not_ready
    assert "unknown_action" in initial.not_ready_reasons

    result = model.update(context(), "turn_on", True, provenance="user")
    assert result.accepted
    sparse = model.predict(context(), "turn_on")
    assert sparse.not_ready
    assert "insufficient_effective_support" in sparse.not_ready_reasons
    assert sparse.confidence == 0


def test_positive_and_negative_online_learning_move_probability() -> None:
    positive = OnlineActionModel(min_effective_support=3)
    train(positive, "turn_on", True, 8)
    positive_prediction = positive.predict(context(), "turn_on")
    assert positive_prediction.probability > 0.75
    assert positive_prediction.ready
    assert positive_prediction.confidence > 0
    assert positive_prediction.top_contributing_features

    negative = OnlineActionModel(min_effective_support=3)
    train(negative, "turn_on", False, 8)
    negative_prediction = negative.predict(context(), "turn_on")
    assert negative_prediction.probability < 0.25
    assert negative_prediction.ready


def test_weighted_update_and_provenance_gate_never_learn_alarm_or_safety_targets() -> (
    None
):
    model = OnlineActionModel(min_effective_support=1)
    assert is_training_eligible("manual")
    assert not is_training_eligible("security_alarm_automation")

    rejected_alarm = model.update(
        context(),
        "turn_on",
        True,
        provenance=ActionProvenance.ALARM,
    )
    rejected_safety = model.update(
        context(),
        "turn_on",
        True,
        provenance="night_safety_override",
    )
    assert not rejected_alarm.accepted
    assert rejected_alarm.reason == "excluded_provenance"
    assert not rejected_safety.accepted
    assert model.action_count == 0

    accepted = model.update(
        context(),
        "turn_on",
        True,
        provenance="manual",
        weight=2.0,
    )
    assert accepted.accepted
    assert model.predict(context(), "turn_on").effective_support == pytest.approx(2)


def test_forgetting_reduces_support_and_staleness_refuses_old_model() -> None:
    model = OnlineActionModel(
        forgetting_half_life=timedelta(hours=1),
        freshness_half_life=timedelta(hours=1),
        stale_after=timedelta(hours=3),
        min_effective_support=1,
    )
    train(model, "turn_on", True, 4)
    fresh = model.predict(context(BASE_TIME + timedelta(minutes=1)), "turn_on")
    old = model.predict(context(BASE_TIME + timedelta(hours=4)), "turn_on")

    assert old.effective_support < fresh.effective_support
    assert old.freshness < fresh.freshness
    assert old.not_ready
    assert "stale_model" in old.not_ready_reasons


def test_feedback_opposite_manual_action_is_strong_negative_and_suppresses_repeat() -> (
    None
):
    model = OnlineActionModel(min_effective_support=1)
    train(model, "turn_on", True, 8)
    proposal_context = context(BASE_TIME + timedelta(hours=1))
    proposal_id = model.register_proposal(proposal_context, "turn_on")

    feedback = model.record_feedback(
        proposal_id,
        observed_action="turn_off",
        provenance=ActionProvenance.MANUAL,
        context=context(BASE_TIME + timedelta(hours=1, minutes=5)),
    )
    assert feedback.accepted
    assert feedback.feedback_kind == "strong_negative_correction"
    assert feedback.effective_weight == 2.5
    assert model.pending_proposal_count == 0

    suppressed = model.predict(
        context(BASE_TIME + timedelta(hours=1, minutes=6)),
        "turn_on",
    )
    other_entity = model.predict(
        context(
            BASE_TIME + timedelta(hours=1, minutes=6),
            entity_id="bedroom",
        ),
        "turn_on",
    )
    assert suppressed.suppressed
    assert suppressed.probability < 0.1
    assert suppressed.confidence == 0
    assert "recent_manual_correction" in suppressed.not_ready_reasons
    assert not other_entity.suppressed


def test_unchanged_feedback_is_weak_positive_and_can_be_settled() -> None:
    model = OnlineActionModel(min_effective_support=1)
    proposal_id = model.record_proposal("turn_on", context())
    result = model.feedback(
        proposal_id,
        outcome=ProposalOutcome.UNCHANGED,
        provenance=ActionProvenance.AUTONOMOUS_OBSERVATION,
        at=BASE_TIME + timedelta(minutes=20),
    )
    assert result.accepted
    assert result.feedback_kind == "weak_positive_unchanged"
    assert result.effective_weight == pytest.approx(0.25)
    assert model.predict(
        context(BASE_TIME + timedelta(minutes=20)),
        "turn_on",
    ).effective_support == pytest.approx(0.25)

    second = model.record_proposal("turn_on", context(BASE_TIME + timedelta(hours=1)))
    settled = model.settle_unchanged(
        BASE_TIME + timedelta(hours=2),
        outcome_window=timedelta(minutes=30),
    )
    assert second in {result.proposal_id for result in settled}
    assert model.pending_proposal_count == 0


def test_pending_and_weight_storage_are_bounded() -> None:
    model = OnlineActionModel(
        max_actions=2,
        max_features=5,
        max_pending_actions=2,
        max_suppressions=2,
    )
    for index in range(5):
        model.record_proposal(
            f"action-{index}",
            context(BASE_TIME + timedelta(minutes=index)),
        )
    assert model.pending_proposal_count == 2

    for index in range(10):
        model.update(
            context(
                BASE_TIME + timedelta(hours=2, minutes=index),
                categories={"occupancy": "occupied", "changing": str(index)},
            ),
            "bounded",
            True,
            provenance="user",
        )
    stats = model.storage_stats()
    assert stats["actions"] <= 2
    assert stats["weights"] <= 5 * 2


def test_persistence_is_json_safe_deterministic_and_preserves_feedback_state() -> None:
    model = OnlineActionModel(min_effective_support=1)
    train(model, "turn_on", True, 4)
    proposal_id = model.register_proposal(context(), "turn_on")
    model.record_feedback(
        proposal_id,
        observed_action="turn_off",
        provenance="manual",
        at=BASE_TIME + timedelta(minutes=2),
    )
    payload = model.dumps()
    json.loads(payload)
    restored = OnlineActionModel.loads(payload)

    assert restored.dumps() == payload
    assert restored.storage_stats() == model.storage_stats()
    original_prediction = model.predict(
        context(BASE_TIME + timedelta(minutes=3)),
        "turn_on",
    )
    restored_prediction = restored.predict(
        context(BASE_TIME + timedelta(minutes=3)),
        "turn_on",
    )
    assert restored_prediction.as_dict() == original_prediction.as_dict()


def test_expired_correction_suppression_is_removed() -> None:
    model = OnlineActionModel(
        min_effective_support=1,
        correction_suppression=timedelta(minutes=10),
    )
    train(model, "turn_on", True, 3)
    proposal_id = model.register_proposal(context(), "turn_on")
    model.record_feedback(
        proposal_id,
        observed_action="turn_off",
        provenance="manual",
        at=BASE_TIME + timedelta(minutes=1),
    )
    assert model.predict(
        context(BASE_TIME + timedelta(minutes=2)),
        "turn_on",
    ).suppressed
    assert not model.predict(
        context(BASE_TIME + timedelta(minutes=12)),
        "turn_on",
    ).suppressed


def persisted_payload(
    *,
    max_actions: int = 4,
    max_features: int = 128,
    max_pending_actions: int = 4,
    max_suppressions: int = 4,
) -> dict[str, object]:
    model = OnlineActionModel(
        min_effective_support=1,
        max_actions=max_actions,
        max_features=max_features,
        max_pending_actions=max_pending_actions,
        max_suppressions=max_suppressions,
    )
    train(model, "turn_on", True, 3)
    return model.to_dict()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("config", "learning_rate"), float("nan")),
        (("config", "regularization"), float("inf")),
        (("actions", "turn_on", "bias"), float("nan")),
        (("actions", "turn_on", "positive_support"), float("inf")),
    ],
)
def test_from_dict_rejects_non_finite_persisted_values(
    path: tuple[str, ...],
    value: float,
) -> None:
    payload = persisted_payload()
    target: dict[str, object] = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError)):
        OnlineActionModel.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("actions", []), ("pending", {}), ("suppressions", {})],
)
def test_from_dict_rejects_wrong_collection_shapes(field: str, value: object) -> None:
    payload = persisted_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        OnlineActionModel.from_dict(payload)


def test_from_dict_rejects_booleans_invalid_timestamps_and_out_of_range_values() -> (
    None
):
    cases = []
    payload = persisted_payload()
    invalid_max_actions = deepcopy(payload)
    invalid_max_actions["config"]["max_actions"] = True  # type: ignore[index]
    cases.append(invalid_max_actions)
    invalid_timestamp = deepcopy(payload)
    invalid_timestamp["actions"]["turn_on"]["last_update"] = "2026-01-12T18:00:00"  # type: ignore[index]
    cases.append(invalid_timestamp)
    invalid_weight = deepcopy(payload)
    invalid_weight["actions"]["turn_on"]["weights"]["local_time_sin"] = 9  # type: ignore[index]
    cases.append(invalid_weight)

    for invalid in cases:
        with pytest.raises((TypeError, ValueError)):
            OnlineActionModel.from_dict(invalid)


def test_from_dict_rejects_oversize_collections_and_feature_maps_before_loading() -> (
    None
):
    oversized_actions = persisted_payload(max_actions=1)
    oversized_actions["actions"]["second"] = deepcopy(  # type: ignore[index]
        oversized_actions["actions"]["turn_on"],  # type: ignore[index]
    )
    oversized_features = persisted_payload(max_features=1)
    oversized_features["actions"]["turn_on"]["weights"]["local_time_cos"] = 0  # type: ignore[index]
    oversized_pending = persisted_payload(max_pending_actions=1)
    oversized_pending["pending"] = [None, None]
    oversized_suppressions = persisted_payload(max_suppressions=1)
    oversized_suppressions["suppressions"] = [None, None]

    for invalid in (
        oversized_actions,
        oversized_features,
        oversized_pending,
        oversized_suppressions,
    ):
        with pytest.raises((TypeError, ValueError)):
            OnlineActionModel.from_dict(invalid)


def test_from_dict_rejects_duplicate_pending_and_suppression_records() -> None:
    pending_model = OnlineActionModel(min_effective_support=1, max_pending_actions=3)
    pending_model.record_proposal("turn_on", context())
    pending_payload = pending_model.to_dict()
    pending_payload["pending"] = [
        deepcopy(pending_payload["pending"][0]),
        deepcopy(pending_payload["pending"][0]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        OnlineActionModel.from_dict(pending_payload)

    suppression_model = OnlineActionModel(min_effective_support=1)
    train(suppression_model, "turn_on", True, 3)
    proposal_id = suppression_model.record_proposal("turn_on", context())
    suppression_model.record_feedback(
        proposal_id,
        observed_action="turn_off",
        provenance="manual",
        at=BASE_TIME + timedelta(minutes=1),
    )
    suppression_payload = suppression_model.to_dict()
    suppression_payload["suppressions"] = [
        deepcopy(suppression_payload["suppressions"][0]),
        deepcopy(suppression_payload["suppressions"][0]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        OnlineActionModel.from_dict(suppression_payload)


def test_from_dict_or_reset_is_explicit_and_does_not_reset_malformed_current_schema() -> (
    None
):
    future = persisted_payload()
    future["schema_version"] = 999
    reset_model, was_reset = OnlineActionModel.from_dict_or_reset(future)
    assert was_reset
    assert reset_model.action_count == 0

    malformed = persisted_payload()
    malformed["actions"] = []
    with pytest.raises((TypeError, ValueError)):
        OnlineActionModel.from_dict_or_reset(malformed)
    with pytest.raises(UnsupportedSchemaError):
        OnlineActionModel.from_dict(future)


def test_feedback_before_proposal_creation_does_not_consume_next_proposal() -> None:
    model = OnlineActionModel(min_effective_support=1)
    rejected = model.record_feedback(
        "proposal-00000001",
        outcome="corrected",
        provenance="manual",
    )
    assert not rejected.accepted
    assert rejected.reason == "unknown_proposal"
    assert model.pending_proposal_count == 0
    assert model.record_proposal("turn_on", context()) == "proposal-00000001"


def test_corrected_outcome_requires_manual_or_human_provenance() -> None:
    model = OnlineActionModel(min_effective_support=1)
    proposal_id = model.record_proposal("turn_on", context())
    rejected = model.record_feedback(
        proposal_id,
        outcome=ProposalOutcome.CORRECTED,
        provenance="autonomous_observation",
    )

    assert not rejected.accepted
    assert rejected.reason == "corrected_requires_manual_provenance"
    assert model.pending_proposal_count == 1
    assert model.action_count == 0


def test_proposal_registration_encodes_context_once() -> None:
    class CountingEncoder(TemporalFeatureEncoder):
        encode_calls = 0

        def encode(self, context: TemporalContext):  # type: ignore[no-untyped-def]
            self.encode_calls += 1
            return super().encode(context)

    encoder = CountingEncoder()
    model = OnlineActionModel(encoder=encoder)
    model.register_proposal(context(), "turn_on")
    assert encoder.encode_calls == 1


def test_small_feature_budget_roundtrips_its_own_pending_proposal() -> None:
    model = OnlineActionModel(max_features=5)
    model.register_proposal(context(), "turn_on")

    restored = OnlineActionModel.from_dict(model.to_dict())

    assert restored.pending_proposal_count == 1
    assert len(restored.to_dict()["pending"][0]["features"]) == 5


def test_runtime_identifier_bounds_match_persistence_bounds() -> None:
    with pytest.raises(ValueError, match="entity_id"):
        context(entity_id="light." + "x" * 123)
    model = OnlineActionModel()
    with pytest.raises(ValueError, match="action"):
        model.predict(context(), "x" * 129)
