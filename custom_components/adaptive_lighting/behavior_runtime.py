# ruff: noqa: ARG002, BLE001, COM812, EM101, EM102, FBT003, PLR0911, PLR0912, PLR0915, TRY003, TRY004, TC006

"""Home Assistant runtime adapter for continuously learned light behavior.

The temporal learner is deliberately kept free of Home Assistant concerns.  This
module owns the boundary: candidate reconciliation, event attribution, storage,
conservative actuation gates, and bounded diagnostics.  A runtime instance owns
the only Store and creates exactly one learner per entity, so one room cannot
silently teach another room's actuator.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from homeassistant.const import (
    EVENT_CALL_SERVICE,
    EVENT_STATE_CHANGED,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import Context, Event, HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .behavior import (
    BehaviorObservation,
    normalize_behavior_event,
    normalize_good_night_routine,
)
from .temporal_model import (
    ActionProvenance,
    OnlineActionModel,
    ProposalOutcome,
    TemporalContext,
    TemporalFeatureEncoder,
)

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
DATA_VERSION = 2
DEFAULT_STORAGE_KEY = "adaptive_lighting_behavior_runtime"
MAX_ENTITIES = 128
MAX_DIAGNOSTIC_DECISIONS = 32
MAX_TRACKED_CONTEXTS = 512
MAX_TRACKED_STATES = 256
MAX_CONTEXT_AGE = timedelta(minutes=10)
DEFAULT_REMOVED_RETENTION = timedelta(days=7)
DEFAULT_EMPTY_DWELL = timedelta(minutes=5)
DEFAULT_OBSERVATION_WINDOW = timedelta(hours=1)
DEFAULT_CORRECTION_WINDOW = timedelta(minutes=15)
DEFAULT_CORRECTION_SUPPRESSION = timedelta(minutes=30)
DEFAULT_MANUAL_ACTION_HOLD = timedelta(minutes=30)
MAX_MANUAL_ACTION_HOLD = timedelta(days=7)

_LIGHT_DOMAINS = frozenset({"light", "switch"})
_BLOCKED_DOMAINS = frozenset(
    {"cover", "door", "garage", "lock", "valve", "window"},
)
_SERVICE_ACTIONS = {"turn_on": "on", "turn_off": "off"}
_STATE_ACTIONS = {STATE_ON: "on", STATE_OFF: "off"}
_GOOD_NIGHT_WORDS = frozenset({"goodnight", "good_night", "good-night"})
_SLEEP_WORDS = frozenset({"sleep", "sleeping", "asleep", "night"})
_SAFETY_WORDS = frozenset(
    {"alarm", "emergency", "evacuation", "fire", "panic", "safety", "security"},
)
_AUTOMATION_WORDS = frozenset(
    {
        "automation",
        "home_assistant",
        "integration",
        "scene",
        "scheduled",
        "script",
        "service",
        "system",
    },
)
_HUMAN_WORDS = frozenset(
    {"human", "manual", "physical", "person", "remote_control", "user", "wall_switch"},
)
_UNKNOWN_WORDS = frozenset({"", "unknown", "none", "null"})
_RAW_EVENT_CONTEXT_FIELDS = frozenset(
    {
        "timestamp",
        "observed_at",
        "at",
        "source",
        "origin",
        "actor",
        "provenance",
        "source_provenance",
        "semantic_routine",
        "routine",
        "routine_name",
        "good_night",
        "entity_state",
        "state",
    },
)
_AREA_CONTEXT_FIELDS = frozenset(
    {
        "occupancy",
        "presence",
        "event_times",
        "events",
        "state_dwell",
        "dwell",
        "categorical_context",
        "categories",
    },
)


class CandidateProvider(Protocol):
    """Return the immutable current inventory of light-like candidates."""

    def __call__(self) -> Iterable[CandidateRecord]:
        """Return the current immutable candidate records."""


class ContextProvider(Protocol):
    """Return the compact current context used by the temporal encoder."""

    def __call__(self) -> Mapping[str, Any]:
        """Return the current compact context mapping."""


class BoolCallback(Protocol):
    """Return whether active actuation is currently allowed."""

    def __call__(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """An immutable, explicitly permissioned actuator candidate."""

    entity_id: str
    area: str = "unknown"
    domain: str = "light"
    supports_brightness: bool = False
    available: bool = True
    explicit_light_switch: bool = False
    manual_hold: bool = False

    @property
    def availability(self) -> bool:
        """Compatibility alias for providers that call this field availability."""
        return self.available

    @property
    def is_light_like_switch(self) -> bool:
        """Return whether a switch has explicit permission to act like a light."""
        return self.domain == "switch" and self.explicit_light_switch


Candidate = CandidateRecord


@dataclass(frozen=True, slots=True)
class BehaviorProposal:
    """A bounded, inspectable decision; it contains no raw HA event data."""

    entity_id: str
    area: str
    domain: str
    action: str
    service: str
    probability: float
    confidence: float
    effective_support: float
    fresh: bool
    active: bool
    reason: str
    proposal_id: str | None = None
    ready: bool = False
    executed: bool = False

    @property
    def service_domain(self) -> str:
        """Return the service domain used by the proposal."""
        return self.domain

    def as_dict(self) -> dict[str, Any]:
        """Return bounded JSON-safe proposal diagnostics."""
        return {
            "entity_id": self.entity_id,
            "area": self.area,
            "domain": self.domain,
            "action": self.action,
            "service": self.service,
            "probability": self.probability,
            "confidence": self.confidence,
            "effective_support": self.effective_support,
            "fresh": self.fresh,
            "active": self.active,
            "ready": self.ready,
            "executed": self.executed,
            "reason": self.reason,
            "proposal_id": self.proposal_id,
        }


@dataclass(frozen=True, slots=True)
class _ServiceAttribution:
    entity_id: str
    action: str
    source: str
    provenance: str
    context_id: str | None
    parent_id: str | None
    at: datetime
    good_night: bool = False
    safety: bool = False


@dataclass(frozen=True, slots=True)
class _TrackedContext:
    context_id: str
    parent_id: str | None
    expires_at: datetime


@dataclass(slots=True)
class _EntityRecord:
    entity_id: str
    area: str
    domain: str
    supports_brightness: bool
    explicit_light_switch: bool
    model: OnlineActionModel
    last_seen_at: datetime
    removed_at: datetime | None = None
    last_access: int = 0
    sample_count: int = 0
    candidate: CandidateRecord | None = None
    last_human_action_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _AttributionDecision:
    source: str
    provenance: str
    context_id: str | None = None
    good_night: bool = False
    safety: bool = False


def _utc(value: datetime | None, fallback: datetime | None = None) -> datetime:
    """Normalize timestamps without ever trusting a naive local timestamp."""
    selected = value or fallback or dt_util.utcnow()
    if selected.tzinfo is None or selected.utcoffset() is None:
        return selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def _text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _words(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    normalized = value.lower()
    for separator in ("-", ".", "+", ":", "/", " "):
        normalized = normalized.replace(separator, "_")
    return {word for word in normalized.split("_") if word}


def _value_words(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set().union(*(_value_words(item) for item in value.values()))
    enum_value = getattr(value, "value", value)
    return _words(enum_value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value) in {"1", "true", "yes", "on", "active", "home"}
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0
    )


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _context_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    identifier = getattr(value, "id", None)
    return identifier if isinstance(identifier, str) and identifier else None


def _parent_context_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        parent = value.get("parent_id")
    else:
        parent = getattr(value, "parent_id", None)
    return parent if isinstance(parent, str) and parent else None


def _user_context(value: Any) -> str | None:
    if isinstance(value, Mapping):
        user_id = value.get("user_id")
    else:
        user_id = getattr(value, "user_id", None)
    return user_id if isinstance(user_id, str) and user_id else None


def _event_data(event: Event | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return event
    data = getattr(event, "data", {})
    return data if isinstance(data, Mapping) else {}


def _event_context(event: Event | Mapping[str, Any]) -> Any:
    data = _event_data(event)
    return data.get("context", getattr(event, "context", None))


def _state_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        raw = value.get("state")
    else:
        raw = getattr(value, "state", value if isinstance(value, str) else None)
    return _text(raw) if isinstance(raw, str) else None


def _event_time(event: Event | Mapping[str, Any], data: Mapping[str, Any]) -> datetime:
    """Use HA's event timestamp, with a deterministic fixture-friendly fallback."""
    explicit_time = data.get("timestamp")
    if isinstance(explicit_time, datetime):
        return _utc(explicit_time)
    event_time = getattr(event, "time_fired", None)
    if isinstance(event_time, datetime):
        return _utc(event_time)
    return _timestamp_from(data, _utc(None))


def _service_targets(data: Mapping[str, Any]) -> tuple[str, ...]:
    service_data = data.get("service_data")
    if not isinstance(service_data, Mapping):
        service_data = data.get("data") if isinstance(data.get("data"), Mapping) else {}
    target = service_data.get("entity_id", data.get("entity_id"))
    if target is None and isinstance(service_data.get("target"), Mapping):
        target = service_data["target"].get("entity_id")
    if target is None and isinstance(data.get("target"), Mapping):
        target = data["target"].get("entity_id")
    if isinstance(target, str):
        raw_targets: Sequence[Any] = (target,)
    elif isinstance(target, Sequence) and not isinstance(target, (bytes, str)):
        raw_targets = target
    else:
        raw_targets = ()
    return tuple(
        item.strip()
        for item in raw_targets
        if isinstance(item, str) and item.strip() and "." in item
    )


def _timestamp_from(mapping: Mapping[str, Any], fallback: datetime) -> datetime:
    value = _first(mapping, "timestamp", "observed_at", "at", "now", default=None)
    if isinstance(value, datetime):
        return _utc(value, fallback)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            pass
    if isinstance(value, str):
        try:
            return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")), fallback)
        except ValueError:
            pass
    return _utc(fallback)


def _explicit_timestamp(mapping: Mapping[str, Any]) -> datetime | None:
    value = _first(mapping, "timestamp", "observed_at", "at", "now", default=None)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = _text(value)
        if normalized in {"true", "yes", "on", "1", "home", "present", "occupied"}:
            return True
        if normalized in {
            "false",
            "no",
            "off",
            "0",
            "away",
            "absent",
            "empty",
            "vacant",
        }:
            return False
    return None


class BehaviorRuntimeAdapter:
    """Continuously learn and conservatively propose light on/off actions."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        candidate_provider: CandidateProvider | Callable[[], Iterable[Any]],
        context_provider: ContextProvider | Callable[[], Mapping[str, Any]],
        actuation_enabled: BoolCallback | Callable[[], bool] | None = None,
        on_change: Callable[[Mapping[str, Any]], Any] | None = None,
        on_accepted_observation: Callable[[BehaviorObservation], Any] | None = None,
        phase_provider: Callable[[], str] | None = None,
        storage_key: str = DEFAULT_STORAGE_KEY,
        max_entities: int = MAX_ENTITIES,
        removed_retention: timedelta | float = DEFAULT_REMOVED_RETENTION,
        observation_window: timedelta | float = DEFAULT_OBSERVATION_WINDOW,
        correction_window: timedelta | float = DEFAULT_CORRECTION_WINDOW,
        correction_suppression: timedelta | float = DEFAULT_CORRECTION_SUPPRESSION,
        manual_action_hold: timedelta | float = DEFAULT_MANUAL_ACTION_HOLD,
        empty_dwell: timedelta | float = DEFAULT_EMPTY_DWELL,
        min_probability: float = 0.90,
        min_confidence: float = 0.75,
        min_effective_support: float = 8.0,
        min_freshness: float = 0.25,
        context_max_age: timedelta | float = MAX_CONTEXT_AGE,
    ) -> None:
        """Configure the HA boundary and its conservative policy defaults."""
        if type(max_entities) is not int or not 1 <= max_entities <= MAX_ENTITIES:
            raise ValueError(f"max_entities must be between 1 and {MAX_ENTITIES}")
        for name, value in (
            ("min_probability", min_probability),
            ("min_confidence", min_confidence),
            ("min_freshness", min_freshness),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if min_effective_support <= 0 or not math.isfinite(
            float(min_effective_support)
        ):
            raise ValueError("min_effective_support must be positive")

        self.hass = hass
        self.storage_key = storage_key
        self._candidate_provider = candidate_provider
        self._context_provider = context_provider
        self._actuation_enabled = actuation_enabled or (lambda: False)
        self._on_change = on_change
        self._on_accepted_observation = on_accepted_observation
        self._phase_provider = phase_provider
        self.max_entities = max_entities
        self.removed_retention = self._duration(removed_retention, "removed_retention")
        self.observation_window = self._duration(
            observation_window, "observation_window"
        )
        self.correction_window = self._duration(correction_window, "correction_window")
        self.correction_suppression = self._duration(
            correction_suppression,
            "correction_suppression",
        )
        self.manual_action_hold = self._duration(
            manual_action_hold,
            "manual_action_hold",
        )
        if self.manual_action_hold > MAX_MANUAL_ACTION_HOLD:
            raise ValueError(
                f"manual_action_hold must not exceed {MAX_MANUAL_ACTION_HOLD.days} days"
            )
        self.empty_dwell = self._duration(empty_dwell, "empty_dwell")
        self.context_max_age = self._duration(context_max_age, "context_max_age")
        self.min_probability = float(min_probability)
        self.min_confidence = float(min_confidence)
        self.min_effective_support = float(min_effective_support)
        self.min_freshness = float(min_freshness)

        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            storage_key,
            private=True,
        )
        self._entities: dict[str, _EntityRecord] = {}
        self._states: dict[str, str] = {}
        self._state_event_times: dict[str, datetime] = {}
        self._attributions: dict[str, _ServiceAttribution] = {}
        self._good_night_contexts: dict[str, datetime] = {}
        self._safety_contexts: dict[str, datetime] = {}
        self._parent_contexts: dict[str, tuple[str | None, datetime]] = {}
        self._own_contexts: dict[str, _TrackedContext] = {}
        self._counter = 0
        self._listeners: list[Callable[[], None]] = []
        self._loaded = False
        self._stopped = False
        self._last_load_reset_reason = "not_loaded"
        self._last_decisions: list[dict[str, Any]] = []
        self._last_rejection_reason: str | None = None
        self._last_change_at: datetime | None = None

    @staticmethod
    def _duration(value: timedelta | float, name: str) -> timedelta:
        seconds = (
            value.total_seconds() if isinstance(value, timedelta) else float(value)
        )
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"{name} must be positive")
        return timedelta(seconds=seconds)

    @property
    def models(self) -> Mapping[str, OnlineActionModel]:
        """Expose the bounded entity-to-model mapping for diagnostics/tests."""
        return {entity_id: record.model for entity_id, record in self._entities.items()}

    @property
    def phase(self) -> str:
        """Return the integration-owned rollout phase, when supplied."""
        if self._phase_provider is None:
            return "shadow"
        try:
            value = self._phase_provider()
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior phase provider failed")
            return "unknown"
        return _text(value) or "unknown"

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return bounded JSON-safe runtime diagnostics."""
        pending = 0
        suppressions = 0
        sample_counts: dict[str, int] = {}
        for entity_id, record in sorted(self._entities.items()):
            stats = record.model.storage_stats()
            pending += stats["pending_proposals"]
            suppressions += stats["suppressions"]
            sample_counts[entity_id] = min(1_000_000, record.sample_count)
        recent_corrections = sum(
            item.get("decision") == "corrected" for item in self._last_decisions
        )
        accepted_observations = sum(sample_counts.values())
        return {
            "phase": self.phase,
            "models": len(self._entities),
            "candidates": sum(
                record.candidate is not None for record in self._entities.values()
            ),
            "available_candidates": sum(
                record.candidate is not None and record.candidate.available
                for record in self._entities.values()
            ),
            "sample_counts": sample_counts,
            "accepted_observations": accepted_observations,
            "behavior_accepted_count": accepted_observations,
            "last_decisions": [dict(item) for item in self._last_decisions],
            "pending": pending,
            "corrections": recent_corrections,
            "corrections_scope": "recent_bounded",
            "active_suppressions": suppressions,
            "suppression": suppressions,
            "last_load_reset_reason": self._last_load_reset_reason,
            "last_rejection_reason": self._last_rejection_reason,
            "last_change_at": (
                self._last_change_at.isoformat()
                if self._last_change_at is not None
                else None
            ),
        }

    @property
    def summary(self) -> Mapping[str, Any]:
        """Compatibility alias for diagnostics."""
        return self.diagnostics

    async def async_start(self) -> None:
        """Load bounded state and register HA listeners without setup failures."""
        if self._loaded and not self._stopped:
            return
        self._stopped = False
        self._last_load_reset_reason = "empty"
        try:
            stored = await self._store.async_load()
            if stored is not None:
                self._restore(stored)
                self._last_load_reset_reason = "loaded"
        except Exception as error:  # corrupt local state must never break HA setup
            reason = (
                "unsupported_schema"
                if "schema" in str(error).lower()
                else "corrupt_state"
            )
            self._reset_load_reason(reason, error)
            self._entities.clear()
        self._reconcile_candidates(_utc(None))
        self._register_listeners()
        self._loaded = True
        await self._persist()
        self._notify()

    async def async_setup(self) -> None:
        """Alias used by integrations that call adapters ``async_setup``."""
        await self.async_start()

    async def async_stop(self) -> None:
        """Persist state and remove every listener registered by this adapter."""
        if self._stopped and not self._loaded:
            return
        self._stopped = True
        for remove in self._listeners:
            try:
                remove()
            except Exception:
                _LOGGER.exception("Adaptive Lighting behavior listener removal failed")
        self._listeners.clear()
        await self._persist(force=True)
        self._loaded = False

    async def async_unload(self) -> None:
        """Alias used by Home Assistant unload paths."""
        await self.async_stop()

    def _reset_load_reason(self, reason: str, error: Exception) -> None:
        self._last_load_reset_reason = reason
        _LOGGER.warning(
            "Adaptive Lighting behavior state reset safely (%s): %s", reason, error
        )

    def _register_listeners(self) -> None:
        if self._listeners:
            return
        self._listeners.extend(
            (
                self.hass.bus.async_listen(
                    EVENT_CALL_SERVICE, self._async_call_service
                ),
                self.hass.bus.async_listen(
                    EVENT_STATE_CHANGED, self._async_state_changed
                ),
                async_track_time_interval(
                    self.hass,
                    self._async_interval,
                    timedelta(minutes=1),
                ),
            ),
        )

    async def _async_interval(self, now: datetime) -> None:
        if self._stopped:
            return
        at = _utc(now)
        changed = self._reconcile_candidates(at)
        for record in self._entities.values():
            if record.model.settle_unchanged(
                at,
                outcome_window=self.observation_window,
            ):
                changed = True
        if changed:
            await self._persist()
            if self._stopped:
                return
            self._notify()
        if self._stopped:
            return
        await self.async_evaluate(now=at)

    async def _async_call_service(self, event: Event) -> None:
        if self._stopped:
            return
        await self.async_process_call_service(event)

    async def _async_state_changed(self, event: Event) -> None:
        if self._stopped:
            return
        await self.async_process_state_change(event)

    async def async_process_call_service(
        self,
        event: Event | Mapping[str, Any],
    ) -> None:
        """Attribute light service calls; never infer actuator permission from them."""
        if self._stopped:
            return
        data = _event_data(event)
        domain = _text(data.get("domain"))
        service = _text(data.get("service"))
        context = _event_context(event)
        context_id = _context_id(context) or _context_id(data.get("context_id"))
        parent_id = _parent_context_id(context)
        at = _event_time(event, data)
        self._remember_parent_context(context_id, parent_id, at)
        service_data = data.get("service_data")
        if not isinstance(service_data, Mapping):
            service_data = (
                data.get("data") if isinstance(data.get("data"), Mapping) else {}
            )
        labels = tuple(
            value
            for value in (
                data.get("source"),
                data.get("origin"),
                data.get("actor"),
                data.get("provenance"),
                service_data.get("source"),
                service_data.get("origin"),
                service_data.get("actor"),
                service_data.get("provenance"),
            )
            if isinstance(value, str)
        )
        entity_targets = _service_targets(data)
        routine = _first(
            data,
            "semantic_routine",
            "routine",
            "routine_name",
            default=_first(service_data, "semantic_routine", "routine", default=""),
        )
        scene_target = any(
            _text(entity).split(".", 1)[-1] in _GOOD_NIGHT_WORDS
            for entity in entity_targets
            if entity.startswith("scene.")
        )
        safety = self._is_safety(
            " ".join(labels),
            service,
            domain,
        ) or self._context_is_safety(context_id, parent_id, at)
        good_night = not safety and (
            self._is_good_night(routine)
            or scene_target
            or self._context_is_good_night(context_id, parent_id, at)
        )
        if safety:
            if context_id is not None:
                self._good_night_contexts.pop(context_id, None)
            self._mark_scoped_context(self._safety_contexts, context_id, at)
        elif good_night:
            self._mark_scoped_context(self._good_night_contexts, context_id, at)

        # An autonomous call receives a context id before async_call is issued;
        # its state event is therefore ignored rather than mistaken for a user.
        if context_id is not None and self._is_own_context(context_id, parent_id, at):
            return

        if domain not in _LIGHT_DOMAINS:
            if good_night:
                for entity_id in entity_targets:
                    record = self._entities.get(entity_id)
                    if record is not None:
                        self._attributions[entity_id] = _ServiceAttribution(
                            entity_id,
                            "off",
                            "automation",
                            "automation",
                            context_id,
                            parent_id,
                            at,
                            True,
                            False,
                        )
            return
        action = _SERVICE_ACTIONS.get(service)
        if action is None:
            return

        user_id = _user_context(context)
        source, provenance = self._classify(
            labels,
            user_id=user_id,
            good_night=good_night,
            parent_id=parent_id,
            service_call=True,
        )
        if safety:
            source = "automation_alarm"
            provenance = "alarm"
        for entity_id in entity_targets:
            record = self._entities.get(entity_id)
            if record is None:
                continue
            attribution = _ServiceAttribution(
                entity_id,
                action,
                source,
                provenance,
                context_id,
                parent_id,
                at,
                good_night,
                safety,
            )
            self._attributions[entity_id] = attribution

    async def async_process_state_change(
        self,
        event: Event | Mapping[str, Any],
    ) -> None:
        """Attribute state changes and learn human on/off actions continuously."""
        if self._stopped:
            return
        data = _event_data(event)
        entity_id = data.get("entity_id")
        if not isinstance(entity_id, str):
            return
        new_state = data.get("new_state")
        if isinstance(new_state, Mapping):
            state_context = new_state.get("context")
        else:
            state_context = getattr(new_state, "context", None)
        state = _state_value(new_state)
        if state is None:
            return
        action = _STATE_ACTIONS.get(state)
        record = self._entities.get(entity_id)
        if record is None or record.candidate is None or action is None:
            return
        at = _event_time(event, data)
        previous_event_at = self._state_event_times.get(entity_id)
        if previous_event_at is not None and at < previous_event_at:
            return
        self._state_event_times[entity_id] = at
        self._states[entity_id] = state
        self._trim_states()
        old_action = _STATE_ACTIONS.get(_state_value(data.get("old_state")) or "")
        if old_action == action:
            return
        state_context = state_context or _event_context(event)
        context_id = _context_id(state_context)
        parent_id = _parent_context_id(state_context)
        if self._is_own_context(context_id, parent_id, at):
            return

        attribution = self._attributions.get(entity_id)
        decision = self._attribute_state(
            entity_id,
            action,
            state_context,
            context_id,
            parent_id,
            at,
            attribution,
            data,
            new_state,
        )
        if decision is None:
            self._last_rejection_reason = "unattributed_state_change"
            return
        learned = await self._learn_from_event(
            record,
            action,
            at,
            source=decision.source,
            provenance=decision.provenance,
            semantic_routine="good_night" if decision.good_night else "ambient",
            context_id=decision.context_id,
            candidate=record.candidate,
            state=state,
            raw_context=data,
        )
        if self._stopped:
            return
        if not learned:
            self._attributions.pop(entity_id, None)
            return
        await self._record_pending_feedback(record, action, at, decision.provenance)
        if self._stopped:
            return
        self._attributions.pop(entity_id, None)
        await self._persist()
        if self._stopped:
            return
        self._notify()

    async def async_record_good_night(
        self,
        actions: Sequence[Any] | Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        context_id: str | None = None,
    ) -> None:
        """Learn explicit Good Night targets through the sole normalizer."""
        if self._stopped:
            return
        if self._context_has_safety(context) or self._is_safety(
            context.get("source"),
            context.get("provenance"),
        ):
            self._last_rejection_reason = "safety_context"
            return
        normalized = normalize_good_night_routine(actions, context)
        at = _timestamp_from(context, _utc(None))
        self._mark_scoped_context(self._good_night_contexts, context_id, at)
        for observation in normalized:
            if self._stopped:
                return
            record = self._entities.get(observation.entity_id)
            if record is None:
                continue
            await self._learn_observation(record, observation, context)
        await self._persist()
        if self._stopped:
            return
        self._notify()

    def _attribute_state(
        self,
        entity_id: str,
        action: str,
        state_context: Any,
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
        attribution: _ServiceAttribution | None,
        event_data: Mapping[str, Any],
        new_state: Any,
    ) -> _AttributionDecision | None:
        if self._context_is_safety(context_id, parent_id, at):
            return None
        if self._context_is_good_night(context_id, parent_id, at):
            return _AttributionDecision(
                "automation",
                "automation",
                context_id,
                True,
            )
        if attribution is not None:
            belongs_to_attribution = (
                attribution.context_id is None
                or self._context_matches(
                    context_id,
                    parent_id,
                    attribution.context_id,
                )
            )
            if not belongs_to_attribution:
                attribution = None
        if attribution is not None:
            if attribution.action != action:
                return None
            if attribution.safety:
                return None
            if attribution.good_night or attribution.source in {
                "human",
                "physical",
            }:
                return _AttributionDecision(
                    attribution.source,
                    attribution.provenance,
                    attribution.context_id,
                    attribution.good_night,
                )
            return None

        attributes = (
            new_state.get("attributes", {})
            if isinstance(new_state, Mapping)
            else getattr(new_state, "attributes", {})
        )
        labels = tuple(
            value
            for value in (
                event_data.get("source"),
                event_data.get("origin"),
                event_data.get("actor"),
                event_data.get("provenance"),
                attributes.get("source") if isinstance(attributes, Mapping) else None,
            )
            if isinstance(value, str)
        )
        if self._is_safety(" ".join(labels)):
            return None
        user_id = _user_context(state_context)
        source, provenance = self._classify(labels, user_id=user_id)
        if source == "automation":
            return None
        if (
            context_id is not None
            and parent_id is not None
            and user_id is None
            and not labels
        ):
            # A parent-bearing, unmatched state change is most safely treated
            # as an automation.  A bare context id is common for physical
            # Zigbee/device updates and is therefore retained as physical.
            return None
        return _AttributionDecision(source, provenance, context_id)

    @staticmethod
    def _classify(
        labels: Iterable[str],
        *,
        user_id: str | None = None,
        good_night: bool = False,
        parent_id: str | None = None,
        service_call: bool = False,
    ) -> tuple[str, str]:
        words = set().union(*(_words(label) for label in labels))
        if words.intersection(_SAFETY_WORDS):
            return "automation", "safety"
        if words.intersection(_AUTOMATION_WORDS):
            return "automation", "automation"
        if good_night:
            return "automation", "automation"
        if user_id or words.intersection(_HUMAN_WORDS):
            return "human", "user" if user_id else "physical"
        if service_call and parent_id is not None:
            return "automation", "automation"
        return "physical", "physical"

    @staticmethod
    def _is_safety(*values: Any) -> bool:
        words = set().union(*(_words(value) for value in values))
        return bool(words.intersection(_SAFETY_WORDS))

    @staticmethod
    def _context_has_safety(context: Mapping[str, Any]) -> bool:
        safety_words = _SAFETY_WORDS | {"problem"}
        for key in ("safety", "emergency", "alarm"):
            value = context.get(key)
            if _truthy(value) or _value_words(value).intersection(safety_words):
                return True
        for key in (
            "security",
            "security_state",
            "security_classification",
            "classified_security",
        ):
            if _value_words(context.get(key)).intersection(safety_words):
                return True
        return False

    @staticmethod
    def _is_good_night(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = _text(value).replace("-", "_").replace(" ", "_")
        return normalized in {word.replace("-", "_") for word in _GOOD_NIGHT_WORDS}

    def _context_matches(
        self,
        context_id: str | None,
        parent_id: str | None,
        expected_id: str,
    ) -> bool:
        return expected_id in {context_id, parent_id}

    def _remember_parent_context(
        self,
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
    ) -> None:
        if context_id is None or context_id == parent_id:
            return
        self._parent_contexts[context_id] = (
            parent_id,
            at + self.correction_window,
        )
        if len(self._parent_contexts) > MAX_TRACKED_CONTEXTS:
            oldest = sorted(self._parent_contexts)[
                : len(self._parent_contexts) - MAX_TRACKED_CONTEXTS
            ]
            for stale_context_id in oldest:
                self._parent_contexts.pop(stale_context_id, None)

    def _context_is_good_night(
        self,
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
    ) -> bool:
        """Walk the bounded HA context chain to preserve Good Night intent."""
        return self._context_is_marked(
            self._good_night_contexts,
            context_id,
            parent_id,
            at,
        )

    def _context_is_safety(
        self,
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
    ) -> bool:
        return self._context_is_marked(
            self._safety_contexts,
            context_id,
            parent_id,
            at,
        )

    def _context_is_marked(
        self,
        contexts: dict[str, datetime],
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
    ) -> bool:
        for child_context_id, (_, expires_at) in list(self._parent_contexts.items()):
            if expires_at <= at:
                del self._parent_contexts[child_context_id]
        for marked_context_id, expires_at in list(contexts.items()):
            if expires_at <= at:
                del contexts[marked_context_id]

        frontier = [context_id, parent_id]
        visited: set[str] = set()
        while frontier:
            current = frontier.pop(0)
            if current is None or current in visited:
                continue
            visited.add(current)
            if current in contexts:
                return True
            remembered_parent = self._parent_contexts.get(current)
            if remembered_parent is not None:
                frontier.append(remembered_parent[0])
        return False

    def _mark_scoped_context(
        self,
        contexts: dict[str, datetime],
        context_id: str | None,
        at: datetime,
    ) -> None:
        if context_id is None:
            return
        contexts[context_id] = at + self.correction_window
        while len(contexts) > MAX_TRACKED_CONTEXTS:
            oldest = min(contexts, key=lambda item: (contexts[item], item))
            del contexts[oldest]

    def _is_own_context(
        self,
        context_id: str | None,
        parent_id: str | None,
        at: datetime,
    ) -> bool:
        self._prune_contexts(at)
        return any(
            tracked.context_id in {context_id, parent_id}
            for tracked in self._own_contexts.values()
        )

    def _prune_contexts(self, at: datetime) -> None:
        for context_id, tracked in list(self._own_contexts.items()):
            if tracked.expires_at <= at:
                del self._own_contexts[context_id]
        if len(self._own_contexts) > MAX_TRACKED_CONTEXTS:
            for context_id in sorted(self._own_contexts)[
                : len(self._own_contexts) - MAX_TRACKED_CONTEXTS
            ]:
                self._own_contexts.pop(context_id, None)

    def _trim_states(self) -> None:
        if len(self._states) <= MAX_TRACKED_STATES:
            return
        for entity_id in sorted(self._states)[: len(self._states) - MAX_TRACKED_STATES]:
            self._states.pop(entity_id, None)
            self._state_event_times.pop(entity_id, None)

    async def _learn_from_event(
        self,
        record: _EntityRecord,
        action: str,
        at: datetime,
        *,
        source: str,
        provenance: str,
        semantic_routine: str,
        context_id: str | None,
        candidate: CandidateRecord | None,
        state: str | None = None,
        raw_context: Mapping[str, Any] | None = None,
    ) -> bool:
        if candidate is None:
            return False
        provider_context = self._context_for_area(
            await self._get_context(),
            record.area,
        )
        if self._stopped:
            return False
        if self._context_has_safety(provider_context):
            self._last_rejection_reason = "safety_context"
            return False
        mapping = {
            key: value
            for key, value in provider_context.items()
            if isinstance(key, str) and not key.startswith("_")
        }
        if raw_context is not None:
            mapping.update(
                {
                    key: raw_context[key]
                    for key in _RAW_EVENT_CONTEXT_FIELDS
                    if key in raw_context
                },
            )
        mapping.update(
            {
                "entity_id": record.entity_id,
                "area": candidate.area,
                "domain": candidate.domain,
                "action": action,
                "timestamp": at,
                "available": candidate.available,
                "supports_brightness": candidate.supports_brightness,
                "explicit_light_switch": candidate.explicit_light_switch,
                "source": source,
                "provenance": provenance,
                "semantic_routine": semantic_routine,
                "entity_state": state or action,
            },
        )
        result = normalize_behavior_event(mapping)
        if not result.accepted or result.observation is None:
            self._last_rejection_reason = result.reason
            return False
        learned = await self._learn_observation(record, result.observation, mapping)
        if (
            learned
            and semantic_routine != "good_night"
            and _words(f"{source} {provenance}").intersection(_HUMAN_WORDS)
        ):
            observed_at = _utc(result.observation.timestamp)
            if (
                record.last_human_action_at is None
                or observed_at > record.last_human_action_at
            ):
                record.last_human_action_at = observed_at
        return learned

    async def _learn_observation(
        self,
        record: _EntityRecord,
        observation: BehaviorObservation,
        mapping: Mapping[str, Any],
    ) -> bool:
        mapping = self._context_for_area(mapping, record.area)
        temporal = self._temporal_context(
            mapping, record.entity_id, record.area, observation.action
        )
        provenance: ActionProvenance | str = observation.provenance
        # Every human action trains both sides.  This makes an on event teach
        # both "on is desirable" and "off is undesirable", rather than making
        # the two binary decisions drift independently.
        positive = record.model.update(
            temporal,
            observation.action,
            True,
            provenance=provenance,
        )
        opposite = "off" if observation.action == "on" else "on"
        negative = record.model.update(
            temporal,
            opposite,
            False,
            provenance=provenance,
        )
        if positive.accepted or negative.accepted:
            record.sample_count = min(1_000_000, record.sample_count + 1)
            record.last_access = self._next_access()
            self._last_change_at = observation.timestamp
            self._last_rejection_reason = None
            await self._notify_accepted_observation(observation)
            return True
        return False

    async def _notify_accepted_observation(
        self,
        observation: BehaviorObservation,
    ) -> None:
        """Notify commissioning once, after both binary model updates succeed."""
        if self._stopped or self._on_accepted_observation is None:
            return
        try:
            result = self._on_accepted_observation(observation)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # A commissioning observer must not make behavior learning or HA
            # event processing fail after the model has accepted the sample.
            _LOGGER.exception("Adaptive Lighting accepted-observation callback failed")

    def _temporal_context(
        self,
        mapping: Mapping[str, Any],
        entity_id: str,
        area: str,
        action: str | None = None,
    ) -> TemporalContext:
        now = _timestamp_from(mapping, _utc(None))
        raw_events = _first(mapping, "event_times", "events", default={})
        event_times = dict(raw_events) if isinstance(raw_events, Mapping) else {}
        raw_dwell = _first(mapping, "state_dwell", "dwell", default={})
        state_dwell = dict(raw_dwell) if isinstance(raw_dwell, Mapping) else {}
        if "occupancy_dwell_seconds" in mapping:
            state_dwell.setdefault("occupancy", mapping["occupancy_dwell_seconds"])
        if "presence_dwell_seconds" in mapping:
            state_dwell.setdefault("presence", mapping["presence_dwell_seconds"])
        categories_raw = _first(
            mapping, "categorical_context", "categories", default={}
        )
        categories = (
            {
                str(key): str(value)
                for key, value in categories_raw.items()
                if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            }
            if isinstance(categories_raw, Mapping)
            else {}
        )
        occupancy = _first(mapping, "occupancy", "presence", default="unknown")
        home_away = _first(
            mapping, "home_away", "home_state", "household_state", default="unknown"
        )
        routine = _first(mapping, "semantic_routine", "routine", default="ambient")
        media = _first(mapping, "media", "media_type", "media_state", default="idle")
        categories.update(
            {
                "occupancy": str(occupancy),
                "media": str(media),
                "mode": "sleep" if self._is_good_night(routine) else str(home_away),
                "lighting": action or str(mapping.get("lighting", "unknown")),
                "routine": str(routine),
                "home_away": str(home_away),
                "area": area,
                "daylight": str(
                    _first(
                        mapping,
                        "daylight_band",
                        "daylight",
                        "solar_band",
                        default="unknown",
                    )
                ),
                "weather": str(
                    _first(mapping, "weather_band", "weather", default="unknown")
                ),
            },
        )
        holiday_name = _first(
            mapping, "holiday_name", "holiday_provenance", default=None
        )
        return TemporalContext(
            timestamp=now,
            event_times=event_times,
            state_dwell=state_dwell,
            categorical_context=categories,
            entity_id=entity_id,
            is_holiday=_truthy(
                _first(
                    mapping, "is_holiday", "holiday", "public_holiday", default=False
                )
            ),
            holiday_name=holiday_name if isinstance(holiday_name, str) else None,
        )

    async def _get_context(self) -> Mapping[str, Any]:
        try:
            value = self._context_provider()
            if inspect.isawaitable(value):
                value = await cast(Awaitable[Mapping[str, Any]], value)
            return value if isinstance(value, Mapping) else {}
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior context provider failed")
            return {}

    @staticmethod
    def _context_for_area(
        context: Mapping[str, Any],
        area: str,
    ) -> dict[str, Any]:
        merged = {
            key: value
            for key, value in context.items()
            if isinstance(key, str) and key not in {"area_context", "areas"}
        }
        raw_areas = _first(context, "area_context", "areas", default={})
        if not isinstance(raw_areas, Mapping):
            return merged
        bounded_keys = sorted(key for key in raw_areas if isinstance(key, str))[
            :MAX_ENTITIES
        ]
        area_key = next((key for key in bounded_keys if _text(key) == area), None)
        if area_key is None:
            return merged
        area_context = raw_areas.get(area_key)
        if not isinstance(area_context, Mapping):
            return merged

        # Area occupancy is authoritative for the actuator's room, but nested
        # temporal context is additive.  Replacing these maps would discard
        # house-wide arrival/media/security evidence whenever an area supplied
        # even one local motion or dwell value.
        for key in ("occupancy", "presence"):
            if key in area_context:
                merged[key] = area_context[key]
        for canonical, alias in (
            ("event_times", "events"),
            ("state_dwell", "dwell"),
            ("categorical_context", "categories"),
        ):
            if canonical not in area_context and alias not in area_context:
                continue
            global_values = _first(merged, canonical, alias, default={})
            area_values = _first(area_context, canonical, alias, default={})
            combined = dict(global_values) if isinstance(global_values, Mapping) else {}
            if isinstance(area_values, Mapping):
                combined.update(area_values)
            merged[canonical] = combined
            merged.pop(alias, None)
        return merged

    async def _record_pending_feedback(
        self,
        record: _EntityRecord,
        observed_action: str,
        at: datetime,
        provenance: str,
    ) -> None:
        if not _words(provenance).intersection(_HUMAN_WORDS):
            return
        context_mapping = self._context_for_area(
            await self._get_context(),
            record.area,
        )
        if self._stopped:
            return
        context_mapping["timestamp"] = at
        temporal = self._temporal_context(
            context_mapping,
            record.entity_id,
            record.area,
            observed_action,
        )
        pending = record.model.pending_proposal_ids[:1]
        pending_actions = {
            item["proposal_id"]: item["action"]
            for item in record.model.to_dict()["pending"]
        }
        for proposal_id in pending:
            pending_action = pending_actions.get(proposal_id)
            opposite = {"on": "off", "off": "on"}.get(pending_action)
            outcome = (
                ProposalOutcome.CORRECTED
                if opposite == observed_action
                else ProposalOutcome.ACCEPTED
            )
            result = record.model.record_feedback(
                proposal_id,
                outcome=outcome,
                observed_action=observed_action,
                provenance=provenance,
                context=temporal,
                at=at,
            )
            if result.feedback_kind in {
                "strong_negative_correction",
                "negative_correction",
            }:
                self._last_decisions.append(
                    {
                        "entity_id": record.entity_id,
                        "action": result.action,
                        "decision": "corrected",
                        "reason": (
                            "quick_manual_reversal"
                            if result.feedback_kind == "strong_negative_correction"
                            else "late_manual_reversal"
                        ),
                    },
                )
                self._last_decisions = self._last_decisions[-MAX_DIAGNOSTIC_DECISIONS:]

    def _new_model(self) -> OnlineActionModel:
        encoder = TemporalFeatureEncoder(
            categorical_schema={
                "occupancy": ("empty", "occupied", "unknown"),
                "media": ("idle", "playing", "paused", "unknown"),
                "mode": ("home", "away", "sleep", "unknown"),
                "lighting": ("off", "on", "unknown"),
            },
        )
        return OnlineActionModel(
            encoder=encoder,
            min_effective_support=self.min_effective_support,
            max_actions=2,
            max_pending_actions=1,
            correction_window=self.correction_window,
            correction_suppression=self.correction_suppression,
            max_suppressions=32,
        )

    def _candidate(self, value: Any) -> CandidateRecord | None:
        if isinstance(value, CandidateRecord):
            candidate = value
        elif isinstance(value, Mapping):
            entity_id = value.get("entity_id", value.get("entity"))
            domain = value.get("domain")
            if not isinstance(entity_id, str):
                return None
            inferred_domain = entity_id.split(".", 1)[0]
            candidate = CandidateRecord(
                entity_id=entity_id,
                area=str(value.get("area", value.get("zone", "unknown"))),
                domain=str(domain or inferred_domain),
                supports_brightness=_truthy(
                    value.get(
                        "supports_brightness", value.get("brightness_supported", False)
                    ),
                ),
                available=_truthy(
                    value.get("available", value.get("availability", True))
                ),
                explicit_light_switch=_truthy(
                    value.get(
                        "explicit_light_switch",
                        value.get("is_light_switch", value.get("light_switch", False)),
                    ),
                ),
                manual_hold=_truthy(
                    value.get(
                        "manual_hold",
                        value.get(
                            "manual_control",
                            value.get("manual_controlled", value.get("hold", False)),
                        ),
                    ),
                ),
            )
        else:
            return None
        entity_id = _text(candidate.entity_id)
        domain = _text(candidate.domain)
        if not entity_id or "." not in entity_id or domain in _BLOCKED_DOMAINS:
            return None
        if domain not in _LIGHT_DOMAINS:
            return None
        if entity_id.split(".", 1)[0] != domain:
            return None
        if domain == "switch" and not candidate.explicit_light_switch:
            return None
        return CandidateRecord(
            entity_id=entity_id,
            area=_text(candidate.area) or "unknown",
            domain=domain,
            supports_brightness=bool(candidate.supports_brightness),
            available=bool(candidate.available),
            explicit_light_switch=bool(candidate.explicit_light_switch),
            manual_hold=bool(candidate.manual_hold),
        )

    def _reconcile_candidates(self, at: datetime) -> bool:
        changed = False
        try:
            provided = self._candidate_provider()
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior candidate provider failed")
            self._last_rejection_reason = "candidate_provider_error"
            return False
        if inspect.isawaitable(provided):
            # Candidate providers are intentionally narrow and synchronous.  A
            # coroutine is ignored here rather than blocking HA's event loop.
            self._last_rejection_reason = "async_candidate_provider_unsupported"
            return False
        current = {
            candidate.entity_id: candidate
            for candidate in (self._candidate(item) for item in provided)
            if candidate is not None
        }
        for entity_id, record in list(self._entities.items()):
            if entity_id not in current:
                if record.candidate is not None:
                    record.candidate = None
                    record.removed_at = at
                    self._states.pop(entity_id, None)
                    self._state_event_times.pop(entity_id, None)
                    changed = True
                elif (
                    record.removed_at is not None
                    and at - record.removed_at >= self.removed_retention
                ):
                    del self._entities[entity_id]
                    changed = True

        for entity_id, candidate in sorted(current.items()):
            record = self._entities.get(entity_id)
            if record is None:
                self._entities[entity_id] = _EntityRecord(
                    entity_id,
                    candidate.area,
                    candidate.domain,
                    candidate.supports_brightness,
                    candidate.explicit_light_switch,
                    self._new_model(),
                    at,
                    last_access=self._next_access(),
                    candidate=candidate,
                )
                self._seed_candidate_state(entity_id)
                changed = True
                continue
            was_removed = record.candidate is None
            if record.area != candidate.area:
                # Area is part of the behavioral context.  Resetting here is
                # safer than allowing observations from the old room to vote.
                record.model = self._new_model()
                record.sample_count = 0
                changed = True
            if record.candidate != candidate or record.removed_at is not None:
                changed = True
            record.area = candidate.area
            record.domain = candidate.domain
            record.supports_brightness = candidate.supports_brightness
            record.explicit_light_switch = candidate.explicit_light_switch
            record.candidate = candidate
            record.removed_at = None
            record.last_seen_at = at
            record.last_access = self._next_access()
            if was_removed:
                self._seed_candidate_state(entity_id)

        while len(self._entities) > self.max_entities:
            victim = min(
                self._entities.values(),
                key=lambda record: (
                    record.candidate is not None,
                    record.last_access,
                    record.last_seen_at,
                    record.entity_id,
                ),
            )
            del self._entities[victim.entity_id]
            self._states.pop(victim.entity_id, None)
            self._state_event_times.pop(victim.entity_id, None)
            changed = True
        self._prune_attributions(at)
        return changed

    def _prune_attributions(self, at: datetime) -> None:
        cutoff = at - max(self.correction_window, timedelta(minutes=15))
        for entity_id, attribution in list(self._attributions.items()):
            if attribution.at < cutoff:
                del self._attributions[entity_id]

    def _next_access(self) -> int:
        self._counter = min(1_000_000_000, self._counter + 1)
        return self._counter

    def _is_actuation_enabled(self) -> bool:
        """Read the integration-owned phase/shadow gate defensively."""
        try:
            return bool(self._actuation_enabled())
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior actuation gate failed")
            return False

    def _restore(self, stored: Mapping[str, Any]) -> None:
        if type(stored) is not dict:
            raise ValueError("invalid behavior runtime envelope")
        if stored["data_version"] != DATA_VERSION:
            raise ValueError("unsupported behavior runtime schema")
        if set(stored) != {"data_version", "counter", "entities"}:
            raise ValueError("invalid behavior runtime envelope")
        counter = stored["counter"]
        entities = stored["entities"]
        if type(counter) is not int or not 0 <= counter <= 1_000_000_000:
            raise ValueError("invalid behavior runtime counter")
        if type(entities) is not dict or len(entities) > self.max_entities:
            raise ValueError("behavior runtime entity bounds exceeded")
        restored: dict[str, _EntityRecord] = {}
        for entity_id, raw in entities.items():
            if not isinstance(entity_id, str) or type(raw) is not dict:
                raise ValueError("invalid behavior runtime entity")
            expected = {
                "entity_id",
                "area",
                "domain",
                "supports_brightness",
                "explicit_light_switch",
                "last_seen_at",
                "removed_at",
                "last_access",
                "sample_count",
                "last_human_action_at",
                "model",
            }
            if set(raw) != expected or raw["entity_id"] != entity_id:
                raise ValueError("invalid behavior runtime entity fields")
            if not isinstance(raw["area"], str) or not isinstance(raw["domain"], str):
                raise ValueError("invalid behavior runtime candidate")
            if raw["domain"] not in {"light", "switch"}:
                raise ValueError("invalid behavior runtime domain")
            if entity_id.split(".", 1)[0] != raw["domain"]:
                raise ValueError("behavior runtime entity domain mismatch")
            if raw["domain"] == "switch" and raw["explicit_light_switch"] is not True:
                raise ValueError("switch is not explicitly light-like")
            if (
                type(raw["supports_brightness"]) is not bool
                or type(raw["explicit_light_switch"]) is not bool
            ):
                raise ValueError("invalid behavior runtime capability")
            last_seen = self._parse_stored_time(raw["last_seen_at"])
            removed = (
                None
                if raw["removed_at"] is None
                else self._parse_stored_time(raw["removed_at"])
            )
            last_human_action_at = (
                None
                if raw["last_human_action_at"] is None
                else self._parse_stored_time(raw["last_human_action_at"])
            )
            if type(raw["last_access"]) is not int or raw["last_access"] < 0:
                raise ValueError("invalid behavior runtime access")
            if (
                type(raw["sample_count"]) is not int
                or not 0 <= raw["sample_count"] <= 1_000_000
            ):
                raise ValueError("invalid behavior runtime sample count")
            model = OnlineActionModel.from_dict(raw["model"])
            restored[entity_id] = _EntityRecord(
                entity_id,
                raw["area"],
                raw["domain"],
                raw["supports_brightness"],
                raw["explicit_light_switch"],
                model,
                last_seen,
                removed,
                raw["last_access"],
                raw["sample_count"],
                None,
                last_human_action_at,
            )
        self._entities = restored
        self._counter = counter

    @staticmethod
    def _parse_stored_time(value: Any) -> datetime:
        if not isinstance(value, str):
            raise ValueError("invalid stored timestamp")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("stored timestamp must be timezone-aware")
        return parsed.astimezone(UTC)

    def _export(self) -> dict[str, Any]:
        return {
            "data_version": DATA_VERSION,
            "counter": self._counter,
            "entities": {
                entity_id: {
                    "entity_id": entity_id,
                    "area": record.area,
                    "domain": record.domain,
                    "supports_brightness": record.supports_brightness,
                    "explicit_light_switch": record.explicit_light_switch,
                    "last_seen_at": record.last_seen_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "removed_at": (
                        record.removed_at.isoformat().replace("+00:00", "Z")
                        if record.removed_at is not None
                        else None
                    ),
                    "last_access": record.last_access,
                    "sample_count": record.sample_count,
                    "last_human_action_at": (
                        record.last_human_action_at.isoformat().replace("+00:00", "Z")
                        if record.last_human_action_at is not None
                        else None
                    ),
                    "model": record.model.to_dict(),
                }
                for entity_id, record in sorted(self._entities.items())
            },
        }

    async def _persist(self, *, force: bool = False) -> None:
        if self._stopped and not force:
            return
        try:
            await self._store.async_save(self._export())
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior state save failed")
            self._last_load_reset_reason = "save_error"

    def _notify(self) -> None:
        if self._stopped or self._on_change is None:
            return
        snapshot = self.diagnostics
        try:
            result = self._on_change(snapshot)
            if inspect.isawaitable(result):
                self.hass.async_create_task(result)
        except Exception:
            _LOGGER.exception("Adaptive Lighting behavior diagnostics callback failed")

    def _fresh_context(self, context: Mapping[str, Any], now: datetime) -> bool:
        observed = _explicit_timestamp(context)
        if observed is None:
            return False
        return (
            0
            <= (now - observed).total_seconds()
            <= self.context_max_age.total_seconds()
        )

    @staticmethod
    def _home_state(context: Mapping[str, Any]) -> bool | None:
        return _as_bool(
            _first(context, "home_away", "home_state", "household_state", default=None)
        )

    @staticmethod
    def _presence_state(context: Mapping[str, Any]) -> bool | None:
        value = _first(context, "occupancy", "presence", "occupied", default=None)
        return _as_bool(value)

    def _recent_arrival(self, context: Mapping[str, Any], now: datetime) -> bool:
        value = _first(
            context, "recent_arrival", "arrival", "arrival_state", default=False
        )
        if _as_bool(value) is True or _text(value).replace("-", "_") in {
            "arrival",
            "recent_arrival",
            "just_arrived",
        }:
            return True
        timestamp = _first(context, "arrival_timestamp", "last_arrival", default=None)
        if isinstance(timestamp, datetime):
            return 0 <= (now - _utc(timestamp)).total_seconds() <= 15 * 60
        return False

    def _is_manual_hold(self, context: Mapping[str, Any]) -> bool:
        return _truthy(
            _first(context, "manual_hold", "manual_control", "hold", default=False)
        )

    def _is_suppressed_context(self, context: Mapping[str, Any]) -> bool:
        return self._is_good_night(
            _first(context, "semantic_routine", "routine", default="")
        )

    def _empty_dwell_sufficient(self, context: Mapping[str, Any]) -> bool:
        values: list[float] = []
        for key in ("occupancy_dwell_seconds", "presence_dwell_seconds"):
            value = context.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values.append(float(value))
        raw = _first(context, "state_dwell", "dwell", default={})
        if isinstance(raw, Mapping):
            for key in ("occupancy", "presence", "empty", "away"):
                value = raw.get(key)
                if isinstance(value, timedelta):
                    values.append(value.total_seconds())
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
        return max(values, default=0.0) >= self.empty_dwell.total_seconds()

    def _gate_reason(
        self, action: str, context: Mapping[str, Any], now: datetime
    ) -> str | None:
        if not self._fresh_context(context, now):
            return "stale_context"
        if self._is_safety(
            _first(context, "safety", "security", "emergency", "alarm", default=""),
        ) or _truthy(_first(context, "safety", "emergency", "alarm", default=False)):
            return "safety_context"
        if self._is_manual_hold(context):
            return "manual_hold"
        routine = _first(
            context, "semantic_routine", "routine", "routine_name", default=""
        )
        home = self._home_state(context)
        presence = self._presence_state(context)
        if action == "on":
            if home is not True:
                return "household_not_home"
            if self._is_good_night(routine) or _text(routine) in _SLEEP_WORDS:
                return "sleep_or_good_night"
            if presence is not True and not self._recent_arrival(context, now):
                return "no_occupancy_or_fresh_arrival"
            return None
        if self._is_good_night(routine):
            return None
        if home is False:
            return None
        if presence is False and self._empty_dwell_sufficient(context):
            return None
        return "off_requires_good_night_away_or_empty_dwell"

    def _current_state(self, entity_id: str) -> str | None:
        try:
            current = self.hass.states.get(entity_id)
        except Exception:
            current = None
        if current is not None:
            value = _state_value(current)
            if value in {STATE_ON, STATE_OFF}:
                self._states[entity_id] = value
                return value
            self._states.pop(entity_id, None)
            return None
        state = self._states.get(entity_id)
        return state if state in {STATE_ON, STATE_OFF} else None

    def _seed_candidate_state(self, entity_id: str) -> None:
        self._states.pop(entity_id, None)
        self._state_event_times.pop(entity_id, None)
        try:
            current = self.hass.states.get(entity_id)
        except Exception:
            current = None
        value = _state_value(current)
        if value in {STATE_ON, STATE_OFF}:
            self._states[entity_id] = value

    async def evaluate(
        self, *, now: datetime | None = None
    ) -> tuple[BehaviorProposal, ...]:
        """Synchronous-style shadow evaluation entry point (returned as a coroutine)."""
        return await self.async_evaluate(now=now)

    async def async_evaluate(
        self, *, now: datetime | None = None
    ) -> tuple[BehaviorProposal, ...]:
        """Evaluate in shadow and optionally execute only fully gated decisions."""
        if self._stopped:
            return ()
        at = _utc(now)
        changed = self._reconcile_candidates(at)
        context_mapping = dict(await self._get_context())
        if self._stopped:
            return ()
        proposals: list[BehaviorProposal] = []
        for entity_id, record in sorted(self._entities.items()):
            candidate = record.candidate
            if candidate is None or not candidate.available:
                continue
            pending_items = record.model.to_dict()["pending"]
            if pending_items:
                pending = pending_items[0]
                proposal = BehaviorProposal(
                    entity_id=entity_id,
                    area=candidate.area,
                    domain=candidate.domain,
                    action=pending["action"],
                    service=f"turn_{pending['action']}",
                    probability=pending["predicted_probability"],
                    confidence=0.0,
                    effective_support=0.0,
                    fresh=False,
                    active=False,
                    reason="pending_proposal",
                    proposal_id=pending["proposal_id"],
                    ready=False,
                    executed=False,
                )
                proposals.append(proposal)
                self._last_decisions.append(proposal.as_dict())
                self._last_decisions = self._last_decisions[-MAX_DIAGNOSTIC_DECISIONS:]
                record.last_access = self._next_access()
                continue
            current = self._current_state(entity_id)
            if current is None:
                self._last_rejection_reason = "unknown_entity_state"
                continue
            action = "on" if current == STATE_OFF else "off"
            entity_context = self._context_for_area(context_mapping, record.area)
            gate_reason = (
                "recent_manual_action"
                if record.last_human_action_at is not None
                and at < record.last_human_action_at + self.manual_action_hold
                else (
                    "manual_hold_entity"
                    if candidate.manual_hold
                    else self._gate_reason(action, entity_context, at)
                )
            )
            temporal = self._temporal_context(
                entity_context, entity_id, record.area, action
            )
            prediction = record.model.predict(temporal, action)
            reason = gate_reason
            if reason is None and prediction.not_ready:
                reason = (
                    prediction.not_ready_reasons[0]
                    if prediction.not_ready_reasons
                    else "not_ready"
                )
            if reason is None and prediction.probability < self.min_probability:
                reason = "probability_below_threshold"
            if reason is None and prediction.confidence < self.min_confidence:
                reason = "confidence_below_threshold"
            if (
                reason is None
                and prediction.effective_support < self.min_effective_support
            ):
                reason = "support_below_threshold"
            if reason is None and prediction.freshness < self.min_freshness:
                reason = "stale_model"
            if reason is None and prediction.suppressed:
                reason = "suppressed_after_correction"
            ready = reason is None
            executed = False
            proposal_id: str | None = None
            if ready and self._is_actuation_enabled():
                if self._stopped:
                    return tuple(proposals)
                try:
                    confirmed = await self._async_actuate(candidate, action, at)
                except Exception:
                    _LOGGER.exception(
                        "Adaptive Lighting behavior actuation failed for %s", entity_id
                    )
                    reason = "actuation_failed"
                else:
                    if self._stopped:
                        return tuple(proposals)
                    if not confirmed:
                        reason = "actuation_unconfirmed"
                    else:
                        proposal_id = record.model.register_proposal(temporal, action)
                        executed = True
                        changed = True
                        reason = "executed"
            proposal = BehaviorProposal(
                entity_id,
                candidate.area,
                candidate.domain,
                action,
                f"turn_{action}",
                prediction.probability,
                prediction.confidence,
                prediction.effective_support,
                prediction.freshness >= self.min_freshness,
                executed,
                reason or "ready_shadow",
                proposal_id,
                ready,
                executed,
            )
            proposals.append(proposal)
            self._last_decisions.append(proposal.as_dict())
            self._last_decisions = self._last_decisions[-MAX_DIAGNOSTIC_DECISIONS:]
            record.last_access = self._next_access()
        if proposals or changed:
            await self._persist()
            if self._stopped:
                return tuple(proposals)
            self._notify()
        return tuple(proposals)

    async def _async_actuate(
        self,
        candidate: CandidateRecord,
        action: str,
        at: datetime,
    ) -> bool:
        if self._stopped:
            return False
        if candidate.domain not in _LIGHT_DOMAINS or (
            candidate.domain == "switch" and not candidate.explicit_light_switch
        ):
            raise ValueError(
                "candidate is not an explicitly permitted light-like actuator"
            )
        context = Context()
        context_id = _context_id(context)
        if context_id is not None:
            expiry_base = max(at, _utc(None))
            self._own_contexts[context_id] = _TrackedContext(
                context_id,
                _parent_context_id(context),
                expiry_base + timedelta(minutes=2),
            )
        result = self.hass.services.async_call(
            candidate.domain,
            f"turn_{action}",
            {"entity_id": candidate.entity_id},
            blocking=True,
            context=context,
        )
        if inspect.isawaitable(result):
            await result
        if self._stopped:
            return False
        try:
            current = self.hass.states.get(candidate.entity_id)
        except Exception:
            return False
        return _state_value(current) == action


AdaptiveLightingBehaviorRuntime = BehaviorRuntimeAdapter
BehaviorRuntime = BehaviorRuntimeAdapter

__all__ = [
    "DATA_VERSION",
    "DEFAULT_STORAGE_KEY",
    "MAX_ENTITIES",
    "STORAGE_VERSION",
    "AdaptiveLightingBehaviorRuntime",
    "BehaviorProposal",
    "BehaviorRuntime",
    "BehaviorRuntimeAdapter",
    "Candidate",
    "CandidateProvider",
    "CandidateRecord",
    "ContextProvider",
]
