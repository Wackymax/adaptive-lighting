"""Pure intent classification for context-aware adaptive lighting."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType

from .context import ContextSignal, ContextSnapshot, InputProvenance


class Intent(str, Enum):
    """The supported semantic intents, ordered by safety and user control."""

    EMERGENCY = "emergency"
    MANUAL = "manual"
    SLEEP = "sleep"
    NIGHT_PATH = "night_path"
    TASK = "task"
    VIDEO = "video"
    ARRIVAL = "arrival"
    AMBIENT = "ambient"
    VACANT = "vacant"


class IntentPriority(IntEnum):
    """Explicit priority values; larger values win."""

    VACANT = 1
    AMBIENT = 2
    ARRIVAL = 3
    VIDEO = 4
    TASK = 5
    NIGHT_PATH = 6
    SLEEP = 7
    MANUAL = 8
    EMERGENCY = 9


INTENT_PRIORITY = MappingProxyType(
    {
        Intent.EMERGENCY: IntentPriority.EMERGENCY,
        Intent.MANUAL: IntentPriority.MANUAL,
        Intent.SLEEP: IntentPriority.SLEEP,
        Intent.NIGHT_PATH: IntentPriority.NIGHT_PATH,
        Intent.TASK: IntentPriority.TASK,
        Intent.VIDEO: IntentPriority.VIDEO,
        Intent.ARRIVAL: IntentPriority.ARRIVAL,
        Intent.AMBIENT: IntentPriority.AMBIENT,
        Intent.VACANT: IntentPriority.VACANT,
    },
)


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    """One candidate and the evidence that made it active or unavailable."""

    intent: Intent
    active: bool
    available: bool
    confidence: float
    priority: int
    reason: str
    provenance: tuple[InputProvenance, ...]


@dataclass(frozen=True, slots=True)
class RejectedAlternative:
    """An intent that was considered but lost to a higher-priority intent."""

    intent: Intent
    priority: int
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class IntentResolution:
    """Deterministic result of resolving all intent candidates."""

    intent: Intent
    priority: int
    confidence: float
    reason: str
    candidates: tuple[IntentCandidate, ...]
    rejected_alternatives: tuple[RejectedAlternative, ...]

    @property
    def active_intent(self) -> Intent:
        """Alias that reads naturally at policy call sites."""
        return self.intent


_SIGNAL_NAMES: dict[Intent, tuple[str, ...]] = {
    Intent.EMERGENCY: ("emergency",),
    Intent.MANUAL: ("manual", "manual_hold"),
    Intent.SLEEP: ("sleep", "sleep_mode"),
    Intent.NIGHT_PATH: ("night_path", "night_path_active"),
    Intent.TASK: ("task", "task_mode"),
    Intent.VIDEO: ("video", "video_playing", "media_playing"),
    Intent.ARRIVAL: ("arrival", "arrival_active"),
    Intent.AMBIENT: ("ambient",),
    Intent.VACANT: ("vacant",),
}

# Semantic aliases are deliberately narrow.  In particular, ``away`` is not
# proof that a room is vacant, and ``prelight`` is not proof of a night path.
# Those ambiguous labels may inform a future predictor but must not authorize
# current-room actuation in this policy.
_INTENT_HINTS: dict[Intent, frozenset[str]] = {
    Intent.EMERGENCY: frozenset({"emergency", "alarm"}),
    Intent.MANUAL: frozenset({"manual"}),
    Intent.SLEEP: frozenset({"sleep"}),
    Intent.NIGHT_PATH: frozenset({"night", "night_path"}),
    Intent.TASK: frozenset({"task", "focus", "work"}),
    Intent.VIDEO: frozenset({"video", "movie", "cinema"}),
    Intent.ARRIVAL: frozenset({"arrival", "welcome"}),
    Intent.AMBIENT: frozenset({"ambient", "relax", "chill"}),
    Intent.VACANT: frozenset({"vacant"}),
}


def _provenance(name: str, value: ContextSignal[object]) -> InputProvenance:
    return InputProvenance(
        name=name,
        source=value.source,
        available=value.usable(),
        confidence=value.confidence,
        age_seconds=value.age_seconds,
        detail=value.detail,
    )


def _best_boolean_signal(
    snapshot: ContextSnapshot,
    names: Iterable[str],
) -> tuple[str, ContextSignal[bool]] | None:
    """Find the first active alias, otherwise the best usable false signal."""
    available: list[tuple[str, ContextSignal[bool]]] = []
    for name in names:
        value = getattr(snapshot, name)
        if value.usable():
            available.append((name, value))
            if value.value is True:
                return name, value
    return available[0] if available else None


def _candidate(
    intent: Intent,
    snapshot: ContextSnapshot,
) -> IntentCandidate:
    names = _SIGNAL_NAMES[intent]
    selected = _best_boolean_signal(snapshot, names)

    if selected is None:
        # Semantic inputs are advisory labels, but when they explicitly name
        # an intent they are still stronger evidence than the ambient fallback.
        # They never create an intent for unrelated words such as "adaptive".
        for name in ("intent_hint", "semantic_intent"):
            value = getattr(snapshot, name)
            if (
                value.usable()
                and isinstance(value.value, str)
                and value.value in _INTENT_HINTS[intent]
            ):
                selected = (name, value)
                break

    # Vacancy can be derived only from a *known* false occupancy signal.  An
    # unavailable occupancy sensor is never permission to turn lights off.
    if intent is Intent.VACANT and selected is None:
        occupancy = snapshot.occupancy
        if occupancy.usable() and occupancy.value is False:
            selected = ("occupancy", occupancy)

    if selected is None:
        return IntentCandidate(
            intent=intent,
            active=False,
            available=False,
            confidence=0.0,
            priority=int(INTENT_PRIORITY[intent]),
            reason="signal unavailable or stale",
            provenance=(),
        )

    name, value = selected
    active = value.value is True or (
        isinstance(value.value, str)
        and value.value in _INTENT_HINTS[intent]
    )
    if intent is Intent.VACANT and name == "occupancy":
        active = value.value is False
    reason = (
        f"{name} reported {'active' if active else 'inactive'}"
        if active
        else f"{name} reported inactive"
    )
    return IntentCandidate(
        intent=intent,
        active=active,
        available=True,
        confidence=value.confidence,
        priority=int(INTENT_PRIORITY[intent]),
        reason=reason,
        provenance=(_provenance(name, value),),
    )


def resolve_intent(snapshot: ContextSnapshot) -> IntentResolution:
    """Select the highest-priority usable intent from ``snapshot``.

    Safety and control are encoded here rather than left to incidental rule
    ordering: emergency always beats manual, manual always beats sleep, and a
    missing signal never becomes an active intent.  Ambient is the explicit,
    safe fallback when no stronger intent is currently evidenced.
    """
    candidates = tuple(_candidate(intent, snapshot) for intent in Intent)
    active = tuple(candidate for candidate in candidates if candidate.active)

    if active:
        winner = max(active, key=lambda candidate: candidate.priority)
    else:
        ambient = next(candidate for candidate in candidates if candidate.intent is Intent.AMBIENT)
        if not ambient.available:
            ambient = IntentCandidate(
                intent=Intent.AMBIENT,
                active=True,
                available=False,
                confidence=0.25,
                priority=int(INTENT_PRIORITY[Intent.AMBIENT]),
                reason="no stronger intent was available; using safe ambient fallback",
                provenance=(),
            )
        else:
            ambient = IntentCandidate(
                intent=Intent.AMBIENT,
                active=True,
                available=True,
                confidence=ambient.confidence,
                priority=ambient.priority,
                reason="ambient signal is active and no stronger intent won",
                provenance=ambient.provenance,
            )
        winner = ambient
        candidates = tuple(
            ambient if candidate.intent is Intent.AMBIENT else candidate
            for candidate in candidates
        )

    rejected = tuple(
        RejectedAlternative(
            intent=candidate.intent,
            priority=candidate.priority,
            confidence=candidate.confidence,
            reason=(
                f"{candidate.reason}; lower priority than {winner.intent.value}"
                if candidate.active
                else f"{candidate.intent.value} not selected: {candidate.reason}"
            ),
        )
        for candidate in sorted(
            (candidate for candidate in candidates if candidate.intent is not winner.intent),
            key=lambda candidate: (-candidate.priority, candidate.intent.value),
        )
    )
    return IntentResolution(
        intent=winner.intent,
        priority=winner.priority,
        confidence=winner.confidence,
        reason=winner.reason,
        candidates=candidates,
        rejected_alternatives=rejected,
    )


select_intent = resolve_intent


__all__ = [
    "INTENT_PRIORITY",
    "Intent",
    "IntentCandidate",
    "IntentPriority",
    "IntentResolution",
    "RejectedAlternative",
    "resolve_intent",
    "select_intent",
]
