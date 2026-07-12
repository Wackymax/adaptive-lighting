"""Home-Assistant-independent context types for adaptive-lighting decisions.

This module deliberately contains data only.  A Home Assistant adapter can turn
states, events, and configuration into these values, while tests and replay
tools can exercise the decision engine without importing Home Assistant.
"""

from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ContextSignal(Generic[T]):
    """A normalized observation with freshness, quality, and provenance.

    ``available`` is intentionally separate from the value.  ``False`` is a
    meaningful observation (for example, ``occupancy=False``), whereas an
    unavailable occupancy sensor must not be treated as vacancy.  The policy
    uses :meth:`usable` before acting on any signal.
    """

    value: T | None = None
    source: str = "unknown"
    available: bool = True
    confidence: float = 1.0
    age_seconds: float | None = None
    max_age_seconds: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        """Normalize quality numbers so callers cannot create unsafe metadata."""
        confidence = self.confidence
        if not isfinite(confidence):
            confidence = 0.0
        object.__setattr__(self, "confidence", min(1.0, max(0.0, confidence)))

        for name in ("age_seconds", "max_age_seconds"):
            value = getattr(self, name)
            if value is not None:
                value = None if not isfinite(value) else max(0.0, value)
                object.__setattr__(self, name, value)

        # A signal with no value, or with an explicitly false availability bit,
        # is missing.  Normalizing confidence to zero prevents accidental use
        # by a future policy rule that forgets to check ``available``.
        if self.value is None or not self.available:
            object.__setattr__(self, "available", False)
            object.__setattr__(self, "confidence", 0.0)

        if not self.source:
            object.__setattr__(self, "source", "unknown")

    @property
    def fresh(self) -> bool:
        """Whether this signal is within its optional freshness limit."""
        return self.max_age_seconds is None or (
            self.age_seconds is not None
            and self.age_seconds <= self.max_age_seconds
        )

    def usable(self) -> bool:
        """Return whether this observation is safe for an automatic decision."""
        return self.available and self.value is not None and self.confidence > 0 and self.fresh


def signal(
    value: T,
    *,
    source: str,
    confidence: float = 1.0,
    age_seconds: float | None = None,
    max_age_seconds: float | None = None,
    detail: str = "",
) -> ContextSignal[T]:
    """Build an available signal with explicit provenance.

    The factory makes call sites readable and avoids accidentally omitting the
    source of a value that can later affect a light.
    """
    return ContextSignal(
        value=value,
        source=source,
        confidence=confidence,
        age_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        detail=detail,
    )


def unavailable(*, source: str = "unavailable", detail: str = "") -> ContextSignal[T]:
    """Build an explicitly unavailable signal for a missing input."""
    return ContextSignal(
        value=None,
        source=source,
        available=False,
        confidence=0.0,
        detail=detail,
    )


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """The auditable part of a signal retained on a policy decision."""

    name: str
    source: str
    available: bool
    confidence: float
    age_seconds: float | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable normalized inputs for one policy evaluation.

    The aliases (``sleep_mode``, ``video_playing``, and so on) are intentional:
    adapters can map their native vocabulary here without leaking that
    vocabulary into the pure intent engine.  Intent resolution considers the
    aliases in a fixed order, so the same snapshot always produces the same
    decision.
    """

    emergency: ContextSignal[bool] = field(default_factory=unavailable)
    manual: ContextSignal[bool] = field(default_factory=unavailable)
    manual_hold: ContextSignal[bool] = field(default_factory=unavailable)
    sleep: ContextSignal[bool] = field(default_factory=unavailable)
    sleep_mode: ContextSignal[bool] = field(default_factory=unavailable)
    night_path: ContextSignal[bool] = field(default_factory=unavailable)
    night_path_active: ContextSignal[bool] = field(default_factory=unavailable)
    task: ContextSignal[bool] = field(default_factory=unavailable)
    task_mode: ContextSignal[bool] = field(default_factory=unavailable)
    video: ContextSignal[bool] = field(default_factory=unavailable)
    video_playing: ContextSignal[bool] = field(default_factory=unavailable)
    arrival: ContextSignal[bool] = field(default_factory=unavailable)
    arrival_active: ContextSignal[bool] = field(default_factory=unavailable)
    ambient: ContextSignal[bool] = field(default_factory=unavailable)
    vacant: ContextSignal[bool] = field(default_factory=unavailable)
    occupancy: ContextSignal[bool] = field(default_factory=unavailable)
    motion: ContextSignal[bool] = field(default_factory=unavailable)
    daylight: ContextSignal[float] = field(default_factory=unavailable)
    illuminance: ContextSignal[float] = field(default_factory=unavailable)
    ambient_brightness: ContextSignal[float] = field(default_factory=unavailable)
    requested_brightness: ContextSignal[float] = field(default_factory=unavailable)
    current_brightness: ContextSignal[float] = field(default_factory=unavailable)
    companion_on: ContextSignal[bool] = field(default_factory=unavailable)
    # These adapter-facing aliases keep the pure seam tolerant of different
    # event vocabularies without making the policy depend on entity objects.
    media_playing: ContextSignal[bool] = field(default_factory=unavailable)
    semantic_intent: ContextSignal[str] = field(default_factory=unavailable)
    intent_hint: ContextSignal[str] = field(default_factory=unavailable)

    def __post_init__(self) -> None:
        """Normalize simple adapter values into typed, immutable signals.

        A Home Assistant adapter may initially pass a boolean or string while
        building a snapshot.  Accepting and immediately wrapping that scalar
        keeps this boundary additive, while still ensuring all policy code sees
        the same ``ContextSignal`` contract.  Unknown or ``None`` values remain
        unavailable and therefore cannot trigger an automatic action.
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, ContextSignal):
                continue
            normalized = (
                unavailable(source="adapter")
                if value is None
                else signal(value, source="adapter")
            )
            object.__setattr__(self, item.name, normalized)

    def signals(self) -> tuple[tuple[str, ContextSignal[object]], ...]:
        """Return all inputs in declaration order for deterministic auditing."""
        return (
            ("emergency", self.emergency),
            ("manual", self.manual),
            ("manual_hold", self.manual_hold),
            ("sleep", self.sleep),
            ("sleep_mode", self.sleep_mode),
            ("night_path", self.night_path),
            ("night_path_active", self.night_path_active),
            ("task", self.task),
            ("task_mode", self.task_mode),
            ("video", self.video),
            ("video_playing", self.video_playing),
            ("arrival", self.arrival),
            ("arrival_active", self.arrival_active),
            ("ambient", self.ambient),
            ("vacant", self.vacant),
            ("occupancy", self.occupancy),
            ("motion", self.motion),
            ("daylight", self.daylight),
            ("illuminance", self.illuminance),
            ("ambient_brightness", self.ambient_brightness),
            ("requested_brightness", self.requested_brightness),
            ("current_brightness", self.current_brightness),
            ("companion_on", self.companion_on),
            ("media_playing", self.media_playing),
            ("semantic_intent", self.semantic_intent),
            ("intent_hint", self.intent_hint),
        )

    @property
    def input_provenance(self) -> tuple[InputProvenance, ...]:
        """Expose every input, including unavailable ones, for safe diagnosis."""
        return tuple(
            InputProvenance(
                name=name,
                source=value.source,
                available=value.usable(),
                confidence=value.confidence,
                age_seconds=value.age_seconds,
                detail=value.detail,
            )
            for name, value in self.signals()
        )


__all__ = [
    "ContextSignal",
    "ContextSnapshot",
    "InputProvenance",
    "signal",
    "unavailable",
]
