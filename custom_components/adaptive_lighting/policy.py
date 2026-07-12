"""Pure, bounded target policy for context-aware adaptive lighting."""

from dataclasses import dataclass
from math import isfinite

from .context import ContextSnapshot, InputProvenance
from .intent import (
    Intent,
    IntentResolution,
    RejectedAlternative,
    resolve_intent,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a percentage while tolerating hostile or malformed inputs."""
    if not isfinite(value):
        return lower
    return min(upper, max(lower, value))


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """User-tunable targets and hard safety caps, expressed as percentages."""

    min_brightness: float = 1.0
    max_brightness: float = 100.0
    emergency_brightness: float = 100.0
    task_brightness: float = 80.0
    video_brightness: float = 15.0
    arrival_brightness: float = 30.0
    ambient_brightness: float = 30.0
    vacant_brightness: float = 0.0
    sleep_brightness: float = 1.0
    sleep_max_brightness: float = 10.0
    night_path_brightness: float = 10.0
    night_path_max_brightness: float = 10.0
    # Short aliases are useful to integrations that call these values caps.
    sleep_cap: float | None = None
    night_path_cap: float | None = None
    # Compatibility names for an adapter that supplies upper caps rather than
    # complete intent targets.  They are folded into the target/cap fields once
    # and do not alter the immutable decision contract.
    task_brightness_cap: float | None = None
    ambient_brightness_cap: float | None = None
    video_brightness_cap: float | None = None
    night_brightness_cap: float | None = None
    prelight_brightness_cap: float | None = None

    def __post_init__(self) -> None:
        """Normalize configuration once, keeping every later decision pure."""
        percentage_fields = (
            "min_brightness",
            "max_brightness",
            "emergency_brightness",
            "task_brightness",
            "video_brightness",
            "arrival_brightness",
            "ambient_brightness",
            "vacant_brightness",
            "sleep_brightness",
            "sleep_max_brightness",
            "night_path_brightness",
            "night_path_max_brightness",
        )
        for name in percentage_fields:
            value = getattr(self, name)
            if not isfinite(value):
                value = 0.0
            object.__setattr__(self, name, _clamp(value))

        # A reversed general range is repaired conservatively to the lower
        # endpoint.  This avoids an exception in a safety path and guarantees a
        # target can never escape the configured [min, max] interval.
        if self.max_brightness < self.min_brightness:
            object.__setattr__(self, "min_brightness", self.max_brightness)

        for name, cap_name in (
            ("sleep_cap", "sleep_max_brightness"),
            ("night_path_cap", "night_path_max_brightness"),
        ):
            cap = getattr(self, name)
            if cap is not None:
                if not isfinite(cap):
                    cap = 0.0
                cap = _clamp(cap)
                object.__setattr__(self, name, cap)
                object.__setattr__(self, cap_name, min(getattr(self, cap_name), cap))

        for name in (
            "task_brightness_cap",
            "ambient_brightness_cap",
            "video_brightness_cap",
            "night_brightness_cap",
            "prelight_brightness_cap",
        ):
            cap = getattr(self, name)
            if cap is not None:
                if not isfinite(cap):
                    cap = 0.0
                object.__setattr__(self, name, _clamp(cap))

        for target_name, cap_name in (
            ("task_brightness", "task_brightness_cap"),
            ("ambient_brightness", "ambient_brightness_cap"),
            ("video_brightness", "video_brightness_cap"),
        ):
            cap = getattr(self, cap_name)
            if cap is not None:
                object.__setattr__(self, target_name, min(getattr(self, target_name), cap))
        if self.night_brightness_cap is not None:
            object.__setattr__(
                self,
                "night_path_max_brightness",
                min(self.night_path_max_brightness, self.night_brightness_cap),
            )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A complete, immutable policy result suitable for preview or execution."""

    intent: Intent
    priority: int
    brightness_target: float | None
    companion_on: bool | None
    confidence: float
    reasons: tuple[str, ...]
    rejected_alternatives: tuple[RejectedAlternative, ...]
    input_provenance: tuple[InputProvenance, ...]
    should_apply: bool

    @property
    def target_brightness(self) -> float | None:
        """Common alias for callers that use target-first terminology."""
        return self.brightness_target

    @property
    def target_brightness_pct(self) -> float | None:
        """Percentage alias for adapters that expose explicit units."""
        return self.brightness_target

    @property
    def active_intent(self) -> Intent:
        """Alias for observability adapters."""
        return self.intent

    @property
    def companion_recommendation(self) -> bool | None:
        """``None`` means hold the current companion state."""
        return self.companion_on

    @property
    def provenance(self) -> tuple[InputProvenance, ...]:
        """Short alias for the auditable input list."""
        return self.input_provenance

    @property
    def reason(self) -> str:
        """Flatten structured reasons for simple state-attribute consumers."""
        return "; ".join(self.reasons)


_SIMPLE_TARGETS: dict[Intent, tuple[str, str]] = {
    Intent.EMERGENCY: (
        "emergency_brightness",
        "emergency target is bounded by the global brightness range",
    ),
    Intent.TASK: ("task_brightness", "task target favors clear work visibility"),
    Intent.VIDEO: (
        "video_brightness",
        "video target keeps peripheral light restrained",
    ),
    Intent.ARRIVAL: (
        "arrival_brightness",
        "arrival target provides a low welcome level",
    ),
    Intent.VACANT: (
        "vacant_brightness",
        "vacant target is the configured empty-room level",
    ),
}


def _signal_value(snapshot: ContextSnapshot, name: str) -> float | None:
    value = getattr(snapshot, name)
    if value.usable() and isinstance(value.value, (int, float)):
        return float(value.value)
    return None


def _bounded_general(value: float, config: PolicyConfig) -> float:
    return _clamp(value, config.min_brightness, config.max_brightness)


def _bounded_capped(value: float, cap: float, config: PolicyConfig) -> float:
    # Caps are upper safety bounds.  Do not re-apply min_brightness after a cap:
    # a configured 5% night-path cap must be allowed even if the ambient
    # minimum is 20%.
    return _clamp(value, 0.0, min(config.max_brightness, cap))


def _target_for(
    intent: Intent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[float | None, str]:
    target: float | None
    reason: str
    if intent is Intent.MANUAL:
        requested = _signal_value(snapshot, "requested_brightness")
        current = _signal_value(snapshot, "current_brightness")
        if requested is not None:
            target = _bounded_general(requested, config)
            reason = "manual brightness is retained without adaptation"
        elif current is not None:
            target = _bounded_general(current, config)
            reason = "current brightness is retained without adaptation"
        else:
            target = None
            reason = "manual hold is active and no brightness value was available"
    elif intent is Intent.SLEEP:
        target = _bounded_capped(config.sleep_brightness, config.sleep_max_brightness, config)
        reason = f"sleep target is constrained by the {config.sleep_max_brightness:g}% sleep cap"
    elif intent is Intent.NIGHT_PATH:
        target = _bounded_capped(
            config.night_path_brightness,
            config.night_path_max_brightness,
            config,
        )
        reason = f"night-path target is constrained by the {config.night_path_max_brightness:g}% cap"
    elif intent is Intent.AMBIENT:
        ambient = _signal_value(snapshot, "ambient_brightness")
        if ambient is not None:
            target = _bounded_general(ambient, config)
            reason = "ambient target came from the available room estimate"
        else:
            target = _bounded_general(config.ambient_brightness, config)
            reason = "ambient target came from the configured fallback"
    else:
        field_name, reason = _SIMPLE_TARGETS[intent]
        target = _bounded_general(getattr(config, field_name), config)
    return target, reason


def _companion_for(
    intent: Intent,
    snapshot: ContextSnapshot,
) -> tuple[bool | None, str]:
    recommendation: bool | None
    reason: str
    if intent is Intent.EMERGENCY:
        recommendation = True
        reason = "emergency is permitted to turn the companion on immediately"
    elif intent is Intent.MANUAL:
        recommendation = None
        reason = "manual control owns the companion; policy will not fight it"
    elif intent is Intent.VACANT:
        recommendation = False
        reason = "confirmed vacancy permits a safe turn-off recommendation"
    elif intent in {
        Intent.SLEEP,
        Intent.NIGHT_PATH,
        Intent.TASK,
        Intent.VIDEO,
        Intent.ARRIVAL,
    }:
        recommendation = True
        reason = f"explicit {intent.value} context permits the companion on"
    elif snapshot.occupancy.usable():
        occupancy = snapshot.occupancy
        if occupancy.value is True:
            recommendation = True
            reason = "occupancy is available and confirms the ambient light may be on"
        else:
            recommendation = False
            reason = "occupancy is available and confirms the room is vacant"
    else:
        recommendation = None
        reason = "occupancy is unavailable; preserving the companion state is safer"
    return recommendation, reason


def _confidence(
    resolution: IntentResolution,
    snapshot: ContextSnapshot,
    intent: Intent,
) -> float:
    confidence = resolution.confidence
    if intent is Intent.AMBIENT and not snapshot.occupancy.usable():
        # Ambient brightness may still be adjusted for an already-on light, but
        # an unavailable occupancy input makes an automatic turn-on unsafe.
        confidence = min(confidence, 0.35)
    return _clamp(confidence, 0.0, 1.0)


def evaluate_policy(
    snapshot: ContextSnapshot,
    config: PolicyConfig | None = None,
) -> PolicyDecision:
    """Evaluate one snapshot without side effects or Home Assistant imports."""
    config = config or PolicyConfig()
    resolution = resolve_intent(snapshot)
    target, target_reason = _target_for(resolution.intent, snapshot, config)
    companion_on, companion_reason = _companion_for(resolution.intent, snapshot)
    confidence = _confidence(resolution, snapshot, resolution.intent)

    reasons = (
        f"selected {resolution.intent.value} intent at priority {resolution.priority}",
        resolution.reason,
        target_reason,
        companion_reason,
    )
    should_apply = resolution.intent is not Intent.MANUAL and target is not None

    return PolicyDecision(
        intent=resolution.intent,
        priority=resolution.priority,
        brightness_target=target,
        companion_on=companion_on,
        confidence=confidence,
        reasons=reasons,
        rejected_alternatives=resolution.rejected_alternatives,
        input_provenance=snapshot.input_provenance,
        should_apply=should_apply,
    )


def decide(
    snapshot: ContextSnapshot,
    config: PolicyConfig | None = None,
) -> PolicyDecision:
    """Concise public spelling for :func:`evaluate_policy`."""
    return evaluate_policy(snapshot, config)


class TargetPolicy:
    """Reusable configured policy object for adapters and replay harnesses."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        """Store the normalized configuration used for every evaluation."""
        self.config = config or PolicyConfig()

    def decide(self, snapshot: ContextSnapshot) -> PolicyDecision:
        """Evaluate a snapshot with this immutable configuration."""
        return evaluate_policy(snapshot, self.config)


Policy = TargetPolicy


__all__ = [
    "Policy",
    "PolicyConfig",
    "PolicyDecision",
    "TargetPolicy",
    "decide",
    "evaluate_policy",
]
