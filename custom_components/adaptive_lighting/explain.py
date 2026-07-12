"""Structured and deterministic explanations for pure policy decisions."""

from dataclasses import dataclass

from .context import InputProvenance
from .intent import Intent, RejectedAlternative
from .policy import PolicyDecision


@dataclass(frozen=True, slots=True)
class DecisionExplanation:
    """A presentation-neutral explanation suitable for diagnostics or logs."""

    summary: str
    intent: Intent
    confidence: float
    reasons: tuple[str, ...]
    rejected_alternatives: tuple[RejectedAlternative, ...]
    input_provenance: tuple[InputProvenance, ...]

    @property
    def active_intent(self) -> Intent:
        """Alias used by diagnostic consumers."""
        return self.intent

    def as_text(self) -> str:
        """Render a stable one-line explanation without timestamps or HA state."""
        reasons = "; ".join(self.reasons)
        rejected = ", ".join(
            alternative.intent.value for alternative in self.rejected_alternatives
        ) or "none"
        return (
            f"{self.summary} Reasons: {reasons}. "
            f"Rejected alternatives: {rejected}. "
            f"Confidence: {self.confidence:.2f}."
        )

    def __str__(self) -> str:
        """Return the stable human-readable rendering."""
        return self.as_text()


def explain_decision(decision: PolicyDecision) -> DecisionExplanation:
    """Convert a decision into a structured explanation without re-evaluating it."""
    target = "hold" if decision.brightness_target is None else f"{decision.brightness_target:g}%"
    companion = (
        "hold" if decision.companion_on is None else "on" if decision.companion_on else "off"
    )
    return DecisionExplanation(
        summary=(
            f"Use {decision.intent.value} lighting at {target}; "
            f"companion recommendation is {companion}"
        ),
        intent=decision.intent,
        confidence=decision.confidence,
        reasons=decision.reasons,
        rejected_alternatives=decision.rejected_alternatives,
        input_provenance=decision.input_provenance,
    )


def explain(decision: PolicyDecision) -> DecisionExplanation:
    """Concise alias for :func:`explain_decision`."""
    return explain_decision(decision)


format_explanation = explain_decision


__all__ = [
    "DecisionExplanation",
    "explain",
    "explain_decision",
    "format_explanation",
]
