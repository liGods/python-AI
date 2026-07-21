"""Structured score inputs retained from the legacy candidate record."""

from __future__ import annotations

from typing import Any

from .candidate import CandidateDecision


def score_components(candidate: CandidateDecision, labels: tuple[str, ...]) -> dict[str, Any]:
    """Expose every legacy score component without changing sort semantics."""

    values = candidate.score[:-1]
    return {
        label: values[index] if index < len(values) else None
        for index, label in enumerate(labels)
    }


def projection_features(candidate: CandidateDecision) -> dict[str, Any]:
    projection = candidate.projection
    return {
        "expected_remaining_turns": projection.expected_remaining_turns,
        "worst_remaining_turns": projection.worst_remaining_turns,
        "expected_remaining_cards": projection.expected_remaining_cards,
        "worst_remaining_cards": projection.worst_remaining_cards,
        "enemy_finish_risk": projection.enemy_finish_risk,
        "control_card_cost": projection.control_card_cost,
    }
